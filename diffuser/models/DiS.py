import torch
import torch.nn as nn
import math
import einops

from functools import partial
from torch import Tensor
from typing import Optional
import copy

from torch.distributions import Bernoulli
from .helpers import SinusoidalPosEmb

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, skip=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, skip=None, **mixer_kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        skip=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        skip=skip
    )
    block.layer_idx = layer_idx
    return block


class TemporalMamba(nn.Module):
    def __init__(
        self,
        horizon = 56,
        input_dim = 37,
        dim=128,
        depth=3,
        returns_condition=False,
        condition_dropout=0.1,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        skip=True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.proj_up = nn.Linear(input_dim, dim)
        self.proj_down = nn.Linear(dim, input_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(4,dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)

        self.time_dim = dim
        self.returns_dim = dim

        # Block configurations
        d_intermediate = 2 * dim
        factory_kwargs = {"device": device, "dtype": dtype}

        self.downs = nn.ModuleList([
            create_block(
                d_model=dim,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        self.mid_block = create_block(
            d_model=dim,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=depth,
            **factory_kwargs,
        )

        self.ups = nn.ModuleList([
            create_block(
                d_model=dim,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i + depth + 1,
                skip=skip,
                **factory_kwargs,
            )
            for i in range(depth)
        ])


    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False, inference_params=None):

        x = self.proj_up(x)  # (B,L,37) -> (B,L,D)
        B,L,D = x.shape

        time_embed = self.time_mlp(time)  # (B,) -> (B, D)
        time_embed = time_embed.unsqueeze(1).expand(-1,L,-1)

        x = x + time_embed

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            returns_embed = returns_embed.unsqueeze(1).expand(-1,L,-1)
            x = x + returns_embed

        residual = None
        hidden_states = x

        h = []

        for block in self.downs:
            hidden_states, residual = block(hidden_states, residual, inference_params=inference_params)
            h.append(hidden_states)

        hidden_states, residual = self.mid_block(hidden_states, residual, inference_params=inference_params)

        for block in self.ups:
            hidden_states, residual = block(hidden_states, residual, inference_params=inference_params, skip=h.pop())

        x = hidden_states

        # final layer
        # if returns is not None:
        #     x = self.final_layer(x, c=returns_embed)
        # else:
        #     x = self.final_layer(x)

        x = self.proj_down(x)

        return x