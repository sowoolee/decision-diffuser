import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange
from typing import Optional
from functools import partial
import copy

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

class MambaR(Mamba):
    def forward(self, hidden_states, inference_params=None, scale_factor=1.0):
        """
        Custom Mamba with slow path and scale_factor applied to dt.

        Args:
            hidden_states: Input tensor (B, L, D).
            inference_params: Optional inference-related parameters.
            scale_factor: Scaling factor applied to dt in slow path.

        Returns:
            Tensor of the same shape as hidden_states.
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # Use the step function for inference (1 token at a time)
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Apply scale factor to dt
        dt = scale_factor * (self.dt_proj.weight @ dt.t())
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        # Rearrange B and C
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # Compute SSM using selective scan
        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        # Output projection
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
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

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, step_scale=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
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
        if step_scale == None:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, scale_factor=step_scale, **mixer_kwargs)

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

    # 선택적 SSM Layer
    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg)
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer == "MambaR":
            mixer_cls = partial(
                MambaR,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
        elif ssm_layer in ["Mamba1", "Mamba2"]:
            mixer_cls = partial(
                Mamba2 if ssm_layer == "Mamba2" else Mamba,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
        else:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}")
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
        # skip=skip
    )
    block.layer_idx = layer_idx
    return block


def main():
    d_model = 37
    d_state = 16
    d_intermediate = 2 * d_model
    seq_len = 56
    batch_size = 1

    # Params
    norm_epsilon = 1e-5
    fused_add_norm = True  # LayerNorm과 Residual 결합 비활성화
    residual_in_fp32 = False  # Residual FP32 유지 비활성화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Dummy input
    dummy_input = torch.randn(batch_size, seq_len, d_model, dtype=dtype).to(device)  # (B, L, D)

    print("\n===== Testing with Original Mamba Block =====")
    # Original Mamba Block
    original_mamba_block = Block(
        dim=d_model,
        mixer_cls=partial(Mamba, d_state=d_state, device=device),
        mlp_cls=partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, device=device),
        norm_cls=partial(nn.LayerNorm, eps=norm_epsilon, device=device),
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )

    residual = None  # 초기 Residual은 None으로 설정
    output, residual = original_mamba_block(dummy_input, residual)  # Forward 실행

    print("Original Mamba Block Output Shape:", output.shape)  # (B, L, D)
    print("Original Mamba Block Residual Shape:", residual.shape if residual is not None else None)

    print("\n===== Testing with Custom Mamba-R Block =====")
    # Custom Mamba-R Block
    custom_mamba_block = Block(
        dim=d_model,
        mixer_cls=partial(MambaR, d_state=d_state, device=device),
        mlp_cls=partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, device=device),
        norm_cls=partial(nn.LayerNorm, eps=norm_epsilon, device=device),
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )

    residual = None  # 초기 Residual은 None으로 설정
    # Different scale factors for testing
    scale_factors = [0.5, 1.0, 2.0]
    for scale in scale_factors:
        print(f"\nTesting with scale_factor={scale}")
        output, residual = custom_mamba_block(dummy_input, residual, step_scale=scale)  # Forward 실행
        print("Custom Mamba-R Block Output Shape:", output.shape)  # (B, L, D)
        print("Custom Mamba-R Block Residual Shape:", residual.shape if residual is not None else None)


def main2():
    d_model = 37
    d_state = 16
    d_intermediate = 2 * d_model
    seq_len = 56
    batch_size = 1

    # Params
    norm_epsilon = 1e-5
    fused_add_norm = True  # LayerNorm과 Residual 결합 비활성화
    residual_in_fp32 = False  # Residual FP32 유지 비활성화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Dummy input
    dummy_input = torch.randn(batch_size, seq_len, d_model, dtype=dtype).to(device)  # (B, L, D)

    ssm_cfg = {
        "layer": "MambaR",  # Use MambaR
        "d_state": d_state,
    }

    # Create Block using create_block
    block = create_block(
        d_model=d_model,
        d_intermediate=d_intermediate,
        ssm_cfg=ssm_cfg,
        norm_epsilon=norm_epsilon,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        device=device,
        dtype=dtype,
    )

    # Test with multiple step scales
    residual = None  # Initialize Residual
    scale_factors = [0.5, 1.0, 2.0]  # Different step scales for testing
    for scale in scale_factors:
        print(f"\nTesting with step_scale={scale}")
        output, residual = block(dummy_input, residual, step_scale=scale)  # Forward execution with step scale
        print("Block Output Shape:", output.shape)  # (B, L, D)
        print("Block Residual Shape:", residual.shape if residual is not None else None)



if __name__ == "__main__":
    main2()
