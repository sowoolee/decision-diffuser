# mamba block
import isaacgym
import torch
import time as TIME

from diffuser.models.temporal import TemporalUnet
from mamba_ssm.modules.mamba_simple import Mamba
from diffuser.models.DiS import TemporalMamba

# input data config
batch_size = 32
horizon = 56
input_dim = 37
dim = 128

# dummy input data
x = torch.randn(batch_size, horizon, input_dim).to("cuda")  # (B, L, D)
cond = torch.randn(batch_size, 4).to("cuda")
time = torch.randint(1, 10, (batch_size,)).to("cuda")

# models for validation
temporal_mamba = TemporalMamba(
    horizon=horizon,
    input_dim=input_dim,
    dim=dim,
    returns_condition=True,
).to("cuda")

temporal_unet = TemporalUnet(
    horizon=horizon,
    transition_dim=input_dim,
    cond_dim=cond.size(-1)
).to("cuda")

start = TIME.time()
mamba_output = temporal_mamba(x, cond, time, returns=cond)
end = TIME.time()
mamba_t = end - start

start = TIME.time()
unet_output = temporal_unet(x, cond, time)
end = TIME.time()
unet_t = end - start

print("\n")
print('TemporalMamba Output Shape:', mamba_output.shape)
print('TemporalUnet Output Shape:', unet_output.shape)
print("\n")
print('TemporalMamba inference time: ', mamba_t, ' s')
print('TemporalUnet inference time: ', unet_t, ' s')