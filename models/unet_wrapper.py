import torch
import torch.nn as nn
from diffusers import UNet1DModel


class UnetWrapper(nn.Module):
    def __init__(self, unet_params):
        super().__init__()
        self.model = UNet1DModel(**unet_params)

    def timestep_forward(self, x):
        timesteps = torch.ones(x.shape[0]).to('cuda')  # here we don't want diffusion so use a cnstant
        y_hat = self.model(x, timesteps, return_dict=False)[0]
        return y_hat