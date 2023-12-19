import torch
import torch.nn as nn
from diffusers import UNet1DModel

class UnetWrapper(nn.Module):
    def __init__(self, unet_params, pred_diff):
        super().__init__()
        self.pred_diff = pred_diff
        self.model = UNet1DModel(**unet_params)

    def timestep_forward(self, x):
        timesteps = torch.zeros(x.shape[0]).float().to('cuda')  # here we don't want diffusion so use a cnstnt
        y_hat = self.model(x, timesteps, return_dict=False)[0]

        if self.pred_diff:
            y_hat = x + y_hat

        return y_hat