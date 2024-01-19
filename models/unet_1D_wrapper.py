# this class wraps a unet model and provides methods that make the unet usable with the training class (currently called train_generative.py)

import torch
import torch.nn as nn
from diffusers import UNet1DModel

class Unet1DWrapper(nn.Module):
    def __init__(self, unet_params, pred_diff):
        super().__init__()
        self.name = "unet_1D"
        self.pred_diff = pred_diff
        self.model = UNet1DModel(**unet_params)
        
    # overrides the forward method of torch.nn.Module
    def forward(self, x):
        timesteps = torch.zeros(x.shape[0]).float().to('cuda')  # here we don't want diffusion so use a cnstnt
        y_hat = self.model(x, timesteps, return_dict=False)[0]

        if self.pred_diff:
            y_hat = x + y_hat

        # force output to be in [0,1] (because the transforms assume that).
        y_hat = torch.sigmoid(y_hat)
        return y_hat