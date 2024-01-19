# this class contains everything needed for a diffusion model (a unet, a method for generating an image 
# (i.e. repeatedly invoking the unet wrapped by this class), a method for training the unet that is wrapped by this class)

import sys
sys.path.append("../")
from models.diffusion_pipeline import DDPMPipeline1DCond
import torch
import torch.nn as nn
from diffusers import UNet1DModel
import global_objects

class DiffusionModel(nn.Module):
    def __init__(self, unet_params, scheduler):
        super().__init__()
        self.prediction_type = scheduler.config.prediction_type
        self.name = "diffusion_model"
        self.model = UNet1DModel(**unet_params) 
        self.noise_scheduler = scheduler
        self.pipeline = DDPMPipeline1DCond(self.model, self.noise_scheduler)

    # method for generating an image (i.e. repeatedly invoking the unet wrapped by this class) - this is the forward method of the diffusion model-
    # inputs to diffusion model: mel spec (= conditioning input)
    # - the pipeline will then sample noise and use the conditioning input to guide the proceess of iteratively_removing_noise_to_obtain_a_spec
    # output of diffusion model: (hopefully high quality) spec
    def generate(self, conditioning_input):
        sig = self.pipeline(
            low_res=conditioning_input,
            generator=torch.manual_seed(42),
            num_inference_steps=global_objects.config["diff_num_inference_steps"],  
        ).audios

        # some transforms require their input to be in [0, 1]. 
        # in the other models I am using sigmoid for this (because there the forward method returns the generated image) BUT here the generate method returns
        # the generated image i.e. if I would use sigmoid here I would not inculude the sigmoid transformation in the training process (which goes through forward)
        # i.e. the trained u-net would not "know" (and therefore cannot adjust itself to the fact that) that there is a sigmoid at the end of the generation
        # which is why I am choosing to clip here (because I think clipping a less "invasisve" transformation than using a sigmoid). 
        sig[sig > 1] = 1
        sig[sig < 0] = 0

        return sig

    # method for training the unet that is wrapped by this class (overrides the forward method of torch.nn.Module) - this is the forward method of the unet.
    # inputs to the unet: mel spec (= conditioning input) + noisy target spec
    # output of the unet: noise added in last timestep
    def forward(self, data):
        conditioning_input = data[0]  # mel spec
        target = data[1]  # clean, target spec
        
        # Sample noise
        noise = torch.randn(conditioning_input.shape).to(conditioning_input.device)

        # Sample a random timestep for each signal
        assert self.noise_scheduler.config.num_train_timesteps == global_objects.config["diff_num_inference_steps"]  # just to make sure
        batch_size = conditioning_input.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=conditioning_input.device).long()

        # Add noise to the clean high_res according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)

        # concat the conditioning input with the noisy input to form the input to the unet
        unet_input = torch.cat((conditioning_input, noisy_target), dim=1)

        # invoke the unet
        output = self.model(unet_input, timesteps, return_dict=False)[0]

        target = noise if self.prediction_type == "epsilon" else target
        
        # here we also have to return the target (= the noise that was added) since it was created in this method i.e. the caller can't know the target
        return (output, target) 