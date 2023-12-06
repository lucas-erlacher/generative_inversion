# this file has been taken from the tqdne repo (and then modified when needed)

import torch
from torch.nn import functional as F
from typing import List
from diffusers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from tqdne_diffusers import DDPMPipeline1DCond
from tqdne_diffusers import to_inputs
from torch import nn
from diffusers import UNet1DModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning.loggers import WandbLogger
from loader import Loader
import yaml
from train_utils import to_starting_point, to_learning_space, from_learning_space

config = yaml.safe_load(open("config.yaml", "r"))

# TODOs: 
# get the cnn to learn someting useful again (now with the reduced image size)

################    DIFFUSION WRAPPER    ################
class LightningDDMP(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: DDPMScheduler,
        optimizer_params: dict,
        prediction_type: str = "epsilon"
    ):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.noise_scheduler = noise_scheduler
        if prediction_type not in ["epsilon", "sample"]:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        self.prediction_type = prediction_type
        self.pipeline = DDPMPipeline1DCond(self.net, self.noise_scheduler)
        self.save_hyperparameters()

    def evaluate(self, low_res):
        # Sample some signal from random noise (this is the backward diffusion process).
        sig = self.pipeline(
            low_res=low_res,
            generator=torch.manual_seed(self.optimizer_params["seed"]),
        ).audios

        return sig

    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward_step(self, batch: list, timestep: int):
        # Sample a random timestep for each signal
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device="cuda",
        ).long()
        output = self.net(batch, timesteps, return_dict=False)[0]
        return output

    def global_step(self, batch: List, timestep, train: bool = False):
        x, y = batch[0], batch[1]

        x = x.to('cpu')
        y = y.to('cpu')
        x = self.to_starting_point(x)
        x, y = self.to_learning_space(x, y)
        x = x.to('cuda')
        y = y.to('cuda')

        # reshape input to the format that the huggingface lib expects
        x = x.flatten(1).float()
        y = y.flatten(1).float()

        x.unsqueeze_(1)  # add channel dimension
        y_hat = self.forward_step(x, timestep)
        x = x.squeeze(1)  # remove channel dimension

        loss = F.mse_loss(y_hat, y)

        print(loss)

        self.log_value(loss, "loss", train=train, prog_bar=True)

        return loss

    def training_step(self, train_batch: List, batch_idx: int):
        return self.global_step(train_batch, timestep, train=True)

    def validation_step(self, val_batch: List, batch_idx: int):
        return self.global_step(val_batch, timestep)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_params["lr_warmup_steps"],
            num_training_steps=(
                self.optimizer_params["n_train"] * self.optimizer_params["max_epochs"]
            ),
        )
        return [optimizer], [lr_scheduler]




################    TRAINING    ################
if __name__ == "__main__":
    batch_size = 1
    # training params
    loader = Loader()
    train_dataloader = loader.get_train_loader(batch_size)
    print(type(train_dataloader))
    test_dataloader = loader.get_test_loader(batch_size)
    print(type(test_dataloader))

    ######## TRAINING PARAMS ########
    prediction_type = "sample"  # `epsilon` (predicts the noise of the diffusion process) or `sample` (directly predicts the noisy sample` 
    max_epochs = 10
    channels = config["stft_num_channels"]
    timestep = 1  # FIX TO A CONSTANT VALUE IN ORDER TO DEACTIVATE DIFFUSION AND ONLY TRAIN A U-NET
    unet_params = {
        "sample_size":timestep,
        "in_channels":1,  # if we were working on rgb ims this would have to be 3 
        "out_channels":1,
        "block_out_channels":  (32, 64, 128),
        "down_block_types": ('DownBlock1D', 'DownBlock1D', 'AttnDownBlock1D'),
        "up_block_types": ('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1D'),
        "mid_block_type": 'UNetMidBlock1D',
        "out_block_type": "OutConv1DBlock",
        "extra_in_channels" : 0,
        "act_fn": "swish",  # if I don't pick an act fn I get error ...
    }
    scheduler_params = {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_train_timesteps": 1000,
        "prediction_type": prediction_type,
        "clip_sample": False,
    }
    optimizer_params = {
        "learning_rate": 1e-4,
        "lr_warmup_steps": 500,
        "n_train": len(train_dataloader) // batch_size,
        "seed": 0,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
    }
    trainer_params = {
        # trainer parameters
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1,
        "precision": "32-true",  
        # Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
        # 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        # Can be used on CPU, GPU, TPUs, HPUs or IPUs.
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "num_nodes": 1}
    
    ######## INVOKE TRAINING ########
    net = UNet1DModel(**unet_params)
    scheduler = DDPMScheduler(**scheduler_params)
    pipeline = DDPMPipeline1DCond(net, scheduler)
    model = LightningDDMP(net, scheduler, prediction_type=prediction_type, optimizer_params=optimizer_params)
    # wandb_logger = WandbLogger(project='generative inversion')
    # wandb_logger.experiment.batch_size = batch_size
    trainer = pl.Trainer(**trainer_params)
    # trainer = pl.Trainer(**trainer_params, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader)