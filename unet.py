# this file has been taken from the tqdne repo (and then modified when needed)

import torch
from diffusers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from tqdne_diffusers import DDPMPipeline1DCond
from torch import nn
from diffusers import UNet1DModel
from pytorch_lightning.loggers import WandbLogger
from loader import Loader
import yaml
from utils_train import to_starting_point, to_learning_space, quick_eval

config = yaml.safe_load(open("config.yaml", "r"))

# TODOs:
# get the cnn to learn someting useful again (now with the reduced image size)
# get unet to learn something useful

################  PARAMETERS ################
#training
eval_freq = 500
batch_size = 1
max_epochs = 10
loader = Loader()
train_dataloader = loader.get_train_loader(batch_size)
test_dataloader = loader.get_test_loader(batch_size)
optimizer_params = {
    "learning_rate": 1e-4,
    "lr_warmup_steps": 500,
    "n_train": len(train_dataloader) // batch_size,
    "seed": 0,
    "batch_size": batch_size,
    "max_epochs": max_epochs,
}
trainer_params = {
    "accumulate_grad_batches": 1,
    "gradient_clip_val": 1,
    "precision": "32-true",  
    "max_epochs": max_epochs,
    "accelerator": "auto",
    "devices": "auto",
    "num_nodes": 1}
# model
prediction_type = "sample"
channels = config["stft_num_channels"]
timestep = 1 
unet_params = {
    "sample_size":timestep,
    "in_channels":1,
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

################    MODEL   ################

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

    def prepare_call_loss(self, batch):
        x, y = batch[0], batch[1]

        x = x.to('cpu')
        y = y.to('cpu')
        x = to_starting_point(x)
        x, y = to_learning_space(x, y)
        x = x.to('cuda')
        y = y.to('cuda')

        batch_size = x.shape[0]
        second_dim = x.shape[2]

        # reshape input to the format that the huggingface lib expects
        x = x.flatten(1).float()
        y = y.flatten(1).float()

        x.unsqueeze_(1)  # add channel dimension

        # Sample a random timestep for each signal
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device="cuda",
        ).long()
        y_hat = self.net(x, timesteps, return_dict=False)[0]
        # some transforms need the input to be in [0,1] 
        y_hat = torch.sigmoid(y_hat)

        x = x.squeeze(1)  # remove channel dimension

        loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log("train_loss", loss)

        # reshape back into a 2D matrix
        x = x.reshape(batch_size, -1, second_dim)
        y_hat = y_hat.reshape(batch_size, -1, second_dim)
        y = y.reshape(batch_size, -1, second_dim)
        return x, y, y_hat, loss

    def training_step(self, batch, batch_idx):
        _, _, _, loss = self.prepare_call_loss(batch)
        self.log("train_loss", loss)
        if batch_idx != 0 and batch_idx % eval_freq == 0:
            quick_eval(batch_size, self.prepare_call_loss, test_dataloader, self)
        return {"loss": loss}

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

################  TRAINING ################

if __name__ == "__main__":
    scheduler = DDPMScheduler(**scheduler_params)
    net = UNet1DModel(**unet_params)
    model = LightningDDMP(net, scheduler, prediction_type=prediction_type, optimizer_params=optimizer_params)
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader)