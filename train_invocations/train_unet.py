from train_generative import TrainGenerative
from models.unet_wrapper import UnetWrapper
from pytorch_lightning.loggers import WandbLogger
from diffusers import DDPMScheduler
import pytorch_lightning as pl
import yaml

config = yaml.safe_load(open("config.yaml", "r"))

################  PARAMETERS  ################
#training
eval_freq = 500
batch_size = 1
max_epochs = 10
optimizer_params = {
    "learning_rate": 1e-4,
    "lr_warmup_steps": 500,
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
    "sample_size": config["spec_x_len"],  # x-axis len = x_len/hop_size
    "in_channels": config["stft_num_channels"],  # channels = "y-axis" = freq dims of specs
    "out_channels": config["stft_num_channels"],  # should be same as in_channels
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

################  TRAINING INVOCATION  ################

if __name__ == "__main__":
    scheduler = DDPMScheduler(**scheduler_params)
    model_wrapper = UnetWrapper(unet_params)
    training_class = TrainGenerative(model_wrapper, scheduler, optimizer_params=optimizer_params)
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=training_class, train_dataloaders=training_class.train_dataloader)

    # final eval
    final_loss = training_class.final_eval(training_class, training_class.final_eval_dataloader)