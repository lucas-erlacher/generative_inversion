import sys
sys.path.append("../")
from train_classes.train_generative import TrainGenerative
from models.unet_wrapper import UnetWrapper
from pytorch_lightning.loggers import WandbLogger
from diffusers import DDPMScheduler
import pytorch_lightning as pl
import global_objects
import math

# TODOs: 

################  PARAMETERS  ################
#training
eval_freq = 5
batch_size = 1
max_epochs = 10 
optimizer_params = {
    "learning_rate": 1e-4,
    "lr_warmup_steps": 500,
    "seed": 0,
    "batch_size": batch_size,
    "max_epochs": max_epochs}
trainer_params = {
    "accumulate_grad_batches": 1,
    "gradient_clip_val": 1,
    "precision": "32-true",  
    "max_epochs": max_epochs,
    "accelerator": "auto",
    "devices": "auto",
    "num_nodes": 1}
# model
pred_diff = True
prediction_type = "sample"
channels = global_objects.config["stft_num_channels"] 
stft_num_channels = math.floor(global_objects.config["stft_num_channels"] / 2) + 1
unet_params = {
    "sample_size": global_objects.config["spec_x_len"],  # x-axis len = x_len/hop_size
    "in_channels": stft_num_channels,  # channels = "y-axis" = freq dims of specs
    "out_channels": stft_num_channels,  # should be same as in_channels
    "block_out_channels":  (32, 64, 128),
    "down_block_types": ('DownBlock1D', 'DownBlock1D', 'AttnDownBlock1D'),
    "up_block_types": ('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1D'),
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
    model_wrapper = UnetWrapper(unet_params, pred_diff)
    training_class = TrainGenerative(model_wrapper, scheduler, optimizer_params, batch_size, eval_freq)
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=training_class, train_dataloaders=training_class.train_dataloader)

    # final eval
    final_loss = training_class.final_eval(training_class, training_class.final_eval_dataloader)