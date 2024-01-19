import sys
sys.path.append("../")
from train_classes.train_generative import TrainGenerative
from models.unet_1D_wrapper import Unet1DWrapper
from pytorch_lightning.loggers import WandbLogger
from diffusers import DDPMScheduler
import pytorch_lightning as pl
import global_objects
import math

################  PARAMETERS  ################
# training
eval_freq = 2
batch_size = 32
max_epochs = 10 
optimizer_params = {
    "learning_rate": 1e-4,
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

################  TRAINING INVOCATION  ################

if __name__ == "__main__":
    model_wrapper = Unet1DWrapper(unet_params, pred_diff)
    training_class = TrainGenerative(model_wrapper, optimizer_params, batch_size, eval_freq)
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=training_class, train_dataloaders=training_class.train_dataloader)

    # final eval
    training_class.final_eval(training_class, training_class.final_eval_dataloader)