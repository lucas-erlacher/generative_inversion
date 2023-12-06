from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import yaml
from loader import Loader
import pytorch_lightning as pl
from torch.optim import Adam
import wandb
from pytorch_lightning.loggers import WandbLogger
from train_utils import to_starting_point, to_learning_space, from_learning_space
import global_objects
import numpy as np
import matplotlib.pyplot as plt

config = yaml.safe_load(open("config.yaml", "r"))

# training params
batch_size = 4
lr = 0.0001
max_epochs = 10

# wandb histogram plotting parameters
bins = 200
alpha = 0.5

################  MODEL ################

pred_diff = True  # switch betwwen predicting the full spec or the diff to the target spec

class Baseline_CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # HYPERPARAMETERS
        self.kernel_size = 5
        d = 1
        self.save_hyperparameters()  # I think this is necessary to save the hyperparameters to wandb
        # MODEL
        self.act_func = nn.ReLU()
        self.conv1 = nn.Conv2d(kernel_size=self.kernel_size, in_channels=1, out_channels=8, padding=2) 
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(kernel_size=self.kernel_size, in_channels=8, out_channels=16, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(kernel_size=self.kernel_size, in_channels=16, out_channels=32, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(kernel_size=self.kernel_size, in_channels=32, out_channels=16, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(kernel_size=self.kernel_size, in_channels=16, out_channels=8, padding=2)
        self.bn5 = nn.BatchNorm2d(8)
        self.conv6 = nn.Conv2d(kernel_size=self.kernel_size, in_channels=8, out_channels=1, padding=2)

    def forward(self, x):
        init = x  # save input for later use (in case we want to predict the diff)
        x = self.bn1(self.act_func(self.conv1(x)))
        x = self.bn2(self.act_func(self.conv2(x)))
        x = self.bn3(self.act_func(self.conv3(x)))
        x = self.bn4(self.act_func(self.conv4(x)))
        x = self.bn5(self.act_func(self.conv5(x)))
        x = self.conv6(x)

        if pred_diff:
            x = x + init

        # force output to be in [0,1] (because the transforms assume that).
        x = torch.sigmoid(x) # maybe no act funct at end ... but some transforms require output to be in [0,1] and I am not sure clipping is good for gradients

        return x
    
    def train_dataloader(self):
        loader = Loader()
        return loader.get_train_loader(batch_size)
    
    def test_dataloader(self):
        loader = Loader()
        return loader.get_test_loader(batch_size)
    
    # does all the computations that can be reused for training and validation
    def prepare_call_loss(self, batch):
        x, y = batch

        x = x.to('cpu')
        y = y.to('cpu')
        x = to_starting_point(x)
        x, y = to_learning_space(x, y)
        x = x.to('cuda')
        y = y.to('cuda')

        x = x.float()
        y = y.float()

        x = x.unsqueeze(1)  # add channel dimension

        y_hat = self(x)  # forward pass

        x = x.squeeze(1)  # remove channel dimension
        y_hat = y_hat.squeeze(1)  # remove channel dimension

        y_hat = y_hat.to('cuda') # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval
        y = y.to('cuda')         # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval

        loss = torch.nn.functional.mse_loss(y_hat, y)
        return x, y, y_hat, loss

    def training_step(self, batch, batch_idx):
        _, _, _, loss = self.prepare_call_loss(batch)
        self.log("train_loss", loss)
        if batch_idx != 0 and batch_idx % 500 == 0:
            self.quick_eval()
        return {"loss": loss}
    
    def quick_eval(self):  # TODO: the differences (image and histogram) are really more interesting over the entire test set and not just one example
        x = None
        y_hat = None
        y = None
        for batch_idx, batch in enumerate(self.test_dataloader()):
            losses = list()
            x, y, y_hat, loss = self.prepare_call_loss(batch)
            losses.append(loss.detach().cpu().numpy())
            loss = np.mean(losses) / self.batch_size
            self.log("eval_loss", loss)
            if batch_idx == 50:  # don't go over the entire test set (the full test set meant for the final evaluation)
                break
        # log images of one example to wandb
        wandb.log({"specs": [wandb.Image(x[0], caption="input"), 
                              wandb.Image(y_hat[0], caption="output"), 
                              wandb.Image(y[0], caption="target")]})
        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(y[0].detach().cpu().numpy() - x[0].detach().cpu().numpy(), cmap="coolwarm"), ax=ax)
        wandb.log({"changes needed to get from starting point to target": wandb.Image(plt, caption="diffs")})
        plt.close(fig)
        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(y[0].detach().cpu().numpy() - y_hat[0].detach().cpu().numpy(), cmap="coolwarm"), ax=ax)
        wandb.log({"changes needed to get from output to target": wandb.Image(plt, caption="diffs")})
        plt.close(fig)
        # create histograms as images and log them to wandb 
        fig, ax = plt.subplots()
        ax.hist(x[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="input")
        ax.hist(y_hat[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="output")
        ax.hist(y[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="target")
        ax.legend()
        wandb.log({"specs_histograms": wandb.Image(fig, caption="h")})
        plt.close(fig) 
        # clip the hists because they sometimes have peaks that make the rest of the hist hard to see
        fig, ax = plt.subplots()
        ax.hist(x[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="input")
        ax.set_ylim(0, 15000)
        ax.hist(y_hat[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="output")
        ax.set_ylim(0, 15000)
        ax.hist(y[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="target")
        ax.set_ylim(0, 15000)
        ax.legend()
        wandb.log({"specs_histogram_clipped": wandb.Image(fig, caption="h_clipped")})
        plt.close(fig)
        # hists of diffs
        fig, ax = plt.subplots()
        ax.hist(y[0].detach().cpu().numpy().flatten() - x[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="changes needed to get from starting point to target")
        ax.hist(y[0].detach().cpu().numpy().flatten() - y_hat[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="changes needed to get from output to target")
        ax.legend()
        wandb.log({"diffs_histograms": wandb.Image(fig, caption="h2")})   
        plt.close(fig)
        # log audio to wandb
        x = from_learning_space(x[0].detach().cpu().numpy())
        y_hat = from_learning_space(y_hat[0].detach().cpu().numpy())
        y = from_learning_space(y[0].detach().cpu().numpy())
        wandb.log({"audio": [wandb.Audio(global_objects.stft_system.invert_spectrogram(x), caption="input", sample_rate=config["sampling_rate"]), 
                             wandb.Audio(global_objects.stft_system.invert_spectrogram(y_hat), caption="output", sample_rate=config["sampling_rate"]), 
                             wandb.Audio(global_objects.stft_system.invert_spectrogram(y), caption="target", sample_rate=config["sampling_rate"])]})

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    # run training
    wandb_logger = WandbLogger(project='generative inversion')
    wandb_logger.experiment.batch_size = batch_size
    model = Baseline_CNN()
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=model.train_dataloader())