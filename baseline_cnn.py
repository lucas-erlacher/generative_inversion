import torch
import torch.nn as nn
import yaml
from loader import Loader
import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning.loggers import WandbLogger
from utils_train import to_starting_point, to_learning_space, quick_eval

config = yaml.safe_load(open("config.yaml", "r"))

################  PARAMETERS ################

# training
batch_size = 4
lr = 0.0001
max_epochs = 10
eval_freq = 500
# model
kernel_size = 5
pred_diff = True  # switch betwwen predicting the full spec or the diff to the target spec

################  LOADERS  ################
loader = Loader()
train_dataloader = loader.get_train_loader(batch_size)
test_dataloader = loader.get_test_loader(batch_size)

################  MODEL ################

class Baseline_CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()        
        self.act_func = nn.ReLU()
        self.conv1 = nn.Conv2d(kernel_size=kernel_size, in_channels=1, out_channels=8, padding=2) 
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(kernel_size=kernel_size, in_channels=8, out_channels=16, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(kernel_size=kernel_size, in_channels=16, out_channels=32, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(kernel_size=kernel_size, in_channels=32, out_channels=16, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(kernel_size=kernel_size, in_channels=16, out_channels=8, padding=2)
        self.bn5 = nn.BatchNorm2d(8)
        self.conv6 = nn.Conv2d(kernel_size=kernel_size, in_channels=8, out_channels=1, padding=2)

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
        if batch_idx != 0 and batch_idx % eval_freq == 0:
            quick_eval(batch_size, self.prepare_call_loss, test_dataloader, self)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer

################  TRAINING ################

if __name__ == '__main__':
    model = Baseline_CNN()
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader)