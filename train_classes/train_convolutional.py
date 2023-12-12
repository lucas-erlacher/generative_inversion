import torch
import yaml
from train_classes.train_class import TrainClass
import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning.loggers import WandbLogger

config = yaml.safe_load(open("config.yaml", "r")) 

################  LIGHTNING TRAINING WRAPPER  ################

class TrainConcolutional(TrainClass):
    def __init__(self, model, batch_size):
        super().__init__(batch_size)
        self.model = model
    
    # does all the computations that can be reused for training and validation
    def prepare_call_loss(self, batch):
        x, y = batch

        x = x.to('cpu')
        y = y.to('cpu')
        x = self.to_starting_point(x)
        x, y = self.to_learning_space(x, y)
        x = x.to('cuda')
        y = y.to('cuda')

        x = x.float()
        y = y.float()

        x = x.unsqueeze(1)  # add channel dimension

        y_hat = self.model(x)  # forward pass

        x = x.squeeze(1)  # remove channel dimension
        y_hat = y_hat.squeeze(1)  # remove channel dimension

        y_hat = y_hat.to('cuda') # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval
        y = y.to('cuda')         # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval

        loss = torch.nn.functional.mse_loss(y_hat, y)
        return x, y, y_hat, loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer