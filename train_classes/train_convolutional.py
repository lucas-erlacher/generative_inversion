# this train class is currently used for the baseline_cnn

import torch
from train_classes.train_class import TrainClass
from torch.optim import Adam
from models.cnn_1D import Cnn1D
from models.cnn_2D import Cnn2D

################  LIGHTNING TRAINING WRAPPER  ################

class TrainConvolutional(TrainClass):
    def __init__(self, model, batch_size, eval_freq):
        super().__init__(batch_size, eval_freq)
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

        y_hat = self.model(x)  # forward pass

        y_hat = y_hat.to('cuda') # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval
        y = y.to('cuda')         # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval

        loss = torch.nn.functional.mse_loss(y_hat, y)
        return x, y, y_hat, loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer