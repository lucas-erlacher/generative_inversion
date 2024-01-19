# this train class is currently used for the baseline_cnn

import torch
from train_classes.train_class import TrainClass
from torch.optim import Adam

################  LIGHTNING TRAINING WRAPPER  ################

class TrainConvolutional(TrainClass):
    def __init__(self, model, batch_size, eval_freq):
        super().__init__(model, batch_size, eval_freq)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer