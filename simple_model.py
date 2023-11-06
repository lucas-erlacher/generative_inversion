# simple model that just uses spec_to_mel_inverse and then spec_to_preprocessed_spec to create its output

import os
from torch import optim, Tensor
import lightning.pytorch as pl
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec

class SimpleModel(pl.LightningModule):
    # constructor would only contain call to parent.init() so we dont have to write it out

    def forward(self, x):
        return self.spec_to_mel_inverse(self.spec_to_preprocessed_spec(x))

    # this model does not contain any trainable parameters
    def training_step(self, batch, batch_idx):
        return 0

    # this model does not contain any trainable parameters
    def configure_optimizers(self):
        pass

# init the model
model = SimpleModel()