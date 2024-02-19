import sys
sys.path.append('../')
import pytorch_lightning as pl
from models.cnn_1D import Cnn1D
from train_classes.train_convolutional import TrainConvolutional
from pytorch_lightning.loggers import WandbLogger

################  PARAMETERS  ################

# training
batch_size = 32
lr = 0.0001
max_epochs = 1000  # does not have an effect because I have a IterableDataset
eval_freq = 500
# model
kernel_size = 5
pred_diff = True  # switch betwwen predicting the full spec or the diff to the target spec
in_channels = 513  # num_stft_channels / 2 + 1

################  TRAINING INVOCATION  ################

if __name__ == '__main__':
    model = Cnn1D(kernel_size, pred_diff, in_channels)
    training_class = TrainConvolutional(model, batch_size, eval_freq)
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=training_class, train_dataloaders=training_class.train_dataloader)
    
    # final eval
    training_class.final_eval(training_class, training_class.final_eval_dataloader)