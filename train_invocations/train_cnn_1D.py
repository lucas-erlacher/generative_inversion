import pytorch_lightning as pl
import sys
sys.path.append('../')
from models.cnn_1D import Cnn1D
from train_classes.train_convolutional import TrainConvolutional
from pytorch_lightning.loggers import WandbLogger
 
################  PARAMETERS  ################

# training
batch_size = 4
lr = 0.0001
max_epochs = 10
eval_freq = 500
# model
kernel_size = 5
pred_diff = True  # switch betwwen predicting the full spec or the diff to the target spec
in_channels = 513

################  TRAINING INVOCATION  ################

if __name__ == '__main__':
    model = Cnn1D(kernel_size, pred_diff, in_channels)
    training_class = TrainConvolutional(model, batch_size, eval_freq)
    wandb_logger = WandbLogger(project='generative inversion')
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=training_class, train_dataloaders=training_class.train_dataloader)

    # final eval
    final_loss = training_class.final_eval(training_class, training_class.final_eval_dataloader)