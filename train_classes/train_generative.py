# this train_class is currently used for the unet and the diffusion model

import torch
from tqdne_diffusers import DDPMPipeline1DCond
from train_classes.train_class import TrainClass

################  LIGHTNING TRAINING WRAPPER  ################

class TrainGenerative(TrainClass):
    def __init__(self, model_wrapper, scheduler, optimizer_params, batch_size, eval_freq):
        super().__init__(batch_size, eval_freq)
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.net = model_wrapper  # unfortunately pytorch lightning forces me to call the model_wrapper net
        self.noise_scheduler = scheduler
        self.optimizer_params = optimizer_params
        self.pipeline = DDPMPipeline1DCond(self.net, self.noise_scheduler)

    def prepare_call_loss(self, batch):
        x, y = batch[0], batch[1]

        x = x.to('cpu')
        y = y.to('cpu')
        x = self.to_starting_point(x)
        x, y = self.to_learning_space(x, y)
        x = x.to('cuda')
        y = y.to('cuda')

        x = x.float()
        y = y.float()

        y_hat = self.net.timestep_forward(x)  # forward pass
        # some transforms need the input to be in [0,1] 
        y_hat = torch.sigmoid(y_hat)

        loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log("train_loss", loss)

        return x, y, y_hat, loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        return [optimizer], []