# this train_class is currently used for the unet and the diffusion model

import torch
from tqdne_diffusers import DDPMPipeline1DCond
from train_classes.train_class import TrainClass
from diffusers.optimization import get_cosine_schedule_with_warmup

################  LIGHTNING TRAINING WRAPPER  ################

class TrainGenerative(TrainClass):
    def __init__(self, model, optimizer_params, batch_size, eval_freq):
        super().__init__(model, batch_size, eval_freq)
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.optimizer_params = optimizer_params
        self.net = model  # for some reason lightning forces me to have a field called net (even tough I am using self.model)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        return [optimizer]