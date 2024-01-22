# superclass for all the training classes

from loader import Loader
import pytorch_lightning as pl
import torch
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec, spec_to_preprocessed_spec_inverse
import wandb
import numpy as np
import matplotlib.pyplot as plt
import global_objects
import time
import os
from evaluation_class import EvaluationClass

log_memory_usage = False  # for debugging

class TrainClass(pl.LightningModule):

    def __init__(self, model, batch_size, eval_freq):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        # logging params
        self.eval_set_size = 5  # quick eval
        self.bins = 200
        self.alpha = 0.5 
        # set up dataloaders
        loader = Loader()
        self.train_dataloader = loader.get_train_loader(batch_size)
        self.final_eval_dataloader = loader.get_test_loader(batch_size)
        self.quick_eval_dataloader = loader.get_quick_eval_loader(batch_size)
        # set up checkpoint dirs 
        train_start = time.strftime("%B%d-%H%M", time.localtime())
        self.checkpoints_path = "../checkpoints/" + self.model.name + "/" + train_start
        os.makedirs(self.checkpoints_path)
        self.eval_class = EvaluationClass(self.model, self.final_eval_dataloader)


    ################    TRAINING    ################

    def training_step(self, batch, batch_idx):
        _, _, _, loss = self.prepare_call_loss(batch)
        self.log("train_loss", loss)
        if batch_idx != 0 and batch_idx % self.eval_freq == 0:
            self.quick_eval()
        if log_memory_usage:
            print(str(int(torch.cuda.memory_allocated()) / 1e9) + " GB")
        return {"loss": loss}

    def to_starting_point(self, x): 
        dim = (global_objects.config["stft_num_channels"] // 2) + 1
        new_x = torch.zeros(x.shape[0], dim, x.shape[2])  # I will change the y-dim of the specs and pytorch does not allow me to write that back into x
        for i in range(x.shape[0]):  # TODO: very likely can be parallelized over the batch dim
            spec = x[i]
            inv = spec_to_mel_inverse(spec)
            new_x[i] = inv
        return new_x
    
    # does all the computations that can be reused for training and validation
    # cpu indicates where the model is (it is on cpu if cpu is True)
    # diffusion_generate is only relevant to the diffusion model. it indicates which of the two forward methods should be invoked (diffusion-forward or unet-forward)
    def prepare_call_loss(self, batch, cpu=False, diffusion_generate=False):
        x, y = batch

        x = x.to('cpu')
        y = y.to('cpu')
        x = self.to_starting_point(x)
        x, y = self.to_learning_space(x, y)

        if cpu == False:
            x = x.to('cuda')
            y = y.to('cuda')

        x = x.float()
        y = y.float()

        # if we are training a diffusion model AND we want to generate an image we need to invoke the generate method
        # if we are training a diffusion model AND we do not want to generate an image we need to invoke the regular forward method
        # WHICH in case of a diffusion model not only returns the prediction but also the target (= the noise added in the last timestep)
        # if we are not training a diffusion model we always need to invoke the regular forward method
        if self.model.name == "diffusion_model": 
            if diffusion_generate: 
                y_hat = self.model.generate(x)  # generate an image
            else:
                y_hat, y = self.model([x, y])  # predict noise added in last timestep (the adding of the noise is done inside of the forward method)
        else:  
            y_hat = self.model(x) 

        y_hat = y_hat.to('cuda') # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval
        y = y.to('cuda')         # when called from train_step this has already been done by pytorch lightninig, but not when called from quick_eval

        loss = torch.nn.functional.mse_loss(y_hat, y)
        return x, y, y_hat, loss
        
    


    ################    EVALUATION    ################

    # TODO: the differences (image and histogram) are really more interesting over the entire test set and not just one example
    def quick_eval(self):
        if log_memory_usage:
            print(torch.cuda.memory_summary())
        
        x = None
        y = None
        y_hat = None

        ########    LOSS    ########
        losses = list()
        loss = 0
        with torch.no_grad():  # this prevents the memory from overflowing in the loop
            for _ in range(self.eval_set_size): 
                if log_memory_usage:
                    print(str(int(torch.cuda.memory_allocated()) / 1e9) + " GB")
                batch = next(iter(self.quick_eval_dataloader))
                x, y, y_hat, loss = self.prepare_call_loss(batch, diffusion_generate=True)
                losses.append(loss.detach().cpu().numpy())
                loss = np.mean(losses) / self.batch_size
                self.log("eval_loss", loss)
        
        ########    SPECS    ########
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
        
        ########    HISTOGRAMS    ########
        fig, ax = plt.subplots()
        ax.hist(x[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="input")
        ax.hist(y_hat[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="output")
        ax.hist(y[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="target")
        ax.legend()
        wandb.log({"specs_histograms": wandb.Image(fig, caption="h")})
        plt.close(fig) 
        # clip the hists because they sometimes have peaks that make the rest of the hist hard to see
        fig, ax = plt.subplots()
        ax.hist(x[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="input")
        ax.set_ylim(0, 15000)
        ax.hist(y_hat[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="output")
        ax.set_ylim(0, 15000)
        ax.hist(y[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="target")
        ax.set_ylim(0, 15000)
        ax.legend()
        wandb.log({"specs_histogram_clipped": wandb.Image(fig, caption="h_clipped")})
        plt.close(fig)
        # hists of diffs
        fig, ax = plt.subplots()
        ax.hist(y[0].detach().cpu().numpy().flatten() - x[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="changes needed to get from starting point to target")
        ax.hist(y[0].detach().cpu().numpy().flatten() - y_hat[0].detach().cpu().numpy().flatten(), bins=self.bins, alpha=self.alpha, label="changes needed to get from output to target")
        ax.legend()
        wandb.log({"diffs_histograms": wandb.Image(fig, caption="h2")})   
        plt.close(fig)

        ########    AUDIO    ########
        x = self.from_learning_space(x[0].detach().cpu().numpy())
        y_hat = self.from_learning_space(y_hat[0].detach().cpu().numpy())
        y = self.from_learning_space(y[0].detach().cpu().numpy())
        wandb.log({"audio": [wandb.Audio(global_objects.stft_system.invert_spectrogram(x), caption="input", sample_rate=global_objects.config["sampling_rate"]), 
                                wandb.Audio(global_objects.stft_system.invert_spectrogram(y_hat), caption="output", sample_rate=global_objects.config["sampling_rate"]), 
                                wandb.Audio(global_objects.stft_system.invert_spectrogram(y), caption="target", sample_rate=global_objects.config["sampling_rate"])]})

        ########    CHECKPOINT    ########
        self.save_model(False, self.model, loss)

    # called at the end of every training run
    def final_eval(self, training_class, final_eval_dataloader):
        # save the final model
        self.save_model(True, self.model)
        # invoke the final eval
        self.eval_class.eval(self)

        




    ################    HELPERS    ################

    # we might not want to learn in the space that the data lives in, this method moves x and y to a potentially more suitable space for learning.
    # right now we are moving them into the log space (with spec_to_preprocessed_spec) in the hopes that there learning will be easier 
    # (e.g. because lower volume details are pronounced by this transformation). 
    def to_learning_space(self, x, y):
        x = spec_to_preprocessed_spec(x)
        y = spec_to_preprocessed_spec(y)
        return x, y

    def from_learning_space(self, entity):
        entity = spec_to_preprocessed_spec_inverse(entity, numpy=True)
        return entity 
    
    def save_model(self, final_model, model, loss=None):
        # remove all old checkpoints
        for file in os.listdir(self.checkpoints_path):
            os.remove(self.checkpoints_path + "/" + file)
        # create new checkpoint
        checkpoint = {
            'epoch': self.trainer.current_epoch,
            'num_steps_in_epoch': self.trainer.global_step,
            'model_state_dict': model.state_dict(),
            'curr_eval_loss': loss
        }
        # save the new checkpoint
        checkpoint_path = None
        if final_model:  
            checkpoint_path = self.checkpoints_path + "/final.pt"
        else:  # quick eval
            checkpoint_path = self.checkpoints_path + "/" + str(self.trainer.current_epoch) + "_" + str(self.trainer.global_step) + ".pt"
        torch.save(checkpoint, checkpoint_path)

    