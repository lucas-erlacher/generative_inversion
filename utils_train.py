# I also don't like having "utils" and "utils_train" but if I put these functions into utils.py I get a circular import error.
# There is probably a better way to resolve this but I don't wan't to spend time on that at the moment. 

import yaml
import torch
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec, spec_to_preprocessed_spec_inverse
import wandb
import numpy as np
import matplotlib.pyplot as plt
import global_objects

config = yaml.safe_load(open("config.yaml", "r"))

################  PARAMETERS  ################

eval_set_size = 50
bins = 200
alpha = 0.5

################  FUNCTIONS  ################

def to_starting_point(x):
        dim = (config["stft_num_channels"] // 2) + 1
        new_x = torch.zeros(x.shape[0], dim, x.shape[2])  # I will change the y-dim of the specs and pytorch does not allow me to write that back into x
        for i in range(x.shape[0]):  # TODO: very likely can be parallelized over the batch dim
            spec = x[i]
            inv = spec_to_mel_inverse(spec)
            new_x[i] = inv 
        return new_x
    
# we might not want to learn in the space that the data lives in, this method moves x and y to a potentially more suitable space for learning.
# right now we are moving them into the log space (with spec_to_preprocessed_spec) in the hopes that there learning will be easier 
# (e.g. because lower volume details are pronounced by this transformation). 
def to_learning_space(x, y):
    x = spec_to_preprocessed_spec(x)
    y = spec_to_preprocessed_spec(y)
    return x, y

def from_learning_space(entity):
    entity = spec_to_preprocessed_spec_inverse(entity, numpy=True)
    return entity 

# TODO: the differences (image and histogram) are really more interesting over the entire test set and not just one example
def quick_eval(batch_size, prepare_call_loss, test_dataloader, log_obj):  
    x = None
    y_hat = None
    y = None
    losses = list()
    for i in range(eval_set_size): 
        batch = next(iter(test_dataloader))
        x, y, y_hat, loss = prepare_call_loss(batch)
        losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses) / batch_size
        log_obj.log("eval_loss", loss)
    # log images of one example to wandb
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
    # create histograms as images and log them to wandb 
    fig, ax = plt.subplots()
    ax.hist(x[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="input")
    ax.hist(y_hat[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="output")
    ax.hist(y[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="target")
    ax.legend()
    wandb.log({"specs_histograms": wandb.Image(fig, caption="h")})
    plt.close(fig) 
    # clip the hists because they sometimes have peaks that make the rest of the hist hard to see
    fig, ax = plt.subplots()
    ax.hist(x[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="input")
    ax.set_ylim(0, 15000)
    ax.hist(y_hat[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="output")
    ax.set_ylim(0, 15000)
    ax.hist(y[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="target")
    ax.set_ylim(0, 15000)
    ax.legend()
    wandb.log({"specs_histogram_clipped": wandb.Image(fig, caption="h_clipped")})
    plt.close(fig)
    # hists of diffs
    fig, ax = plt.subplots()
    ax.hist(y[0].detach().cpu().numpy().flatten() - x[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="changes needed to get from starting point to target")
    ax.hist(y[0].detach().cpu().numpy().flatten() - y_hat[0].detach().cpu().numpy().flatten(), bins=bins, alpha=alpha, label="changes needed to get from output to target")
    ax.legend()
    wandb.log({"diffs_histograms": wandb.Image(fig, caption="h2")})   
    plt.close(fig)
    # log audio to wandb
    x = from_learning_space(x[0].detach().cpu().numpy())
    y_hat = from_learning_space(y_hat[0].detach().cpu().numpy())
    y = from_learning_space(y[0].detach().cpu().numpy())
    wandb.log({"audio": [wandb.Audio(global_objects.stft_system.invert_spectrogram(x), caption="input", sample_rate=config["sampling_rate"]), 
                            wandb.Audio(global_objects.stft_system.invert_spectrogram(y_hat), caption="output", sample_rate=config["sampling_rate"]), 
                            wandb.Audio(global_objects.stft_system.invert_spectrogram(y), caption="target", sample_rate=config["sampling_rate"])]})