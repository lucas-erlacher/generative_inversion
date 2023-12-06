# I also don't like having "utils" and "train_utils" but if I put these functions into utils.py I get a circular import error.
# There is probably a better way to resolve this but I don't wan't to spend time on that at the moment. 

import yaml
import torch

config = yaml.safe_load(open("config.yaml", "r"))

from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec, spec_to_preprocessed_spec_inverse

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