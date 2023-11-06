# simple model that just uses spec_to_mel_inverse and then spec_to_preprocessed_spec to create its output

import torch
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec

class SimpleModel(torch.nn.Module):
    # constructor would only contain call to parent.init() so we dont have to write it out

    def forward(self, x):

        return spec_to_preprocessed_spec(spec_to_mel_inverse(x))

    # TODO: have a spec transform method that does to_01 internally and only returns 
    # normd specs and then rm all the initial norms 