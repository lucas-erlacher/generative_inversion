# simple model that just uses spec_to_mel_inverse and then spec_to_preprocessed_spec to create its output

import torch
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec

class SimpleModel(torch.nn.Module):
    # constructor would only contain call to parent.init() so we dont have to write it out

    def forward(self, x):

        mel_inv = spec_to_mel_inverse(x)
        return spec_to_preprocessed_spec(mel_inv)