# simple model that just uses spec_to_mel_inverse and then spec_to_preprocessed_spec to create its output

import torch
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec
from loader import Loader
from utils import spec_to_wav

class SimpleModel(torch.nn.Module):
    # constructor would only contain call to parent.init() so we dont have to write it out

    def forward(self, x):
        return spec_to_mel_inverse(x)
        # TODO: this output now does not live in the log space i.e. we would compute the eval score (e.g. MSE) against the unprocessed spec 
        # BUT the loss of the baseline_cnn is computed between two specs that live in the log space i.e. if we compute the eval score 
        # of the baseline_cnn also in the log space then these two eval scores are not comparable.
        #
        # Solution: either compute the eval score of the baseline_cnn in the non-log space (by transfoming the two specs that the eval score
        # would have been computed between by spec_to_preprocessed_spec_inverse) OR compute the eval score of this model in the log space.