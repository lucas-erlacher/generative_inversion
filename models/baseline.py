# simple model that just uses spec_to_mel_inverse to create its output

import torch
from transforms import spec_to_mel_inverse, spec_to_preprocessed_spec
from loader import Loader
from sklearn.metrics import mean_squared_error

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
    
if __name__ == "__main__":
    loader = Loader()
    test_loader = loader.get_test_loader(batch_size=32)
    model = SimpleModel()
    error = 0
    for batch in test_loader:
        x, y = batch
        y_hat = model(x)
        # reduce error over the batch dimension
        for i in range(len(y_hat)):
            error += mean_squared_error(y_hat[i], y[i])
    # save error to file
    with open("eval.txt", "w") as f:
        f.write("cumulative MSE of SimpleModel on test set: " + str(error))