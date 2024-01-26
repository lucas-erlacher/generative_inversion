# Implementations of transformations that are used in multiples places in the project and therefore have to be consistent across each other. 

# ATTENTION: 
# in the ranges that you write in the comments do not use the values from the config file but only the names of the config file values.
# that way changes to the config file values will not break the ranges in the comments.

# TODO (later): all methods need to work on tensors (cant backprop through nparray) and I think the following comment has to be changed:
# The functions internally operate on numpy arrays since that is what the linear algebra operations require. 
# By default the inputs and outputs of the functions are expected to be torch tensors (since these functions will be used as part of a DL model/pipeline). 
# Alternatively the numpy flag can be set to true (is false by default) which will make the function accept and return numpy arrays.

import numpy as np
from utils_lucas import db_transform, to_01, db_inverse, spec_to_wav, normed_spec, print_range, undo_to_01, load
import global_objects
import torch




########  UNPROCESSED SPEC TO PREPROCESSED SPEC  ########

# PURPOSE:
# - generate the target element of a training data pair
#
# ASSUMPTIONS ON INPUT:
# torch tensor that represents a spectrogram that is normalized to [0,1]
def spec_to_preprocessed_spec(normed_spec, numpy=False):
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        normed_spec = normed_spec.numpy() 
    # assert that spec is in [0,1]
    if np.min(normed_spec) < 0 or np.max(normed_spec) > 1:
        raise Exception("spec_to_preprocessed_spec: spec must be in [0,1]")
    # 1) 
    # db transform
    # RANGE: [-infty, 0]
    db_normed_spec = db_transform(normed_spec)
    # 2)
    # clip very negative values
    # RANGE: [stft_min_val, 0]
    db_normed_spec[db_normed_spec < global_objects.config["stft_min_val"]] = global_objects.config["stft_min_val"]
    clipped_db_normed_spec = db_normed_spec  
    # 2.5)
    # shift into the positive range (so that we can use to_01). 
    # note that shifting by abs(stft_min_val) might not be optimal since if no vals got clipped we are shifting by too much 
    # (but I can't think of a better shift factor that wouldn't introduce an analogous problem of shifting too much/not shifting enough).
    # picking np.min(clipped_db_normed_spec) as the shift factor would be the optimal choice BUT then it is not obvious how to undo that shift in the inverse. 
    # RANGE: [0, abs(stft_min_val)]
    clipped_db_normed_spec += abs(global_objects.config["stft_min_val"])
    # 3) 
    # transform to [0, 1] (which might be easier for the NN to learn)
    # RANGE: [0, 1]
    normed_clipped_db_normed_spec = to_01(clipped_db_normed_spec, global_objects.config["stft_dyn_range_upper_bound"]) 
    # convert back to tensor if necessary
    if not numpy:
        normed_clipped_db_normed_spec = torch.from_numpy(normed_clipped_db_normed_spec)
    return normed_clipped_db_normed_spec

# PURPOSE:
# - sanity check the implementation of spec_to_preprocessed_spec
# 
# ASSUMPTIONS ON INPUT:
# spec has to be in [0,1] since that is what spec_to_preprocessed_spec outputs
def spec_to_preprocessed_spec_inverse(spec, numpy=False):
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        spec = spec.numpy()
    # assert that spec is in [0,1]
    if np.min(spec) < 0 or np.max(spec) > 1:
        raise Exception("spec_to_preprocessed_spec_inverse: spec must be in [0,1]") 
    # 3)
    # to_01 normalization can be undone
    # RANGE: [0, stft_dyn_range_upper_bound]
    clipped_db_normed_spec = undo_to_01(spec, global_objects.config["stft_dyn_range_upper_bound"])
    # 2.5)
    # invert 2.5 (otherwise recon will be heavily distorted). 
    # RANGE: [stft_min_val, 0]
    # upper bound being 0 assumes stft_dyn_range_upper_bound = abs(stft_min_val) (which is currently the case in the config file)
    clipped_db_normed_spec -= abs(global_objects.config["stft_min_val"])
    # 2)
    # clipping can't be undone since we don't know which (if any) values got clipped
    # RANGE: RANGE: [stft_min_val, 0]
    db_normed_spec = clipped_db_normed_spec
    # 1)
    # db transform can be undone
    # RANGE: [0, 1]
    normed_spec = db_inverse(db_normed_spec)
    # convert back to tensor if necessary
    if not numpy:
        normed_spec = torch.from_numpy(normed_spec)
    return normed_spec




########  UNPROCESSED SPEC TO MEL SPEC  ########

# PURPOSE:
# - generate the input element of a training data pair
# 
# ASSUMPTIONS ON INPUT:
# spec is assumed to be a fresh spectrogram that is normed to [0,1]
def spec_to_mel(normed_spec, numpy=False, debug=False):
    if debug: print_range(normed_spec)
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        normed_spec = normed_spec.numpy()
    # assert that spec is in [0,1]
    if np.min(normed_spec) < 0 or np.max(normed_spec) > 1:
        raise Exception("spec_to_mel: spec must be in [0,1]")
    # 1) 
    # multiply by mel basis
    # RANGE: [0, 0.04] 
    # - both bounds determined from examining some ranges so not 100% guaranteed (this uncertainty is propagated through the rest of the method)
    mel_normed_spec = np.dot(global_objects.mel_basis, normed_spec)
    if debug: print_range(mel_normed_spec)
    # 2) 
    # db transform (I think this is the "log" operation that everybody is referring to when they say "log mel spectrogram")
    # RANGE: [-infty, -14]
    db_mel_normed_spec = db_transform(mel_normed_spec) 
    if debug: print_range(db_mel_normed_spec)
    # 3) 
    # clip very negative values
    # RANGE: [log_mel_min_val, -14]
    db_mel_normed_spec[db_mel_normed_spec < global_objects.config["log_mel_min_val"]] = global_objects.config["log_mel_min_val"]
    clipped_db_mel_normed_spec = db_mel_normed_spec   
    if debug: print_range(clipped_db_mel_normed_spec)
    # 3.5)
    # shift into the positive range (so that we can use to_01)
    # note that shifting by abs(log_mel_min_val) might not be optimal since if no vals got clipped we are shifting by too much
    # (but I can't think of a better shift factor that wouldn't introduce an analogous problem of shifting too much/not shifting enough).
    # picking np.min(clipped_db_mel_normed_spec) as the shift factor would be the optimal choice BUT then it is not obvious how to undo that shift in the inverse.
    # RANGE: [0, abs(log_mel_min_val) - 14]
    clipped_db_mel_normed_spec += abs(global_objects.config["log_mel_min_val"])
    if debug: print_range(clipped_db_mel_normed_spec)
    # 4) 
    # transform to [0,1] because the output of this method is supposed to be the input to a NN and being in [0,1] might make the NN's life easier)
    # RANGE: [0, 1] 
    normed_clipped_db_mel_normed_spec = to_01(clipped_db_mel_normed_spec, global_objects.config["log_mel_dyn_range_upper_bound"])
    if debug: 
        print_range(normed_clipped_db_mel_normed_spec)
        print()
    # convert back to tensor if necessary
    if not numpy:
        normed_clipped_db_mel_normed_spec = torch.from_numpy(normed_clipped_db_mel_normed_spec)
    return normed_clipped_db_mel_normed_spec

# PURPOSE:
# - sanity check the implementation of spec_to_mel
# - generate a spectrogram from a mel spectrogram that we then apply spec_to_preprocessed_spec to which results in a first approximaton
#   of a preprocessed spectrogram (i.e. the target element of a dataset pair) that can later be refined by adding the output of a NN
#
# ASSUMPTIONS ON INPUT:
# well, it has to come from spec_to_mel but the only thing we can really check/assert about it is that it is in [0,1]
def spec_to_mel_inverse(spec, numpy=False, debug=False):
    if debug: print_range(spec)
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        spec = spec.numpy()
    # assert that spec is in [0,1]
    if np.min(spec) < 0 or np.max(spec) > 1:
        raise Exception("spec_to_mel_inverse: spec must be in [0,1]")
    # 4)
    # to_01 normalization can be undone
    # RANGE: [0, log_mel_dyn_range_upper_bound]
    clipped_db_mel_normed_spec = undo_to_01(spec, global_objects.config["log_mel_dyn_range_upper_bound"])
    if debug: print_range(clipped_db_mel_normed_spec)
    # 3.5)
    # invert 3.5 (otherwise recon will be heavily distorted).
    # RANGE: [log_mel_min_val, 0]
    # upper bound being 0 assumes log_mel_dyn_range_upper_bound = abs(log_mel_min_val) (which is currently the case in the config file) 
    clipped_db_mel_normed_spec -= abs(global_objects.config["log_mel_min_val"])
    if debug: print_range(clipped_db_mel_normed_spec)
    # 3)
    # clipping can't be undone
    # RANGE: [log_mel_min_val, 0] 
    db_mel_normed_spec = clipped_db_mel_normed_spec
    if debug: print_range(db_mel_normed_spec)
    # 2)
    # db transform can be undone
    # RANGE: [db_inverse(log_mel_min_val), 1]
    mel_normed_spec = db_inverse(db_mel_normed_spec)
    if debug: print_range(mel_normed_spec)
    # 1)
    # undoing multiplication with mel basis can be approximated (using the pseudoinverse)
    # RANGE: [0, 1]
    # - both bounds determined from examining some ranges so not 100% guaranteed
    normed_spec = np.dot(np.linalg.pinv(global_objects.mel_basis), mel_normed_spec)
    if debug: 
        print_range(normed_spec)
        print()
    # convert back to tensor if necessary
    if not numpy:
        normed_spec = torch.from_numpy(normed_spec)

    # ATTENTION: I have noticed that normed_spec can slightly go beneath 0 and above 1.
    # I think this is due to the pseudoinverse being an imperfect inverse (my supervisors have confirmed that this could very well be the reason). 
    # Since the deviations I have seen were all very small I will simply clip them back to 0 or 1 respectively.
    normed_spec[normed_spec < 0] = 0
    normed_spec[normed_spec > 1] = 1

    return normed_spec




########  TEST  ########

# main function tests if the implementations of the transforms are "good" by running the
# transforms forward and backward and comparing the result of that against the original input
if __name__ == "__main__":
    import librosa

    # save input as wav for reference
    filename = librosa.util.example('brahms')
    y = load(filename)
    y = y[:int(global_objects.config["spec_x_len"] / 2)]  # less than spec_x_len because the brahms sample is not long enough
    unprocessed_spec = normed_spec(y)
    spec_to_wav(unprocessed_spec, "x_input.wav")
    
    # test spec_to_preprocessed_spec
    spec = spec_to_preprocessed_spec(unprocessed_spec, numpy=True)
    recon = spec_to_preprocessed_spec_inverse(spec, numpy=True)
    spec_to_wav(recon, "x_recon_1.wav")

    # test spec_to_mel
    mel_spec = spec_to_mel(unprocessed_spec, numpy=True)
    recon = spec_to_mel_inverse(mel_spec, numpy=True)
    spec_to_wav(recon, "x_recon_2.wav")