# Implementations of transformations that are used in multiples places in the project and therefore have to be consistent across each other. 

# By default the inputs and outputs of the functions are expected to be torch tensors (since these functions will be used as part of a DL model/pipeline). 
# Alternatively the numpy flag can be set to true (is false by default) which will make the function accept and return numpy arrays.

# ON THE SHIFTING BACK INTO THE NEGATIVE RANGE IN THE INVERSE METHODS:
# has to be done otherwise the reconstruction will be heavily distorted. 
# since we don't know min_val from the to_01 invocation we have to use a proxy for the magnitude of the back-shift. 
# in principle many values could be used here but we might as well make it make as much sense as possible in our ctxt. 
# we can't reconstruct the lower bound of the input array in from the forward pass bc maybe nothing got clipped in the forward pass and then shifting by 
# the clipping threshold will not reconstruct the lower bound. 
# we can however reconstruct the upper bound of the input array from the forward pass because this value is definitely known. 
# therefore we will choose proxy_min_val such that the upper bound of the input array from the forward pass is reconstructed.
# note that this includes scaling by the maximum of the input to the inverse method since this spec might not peak at 1 meaning that 
# if we don't scale by the max we might undershoot the upper bound of the input array (which we aimed to match exactly) which would make the recon low in volume.
# in the above sentences we by "input array" mean the array that goes into the "shift into the positive range" step" from the forward passes of the transforms.

# ATTENTION: in the ranges that you write in the comments do not use the values from the config file but only the names of the config file values.
#            that way changes to the config file values will not break the ranges in the comments.

# TODO (later): all methods need to work on tensors (cant backprop through nparray)

import yaml
import numpy as np
from utils import db_transform, to_01, db_inverse, spec_to_wav, normed_spec, print_range, undo_to_01
import global_objects
import torch

config = yaml.safe_load(open("config.yaml", "r"))




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
    db_normed_spec[db_normed_spec < config["stft_min_val"]] = config["stft_min_val"]
    clipped_db_normed_spec = db_normed_spec   
    # 2.5)
    # shift into the positive range (so that we can use to_01)
    # RANGE: [0, abs(min_val)]
    min_val = np.min(clipped_db_normed_spec)
    if min_val < 0:
        clipped_db_normed_spec += abs(min_val)
    # 3) 
    # transform to [0, 1] (which might be easier for the NN to learn)
    # RANGE: [0, 1]
    normed_clipped_db_normed_spec = to_01(clipped_db_normed_spec, config["stft_dyn_range_upper_bound"]) 
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
    # save this bc spec is unfortunately scaled by undo_to_01 but we need it in 3.5
    max = np.max(spec)  
    # 3)
    # to_01 normalization can partially be undone
    # RANGE: [0, stft_dyn_range_upper_bound]
    clipped_db_normed_spec = undo_to_01(spec, config["stft_dyn_range_upper_bound"])
    # 2.5)
    # shift back into the negative range (otherwise recon will be heavily distorted). 
    # RANGE: [-stft_dyn_range_upper_bound, 0]
    pseudo_min_val = -config["stft_dyn_range_upper_bound"] * max
    clipped_db_normed_spec = clipped_db_normed_spec + pseudo_min_val
    # 2)
    # clipping can't be undone since we don't know which (if any) values got clipped
    # RANGE: RANGE: [-stft_dyn_range_upper_bound, 0]
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
    db_mel_normed_spec[db_mel_normed_spec < config["log_mel_min_val"]] = config["log_mel_min_val"]
    clipped_db_mel_normed_spec = db_mel_normed_spec   
    if debug: print_range(clipped_db_mel_normed_spec)
    # 3.5)
    # shift into the positive range (so that we can use to_01)
    # RANGE: [0, -14 + abs(min_val)]
    min_val = np.min(clipped_db_mel_normed_spec)
    if min_val < 0:
        clipped_db_mel_normed_spec += abs(min_val)
    if debug: print_range(clipped_db_mel_normed_spec)
    # 4) 
    # transform to [0,1] because the output of this method is supposed to be the input to a NN and being in [0,1] might make the NN's life easier)
    # RANGE: [0, 1] 
    normed_clipped_db_mel_normed_spec = to_01(clipped_db_mel_normed_spec, config["log_mel_dyn_range_upper_bound"])
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
    # save this bc spec is unfortunately scaled by undo_to_01 but we need it in 3.5
    max = np.max(spec)  
    # 4)
    # to_01 normalization can partially be undone
    # RANGE: [0, log_mel_dyn_range_upper_bound]
    clipped_db_mel_normed_spec = undo_to_01(spec, config["log_mel_dyn_range_upper_bound"])
    if debug: print_range(clipped_db_mel_normed_spec)
    # 3.5)
    # shift back into the negative range (otherwise recon will be heavily distorted).
    # RANGE: [proxy_min_val, -14] 
    proxy_min_val = (-config["log_mel_dyn_range_upper_bound"] * max -14.668440077126899) 
    clipped_db_mel_normed_spec = clipped_db_mel_normed_spec + proxy_min_val
    if debug: print_range(clipped_db_mel_normed_spec)
    # 3)
    # clipping can't be undone
    # RANGE: [proxy_min_val, -14] 
    db_mel_normed_spec = clipped_db_mel_normed_spec
    if debug: print_range(db_mel_normed_spec)
    # 2)
    # db transform can be undone
    # RANGE: [db_inverse(proxy_min_val), 0.04]
    mel_normed_spec = db_inverse(db_mel_normed_spec)
    if debug: print_range(mel_normed_spec)
    # 1)
    # multiply by mel basis can be undone (using the pseudoinverse)
    # RANGE: [0, 1]
    # - both bounds determined from examining some ranges so not 100% guaranteed
    normed_spec = np.dot(np.linalg.pinv(global_objects.mel_basis), mel_normed_spec)
    if debug: 
        print_range(normed_spec)
        print()
    # convert back to tensor if necessary
    if not numpy:
        normed_spec = torch.from_numpy(normed_spec)

    # ATTENTION: I have noticed that normed_spec can exceed [0, 1]. 
    # - going over 1:
    #   the problem is here the imprecision of the upper bound in 1) in the forward pass. 
    #   this imprecision propagates through the forward method and therefore the upper bound of 3 is not exactly -14 either (it in reality is a bit lower).
    #   in 3.5 of the backwards pass we do however match the upper bound exactly to the (not perfectly precise) upper bound of the ouptut
    #   of 3 from the forward pass (= -14). this matching pull some values up too high which results in them exceeding 1 after the pinv_mel transform. 
    #   this problem could be improved by measuring a tighter upper bound in 1) but as long as I can't precisely prove the uper bound of the 
    #   mel_basis transform this problem will persist. 
    # - going below 0:
    #   the problem here is that we cannot reconstruct the clipped values resulting in some values in the low end of the range are too high which 
    #   by db transform are being mapped to smth that is too high (compared to what it was in the forward pass) which by the inverse mel pinv transform 
    #   are being mapped to a value that is < 0. I don't think I can do anything about this since there is a loss of information happening here. 
    # 
    # not sure this is a good solution but at least the "listen to the reconstruction" test sounds good (even with this):
    normed_spec[normed_spec > 1] = 1  
    normed_spec[normed_spec < 0] = 0

    return normed_spec




########  TEST  ########

# main function tests if the implementations of the transforms are "good" by running the
# transforms forward and backward and comparing the result of that against the original input
if __name__ == "__main__":
    import librosa
    from tifresi.utils import load_signal

    # save input as wav for reference
    filename = librosa.util.example('brahms')
    y, _ = load_signal(filename)
    y = y[:int(config["spec_x_len"] / 2)]  # less than spec_x_len because the brahms sample is not long enough
    unprocessed_spec = normed_spec(y)
    spec_to_wav(unprocessed_spec, "x_input.wav")
    
    # test spec_to_preprocessed_spec
    spec = spec_to_preprocessed_spec(unprocessed_spec, numpy=True)
    recon = spec_to_preprocessed_spec_inverse(spec, numpy=True)
    spec_to_wav(recon, "x_recon_1.wav")

    # test spec_to_mel
    mel_spec = spec_to_mel(unprocessed_spec, numpy=True, debug=True)
    recon = spec_to_mel_inverse(mel_spec, numpy=True, debug=True)
    spec_to_wav(recon, "x_recon_2.wav")