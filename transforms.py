# Implementations of transformations that are used in multiples places in the project and therefore have to be consistent across each other. 
#
# By default the inputs and outputs of the functions are expected to be torch tensors (since these functions will be used as part of a DL model/pipeline). 
# Alternatively the numpy flag can be set to true (is false by default) which will make the function accept and return numpy arrays.

# TODO: do the checks in the main function return a satisfying result?

import yaml
import numpy as np
from utils import db_transform, to_01, db_inverse, print_range
import global_objects
import torch

config = yaml.safe_load(open("config.yaml", "r"))




########  WAVE TO STFT  ########

# PURPOSE:
# - generate the target element of a training data pair
#
# ASSUMPTIONS ON INPUT:
# not much, it is just a torch tensor that represents a spectrogram (no normalization or anything like that required)
def spec_to_preprocessed_spec(spec, numpy=False):
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        spec = spec.numpy()
    # 1) 
    # normalize spec between 0 and 1
    # RANGE: [0, 1]
    normed_spec = to_01(spec)    
    # 2) 
    # db transform
    # RANGE: [-infty, 0]
    db_normed_spec = db_transform(normed_spec)   
    # 3)
    # clip very negative values
    # RANGE: [stft_min_val, 0]
    db_normed_spec[db_normed_spec < config["stft_min_val"]] = config["stft_min_val"]
    clipped_db_normed_spec = db_normed_spec    
    # 4) 
    # transform to [0, 1] (which might be easier for the NN to learn)
    # RANGE: [0, 1]
    normed_clipped_db_normed_spec = to_01(clipped_db_normed_spec)    
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
        audio = audio.numpy()
    # assert that spec is in [0,1]
    if np.min(spec) < 0 or np.max(spec) > 1:
        raise Exception("spec_to_preprocessed_spec_inverse: spec must be in [0,1]")
    # 4)
    # transform from [0, 1] back to [stft_min_val, 0] can't be done 
    # RANGE: [0,1]
    clipped_db_normed_spec = spec
    # 3)
    # clipping can't be undone
    # RANGE: [0,1]
    db_normed_spec = clipped_db_normed_spec
    # 2)
    # db transform can be undone
    # RANGE: [1, 1.2589]
    normed_spec = db_inverse(db_normed_spec)
    # 1)
    # normalization can again not be undone
    # RANGE: [1, 1.2589]
    spec = normed_spec
    # turn spec into array representing waveform
    wave_array = global_objects.stft_system.invert_spectrogram(spec)
    # convert back to tensor if necessary
    if not numpy:
        wave_array = torch.from_numpy(wave_array)
    return wave_array




########  SPECTROGRAM TO MEL  ########

# TODO: 
# - check all ranges of forward and backward transforms

# PURPOSE:
# - generate the input element of a training data pair
# 
# ASSUMPTIONS ON INPUT:
# spec is assumed to be a fresh spectrogram that has not had any processing applied to it. 
# - We don't want it to contain any processing because e.g. if it contains a db transformation then we would be applying a second db transformation
#   in this method (and this is not what we understand a log_mel_sepc to be (see Tagebuch for what we do understand it to be)). 
def spec_to_mel(spec, numpy=False):
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        spec = spec.numpy()
    # 1) 
    # multiply by mel basis
    # RANGE: [0,1] (determined from examining some ranges so not 100% guaranteed)
    mel_spec = np.dot(global_objects.mel_basis, spec) 
    # 2) 
    # db transform (I think that this is the "log" operation that everybody is referring to when they say "log mel spectrogram")
    # RANGE: [-infty,0] (uncertainty inherited from mel_spec)
    db_mel_spec = db_transform(mel_spec)    
    # 3) 
    # clip very negative values
    # RANGE: [log_mel_min_val, 0] (uncertainty inherited from db_mel_spec)
    db_mel_spec[db_mel_spec < config["log_mel_min_val"]] = config["log_mel_min_val"]
    clipped_db_mel_spec = db_mel_spec    
    # 4) 
    # transform to [0,1] because the output of this method is supposed to be the input to a NN and being in [0,1] might make the NN's life easier)
    # RANGE: [0,1] (uncertainty inherited from clipped_db_mel_spec)
    normed_clipped_db_mel_spec = to_01(clipped_db_mel_spec) 
    # convert back to tensor if necessary
    if not numpy:
        normed_clipped_db_mel_spec = torch.from_numpy(normed_clipped_db_mel_spec)
    return normed_clipped_db_mel_spec

# PURPOSE:
# - sanity check the implementation of spec_to_mel
# - generate a spectrogram from a mel spectrogram that we then apply spec_to_preprocessed_spec to which results in a first approximaton
#   of a preprocessed spectrogram (i.e. the target element of a dataset pair) that can later be refined by adding the output of a NN
#
# ASSUMPTIONS ON INPUT:
# well, it has to come from spec_to_mel but the only thing we can really check/assert about it is that it is in [0,1]
def spec_to_mel_inverse(spec, numpy=False):
    # convert to numpy array if we are taking in a torch tensor
    if not numpy:
        log_mel_spec = log_mel_spec.numpy()
    # assert that spec is in [0,1]
    if np.min(spec) < 0 or np.max(spec) > 1:
        raise Exception("spec_to_mel_inverse: spec must be in [0,1]")
    # 4)
    # transform from [0, 1] back to [log_mel_min_val, 0] can't be done
    # RANGE: [0,1]
    clipped_db_mel_spec = spec
    # 3)
    # clipping can't be undone
    # RANGE: [0,1]
    db_mel_spec = clipped_db_mel_spec
    # 2)
    # db transform can be undone
    # RANGE: [1, 1.2589]
    mel_spec = db_inverse(db_mel_spec)
    # 1)
    # multiply by mel basis can be undone (using the pseudoinverse)
    # RANGE: [0, ???] (determined from examining some ranges so not 100% guaranteed)
    spec = np.dot(np.linalg.pinv(global_objects.mel_basis), mel_spec)
    # convert back to tensor if necessary
    if not numpy:
        spec = torch.from_numpy(spec)
    return spec




########  TEST  ########

# main function tests if the implementations of the transforms are "good" by running the
# transforms forward and backward and comparing the result of that against the original input
if __name__ == "__main__":
    import librosa
    from tifresi.utils import load_signal

    # TODO: I dont think the steps in this test method respect the newly added assumptions that the transforms make about their inputs
    '''
    filename = librosa.util.example('brahms')
    y, _ = load_signal(filename)
    y = y[:int(config["spec_x_len"] / 2)]  # less than spec_x_len because the brahms sample is not long enough
    array_to_wav(y, "x_input.wav")
    # test wave to stft
    spec = spec_to_preprocessed_spec(y)
    recon = spec_to_preprocessed_spec_inverse(spec)
    array_to_wav(recon, "x_recon_1.wav")
    # test spec to mel
    mel_spec = spec_to_mel(spec)
    recon = spec_to_mel_inverse(mel_spec)
    wave = global_objects.stft_system.invert_spectrogram(spec)
    array_to_wav(wave, "x_recon_2.wav")
    '''