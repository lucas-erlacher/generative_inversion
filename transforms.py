# implementations of transformations that are used in multiples places in the project and therefore have to be consistent across each other. 

# TODO: do the checks in the main function return a satisfying result?

import yaml
import numpy as np
from tifresi.transforms import log_mel_spectrogram
from utils import plot_spec, array_to_wav, spec_to_wav
import global_objects

config = yaml.safe_load(open("config.yaml", "r"))

######## WAVE TO STFT ########

# TODO: norm to [0,1] (divide by max(spec)), db_transform (10*log_10(x)), clip (x<10^-5 = 10^-5), range is now [-50, 0] so transform into [0,1] (might be easier for NN to learn)

# returns the log of the normalized magnitude part of an STFT spectrogram that is derived from an audio signal
def wave_to_stft(audio):
    spec = global_objects.stft_system.spectrogram(audio)
    # log the spectrogram
    log_spec = np.log(spec)
    # cap large negative values to maybe something like -50
    # we need to do this bc otherwise we would not know what to shift the values by 
    # (but we need to shift them in order to be able to normalize them into [0, 1])
    log_spec[log_spec < config["stft_min_val"]] = config["stft_min_val"]
    # make all vals positive 
    log_spec += np.abs(config["stft_min_val"])
    max_val = np.max(log_spec)
    if max_val == 0: 
        # in this case (which has to be handled in order to avoid an exception)
        # we know that all values in spec must have been zero, so we can just return an array of zeros
        return np.zeros(log_spec.shape)    
    # normalize it between 0 and 1
    normed_log_spec = log_spec / (max_val)
    return normed_log_spec

# TODO: forward will change, so this will have to change too
def wave_to_stft_inverse(normed_log_spec):
    # we don't know what max_value was so we can't multiply by it here. 
    #
    # we do know that we added config["stft_min_val"] to all values HOWEVER we do not subtract it here because 
    # this will drive the values so far into the negative that exp-ing them results in a value that is too small
    # for pghi (which is used by stft_system.invert_spectrogram)
    #
    # the clipping can't be undone (because we don't know which values were clipped)
    #
    # undo the log
    spec = np.exp(normed_log_spec) 
    # inverse stft
    wave = global_objects.stft_system.invert_spectrogram(spec)
    return wave

######## SPECTROGRAM TO MEL ########

# TODO: take in spoectrogram, norm to [0,1] i.e. divide by max(spec), multiply by mel basis (just code it yourself and don't use the Tifresi method),
# clamp since the output of that can have very small negative values, db transformation, normalize to [0,1] 

# returns the log of the mel spectrogram that is derived from an STFT spectrogram
def spec_to_mel(spectrogram):
    log_mel_spec = log_mel_spectrogram(spectrogram, config["stft_num_channels"], config["mel_num_channels"])
    return log_mel_spec

# TODO: I think forward will change, so if that is the case this will have to change too

def spec_to_mel_inverse(log_mel_spec):
    # undo the log
    mel_spec = np.exp(log_mel_spec)
    # invert mel spectrogram)
    pinv = np.linalg.pinv(globals.basis)
    spec = np.dot(pinv, mel_spec)
    return spec

# main function tests if the implementations of the transforms are "good" by running the
# transforms forward and backward and comparing the result of that against the original input
if __name__ == "__main__":
    import librosa
    from tifresi.utils import load_signal

    filename = librosa.util.example('brahms')
    y, _ = load_signal(filename)
    y = y[:int(config["spec_x_len"] / 2)]  # less than spec_x_len because the brahms sample is not long enough
    array_to_wav(y, "x_input.wav")
    # test wave to stft
    spec = wave_to_stft(y)
    
    recon = wave_to_stft_inverse(spec)
    array_to_wav(recon, "x_recon_1.wav")
    # test spec to mel
    mel_spec = spec_to_mel(spec)
    recon = spec_to_mel_inverse(mel_spec)
    wave = global_objects.stft_system.invert_spectrogram(spec)
    array_to_wav(wave, "x_recon_2.wav")