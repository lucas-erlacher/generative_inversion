import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import yaml
import global_objects
import numpy as np
import torch
from tinytag import TinyTag
import librosa

config = yaml.safe_load(open("config.yaml", "r"))

########  NORMALIZATION  ########

# turns the range of any array into [0,1] 
# assumes that the array only contains positive values
def to_01(array, dyn_range_upper_bound):  
    if np.min(array) < 0:
        raise Exception("to_01: array must only contain positive values")  
    # we had prevoiusly divided by the max value but that makes no sense since then every spec would
    # peak towards 1 and so relative loudness between different spectrograms would have been discarded
    array /= dyn_range_upper_bound
    return array

def undo_to_01(array, dyn_range_upper_bound):
    array *= dyn_range_upper_bound
    return array

def normed_spec(wave):
    spec = global_objects.stft_system.spectrogram(wave)
    # normalize to [0,1] (only if necessary)
    if np.max(spec) > 1:
        spec = to_01(spec, config["stft_dyn_range_upper_bound"])
    return spec

########  STATS  ########

# input can be either a numpy array or a torch tensor
def print_range(array):
    if (type(array) == torch.Tensor):
        array = array.detach().cpu().numpy()
    print("range: " + str(np.min(array)) + " " + str(np.max(array)))

def print_stats(array):
    print("mean: " + str(np.mean(array)))
    print("std: " + str(np.std(array)))
    print_range(array)

########  SPECTROGRAM UTILS  ########

def plot_spec(spec, title):
    plt.figure(figsize=(10, 2))
    plt.imshow(spec, cmap="afmhot_r", origin="lower", aspect="auto")
    plt.title(title)
    plt.colorbar()
    # save the plot since it won't open a window when I am on the cluster
    plt.savefig("./" + title + ".png")

########  TO WAV  ########

# ONLY APPLICABLE TO "NORMAL" SPECTROGRAMS (AND E.G. NOT MEL SPECTROGRAMS)!
# useful for finding out which spec in a chain of specs is the first to go south
def spec_to_wav(spec, title):
    wave = global_objects.stft_system.invert_spectrogram(spec)
    write(title, config['sampling_rate'], wave)

########  OTHER  ########

# array could be any numpy array so it can be a spectrogram or a waveform
def db_transform(array):
    # handle zero values
    eps = 1e-10
    array[array == 0] = eps
    return 10 * np.log10(array)

def db_inverse(array):
    return 10 ** (array / 10)

def load(path):
    data = TinyTag.get(path)
    y, _ = librosa.load(path)
    return y