import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import yaml
import global_objects
import numpy as np

config = yaml.safe_load(open("config.yaml", "r"))

########  NORMALIZATION  ########

# array is assumed to be positive valued.
# function returns an array in [0,1]. 
def max_norm(array):
    max_val = np.max(array)
    # edge case:
    if max_val == 0:
        return array  # if max_val is 0 then don't divid but just return the spec (since it is already in [0,1] if all values are 0)
    return array / max_val

# turns the range of any array into [0,1] i.e. no assumptions on the range of the input 
def to_01(array):
    # shift all values to be positive (only needed if the array contains negative values)
    min_val = np.min(array)
    if min_val < 0:
        array += abs(min_val)
    # normalize to [0,1]
    return max_norm(array)

########  STATS  ########

def print_range(array):
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

# useful for fining out which spec in a chain of specs is the first to go south
def spec_to_wav(spec, title):
    wave = global_objects.stft_system.invert_spectrogram(spec)
    write(title, config["sampling_rate"], wave)

########  OTHER  ########

# array could be any numpy array so it can be a spectrogram or a waveform
def db_transform(array):
    return 10 * np.log10(array)

def db_inverse(array):
    return 10 ** (array / 10)