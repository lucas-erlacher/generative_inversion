import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import yaml
import global_objects


config = yaml.safe_load(open("config.yaml", "r"))

def plot_spec(spec, title):
    plt.figure(figsize=(10, 2))
    plt.imshow(spec, cmap="afmhot_r", origin="lower", aspect="auto")
    plt.title(title)
    plt.colorbar()
    # save the plot
    plt.savefig("./" + title + ".png")

# useful for fining out which spec in a chain of specs is the first to go south
def spec_to_wav(spec, title):
    array = global_objects.stft_system.invert_spectrogram(spec)
    array_to_wav(array, title)

# I am putting this into a function such that the same sample rate is used everywhere
def array_to_wav(array, title):
    write(title, config["sampling_rate"], array)