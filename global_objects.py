# file containing instantiations of objects that are used by multilpe classes 
# and therefore have to be consistent in the parameters they were set up with

import yaml
import librosa
from tifresi.stft import GaussTF

config = yaml.safe_load(open("config.yaml", "r"))

stft_system = GaussTF(hop_size=config["stft_hop_size"], stft_channels=config["stft_num_channels"])
mel_basis = librosa.filters.mel(sr=config["sampling_rate"], n_fft=config["stft_num_channels"], n_mels=config["mel_num_channels"])