# generates set of tuples of (spectrogram, log mel spectrogram) from a "corpus" of recordings. 

import os
import yaml
from tifresi.utils import load_signal
from transforms import spec_to_preprocessed_spec, spec_to_mel
from datasets import Dataset
import global_objects

# debug on a small subset of the dataset
debug = True
debug_limit_folders = 1
debug_limit_files = 3

def data_generator():
    config = yaml.safe_load(open("config.yaml", "r"))
    dir = "maestro-v3.0.0"
    # iterate over all subdirectories in the directory
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    for subfolder in subfolders[:(debug_limit_folders if debug else len(subfolders))]: 
        # iterate over all files in the subdirectory
        filenames = os.listdir(subfolder)
        for filename in filenames[:(debug_limit_files if debug else len(filenames))]:
            filepath = subfolder + "/" + filename
            if filepath.endswith(".wav"):
                # load the file
                audio, sr = load_signal(filepath)
                # split the audio into piece of length x_len (audio is np.ndarray)
                if len(audio) < config["spec_x_len"]:
                    # if an audio clip is too short, skip it
                    continue
                pieces = [audio[i:i + config["spec_x_len"]] for i in range(0, len(audio), config["spec_x_len"])]
                # iterate over the pieces
                for piece in pieces:
                    if len(piece) < config["spec_x_len"]:
                        # skip last piece of split if it is shorter than x_len
                        continue
                    # generate the entries of the tuple
                    unprocessed_spec = global_objects.stft_system.spectrogram(piece)
                    preprocesed_spec = spec_to_preprocessed_spec(unprocessed_spec, numpy=True)
                    log_mel_spec = spec_to_mel(unprocessed_spec, numpy=True)
                    # yield the tuple
                    yield {"preprocesed_spec": preprocesed_spec, "log_mel_spec": log_mel_spec}

if __name__ == "__main__":
    ds = Dataset.from_generator(data_generator)
    ds.save_to_disk("dataset")