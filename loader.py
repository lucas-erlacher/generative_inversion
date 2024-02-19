# This class is a wrapper for pytorch Dataloaders (train and test). The wrapper is convenient because I can use it's constructor to load the Huggingface dataset.
# I am (at least for now) choosing to not implement pytorch lightning Dataloaders beacuse I am not sure how to create them from a Huggingface dataset. 

import os
from utils_lucas import normed_spec, load
from transforms import spec_to_mel
from torch.utils.data import DataLoader, IterableDataset
import global_objects


################  DATA GENERATOR  ################

# DEBUGGING
debug = False
debug_limit_folders = 1
debug_limit_files = 20  # code breaks if this is too low e.g. 2

# TODO: this is might not be good since I might have some systematic biases in both the train and test sets. 
dir = "/itet-stor/elucas/net_scratch/maestro-v3.0.0"
train_dirs = ["2004", "2006", "2008", "2009", "2011", "2013", "2014"]
test_dirs = ["2015", "2017", "2018"]

def data_generator(valid_dirs, spec_len):
    # this is a hack that I will leave in beacuse the deadline of my project is approaching and I do not have time to implement a proper solution:
    # with the while True loop I can run more than one epoch of training (which is of course necessary) ...
    # doing this in the except block of the Dataset would be more elegant but that just doesn't want to work for some reason ...
    # with this hack the max_epochs parameter doesn't work and I will have to manually stop training 
    # (but as said above we will keep this hack because we are running out of time) 
    while True:
        # iterate over all subdirectories in the directory
        subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
        # filter for the valid dirs (this is how we enforce the train/test split (for now at least))
        subfolders = [subfolder for subfolder in subfolders if subfolder.split("/")[-1] in valid_dirs]
        for subfolder in subfolders[:(debug_limit_folders if debug else len(subfolders))]: 
            # iterate over all files in the subdirectory
            filenames = os.listdir(subfolder)
            for filename in filenames[:(debug_limit_files if debug else len(filenames))]:
                filepath = subfolder + "/" + filename
                if filepath.endswith(".wav"):
                    # load the file
                    audio = load(filepath)
                    # split the audio into piece of length x_len (audio is np.ndarray)
                    if len(audio) < spec_len:
                        # if an audio clip is too short, skip it
                        continue
                    pieces = [audio[i:i + spec_len] for i in range(0, len(audio), spec_len)]
                    # iterate over the pieces
                    for piece in pieces:
                        if len(piece) < spec_len:
                            # skip last piece of split if it is shorter than x_len
                            continue
                        # generate the entries of the tuple
                        unprocessed_spec = normed_spec(piece)
                        log_mel_spec = spec_to_mel(unprocessed_spec, numpy=True, debug=False)
                        yield (log_mel_spec, unprocessed_spec)


################  DATALOADER  ################
                    
# these two values were obtained by running get_dataset_len
train_len = 341760
test_len = 70042

class CustomDataset(IterableDataset):

    def __init__(self, generator_func, valid_dirs, spec_len):  # split_type is either "train" or "test"
        self.generator = generator_func(valid_dirs, spec_len)

    def __iter__(self):
        return self.generator

    def __getitem__(self, _):
        try:
            return next(self.generator)
        except Exception:
            pass

class Loader:
    def __init__(self):
        self.num_workers = 1  # using more for some reason leads to RuntimeErrors
        self.train_dataset = CustomDataset(data_generator, train_dirs, global_objects.config["spec_x_len"])
        self.test_dataset = CustomDataset(data_generator, test_dirs, global_objects.config["eval_spec_x_len"])
        self.quick_eval_dataloader = CustomDataset(data_generator, test_dirs, global_objects.config["eval_spec_x_len"])

    def get_train_loader(self, batch_size):
        return DataLoader(self.train_dataset, batch_size=batch_size)
    
    def get_test_loader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size)
    
    def get_quick_eval_loader(self, batch_size):
        return DataLoader(self.quick_eval_dataloader, batch_size=batch_size)


################  HELPER FUNCTIONS  ################

# count how many tuples would be produced by the data_generator if fully run
def count_datatset_len(valid_dirs, spec_len):
    counter = 0
    # iterate over all subdirectories in the directory
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    # filter for the valid dirs (this is how we enforce the train/test split (for now at least))
    subfolders = [subfolder for subfolder in subfolders if subfolder.split("/")[-1] in valid_dirs]
    for subfolder in subfolders[:(debug_limit_folders if debug else len(subfolders))]: 
        # iterate over all files in the subdirectory
        filenames = os.listdir(subfolder)
        for filename in filenames[:(debug_limit_files if debug else len(filenames))]:
            filepath = subfolder + "/" + filename
            if filepath.endswith(".wav"):
                # load the file
                audio = load(filepath)
                # split the audio into piece of length x_len (audio is np.ndarray)
                if len(audio) < spec_len:
                    # if an audio clip is too short, skip it
                    continue
                pieces = [audio[i:i + spec_len] for i in range(0, len(audio), spec_len)]
                # iterate over the pieces
                for piece in pieces:
                    if len(piece) < spec_len:
                        # skip last piece of split if it is shorter than x_len
                        continue
                    counter += 1
    return counter

if __name__ == "__main__":
    print(count_datatset_len(train_dirs, global_objects.config["spec_x_len"]))
    print(count_datatset_len(test_dirs, global_objects.config["eval_spec_x_len"]))