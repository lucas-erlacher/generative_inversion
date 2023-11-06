# This class is a wrapper for pytorch Dataloaders (train and test). The wrapper is convenient because I can use it's constructor to load the Huggingface dataset.
# I am (at least for now) choosing to not implement pytorch lightning Dataloaders beacuse I am not sure how to create them from a Huggingface dataset. 

from datasets import concatenate_datasets, Dataset
from torch.utils.data import DataLoader
import os

class Loader:
    def __init__(self):
        # the whole point of a Dataloader is to only load the data that is needed by the current batch into RAM so I am assuming that both the 
        # loading + concatentation of the arrow files and the creation of the DataLoader will not cause the entire dataset to be loaded into RAM.
        dataset_path = "./dataset"
        arrow_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(".arrow")]
        dataset = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])
        dataset_dict = dataset.train_test_split(test_size=0.1)
        self.train_dataset = dataset_dict["train"]
        self.test_dataset = dataset_dict["test"]

    def get_train_loader(self, batch_size):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_test_loader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)