import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import segmentation_transforms as T
import config as C


class TrainingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.images, self.labels, _, self.questions = dataset.load_training_list()

        self.transforms = T.Sequential(
            T.RandAugment(2, 10, dataset.ignore_index),
            T.Resize(longer=C.resize_longer),
            T.Pad(C.crop_size, dataset.ignore_index),
            T.Crop(C.crop_size),
            T.Flip(),
            T.ToTensor()
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.dataset.load_files_pil(self.images[idx], self.labels[idx])
        image, label = self.transforms(image, label)
        return image, label, self.questions[idx]


class ValidationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.images, self.labels, _, self.questions = dataset.load_validation_list()

        self.transforms = T.Sequential(
            T.Resize(longer=max(C.validation_crop_size)),
            T.Pad(C.validation_crop_size, dataset.ignore_index, random=False),
            T.Crop(C.validation_crop_size, random=False),
            T.ToTensor()
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.dataset.load_files_pil(self.images[idx], self.labels[idx])
        image, label = self.transforms(image, label)
        return image, label, self.questions[idx]


def create_data_loaders(dataset):
    training_dataset = TrainingDataset(dataset)
    training_sampler = DistributedSampler(training_dataset, shuffle=True, drop_last=True)
    training_loader = DataLoader(training_dataset, C.batch_size, num_workers=C.n_workers, sampler=training_sampler, drop_last=True, persistent_workers=True)
    validation_dataset = ValidationDataset(dataset)
    validation_sampler = DistributedSampler(validation_dataset, shuffle=False)
    validation_loader = DataLoader(validation_dataset, C.validation_batch_size, num_workers=C.n_workers, sampler=validation_sampler, persistent_workers=True)
    return training_loader, training_sampler, validation_loader
