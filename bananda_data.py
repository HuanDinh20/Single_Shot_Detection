import os
import pandas as pd
import torchvision
from d2l import torch as d2l
import torch
from torch.utils.data import Dataset


def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
    else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        targets.append(list(target))

    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(Dataset):
    """A customized dataset to load the banana detection dataset"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)

        print('read ' + str(len(self.features)) + (f' training examples' if
                                                   is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)
