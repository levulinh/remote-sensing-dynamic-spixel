import albumentations as alb
import cv2
import glob
import os
import pandas as pd
import torch
import pdb

from abc import abstractmethod
from functools import lru_cache
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from torch_geometric.transforms import ToSLIC

cv2.setNumThreads(0)

class IDXDataset(Dataset):
    def __init__(
        self,
        transform: "alb.Transform",
        image_format: str = "*.jpg",
        datapath: Optional[str] = "./datasets/AID/train",
        labels_txt: str = "./UCMerced/multilabels.txt",
        sep: str = ','
    ):
        super().__init__()

        self.df = pd.read_csv(labels_txt, sep=sep)
        self.files = list()
        self.transform = transform

        for subfolder in os.listdir(datapath):  # classes
            paths = os.path.join(datapath, subfolder, image_format)
            self.files += glob.glob(paths)

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def get_label(self, x) -> List[int]:
        one_hot_labels = self.df[self.df["IMAGE\LABEL"] == (x.split("/")[5][:-4])].iloc[0].to_list()[1:]
        return one_hot_labels

    @lru_cache
    def cache(self, file):
        image = cv2.imread(file)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        label = self.get_label(file)
        return torch.tensor(image), torch.tensor(label)

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        f = self.files[index]
        return self.cache(f)

class SpixelDataset(Dataset):
    def __init__(
        self,
        transform: "alb.Transform",
        image_format: str = "*.jpg",
        datapath: Optional[str] = "./datasets/AID/train",
        labels_txt: str = "./UCMerced/multilabels.txt",
        sep: str = ',',
        n_segments: int = 100
    ):
        super().__init__()

        self.df = pd.read_csv(labels_txt, sep=sep)
        self.files = list()
        self.transform = transform
        self.to_slic_transform = ToSLIC(add_seg=True, add_img=True, n_segments=n_segments)

        for subfolder in os.listdir(datapath):  # classes
            paths = os.path.join(datapath, subfolder, image_format)
            self.files += glob.glob(paths)

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def get_label(self, x) -> List[int]:
        one_hot_labels = self.df[self.df["IMAGE\LABEL"] == (x.split("/")[5][:-4])].iloc[0].to_list()[1:]
        return one_hot_labels

    @lru_cache
    def cache(self, file):
        image = cv2.imread(file)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image = torch.tensor(image)
        graph = self.to_slic_transform(image)
        label = self.get_label(file)
        return [[image, graph], torch.tensor(label)]

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        f = self.files[index]
        return self.cache(f)