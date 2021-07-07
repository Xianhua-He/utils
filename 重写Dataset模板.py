import torch
from torch.utils.data import Dataset
import cv2

class MyDataset(Dataset):
    # define variable

    def __init__(self, root='', transform=None):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.data_list = []

    def __getitem__(self, index):
        img_path = self.data_list[index]
        # img = Image.open(_image)
        # img = np.array(img.convert('RGB')).astype(np.float32)
        sample = {}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return len(self.data_list)



import os
from PIL import Image
import numpy as np

class FaceDataset(Dataset):
    # define variable
    def __init__(self, root_path='', transform=None):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.data_list = []
        for root, dirs, files in os.walk(root_path):
            for f in files:
                name = os.path.join(root, f)
                try:
                    tmp_img = Image.open(name)
                    tmp_img = np.array(tmp_img.convert('RGB')).astype(np.float32)
                    self.data_list.append(name)
                except:
                    print("failed open file: {}".format(name))
                    os.remove(name)
                    continue

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path)
        if self.transform is not None:
            q = self.transform(img)
            k = self.transform(img)
        return [q, k]

    def __len__(self):
        return len(self.data_list)