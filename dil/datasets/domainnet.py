from typing import Tuple, Type
from argparse import Namespace
import os
import numpy as np
import random

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from dil.backbones.mnistmlp import MNISTMLP
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from dil.datasets.transforms.permutation import FixedPermutation
from dil.datasets.utils.continual_dataset import ContinualDataset
from dil.datasets.utils.validation import get_train_val
# from utils.conf import base_path_dataset as base_path

class DomainNet(Dataset):
    """
    Overrides the dataset to change the getitem function.
    """
    def __init__(self, domain_id=0, mode='train', transform=None, target_transform=None):
        super().__init__()
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        self.target_transform = target_transform

        self.mode = mode
        self.data_root = "/root/dataset/DIL/DomainNet"
        self.image_list_root = "/root/dataset/DIL/DomainNet/lists"
        self.task_name = ['real','quickdraw','painting','sketch','infograph','clipart']
        self.class_name = {
            'wine_glass': 0,
            'zebra': 1,
            'zigzag': 2,
            'whale': 3,
            'tiger': 4,
            'bee': 5,
            'sun': 6,
            'bird': 7,
            'circle': 8,
            'fish': 9
        }
        self.domain = self.task_name[domain_id]
        self.image_list_path = os.path.join(self.image_list_root, self.domain + "_" + self.mode + ".txt")
        self.path = self.get_path()
        self.length = len(self.path)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.path[index]
        img_path = os.path.join(self.data_root, img_path)
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == 'train':
            return img, target, index
        elif self.mode == 'test':
            return img, target, index
    
    def get_path(self):
        images = []
        image_list = open(self.image_list_path).readlines()
        images += [
            (val.split()[0],
             self.class_name[val.split('/')[1]])
            for val in image_list if val.split('/')[1] in self.class_name
            ]
        if self.mode == 'train':
            random.shuffle(images)  
        return images


class DomainNetAll(Dataset):
    def __init__(self, mode='train', transform=None):
        super().__init__()
        self.transform = transforms.Compose(
            [transforms.Resize([32, 32]),
            transforms.ToTensor()]
            )
        self.mode = mode
        self.data_root = "/root/dataset/DIL/DomainNet"
        self.image_list_root = "/root/dataset/DIL/DomainNet/lists"
        self.task_name = ['real','quickdraw','painting','sketch','infograph','clipart']
        self.class_name = {
            'wine_glass': 0,
            'zebra': 1,
            'zigzag': 2,
            'whale': 3,
            'tiger': 4,
            'bee': 5,
            'sun': 6,
            'bird': 7,
            'circle': 8,
            'fish': 9
        }
        self.path = self.get_path()
        self.length = len(self.path)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index: int):

        img_path, target, domain = self.path[index]
        img_path = os.path.join(self.data_root, img_path)
        img = Image.open(img_path)
        # original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 'train':
            return img, target, index
        elif self.mode == 'test':
            return img, target, index
    
    def get_path(self):
        images = []
        for di, domain in enumerate(self.task_name):
            image_list_path = os.path.join(self.image_list_root, domain + "_" + self.mode + ".txt")
            image_list = open(image_list_path).readlines()
            images += [
                (
                    val.split()[0],
                    self.class_name[val.split('/')[1]],
                    int(di)
                ) for val in image_list if val.split('/')[1] in self.class_name
                ]
        if self.mode == 'train':
            random.shuffle(images)  
        return images


class SequentialDomainNet(ContinualDataset):

    NAME = 'domain-net'
    N_TASKS = 6
    INDIM = (3, 32, 32)
    N_CLASSES = 10
    N_CLASSES_PER_TASK = 10
    
    def __init__(self, args: Namespace, distill):
        super().__init__(args)
        self.distill=distill
        self.setup_loaders()

    def set_joint(self):
        self.train_loaders = [
            DataLoader(DomainNetAll(mode='train'), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers,pin_memory=True),
            ]
        self.N_TASKS=1
    
    def get_current_train_loader(self):
        return self.train_loaders[self.i]
    
    def get_current_test_loader(self):
        return self.test_loaders[:self.i+1]
    
    def get_data_loaders(self):
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]

        next_train, next_test = None, None
        if self.i+1 < self.N_TASKS:
            next_train = self.train_loaders[self.i+1]
            next_test = self.test_loaders[self.i+1]
        
        return current_train, current_test, next_train, next_test
        
    def setup_loaders(self):
        self.test_loaders, self.train_loaders = [], []
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        for i in range(self.N_TASKS):
            if self.distill:
                train_transform = transforms.Compose([
                    transforms.Resize((self.INDIM[1], self.INDIM[2])),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.Resize((self.INDIM[1], self.INDIM[2])),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    normalize
                ])
            
            test_transform = transforms.Compose([
                transforms.Resize((self.INDIM[1], self.INDIM[2])),
                transforms.ToTensor(),
                normalize
            ])
            trainset_full = DomainNet(i, transform=train_transform)
            testset_full = DomainNet(i, transform=test_transform)

            length = len(trainset_full)
            train_length = int(0.8 * length)
            test_length = length - train_length
            train_dataset, _ = random_split(trainset_full, [train_length, test_length], generator=torch.Generator().manual_seed(3407))
            _, test_dataset = random_split(testset_full, [train_length, test_length], generator=torch.Generator().manual_seed(3407))

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)

    @staticmethod
    def get_backbone():
        return MNISTMLP(3 * 128 * 128, SequentialDomainNet.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 128

    @staticmethod
    def get_minibatch_size():
        return SequentialDomainNet.get_batch_size()


