from argparse import Namespace
from pathlib import Path
from typing import List, Tuple
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

from dil.datasets.utils.continual_dataset import ContinualDataset
from dil.backbones.mnistmlp import MNISTMLP


PACS_ROOT = "/workspace/datasets/DIL/PACS"
if not os.path.exists(PACS_ROOT):
    PACS_ROOT = "/root/dataset/DIL/PACS"
IMAGE_DIR = os.path.join(PACS_ROOT, "images")
TEXT_DIR = os.path.join(PACS_ROOT, "texts")

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']  


def _build_class_to_idx(root_dir: str):
    present = set([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
    if not set(PACS_CLASSES).issubset(present):
        pass
    return {c: i for i, c in enumerate(PACS_CLASSES)}

def _dataset_info(txt_file: str,
                  domain_root: str,
                  class_to_idx: dict):

    img_rel_list, labels = [], []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rel = parts[0]
            if len(parts) >= 2:
                y = int(parts[1])
            else:
                cls_name = Path(rel).parts[0]
                y = class_to_idx[cls_name]
            img_rel_list.append(rel)
            labels.append(y-1)

    # 존재 확인과 절대 경로 변환
    img_abs_list = []
    for rel in img_rel_list:
        p = os.path.join(domain_root, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Can't find:  {p}")
        img_abs_list.append(p)
    return img_abs_list, labels

class PACS(Dataset):
    def __init__(self,
                 items: List[Tuple[str, int]],
                 transform=None,
                 domain_label: int = 0):
        self.items = items
        self.transform = transform
        self.loader = default_loader
        self.domain_label = domain_label
        # 선택적으로 접근 가능
        self.targets = [y for _, y in items]
        self.paths = [p for p, _ in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        path, target = self.items[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index

class SequentialPACS(ContinualDataset):
    NAME = 'pacs'
    N_CLASSES_PER_TASK = 7
    N_TASKS = len(PACS_DOMAINS)
    INDIM = (3, 32, 32)
    MAX_N_SAMPLES_PER_TASK = 16000

    def __init__(self,
                 args: Namespace,
                 distill: bool = False,
                 size: int = 32,
                 txt_dir: str = TEXT_DIR,
                 domains: List[str] = None):
        super().__init__(args)
        self.distill = distill
        self.size = size
        self.txt_dir = txt_dir
        self.domains = domains if domains is not None else PACS_DOMAINS
        self.N_TASKS = len(self.domains)
        self.INDIM = (3, size, size)
        
        self.setup_loaders()

    def _build_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        tr = [
            transforms.Resize((self.size, self.size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        te = [
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ]
        if not self.distill:
            tr.append(normalize)
            te.append(normalize)
        return transforms.Compose(tr), transforms.Compose(te)

    def _read_split(self, domain: str, domain_label: int):
        domain_root = IMAGE_DIR
        class_to_idx = _build_class_to_idx(domain_root)

        train_txt    = os.path.join(self.txt_dir, f"{domain}_train_kfold.txt")
        crossval_txt = os.path.join(self.txt_dir, f"{domain}_crossval_kfold.txt")

        tr_names, tr_labels = _dataset_info(train_txt,    domain_root, class_to_idx)
        te_names, te_labels = _dataset_info(crossval_txt, domain_root, class_to_idx)

        train_domain_labels = [domain_label] * len(tr_labels)
        test_domain_labels  = [domain_label] * len(te_labels)

        return (
            tr_names,    tr_labels,    train_domain_labels,
            te_names,    te_labels,    test_domain_labels
        )

    def setup_loaders(self):
        self.test_loaders, self.train_loaders = [], []
        train_tf, test_tf = self._build_transforms()

        for d_idx, domain in enumerate(self.domains):
            tr_names, tr_labels, tr_dlabels, te_names, te_labels, te_dlabels = \
                self._read_split(domain, d_idx)

            tr_items = list(zip(tr_names, tr_labels))
            te_items = list(zip(te_names, te_labels))

            tr_ds = PACS(tr_items, transform=train_tf, domain_label=d_idx)
            te_ds = PACS(te_items, transform=test_tf,  domain_label=d_idx)

            tr_loader = DataLoader(tr_ds,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=True)
            te_loader = DataLoader(te_ds,
                                   batch_size=self.args.batch_size,
                                   shuffle=False,
                                   num_workers=self.args.num_workers,
                                   pin_memory=True)

            tr_ds.targets_domain = tr_dlabels
            te_ds.targets_domain = te_dlabels

            self.train_loaders.append(tr_loader)
            self.test_loaders.append(te_loader)

    def set_joint(self):
        from torch.utils.data import ConcatDataset
        comb = ConcatDataset([ldr.dataset for ldr in self.train_loaders])
        self.train_loaders = [DataLoader(
            comb,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True)]
        self.N_TASKS = 1

    def get_current_train_loader(self):
        return self.train_loaders[self.i]

    def get_current_test_loader(self):
        return self.test_loaders[:self.i + 1]

    def get_data_loaders(self):
        cur_tr = self.train_loaders[self.i]
        cur_te = self.test_loaders[self.i]
        nxt_tr = self.train_loaders[self.i + 1] if self.i + 1 < self.N_TASKS else None
        nxt_te = self.test_loaders[self.i + 1] if self.i + 1 < self.N_TASKS else None
        return cur_tr, cur_te, nxt_tr, nxt_te

    @staticmethod
    def get_backbone():
        C, H, W = SequentialPACS.INDIM
        return MNISTMLP(C * H * W, SequentialPACS.N_CLASSES_PER_TASK)

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
        return SequentialPACS.get_batch_size()
