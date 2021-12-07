

from typing import Callable, Optional, Tuple, List, Any, Iterable
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, VOCDetection

import numpy as np
import os
from PIL import Image

from .utils import getLogger
from .config import ROOT


class Compose(T.Compose):

    def __init__(self, transforms: List):
        super().__init__(transforms)
        assert isinstance(transforms, list), f"List of transforms required, but {type(transforms)} received ..."

    def append(self, transform: Callable):
        self.transforms.append(transform)

class IdentityTransform:

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, x: Any) -> Any:
        return x

class OrderTransform:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def append(self, transform: Callable, index: int = 0):
        self.transforms[index].append(transform)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '['
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n]'
        return format_string

    def __call__(self, data: Tuple) -> List:
        return [transform(item) for item, transform in zip(data, self.transforms)]


class TinyImageNet(ImageFolder):
    filename = "tiny-imagenet-200"
    def __init__(
        self, root, split='train', transform=None, target_transform=None, **kwargs,
    ):
        root = os.path.join(root, self.filename, split)
        super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)


class VOC2007(VOCDetection):
    year_ = '2007'
    filename_ = "VOC2007"
    class_dict = { # 'background': 0
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20
    }
    def __init__(
        self, root, image_set='train', download=False
    ):
        root = os.path.join(root, self.filename_)
        super().__init__(
            root, year=self.year_, image_set=image_set, download=download
        )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        data = target['annotation']
        objs = data['object']
        if isinstance(objs, dict): # avoid a single obj
            objs = [objs]
        boxes = []
        labels = []
        iscrowd = []
        for obj in objs:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # check whether an invalid box
            if xmax <= xmin or ymax <= ymin:
                getLogger().warning(f"[Warning] in '{data['filename']}', there are some bbox w/h <=0")
                continue
           
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        target = dict(
            image_id=torch.tensor([index]),
            boxes=boxes, labels=labels,
            area=area, iscrowd=iscrowd
        )
        return img, target

    @staticmethod
    def collate_fn(batch: Iterable):
        return tuple(zip(*batch))


class VOC2012(VOC2007):
    year_ = "2012"
    filename_ = "VOC2012"

class WrapperSet(Dataset):

    def __init__(
        self, dataset: Dataset, transforms: str
    ) -> None:
        """
        Args:
            dataset: dataset;
            transforms: string spilt by ',', such as "tensor,none'
        """
        super().__init__()

        self.data = dataset

        transforms = transforms.split(',')
        self.transforms = [AUGMENTATIONS[transform] for transform in transforms]
        if len(self.transforms) == 1:
            self.transforms = self.transforms[0]
        else:
            self.transforms = OrderTransform(self.transforms)
        getLogger().info(self.transforms)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transforms(data)

    @property
    def collate_fn(self):
        if hasattr(self.data, "collate_fn"):
            return self.data.collate_fn
        else:
            return None


AUGMENTATIONS = {
    'none' : Compose([IdentityTransform()]),
    'tensor': Compose([T.ToTensor()]),
    'voc': Compose([
        T.RandomHorizontalFlip(0.5),
        T.ToTensor()
    ]),
    'cifar': Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
    ]),
    'tinyimagenet': Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor()
    ]),
}

