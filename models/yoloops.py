

from typing import Iterable, Tuple, List, Dict, Optional, Union
import torch
from torch import Tensor, flatten, tensor, triangular_solve
import torch.nn.functional as F

import random

from .base import ObjDetectionModule
from .rcnnops import ImageList_



class AbsRelTransform(ObjDetectionModule):

    def __init__(
        self, min_size: int, max_size: int, 
        after_batches: int = 10, stride: int = 32
    ) -> None:
        super().__init__()

        assert min_size % stride == 0 and max_size % stride == 0, f"image size should be times of {stride}"
        self.min_size = min_size
        self.max_size = max_size
        self.__size = max_size
        self.batch_count = 0
        self.after_batches = after_batches
        self.stride = stride

    @property
    def size(self):
        if self.training:
            return self.__size
        else:
            return self.max_size

    def step(self):
        self.batch_count += 1
        self.__size = random.choice(
            range(self.min_size, self.max_size, self.stride)
        )

    def resize(self, image: Tensor):
        image = image.unsqueeze(0)
        return F.interpolate(image, self.size, mode='nearest')

    @staticmethod
    def abs2rel(boxes: Tensor, img_size):
        """
        Args:
            boxes: Tensor, N x 4, (xmin, ymin, xmax, ymax)
            img_size: (H, W)
        """
        H, W = img_size
        boxes = boxes.clone()

        boxes[:, 2] -= boxes[:, 0] # w
        boxes[:, 3] -= boxes[:, 1] # h
        boxes[:, 0] += boxes[:, 2] / 2 # x_center
        boxes[:, 1] += boxes[:, 3] / 2 # y_center

        boxes[:, 2] /= W # w [0, 1]
        boxes[:, 3] /= H # h [0, 1]
        boxes[:, 0] /= W # x [0, 1]
        boxes[:, 1] /= H # y [0, 1]

        return boxes # (x, y, w, h)

    @staticmethod
    def rel2abs(boxes: Tensor, img_size):
        """
        Args:
            boxes: (x + ij, y + ij, w, h)
        """
        H, W = img_size
        boxes = boxes.clone()

        boxes[:, 0] *= W
        boxes[:, 1] *= H
        boxes[:, 2] *= W
        boxes[:, 3] *= H

        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        return boxes # (xmin, ymin, xmax, ymax)


    def forward(self, images: List, targets: Optional[List] = None) -> Tuple[ImageList_, Tensor]:
        self.step()
        image_sizes = [(img.size(-2), img.size(-1)) for img in images]
        images = [self.resize(img) for img in images]
        images = torch.cat(images, dim=0)
        images = ImageList_(
            tensors=images,
            image_sizes=image_sizes
        )
        if targets is not None:
            new_targets = []
            for i, (img_size, target) in enumerate(zip(image_sizes, targets)):
                boxes = target['boxes'] # N x 4
                boxes = self.abs2rel(boxes, img_size)
                boxes = torch.cat((torch.zeros((boxes.size(0), 2)).to(boxes.device), boxes), dim=1) # N x 6
                boxes[:, 0] = float(i)
                boxes[:, 1] = target['labels'].float() - 1 # start from zero
                new_targets.append(boxes)
            targets = torch.cat(new_targets, dim=0) # BN x 6
            return images, targets
        else:
            return images, None




def box_iou_wh(boxes1, boxes2):

    area1 = boxes1[:, 0] * boxes1[:, 1]
    area2 = boxes2[:, 0] * boxes2[:, 1]

    inter = torch.min(boxes1, boxes2).prod(dim=-1)

    iou = inter / (area1 + area2 - inter)
    return iou


def box_iou_xywh(boxes1, boxes2):
    b1x1, b1x2 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 0] + boxes1[:, 2] / 2
    b1y1, b1y2 = boxes1[:, 1] - boxes1[:, 3] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
    b2x1, b2x2 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 0] + boxes2[:, 2] / 2
    b2y1, b2y2 = boxes2[:, 1] - boxes2[:, 3] / 2, boxes2[:, 1] + boxes2[:, 3] / 2

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)

    x1, x2 = torch.max(b1x1, b2x1), torch.min(b1x2, b2x2)
    y1, y2 = torch.max(b1y1, b2y1), torch.min(b1y2, b2y2)
    inter = (x2 - x1) * (y1 - y2)

    iou = inter / (area1 + area2 - inter)
    return iou