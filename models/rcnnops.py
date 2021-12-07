

from typing import Iterable, Tuple, List, Dict, Optional, Union
import torch
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import RPNHead, AnchorGenerator
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

from .base import ObjDetectionModule


__all__ = [
    "ImageList_", "GeneralizedRCNNTransform_", 
    "IntermediateLayerGetter_",
    "RPNHead_", "AnchorGenerator_",
    "MultiScaleRoIAlign_", "TwoMLPHead_", "FastRCNNPredictor_"
]

class ImageList_(ImageList):
    """
    .tensors, .image_sizes
    """

class GeneralizedRCNNTransform_(GeneralizedRCNNTransform, ObjDetectionModule):
    """
    Inputs: 
        images: List[Tensor], N * C x Hn x Wn
        targets: Optional[Dict]
    1. normalize images;
    2. resize images; N * C x Hn' x Wn'
    3. scale the boxes accordingly
    4. make a batch of images with the same C, H, W (by padding)
    Return: ImageList[images: N x C x H x W, real_sizes: N * (Hn', Wn')]
    """


class IntermediateLayerGetter_(IntermediateLayerGetter, ObjDetectionModule):
    """
    Module wrapper that returns intermediate layers from a model
    Inputs: images: Tensor
    Returns: Dict[str, Tensors], outputname: features (x)
    """
    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return super().forward(images)


class RPNHead_(RPNHead, ObjDetectionModule):
    """
    Args:
        num_anchors: the number of anchors on each location, K
    Inputs: features: List[Tensor], L * N x C x Hl x Wl
    Returns: 
        logits: List[Tensor], L * N x K x Hl x Wl
        deltas: List[Tensor], L * N x 4K x Hl x Wl
    """
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__(in_channels, num_anchors)


class AnchorGenerator_(AnchorGenerator, ObjDetectionModule):
    """
    Inputs: images: ImageList, features: List[Tensor], L * N x C x Hl x Wl
    1. cell_anchors: List[Tensor], L * K x 4, K ratios of anchors for each level of features;
    2. grid_anchors: List[Tensor], L * (HlxWlxK) x 4, grid anchors with centers over each level;
    3. anchors: List[Tensor], N * (LxHlxWlxK) x 4, repeat it over all images
    """
    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    ):
        super().__init__(sizes=sizes, aspect_ratios=aspect_ratios)


class MultiScaleRoIAlign_(MultiScaleRoIAlign, ObjDetectionModule):
    """
    Inputs:
        features (x): Dict[str, Tensor], L * N x C x Hl x Wl
        boxes (proposals): List[Tensor], N * B x 4
        image_shapes: real image sizes, List[Tuple[int, int]], N * 2
    1. boxes -convert-> rois: Tensor, (NxB) x 5, (image_id, xmin, ymin, xmax, ymax)
    2. set the scale for each level and infer the level of each box
    3. features, rois -roi_align-> 
        roi_features: (NxB) x C x output_size[0] x output_size[1]
    Returns: 
        roi_features: (NxB) x C x output_size[0] x output_size[1]
    """
    def __init__(
        self, featmap_names: List[str], 
        output_size: Union[int, Tuple[int], List[int]] = 7,
        sampling_ratio: int = 2,
        canonical_scale: int = 224, 
        canonical_level: int = 4
    ):
        super().__init__(
            featmap_names, output_size, sampling_ratio, 
            canonical_scale=canonical_scale, canonical_level=canonical_level
        )

    
class TwoMLPHead_(TwoMLPHead, ObjDetectionModule):
    """
    Inputs:
        roi_features: Tensor, (NxB) x C x output_size[0] x out_put_size[1]
    1. roi_features -flatten-Linear-relu-Linear-relu-> box_features: (NxB) x D
    Returns:
        box_features: Tensor, (NxB) x D
    """
    def forward(self, roi_features: Tensor) -> Tensor:
        return super().forward(roi_features)

class FastRCNNPredictor_(FastRCNNPredictor, ObjDetectionModule):
    """
    Inputs:
        box_features: Tensor, (NxB) x D
    1. box_features -Linear-> scores: (NxB) x num_classes
    2. box_features -Linear-> deltas: (NxB) x (4 x num_classes)
    Returns:
        scores: Tensor, (NxB) x num_classes
        deltas: Tensor, (NxB) x (4 x num_classes)
    """
