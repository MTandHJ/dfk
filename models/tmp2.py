# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from typing import Iterable, Tuple, List, Dict, Optional, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import copy
import math
from collections import OrderedDict

from .base import ObjDetectionModule


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Args:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class GeneralizedRCNNTransform(ObjDetectionModule):

    def __init__(
        self, min_size: int, max_size: int, 
        image_mean: Iterable[float], image_std: Iterable[float],
        size_divisible: int = 32
    ):
        super(GeneralizedRCNNTransform, self).__init__()
        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = float(size_divisible)

    def normalize(self, image: Tensor) -> Tensor:
        size = (1,) + (-1,) * (image.ndim - 1)
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype, device).view(size)
        std = torch.as_tensor(self.image_std, dtype, device).view(size)
        return (image - mean) / std

    def resize(self, image: Tensor) -> Tuple:
        h, w = image.shape[-2:]
        min_size = float(min(h, w))
        max_size = float(max(h, w))
        scale_factor = min(
            self.min_size / min_size,
            self.max_size / max_size
        )
        resized = F.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear',
            recompute_scale_factor=True, align_corners=False
        )[0]
        new_h, new_w = resized.shape[-2:]
        ratio_height = float(new_h) / float(h)
        ratio_width = float(new_w) / float(w)
        return resized, ratio_height, ratio_width

    def scale_bbox(self, boxes: Tensor, ratio_height: float, ratio_weight: float) -> Tensor:
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin, xmax = xmin * ratio_weight, xmax * ratio_weight
        ymin, ymax = ymin * ratio_height, ymax * ratio_height
        return torch.cat([xmin, ymin, xmax, ymax], dim=1)
    
    def make_batch(self, images: List[Tensor]) -> Tensor:
        sizes = zip(*[image.size() for image in images])
        sizes = [max(size) for size in sizes]
        for i, size in enumerate(sizes[1:], 1):
            sizes[i] = int(
                math.ceil(
                    float(size / self.size_divisible)
                ) * self.size_divisible
            )
        batch_shape = [len(images)] + sizes
        batched = images[0].new_full(batch_shape, 0.)
        for img, pad_img in zip(images, batched):
            c, h, w = img.size()
            pad_img[:c, :h, :w].copy_(img)
        return batched

    def forward(self, images: List[Tensor], targets: Optional[Dict]):
        # avoid in-place operation
        images = copy.copy(images)
        targets = copy.copy(targets)
        for i, image in enumerate(images):
            images[i], ratio_h, ratio_w = self.resize(self.normalize(image))
            if targets is not None:
                # the boxes should be scaled accordingly
                boxes = targets[i]['boxes']
                targets[i]['boxes'] = self.scale_bbox(boxes, ratio_h, ratio_w)
        real_sizes = [Tuple[img.shape[-2:]] for img in images]
        batched = self.make_batch(images)
        return ImageList(batched, real_sizes), targets
        
        
# FPN

class ExtraFPNBlock(ObjDetectionModule):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """
    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


class FeaturePyramidNetwork(ObjDetectionModule):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    
    ====================================================================
                                                feature5
                                                    |[extra block]
                                                    u
            Stage4 -> x4 -[1x1, ]-> z4 -[3x3]-> feature4
            |                      |x2
            u                      d
            Stage3 -> x3 -[1x1,+]-> z3 -[3x3]-> feature3
            |                      |x2
            u                      d
            Stage2 -> x2 -[1x1,+]-> z2 -[3x3]-> feature2
            |                      |x2
            u                      d
            Stage1 -> x1 -[1x1,+]-> z1 -[3x3]-> feature1
    ====================================================================
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.inner_blocks[-1][x[-1]]
        results = []
        results.append(self.layer_blocks[-1][last_inner])

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx][x[idx]]
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx][last_inner])

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """
    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


class AnchorGenerator(ObjDetectionModule):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):


    1. cell_anchors: L x K x 4, K ratios of anchors for each layer of features;
    2. grid_anchors: L x (HlxWl) x K x 4, grid anchors with centers over each feature map
    3. anchors: N x (LxHlxWlxK) x 4, repeat it over all images
    """

    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(
        self, scales: List[int], aspect_ratios: List[float], 
        dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
    ):
        scales = torch.as_tensor(scales, dtype=dtype, device=device) # 1, 
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device) # 3,
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        # M x 1, 1 x N -> M x N:
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 # (xmin, ymin, xmax, ymax)
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        cell_anchors = [ # ((size,), (ratio1, ratio2, ratio3)) -> a group of anchors for corresponding layer
            self.generate_anchors(
                sizes,
                aspect_ratios,
                dtype,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:5),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors # List[[xmin, ymin, xmax, ymax]] with grid centers

    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:] # all images are in the same size !
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device) # the anchors for each layer: L x
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides) # L x (HxW) x 
        anchors: List[List[torch.Tensor]] = [] # N x L x (HxWx...) x 4
        for i in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors] # N x (LHWx...) x 4
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors

class RPNHead(ObjDetectionModule):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t)) # L * N x K x H x W
            bbox_reg.append(self.bbox_pred(t)) # L * N x 4K x H x W
        return logits, bbox_reg

