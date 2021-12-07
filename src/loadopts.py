
from typing import Callable, Tuple, List, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision

import time
from tqdm import tqdm


from .base import Checker
from .config import *
from .utils import getLogger, mkdirs



class ModelNotDefineError(Exception): pass
class OptimNotIncludeError(Exception): pass
class DatasetNotIncludeError(Exception): pass


# return the num_classes of corresponding data set
def get_num_classes(dataset_type: str) -> int:
    if dataset_type in ('voc2007', 'voc2012'):
        return 21
    else:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


def load_backbone(model_type: str) -> Tuple[nn.Module, List[str]]:
    """
    resnet18|34|50|101|...
    """
    if model_type.startswith("resnet"):
        backbone = torchvision.models.resnet.__dict__[model_type](
            pretrained=False,
            norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
        )
        stages = [('bn1', 'conv1'), 'layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ModelNotDefineError(f"backbone {model_type} is not defined.\n" \
                f"Refer to the following: {load_backbone.__doc__}\n")
    return backbone, stages


def load_fpn(model_type:str, backbone: nn.Module, out_channels: int = 256):
    from models.rcnnflows import BackboneWithFPN_
    if model_type.startswith("resnet"):
        levels = {f'layer{level+1}': str(level) for level in range(4)}
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [int(in_channels_stage2 * 2 ** (i - 1)) for i in range(1, len(levels) + 1)]
    else:
        raise ModelNotDefineError(f"backbone {model_type} is not defined.\n" \
                f"Refer to the following: {load_backbone.__doc__}\n")
    return BackboneWithFPN_(backbone, levels, in_channels_list, out_channels)


def load_transform(dataset_type: str, min_size: int = 800, max_size: int = 1333):
    from models.rcnnops import GeneralizedRCNNTransform_
    mean = MEANS[dataset_type]
    std = STDS[dataset_type]
    return GeneralizedRCNNTransform_(min_size, max_size, mean, std)

def load_anchor_gen(
    sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
):
    from models.rcnnops import AnchorGenerator_
    return AnchorGenerator_(sizes, aspect_ratios)

def load_rpn_head(in_channels: int, num_anchors: int):
    from models.rcnnops import RPNHead_
    return RPNHead_(in_channels, num_anchors)

def load_rpn(
    anchor_gen, rpn_head,
    fg_iou_thresh: float = 0.7, bg_iou_thresh: float = 0.3, 
    batch_size_per_image: int = 256, positive_fraction: float = 0.5, 
    pre_nms_top_n: Tuple[int] = (2000, 1000), 
    post_nms_top_n: Tuple[int] = (2000, 1000), 
    nms_thresh: float = 0.7, score_thresh: float = 0.
):
    from models.rcnnflows import RegionProposalNetwork_
    pre_nms_top_n = dict(training=pre_nms_top_n[0], testing=pre_nms_top_n[1])
    post_nms_top_n= dict(training=post_nms_top_n[0], testing=post_nms_top_n[1])
    return RegionProposalNetwork_(
        anchor_gen, rpn_head,
        fg_iou_thresh, bg_iou_thresh,
        batch_size_per_image, positive_fraction,
        pre_nms_top_n, post_nms_top_n,
        nms_thresh, score_thresh
    )


def load_roi_pool(
    featmap_names: str = "1,2,3,4",
    output_size: Union[int, Tuple[int], List[int]] = 7,
    sampling_ratio: int = 2, *, 
    canonical_scale: int = 224, 
    canonical_level: int = 4
):
    from models.rcnnops import MultiScaleRoIAlign_
    featmap_names = featmap_names.split(',')
    return MultiScaleRoIAlign_(
        featmap_names, output_size,
        sampling_ratio, canonical_scale, canonical_level
    )

def load_box_head(
    in_channels: int, representation_size: int
):
    from models.rcnnops import TwoMLPHead_
    return TwoMLPHead_(in_channels, representation_size)

def load_roi_predictor(
    dataset_type: str, in_channels: int
):
    from models.rcnnops import FastRCNNPredictor_
    num_classes = get_num_classes(dataset_type)
    return FastRCNNPredictor_(in_channels, num_classes)

def load_roi_heads(
    box_roi_pool, box_head, box_predictor, **kwargs
):
    from models.rcnnflows import RoIHeads_
    return RoIHeads_(
        box_roi_pool, box_head, box_predictor,
        **kwargs
    )


def load_faster_rcnn(
    backbone, rpn, roi_heads, transform
):
    from models.rcnnflows import GeneralizedRCNN_
    return GeneralizedRCNN_(
        backbone, rpn, roi_heads, transform
    )


def _dataset(
    dataset_type: str, 
    train_val_test: str = 'train'
) -> torch.utils.data.Dataset:
    """
    Dataset:
    voc2007
    voc2012
    """
    if dataset_type == "voc2007":
        from src.datasets import VOC2007
        if train_val_test == "val":
            train_val_test = "trainval"
        dataset = VOC2007(root=ROOT, image_set=train_val_test, download=DOWNLOAD)
    elif dataset_type == "voc2012":
        from src.datasets import VOC2012
        if train_val_test == "val":
            train_val_test = "trainval"
        dataset = VOC2012(root=ROOT, image_set=train_val_test, download=DOWNLOAD)
    else:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))
        
    return dataset


def load_dataset(
    dataset_type: str, 
    transforms: str ='default', 
    train_val_test: str = 'train'
) -> torch.utils.data.Dataset:
    from .datasets import WrapperSet
    dataset = _dataset(dataset_type, train_val_test)
    transforms = TRANSFORMS[dataset_type] if transforms == 'default' else transforms
    getLogger().info(f"[Dataset] Apply transforms of '{transforms}' to {train_val_test} dataset ...")
    dataset = WrapperSet(dataset, transforms=transforms)
    return dataset

class _TQDMDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(_TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )

def load_dataloader(
    dataset: torch.utils.data.Dataset, 
    batch_size: int, 
    train: bool = True, 
    show_progress: bool = False
) -> torch.utils.data.DataLoader:

    dataloader = _TQDMDataLoader if show_progress else torch.utils.data.DataLoader
    if train:
        loader = dataloader(
            dataset, batch_size=batch_size, 
            collate_fn=dataset.collate_fn, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
    else:
        loader = dataloader(
            dataset, batch_size=batch_size,
            collate_fn=dataset.collate_fn, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
    return loader


def load_optimizer(
    model: torch.nn.Module, 
    optim_type: str, *,
    lr: float = 0.1, momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 1e-4,
    nesterov: bool = False,
    **kwargs: "other hyper-parameters for optimizer"
) -> torch.optim.Optimizer:
    """
    sgd: SGD
    adam: Adam
    """
    try:
        cfg = OPTIMS[optim_type]
    except KeyError:
        raise OptimNotIncludeError(f"Optim {optim_type} is not included.\n" \
                        f"Refer to the following: {load_optimizer.__doc__}")
    
    kwargs.update(lr=lr, momentum=momentum, betas=betas, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    cfg.update(**kwargs) # update the kwargs needed automatically
    logger = getLogger()
    logger.info(cfg)
    if optim_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), **cfg)
    elif optim_type == "adam":
        optim = torch.optim.Adam(model.parameters(), **cfg)

    return optim


def load_learning_policy(
    optimizer: torch.optim.Optimizer,
    learning_policy_type: str,
    **kwargs: "other hyper-parameters for learning scheduler"
) -> "learning policy":
    """
    default: (100, 105), 110 epochs suggested
    null:
    STD: (82, 123), 164 epochs suggested
    STD-wrn: (60, 120, 160), 200 epochs suggested
    AT: (102, 154), 200 epochs suggested
    TRADES: (75, 90, 100), 76 epochs suggested
    TRADES-M: (55, 75, 90), 100 epochs suggested
    cosine: CosineAnnealingLR, kwargs: T_max, eta_min, last_epoch
    """
    try:
        learning_policy_ = LEARNING_POLICY[learning_policy_type]
    except KeyError:
        raise NotImplementedError(f"Learning_policy {learning_policy_type} is not defined.\n" \
            f"Refer to the following: {load_learning_policy.__doc__}")

    lp_type = learning_policy_[0]
    lp_cfg = learning_policy_[1]
    lp_cfg.update(**kwargs) # update the kwargs needed automatically
    logger = getLogger()
    logger.info(f"{lp_cfg}    {lp_type}")
    learning_policy = getattr(
        torch.optim.lr_scheduler, 
        lp_type
    )(optimizer, **lp_cfg)
    
    return learning_policy


def load_valider(
    model: torch.nn.Module, iou_threshes: Tuple = (0.5, 0.75), device: torch.device = DEVICE,
):
    valider = Checker(
        model=model, iou_threshes=iou_threshes, device=device
    )
    return valider


def generate_path(
    method: str, dataset_type: str, model:str, description: str
) -> Tuple[str, str]:
    info_path = INFO_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description
    )
    log_path = LOG_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description,
        time=time.strftime(TIMEFMT)
    )
    mkdirs(info_path, log_path)
    return info_path, log_path

