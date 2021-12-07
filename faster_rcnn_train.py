#!/usr/bin/env python


from typing import Tuple
from torchvision.models.detection._utils import overwrite_eps

import argparse
from src.loadopts import *
from src.utils import timemeter
from src.config import SAVED_FILENAME



METHOD = "FasterRCNN"
SAVE_FREQ = 5
FMT = "{description}={min_size}-{max_size}_{representation_size}" \
        "={learning_policy}-{optimizer}-{lr}-{weight_decay}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default="resnet50")
parser.add_argument("--dataset", type=str, default="voc2012")
parser.add_argument("--pretrained-path", type=str, default=None)
parser.add_argument("--pretrained-name", type=str, default=SAVED_FILENAME)
parser.add_argument("--trainable-stages", type=int, default=5)

# transform
parser.add_argument("--min-size", type=int, default=300,
                help="for GeneralizedRCNNTransform")
parser.add_argument("--max-size", type=int, default=500,
                help="for GeneralizedRCNNTransform")

# rpn
parser.add_argument("--batch-size-per-image", type=int, default=256)

# roi pool
parser.add_argument("--featmap-names", type=str, default="1,2,3,4",
                help="the features used for next roi pooling")

# box head
parser.add_argument("--representation-size", type=int, default=1024)


parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.005)
parser.add_argument("-lp", "--learning_policy", type=str, default="default", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("-b", "--batch_size", type=int, default=2)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentations which will be applied during training.")

# eval
parser.add_argument("--eval-train", action="store_true", default=False)
parser.add_argument("--eval-valid", action="store_false", default=True)
parser.add_argument("--eval-freq", type=int, default=5,
                help="for valid dataset only")

parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--benchmark", action="store_false", default=True, 
                help="cudnn.benchmark == True ?")
parser.add_argument("-m", "--description", type=str, default=METHOD)
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



@timemeter("Setup")
def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.base import Coach
    from src.utils import set_seed, activate_benchmark, \
                            load, load_checkpoint, set_logger
    from models.base import ODArch
    from cfgs.faster_rcnn_cfg import settings

    # load new settings and check them
    for setting in settings.values():
        setting.update(**opts.__dict__)
    settings.check()

    cfg = Config()
    
    # generate the path for logging information and saving parameters
    cfg['info_path'], cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset, 
        model=opts.backbone, description=opts.description
    )
    # set logger
    logger = set_logger(
        path=cfg.log_path, 
        log2file=opts.log2file, 
        log2console=opts.log2console
    )

    activate_benchmark(opts.benchmark)
    set_seed(opts.seed)

    # the model and other settings for training
    backbone, stages = load_backbone(opts.backbone)
    if opts.pretrained_path is not None:
        load(
            model=backbone,
            path=opts.pretrained_path,
            filename=opts.pretrained_name
        )
        # overwrite_eps(backbone, 0.0)
    # freeze some layers
    stages = stages[::-1][:opts.trainable_stages]
    layers_to_train = []
    for stage in stages:
        if isinstance(stage, str):
            layers_to_train.append(stage)
        else:
            layers_to_train.extend(stage)
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    
    # fpn
    fpn = load_fpn(
        model_type=opts.backbone,
        backbone=backbone
    )

    # transform
    transform = load_transform(
        dataset_type=opts.dataset, **settings.transform
    )

    # rpn
    anchor_gen = load_anchor_gen(**settings.anchor_gen)
    rpn_head = load_rpn_head(
        num_anchors=anchor_gen.num_anchors_per_location()[0],
        **settings.rpn_head
    )
    rpn = load_rpn(
        anchor_gen=anchor_gen, rpn_head=rpn_head,
        **settings.rpn
    )

    # roi
    roi_pool = load_roi_pool(**settings.roi_pool)
    box_head = load_box_head(**settings.box_head)
    roi_predictor = load_roi_predictor(
        dataset_type=opts.dataset,
        **settings.roi_predictor
    )
    roi_heads = load_roi_heads(
        box_roi_pool=roi_pool, 
        box_head=box_head, 
        box_predictor=roi_predictor,
        **settings.roi_heads
    )

    # faster rcnn
    model = load_faster_rcnn(
        backbone=fpn, rpn=rpn, roi_heads=roi_heads, transform=transform
    )
    model = ODArch(
        model=model
    )

    # load the dataset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        transforms=opts.transform,
        train_val_test='train'
    )
    validset = load_dataset(
        dataset_type=opts.dataset,
        transforms="tensor,none",
        train_val_test='val'
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset,
        batch_size=opts.batch_size,
        train=True,
        show_progress=opts.progress
    )
    cfg['validloader'] = load_dataloader(
        dataset=validset,
        batch_size=opts.batch_size,
        train=False,
        show_progress=opts.progress
    )

    # load the optimizer and learning_policy
    optimizer = load_optimizer(
        model=model, optim_type=opts.optimizer, lr=opts.lr,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy = load_learning_policy(
        optimizer=optimizer, 
        learning_policy_type=opts.learning_policy,
        T_max=opts.epochs
    )

    if opts.resume:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path, model=model, 
            optimizer=optimizer, lr_scheduler=learning_policy
        )
    else:
        cfg['start_epoch'] = 0

    cfg['coach'] = Coach(
        model=model,
        optimizer=optimizer,
        learning_policy=learning_policy
    )

    # for validation
    cfg['valider'] = load_valider(
        model=model
    )
    return cfg


def preparation(valider):
    from src.utils import TrackMeter, ImageMeter, getLogger
    from src.dict2obj import Config
    logger = getLogger()
    mAP_logger = Config()
    for iou_thresh in valider.iou_threshes:
        mAP_logger[str(iou_thresh)] = TrackMeter(str(iou_thresh))

    mAP_logger.plotter = ImageMeter(*mAP_logger.values(), title="mAP")

    @timemeter("Evaluation")
    def evaluate(dataloader, prefix='Valid', epoch=8888):
        results = valider.evaluate(dataloader)
        fmtstr = "[mAP({iou_thresh}): {mAP:.6f}]"
        strings = []
        for iou_thresh, mAP in results.items():
            strings.append(fmtstr.format(iou_thresh=iou_thresh, mAP=mAP))
            mAP_logger[iou_thresh](data=mAP, T=epoch)
        strings = "    ".join(strings)
        logger.info(f"{prefix} >>> {strings}")
        return results[str(iou_thresh)]

    return mAP_logger, evaluate


@timemeter("Main")
def main(
    coach, valider, 
    trainloader, validloader, start_epoch, 
    info_path, log_path
):  
    from src.utils import save_checkpoint

    losses_weights = {
        "loss_objectness": 1.,
        "loss_rpn_box_reg": 1.,
        "loss_classifier": 1.,
        "loss_box_reg": 1.
    }

    # preparation
    mAP_logger, evaluate = preparation(valider)

    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        if epoch % opts.eval_freq == 0:
            if opts.eval_train:
                evaluate(trainloader, prefix='Train', epoch=epoch)
            if opts.eval_valid:
                mAP = evaluate(validloader, prefix="Valid", epoch=epoch)
                coach.check_best(mAP, info_path, epoch=epoch)

        running_loss = coach.train(trainloader, losses_weights=losses_weights, epoch=epoch)

    # save the model
    coach.save(info_path)

    # final evaluation
    evaluate(trainloader, prefix='Train', epoch=opts.epochs)
    acc_nat, acc_rob = evaluate(validloader, prefix="Valid", epoch=opts.epochs)
    coach.check_best(acc_nat, acc_rob, info_path, epoch=opts.epochs) 

    mAP_logger.plotter.plot()
    mAP_logger.plotter.save(log_path)


if __name__ ==  "__main__":
    from src.utils import readme
    cfg = load_cfg()
    opts.log_path = cfg.log_path
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)

