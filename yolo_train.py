#!/usr/bin/env python


from typing import Tuple

import argparse
from src.loadopts import *
from src.utils import timemeter
from src.config import SAVED_FILENAME



METHOD = "YoLoTrain"
SAVE_FREQ = 5
FMT = "{description}=" \
        "={learning_policy}-{optimizer}-{lr}-{weight_decay}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default="darknet")
parser.add_argument("--dataset", type=str, default="voc2012")
parser.add_argument("--pretrained-path", type=str, default=None)
parser.add_argument("--pretrained-name", type=str, default=SAVED_FILENAME)

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
parser.add_argument("--eval-valid", action="store_true", default=False)
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
    from models.yoloflows import YOLO
    from cfgs.yolo_cfg import settings

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
    backbone, _ = load_backbone(opts.backbone)
    backbone = backbone(get_num_classes(opts.dataset) - 1)
    if opts.pretrained_path is not None:
        load(
            model=backbone,
            path=opts.pretrained_path,
            filename=opts.pretrained_name
        )
    
    model = YOLO(
        darknet=backbone, **settings.yolo
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
        'loss_boxes': 0.05,
        'loss_objectness': 1.,
        'loss_classification': 0.5
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
    mAP = evaluate(validloader, prefix="Valid", epoch=opts.epochs)
    coach.check_best(mAP, info_path, epoch=opts.epochs) 

    mAP_logger.plotter.plot()
    mAP_logger.plotter.save(log_path)


if __name__ ==  "__main__":
    from src.utils import readme
    cfg = load_cfg()
    opts.log_path = cfg.log_path
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)

