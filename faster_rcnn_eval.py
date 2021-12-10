#!/usr/bin/env python


from typing import Tuple
import argparse
from src.loadopts import *
from src.utils import timemeter
from src.config import SAVED_FILENAME, BOXES



METHOD = "FasterRCNNEval"
SAVE_FREQ = 5
FMT = "{description}={score_thresh}-{nms_thresh}-{detections_per_img}" \
        "={min_size}-{max_size}-{representation_size}" \
        "={batch_size}={transform}"


parser = argparse.ArgumentParser()
parser.add_argument("info_path", type=str)
parser.add_argument("--backbone", type=str, default="resnet50")
parser.add_argument("--dataset", type=str, default="voc2012")
parser.add_argument("--filename", type=str, default=SAVED_FILENAME)

parser.add_argument("--rank", type=int, default=-1)
parser.add_argument("-tvt", "--train-val-test", choices=('train', 'val', 'test'), default='test')

# roi head
parser.add_argument("-st", "--score-thresh", type=float, default=0.5,
                help="only return the bounding boxes greater than score_thresh")
parser.add_argument("-nt", "--nms-thresh", type=float, default=0.35,
                help="lower for fewer")
parser.add_argument("-dpi", "--detections-per-img", type=int, default=10)

# transform
parser.add_argument("--min-size", type=int, default=800,
                help="for GeneralizedRCNNTransform")
parser.add_argument("--max-size", type=int, default=1333,
                help="for GeneralizedRCNNTransform")

# rpn
parser.add_argument("--batch-size-per-image", type=int, default=256)

# roi pool
parser.add_argument("--featmap-names", type=str, default="1,2,3,4",
                help="the features used for next roi pooling")

# box head
parser.add_argument("--representation-size", type=int, default=1024)


parser.add_argument("-b", "--batch_size", type=int, default=2)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentations which will be applied during training.")

parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=-1)
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
    cfg['model'] = ODArch(
        model=model
    )

    load(
        model=cfg.model,
        path=opts.info_path,
        filename=opts.filename
    )

    # load the dataset
    cfg['dataset'] = load_dataset(
        dataset_type=opts.dataset,
        transforms="tensor,none",
        train_val_test=opts.train_val_test
    )

    return cfg



@timemeter("Main")
def main(
    model, dataset, 
    info_path, log_path, device=DEVICE
):
    from freeplot.base import FreePlot, FreePatches
    import random
    if opts.rank == -1:
        opts.rank = random.randint(0, len(dataset))
    patches = FreePatches(linewidth=BOXES.width)
    image, targets = dataset[opts.rank]
    classnames = {v:k for k, v in dataset.data.class_dict.items()}
    
    model.eval()
    predictions = model([image.to(device)])[0]
    boxes, labels, scores = predictions['boxes'], predictions['labels'], predictions['scores']
    boxes = boxes.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    _, H, W = image.size()
    image = image.permute((1, 2, 0)).numpy()
    fp = FreePlot(figsize=(H / 100, W / 100), dpi=100)
    fp.imageplot(image, show_ticks=True)
    for box, label, score in zip(boxes, labels, scores):
        assert score >= opts.score_thresh
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rectangle = patches.Rectangle(x, y, w, h, color=BOXES.color)
        fp.add_patch(rectangle)

        text = f"{classnames[label]}: {score:.3f}"
        fontsize = min(BOXES.fontsize, w * BOXES.proportion)
        w, h = len(text) * fontsize, fontsize * BOXES.ratio
        rectangle = patches.Rectangle(x, y, w, h, color=BOXES.background, fill=True)
        fp.add_patch(rectangle)
        fp.set_text(x, y + h, text, color=BOXES.fontcolor, fontsize=BOXES.fontsize)
    fp.show()


if __name__ ==  "__main__":
    from src.utils import readme
    cfg = load_cfg()
    opts.log_path = cfg.log_path
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)

