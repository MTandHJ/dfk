

from src.dict2obj import Config


__all__ = ['settings']

class CfgConflictError(Exception): ...


cfg = Config()

cfg['fpn'] = Config(out_channels=256)
cfg['transform'] = Config(min_size=800, max_size=1333)

cfg['anchor_gen'] = Config(
    sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)
cfg['rpn_head'] = Config(in_channels=cfg.fpn.out_channels)
cfg['rpn'] = Config(
    fg_iou_thresh = 0.7, bg_iou_thresh = 0.3, 
    batch_size_per_image = 256, positive_fraction = 0.5, 
    pre_nms_top_n = (2000, 1000), 
    post_nms_top_n = (2000, 1000), 
    nms_thresh = 0.7, score_thresh = 0.
)

cfg['roi_pool'] = Config(
    featmap_names = "1,2,3,4", output_size = 7,
    sampling_ratio = 2, canonical_scale = 224, canonical_level =4
)
cfg['box_head'] = Config(
    in_channels = cfg.fpn.out_channels * cfg.roi_pool.output_size ** 2,
    representation_size = 1024
)
cfg['roi_predictor'] = Config(
    in_channels = cfg.box_head.representation_size
)
cfg['roi_heads'] = Config(
    fg_iou_thresh=0.5, bg_iou_thresh=0.5,
    batch_size_per_image=512, positive_fraction=0.25,
    bbox_reg_weights=None,
    score_thresh=0.05, nms_thresh=0.5, detections_per_img=100
)

def _check():
    try:
        assert cfg.fpn.out_channels == cfg.rpn_head.in_channels
        assert len(cfg.anchor_gen.sizes) == len(cfg.anchor_gen.aspect_ratios)
        assert cfg.box_head.in_channels == cfg.fpn.out_channels * cfg.roi_pool.output_size ** 2
        assert cfg.roi_predictor.in_channels == cfg.box_head.representation_size
    except AssertionError as e:
        raise CfgConflictError(
            f"Please check the following config: \n{e}"
        )


cfg.check = _check
settings = cfg

