

from src.dict2obj import Config


__all__ = ['settings']

class CfgConflictError(Exception): ...


cfg = Config()



cfg['darknet'] = Config()
cfg['yolo'] = Config(
    min_size=288, max_size=480, after_batches=10, 
    conf_thresh=.5, iou_thresh=.5
)

def _check():
    try:
        assert cfg.min_size % 32 == 0
        assert cfg.max_size % 32 == 0
    except AssertionError as e:
        raise CfgConflictError(
            f"Please check the following config: \n{e}"
        )

cfg.check = _check
settings = cfg

