




from typing import Optional, Any, Union, List, NoReturn, Dict
import torch
import torch.nn as nn
import numpy as np
from freeplot.base import FreePlot

import logging
import time
import random
import os
import copy
import pickle

from .config import SAVED_FILENAME, LOGGER, DEVICE



def set_logger(
    path: str, 
    log2file: bool = True, log2console: bool = True
) -> None:
    logger = logging.getLogger(LOGGER.name)
    logger.setLevel(LOGGER.level)

    if log2file:
        handler = logging.FileHandler(
            os.path.join(path, LOGGER.filename), 
            encoding='utf-8'
        )
        handler.setLevel(LOGGER.filelevel)
        handler.setFormatter(LOGGER.formatter.filehandler)
        logger.addHandler(handler)
    if log2console:
        handler = logging.StreamHandler()
        handler.setLevel(LOGGER.consolelevel)
        handler.setFormatter(LOGGER.formatter.consolehandler)
        logger.addHandler(handler)
    logger.debug("========================================================================")
    logger.debug("========================================================================")
    logger.debug("========================================================================")
    return logger

def getLogger():
    return logging.getLogger(LOGGER.name)

def timemeter(prefix=""):
    def decorator(func):
        logger = getLogger()
        def wrapper(*args, **kwargs):
            start = time.time()
            results = func(*args, **kwargs)
            end = time.time()
            logger.info(f"[Wall TIME]- {prefix} takes {end-start:.6f} seconds ...")
            return  results
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator
        

class AverageMeter:

    def __init__(self, name: str, fmt: str = ".5f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1, mode: str = "mean") -> None:
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        else:
            raise ValueError(f"Receive mode {mode} but [mean|sum] expected ...")
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} Avg:{avg:{fmt}}"
        return fmtstr.format(**self.__dict__)

class TrackMeter:

    def __init__(self, name: str, fmt: str = ".5f"):
        self.name = name
        self.__history = []
        self.__timeline = []

    @property
    def history(self) -> List:
        return copy.deepcopy(self.__history)

    @property
    def timeline(self) -> List:
        return copy.deepcopy(self.__timeline)

    def track(self, data: float, T: int) -> None:
        self.__history.append(data)
        self.__timeline.append(T)

    def __call__(self, *, data: float, T: int) -> None:
        self.track(data, T)


class ProgressMeter:
    def __init__(self, *meters: AverageMeter, prefix: str = ""):
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, *, epoch: int = 8888) -> None:
        entries = [f"[Epoch: {epoch:<4d}]" + self.prefix]
        entries += [str(meter) for meter in self.meters]
        logger = getLogger()
        logger.info('\t'.join(entries))

    def add(self, *meters: AverageMeter) -> None:
        self.meters += list(meters)

    def step(self) -> None:
        for meter in self.meters:
            meter.reset()

class ImageMeter:
    def __init__(
        self, *meters: TrackMeter, title: str = ""
    ):
        self.meters = list(meters)
        self.title = title
        

    def add(self, *meters: TrackMeter) -> None:
        self.meters += list(meters)

    def plot(self) -> None:
        self.fp = FreePlot(
            shape=(1, 1),
            figsize=(2.2, 2),
            titles=(self.title,),
            dpi=300
        )
        self.fp.set_style('no-latex')
        for meter in self.meters:
            x = meter.timeline
            y = meter.history
            self.fp.lineplot(x, y, label=meter.name)
        self.fp.set_title(y=.98)
        self.fp[0, 0].legend()
    
    def save(self, path: str, postfix: str = '') -> None:
        filename = f"{self.title}{postfix}.png"
        _file = os.path.join(path, filename)
        self.fp.savefig(_file)

def mkdirs(*paths: str) -> None:
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def readme(path: str, opts: "parser", mode: str = "w") -> None:
    """
    opts: the argparse
    """
    import time
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = path + "/README.md"
    s = "|  {0[0]}  |   {0[1]}    |\n"
    info = "\n## {0} \n\n\n".format(time_)
    info += "|  Attribute   |   Value   |\n"
    info += "| :-------------: | :-----------: |\n"
    for item in opts._get_kwargs():
        info += s.format(item)
    with open(filename, mode, encoding="utf8") as fh:
        fh.write(info)

# load model's parameters
def load(
    model: nn.Module, 
    path: str, 
    device: torch.device = DEVICE,
    filename: str = SAVED_FILENAME,
    strict: bool = True, 
    except_key: Optional[str] = None
) -> None:

    filename = os.path.join(path, filename)
    if str(device) == "cpu":
        state_dict = torch.load(filename, map_location="cpu")
        
    else:
        state_dict = torch.load(filename)
    if except_key is not None:
        except_keys = list(filter(lambda key: except_key in key, state_dict.keys()))
        for key in except_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

# save the checkpoint
def save_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: "learning rate policy",
    epoch: int
) -> None:
    path = path + "/model-optim-lr_sch-epoch.tar"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        },
        path
    )

# load the checkpoint
def load_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: "learning rate policy"
) -> int:
    path = path + "/model-optim-lr_sch-epoch.tar"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    return epoch

def export_pickle(data: Dict, file_: str) -> NoReturn:
    logger = getLogger()
    logger.info(">>> Export File ...")
    fh = None
    try:
        fh = open(file_, "wb")
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
    except (EnvironmentError, pickle.PicklingError) as err:
        ExportError_ = type("ExportError", (Exception,), dict())
        raise ExportError_(f"Export Error: {err}")
    finally:
        if fh is not None:
            fh.close()

def import_pickle(file_: str) -> Dict:
    logger = getLogger()
    logger.info(">>> Import File ...")
    fh = None
    try:
        fh = open(file_, "rb")
        return pickle.load(fh)
    except (EnvironmentError, pickle.UnpicklingError) as err:
        raise ImportError(f"Import Error: {err}")
    finally:
        if fh is not None:
            fh.close()


def activate_benchmark(benchmark: bool) -> None:
    from torch.backends import cudnn
    if benchmark:
        getLogger().info(f"[Seed] >>> Activate benchmark")
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        getLogger().info(f"[Seed] >>> Deactivate benchmark")
        cudnn.benchmark, cudnn.deterministic = False, True

def set_seed(seed: int) -> None:
    if seed == -1:
        seed = random.randint(0, 125808521)
        logger = getLogger()
        logger.info(f"[Seed] >>> Set seed randomly: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# caculate the lp distance along the dim you need,
# dim could be tuple or list containing multi dims.
def distance_lp(
    x: torch.Tensor, 
    y: torch.Tensor, 
    p: Union[int, float, str], 
    dim: Optional[int] = None
) -> torch.Tensor:
    return torch.norm(x-y, p, dim=dim)