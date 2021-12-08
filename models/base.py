

from typing import Any, Callable, List, Dict, Optional
import torch
import torch.nn as nn
import abc

from src.config import DEVICE


class ODType(abc.ABC): ...

class ObjDetectionModule(nn.Module, ODType):
    """
    Define some basic properties.
    """
    def __init__(self) -> None:
        super(ObjDetectionModule, self).__init__()
        self.detecting: bool = False
        
    def detect(self, mode: bool = True) -> None:
        self.detecting = mode
        for module in self.children():
            if isinstance(module, ODType):
                module.detect(mode)


class DataParallel(nn.DataParallel, ObjDetectionModule): ...

class ODArch(ObjDetectionModule):

    def __init__(
        self, model: ObjDetectionModule,
        device: torch.device = DEVICE
    ) -> None:
        super().__init__()
        self.arch = model
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(model)
        else:
            self.model = model.to(device)

    def state_dict(self, *args, **kwargs):
        return self.arch.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.arch.load_state_dict(*args, **kwargs)

    def __call__(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None, **kwargs: Any) -> Any:
        return  self.model(images, targets, **kwargs)
