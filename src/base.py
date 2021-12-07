


from typing import List, Tuple, Dict, Iterable, cast
import torch
import os

from models.base import ODArch
from .utils import AverageMeter, ProgressMeter, timemeter, getLogger
from .mAP import MeanAveragePrecision
from .config import SAVED_FILENAME, PRE_BEST, DEVICE



class Coach:
    
    def __init__(
        self, model: ODArch,
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy",
        device: torch.device = DEVICE
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.progress = ProgressMeter(self.loss)

        self._best = 0.

    def save_best(self, mAP: float, path: str, prefix: str = PRE_BEST):
        if mAP > self._best:
            self._best_nat = mAP
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0
    
    def check_best(
        self, mAP: float, path: str, epoch: int = 8888
    ):
        logger = getLogger()
        if self.save_best(mAP, path):
            logger.debug(f"[Coach] Saving the best nat ({mAP:.6f}) model at epoch [{epoch}]")
        
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    @timemeter("Train/Epoch")
    def train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        losses_weights: Dict,
        *, epoch: int = 8888
    ) -> float:

        self.progress.step() # reset the meter
        self.model.train()
        for step, (images, targets) in enumerate(trainloader):
            images = [image.to(self.device) for image in images]
            targets = [{k:v.to(self.device) for k, v in target.items()} for target in targets]

            losses: Dict = self.model(images, targets)
            loss = 0.
            for loss_name, weight in losses_weights.items():
                loss += weight * losses[loss_name]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss.update(loss.item(), len(images), mode="mean")

        self.progress.display(epoch=epoch) 
        self.learning_policy.step() # update the learning rate
        return self.loss.avg


class Checker:

    def __init__(
        self, model: ODArch, iou_threshes: Tuple = (0.5, 0.75),
        device: torch.device = DEVICE
    ) -> None:
        self.model = model
        self.device = device
        self.iou_threshes = iou_threshes
        self.mAPer = MeanAveragePrecision()

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable):
        self.model.eval()
        self.mAPer.reset()
        for images, targets in dataloader:
            images = [image.to(self.device) for image in images]
            predictions: List[Dict] = self.model(images)
            self.mAPer.update(predictions, targets)
        results = {}
        for iou_thresh in self.iou_threshes:
            mAP, aps = self.mAPer(iou_thresh)
            results[str(iou_thresh)] = mAP
        return results
