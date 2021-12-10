

"""
Reference:
facebookresearch/Detectron/detectron/datasets/voc_eval.py
https://github.com/facebookresearch/Detectron/blob/main/detectron/datasets/voc_eval.py
"""

from typing import Iterable, List, Dict, Tuple, Optional
import torch
import numpy as np
from torchvision.ops.boxes import box_iou




class MeanAveragePrecision:

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.gt_boxes = []
        self.gt_labels = []
        self.image_ids = torch.empty(0, dtype=torch.int64)
        self.scores = torch.empty(0, dtype=torch.float32) # confidence
        self.labels = torch.empty(0, dtype=torch.int64)
        self.detections = torch.empty(0, 4, dtype=torch.float32)

    def update(self, predictions: List[Dict], targets: List[Dict], difficult: int = 1e10):
        """
        Any object more difficult than 'difficult' will be removed.
        """
        predictions = [
            {k:v.detach().clone().cpu() for k, v in pred.items()}
            for pred in predictions
        ]
        targets = [
            {k:v.detach().clone().cpu() for k, v in target.items()}
            for target in targets
        ]
        # avoid using target['image_id'], which will be influenced by the shuffle operation !
        image_id = len(self.gt_labels) 
        for pred, target in zip(predictions, targets):
            scores = pred['scores']
            detections = pred['boxes']
            labels = pred['labels']
            self.image_ids = torch.cat((self.image_ids, torch.full_like(labels, image_id)))
            self.labels = torch.cat((self.labels, labels))
            self.scores = torch.cat((self.scores, scores))
            self.detections = torch.cat((self.detections, detections))

            gt_box = target['boxes']
            gt_labels = target['labels']
            iscrowd = target['iscrowd']
            self.gt_boxes.append(gt_box[iscrowd <= difficult])
            self.gt_labels.append(gt_labels[iscrowd <= difficult])

            image_id += 1

    def sort(self):
        orders = torch.argsort(self.scores, descending=True)
        self.image_ids = self.image_ids[orders]
        self.scores = self.scores[orders]
        self.labels = self.labels[orders]
        self.detections = self.detections[orders]
        return orders

    def _choose_by_label(self, label: int):
        indices = self.labels == label
        image_ids = self.image_ids[indices]
        scores = self.scores[indices]
        detections = self.detections[indices]
        gt_boxes = [boxes[labels == label] for boxes, labels in zip(self.gt_boxes, self.gt_labels)]
        return gt_boxes, image_ids, scores, detections

    def average_precision(self, label: int, iou_thresh: float = 0.5):
        self.sort()
        gt_boxes, image_ids, scores, detections = self._choose_by_label(label)
        nums = len(scores)
        fns = [torch.ones(boxes.size(0)) for boxes in gt_boxes]
        tp = torch.zeros(nums)
        fp = torch.zeros(nums)
        for i in range(nums):
            image_id = image_ids[i]
            detection = detections[[i]]
            boxes = gt_boxes[image_id]

            iou_max = -float('inf')
            if boxes.numel() > 0:
                similarity = box_iou(detection, boxes).view(-1)
                iou_max, iou_where = torch.max(similarity, 0)

            if iou_max > iou_thresh and fns[image_id][iou_where]:
                tp[i] = 1
                fns[image_id][iou_where] = 0
            else:
                fp[i] = 1
        
        # tp[i] + fp[i] == 1
        # tp[-1] + fn == the number of gt_boxes
        tp = torch.cumsum(tp, dim=0).numpy()
        fp = torch.cumsum(fp, dim=0).numpy()
        fn = sum(fn.sum().item() for fn in fns)

        ppv = tp / (tp + fp) 
        # I adopts tpr = tp / np.maximum((tp + fn), 0.5) at first,
        # but official code adopts the following formula:
        tpr = tp / (tp[-1] + fn)
        ppv = np.concatenate(([0.], ppv, [0.]))
        tpr = np.concatenate(([0.], tpr, [1.]))

        for i in range(ppv.size - 1, 0, -1):
            ppv[i-1] = np.maximum(ppv[i-1], ppv[i])

        marks = np.where(tpr[1:] != tpr[:-1])[0]
        ap = np.sum((tpr[marks + 1] - tpr[marks]) * ppv[marks + 1]).item()
        return ap

    def mAP(self, iou_thresh: float = 0.5, labels: Optional[Iterable[int]] = None) -> Tuple[float, Dict]:
        if labels is None:
            labels = torch.cat(self.gt_labels, dim=0).unique()
        
        aps = {str(label): self.average_precision(label, iou_thresh) for label in labels}
        return np.mean(list(aps.values())).item(), aps

    def __call__(self, iou_thresh: float = 0.5, labels: Optional[Iterable[int]] = None) -> Tuple[float, Dict]:
        return self.mAP(iou_thresh, labels)

            


            