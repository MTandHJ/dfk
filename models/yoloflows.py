

from typing import Dict, List, Optional, Tuple
import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torchvision.ops.boxes import nms


from .base import ObjDetectionModule
from .darknet import DarkNet
from .yoloops import AbsRelTransform, box_iou_wh, box_iou_xywh




class YOLO(ObjDetectionModule):

    def __init__(
        self, darknet: DarkNet, 
        min_size: int = 288, max_size: int = 480, after_batches: int = 10,
        conf_thresh: float = .5, iou_thresh: float = 0.5,
    ) -> None:
        super().__init__()

        self.darknet = darknet
        self.transform = AbsRelTransform(min_size, max_size, after_batches)
        self.criterion = BCEWithLogitsLoss()

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def align_targets(self, results, targets, H: float, W: float):
        device = targets.device
        multi_preds = [preds for (preds, _) in results]
        anchors = [yolo.anchors.to(device).clone() for (_, yolo) in results]
        strides = [yolo.stride for (_, yolo) in results]

        anchors = torch.cat(anchors, dim=0) # 9 x 2, (w, h)
        anchors[:, 0] /= W
        anchors[:, 1] /= H

        num_of_anchors, num_of_targets = len(anchors), len(targets)
        targets = torch.cat((
            targets.repeat(num_of_anchors, 1, 1), # 9 x N x 6
            torch.arange(num_of_anchors, device=device).view(-1, 1).repeat(1, num_of_targets)[:,:,None], # 9 x N, (ids,)
            2
        )).view(-1, 7) # (img_id, label, x, y, w, h, anchor_id), # (9xN) x 7

        anchors = anchors.repeat(1, num_of_targets, 1).view(-1, 2) # (9xN) x 2

        ious = box_iou_wh(anchors, targets)
        targets = targets[ious > 0.5]
        positives = targets[:, -1]

        multi_targets = []
        multi_indices = []
        for l, stride in enumerate(strides):
            indices = (positives >= 3 * l) & (positives < 3 * (l + 1))
            layer_postives = positives[indices].long()
            layer_targets = targets[indices]
            layer_ids = layer_targets[:, 0].long()
            layer_labels = layer_targets[:, 1].long()
            layer_offset_x = (layer_targets[:, 2] * W / stride).long()
            layer_offset_y = (layer_targets[:, 3] * H / stride).long()
            layer_boxes = layer_targets[:, 2:]
            layer_boxes[:, 0] = layer_boxes[:, 0] * W % stride
            layer_boxes[:, 1] = layer_boxes[:, 1] * H % stride

            multi_targets.append(layer_boxes)
            multi_indices.append([layer_ids, layer_labels, layer_postives, layer_offset_x, layer_offset_y])

        return multi_preds, multi_targets, multi_indices

    def forward(self, images: List, targets: Optional[List] = None):
        images, targets = self.transform(images, targets)
        results = self.darknet(images)
        H, W = float(images.tensors.size(-2)), float(images.tensors.size(-1))

        if not self.training:
            multi_preds = [preds.flatten(1, 3) for (preds, _) in results]
            predications = torch.cat(multi_preds, dim=1)
            keep = predications[..., :4] = self.conf_thresh
            predications = predications[keep]
            results = []
            for i, original_size in enumerate(images.image_sizes):
                preds_per_img = predications[i]
                boxes, scores = preds_per_img[:, :4], preds_per_img[:, 5:]
                boxes[:, 0:4:2] /= W # convert to relative coordinates
                boxes[:, 1:4:2] /= H
                boxes = self.transform.rel2abs(boxes, original_size) # boxes for original image size
                boxes[:, 0:4:2].clamp_(0., original_size[1]) # clip to [0, W]
                boxes[:, 1:4:2].clamp_(0., original_size[0]) # clip to [0, H]
                scores, labels = torch.max(scores, dim=-1)
                boxes = nms(boxes, scores, self.iou_thresh)
                results.append({
                    'boxes': boxes,
                    'labels': labels + 1, # start from 1
                    'scores': scores
                })
            return results


        multi_preds, multi_targets, multi_indices = self.align_targets(results, targets, H, W)
        running_box_loss = 0.
        running_obj_loss = 0.
        running_cls_loss = 0.
        for preds, layer_boxes, indices in zip((multi_preds, multi_targets, multi_indices)):
            layer_obj = torch.zeros_like(preds[..., 0])
            layer_ids, layers_labels, layer_positives, layer_offset_x, layer_offset_y = indices
            if len(layer_ids):
                layer_preds = preds[layer_ids, layer_offset_y, layer_offset_x, layer_positives, :]
                layer_pred_boxes = self.transform.abs2rel(layer_preds[:, 2:6], (H, W))
                layer_pred_cls = layer_preds[..., 5:]

                ious = box_iou_xywh(layer_pred_boxes, layer_boxes)
                running_box_loss += (1 - ious).mean()

                layer_obj[layer_ids, layer_offset_y, layer_offset_x, layer_positives] = ious.detach()

                layer_cls = torch.zeros_like(layer_pred_cls)
                layer_cls[:, layers_labels] = 1.
                running_cls_loss += self.criterion(layer_pred_cls, layer_cls)

            
            running_obj_loss += self.criterion(
                preds[..., 4], layer_obj
            )

        losses = {
            'loss_boxes': running_box_loss,
            'loss_objectness': running_obj_loss,
            'loss_classification': running_cls_loss
        }
        return losses


                
                

