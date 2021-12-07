

from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchvision.models import detection
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads

from .base import ObjDetectionModule
from .rcnnops import GeneralizedRCNNTransform_, IntermediateLayerGetter_, \
                        RPNHead_, AnchorGenerator_, \
                        MultiScaleRoIAlign_, TwoMLPHead_, FastRCNNPredictor_

__all__ = [
    "FeaturePyramidNetwork_", "BackboneWithFPN_",
    "RegionProposalNetwork_", "RoIHeads_",
    "GeneralizedRCNN_"
]


class FeaturePyramidNetwork_(FeaturePyramidNetwork, ObjDetectionModule):
    """
    Inputs: Dict[str, Tensor] | Stage: x
    ====================================================================
                                                feature5
                                                    |[extra block]
                                                    u
            Stage4 -> x4 -[1x1, ]-> z4 -[3x3]-> feature4
            |                      |x2
            u                      d
            Stage3 -> x3 -[1x1,+]-> z3 -[3x3]-> feature3
            |                      |x2
            u                      d
            Stage2 -> x2 -[1x1,+]-> z2 -[3x3]-> feature2
            |                      |x2
            u                      d
            Stage1 -> x1 -[1x1,+]-> z1 -[3x3]-> feature1
    ====================================================================
    Returns: OrderDict[level: str, features: N x C x H x W]
    """

class BackboneWithFPN_(ObjDetectionModule):
    """
    Adds a FPN on top of a model.
    Inputs: images: Tensor
    1. images -> backbone -> Dict[str, Tensor] including features on different layers
    2. The FPN transfroms these features into pyramid features
    Returns: Dict[str, Tensor] L * N x C x Hl x Wl
    """
    def __init__(
        self, backbone: nn.Module, return_layers: Dict[str, str], 
        in_channels_list: List[int], out_channels: int, 
        extra_blocks: Optional[nn.Module] = None
    ):
        super(BackboneWithFPN_, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter_(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork_(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        x = self.body(images) # Tensor -> Dict['layer':features]
        x = self.fpn(x) # Dict['layer':PyramidFeatures]
        return x


class RegionProposalNetwork_(RegionProposalNetwork, ObjDetectionModule):
    """
    Inputs: 
        images: ImageList, N x 3 x H x W
        features: Dict[str, Tensor], L * N x Cl x Hl x Wl
    1. inputs -> head -> 
            objectness: List[Tensor], L * N x K x Hl x Wl; 
            deltas: List[Tensor], L * N x 4K x Hl x Wl;
    2. inputs -> AnchorGenerator -> 
            anchors: List[Tensor] N * (LxHlxWlxK) x 4
    3. permute them into:
            objectness: Tensor, N x (LxHlxWlxK) x 1; 
            deltas: Tensor, N x (LxHlxWlxK) x 4;
    4. the proposals can be deduced from the following transformation:
            Px = Ax + delta_x * Aw
            Py = Ay + delta_y * Ah
            Pw = Aw * exp(delta_w)
            Ph = Ah * exp(delta_h)
        where (x, y) is the center. To obtain the final proposals,
        one should transform the (x, y, w, h) into (xmin, ymin, xmax, ymax).
        Finally:
            proposals: Tensor, N x (LxHlxWlxK) x 4
    5. filter_proposals:
        1. select the proposals with top n scores
        2. clip them into the range of images
        3. remove some small proposals
        4. nms
        5. final proposals: List[Tensor], N * n x 4
    6. assign 'ground truth' bbox to each anchor (not proposals!):
        labels: List[Tensor], N * (LxHlxWlxK), 
            0: background, -1: discarded, 1: object
        matched_gt_boxes: List[Tensor], N * (LxHlxWlxK) x 4
    7. calculate the regression deltas between matched_gt_boxes and anchors:
            delta_x = (Gx - Ax) / Aw
            delta_y = (Gy - Ay) / Ah
            delta_w = log(Gw / Aw)
            delta_h = log(Gh / Ah)
        where (x, y) is the center.
        regression_targets: Tuple[Tensor], N * (LxHlxWlxK) x 4
    8. calculate the loss:
        1. sample a balanced batch of positive and negative deltas
        2. box_loss: the smooth_l1_loss regarding the positive predicted delta and its target
        3. objectness_loss: BCE loss regarding object and background
        4. losses = {loss_objectness:objectness_loss, loss_rpn_box_reg: box_loss}
    Returns: boxes [final proposals]: List[Tensor], N * n x 4, losses
    """

    def __init__(
        self, anchor_generator: AnchorGenerator_, head: RPNHead_, 
        fg_iou_thresh: float, bg_iou_thresh: float, 
        batch_size_per_image: int, positive_fraction: float, 
        pre_nms_top_n: Dict[str, int], post_nms_top_n: Dict[str, int], 
        nms_thresh: float, score_thresh: float = 0.
    ):
        super().__init__(
            anchor_generator, head, 
            fg_iou_thresh, bg_iou_thresh, 
            batch_size_per_image, positive_fraction, 
            pre_nms_top_n, post_nms_top_n, 
            nms_thresh, score_thresh=score_thresh
        )


class RoIHeads_(RoIHeads, ObjDetectionModule):
    """
    Inputs:
        features: Dict[str, Tensor], L * N x C x Hl x Wl
        proposals: List[Tensor], N * n x 4
        image_shapes: real image sizes, List[Tuple[int, int]], N * 2
        targets: Optional[List[Dict[str, Tensor]]]
    1. select training samples:
        1. gt_boxes: N * gn x 4 -append-> proposals: N * (n+gn) x 4
        2. assign targets to proposals: 
            matched_idxs: ...
            labels: List[Tensor], N * (n+gn), {-1, 0, 1, 2 ...}
        3. sample a balanced batch of positives and negatives according labels
        4. Returns:
            proposals: List[Tensor], N * B x 4
            labels: List[Tensor], N * B
            regression_targets (deltas): List[Tensor], N * B x 4
    2. (features, proposals, image_shapes) -roiPooling(roiAlign)->
            roi_features: (NxB) x C x output_size[0](7) x output_size[1](7)
    3. (roi_features,) -box_head (TwoMLP)->
            box_features: (NxB) x D
    4. (box_features,) -box_predictor->
            logits: Tensor, (NxB) x num_classes
            deltas: Tensor, (NxB) x (4 x num_classes)
    5. if training: 
            (logits, deltas, labels, regression_targets) -fastrcnn_loss->
            losses = Dict[
                loss_classifier: Tensor,
                loss_box_reg: Tensor
            ]
        else:
            (logits, deltas, proposals, image_shapes) -postprocess_detections:
            1. (deltas, proposals) -> pred_boxes: Tensor, (NxB) x num_class x 4
            2. logits -softmax-> pred_scores: Tensor, (NxB) x num_classes
            3. clip boxes to image
            4. remove boxes belonging to background
            5. remove low scoring boxes
            6. remove empty boxes
            7. nms
            8. keep topk
            9. result: List[N *
                Dict[
                    boxes: Tensor, K x 4
                    labels: Tensor, K
                    scores: Tensor, K
                ]
            ]
        Returns:
            result, losses
    """

    def __init__(
        self, box_roi_pool: MultiScaleRoIAlign_, 
        box_head: TwoMLPHead_, box_predictor: FastRCNNPredictor_, 
        fg_iou_thresh: float, bg_iou_thresh: float,
        batch_size_per_image: int, positive_fraction: float, 
        bbox_reg_weights: Tuple[float], score_thresh: float,
        nms_thresh: float, detections_per_img: int,
        # for mask
        mask_roi_pool=None, mask_head=None, 
        mask_predictor=None, keypoint_roi_pool=None, 
        keypoint_head=None, keypoint_predictor=None
    ):
        super().__init__(
            box_roi_pool, box_head, box_predictor, 
            fg_iou_thresh, bg_iou_thresh, 
            batch_size_per_image, positive_fraction, 
            bbox_reg_weights, score_thresh, 
            nms_thresh, detections_per_img, 
            mask_roi_pool=mask_roi_pool, mask_head=mask_head, 
            mask_predictor=mask_predictor, keypoint_roi_pool=keypoint_roi_pool, 
            keypoint_head=keypoint_head, keypoint_predictor=keypoint_predictor
        )


class GeneralizedRCNN_(GeneralizedRCNN, ObjDetectionModule):
    """
    Inputs: 
        images: List[Tensors],
        targets: Optional[List[Dict[str, Tensor]]]
    1. (images, targets) -transform-> images: ImagesList, targets
    2. (images.tensors,) -backbone-> features: Dict[str, Tensor] L * N x C x Hl x Wl
    3. (images, features, targets) -rpn->
            proposals: List[Tensor], N * n x 4
            proposal_losses: Dict[
                loss_objectness: Tensor,
                loss_rpn_box_reg: Tensor
            ]
    4. (features, proposals, images.image_sizes, targets) -roi_heads->
            detections: List[N *
                Dict[
                    boxes: Tensor, K x 4
                    labels: Tensor, K
                    scores: Tensor, K
                ]
            ]
            detector_losses: Dict[
                loss_classifier: Tensor,
                loss_box_reg: Tensor
            ]
    5. (detections, images.image_sizes, original_image_sizes) -postprocess->
        Note that the detections is based on the resized image (i.e. images.image_sizes),
        so we should transform them back to the original sizes for inference.
            detections: ...
    Returns: if training:
                return losses: Dict[
                    loss_objectness: Tensor,
                    loss_rpn_box_reg: Tensor
                    loss_classifier: Tensor,
                    loss_box_reg: Tensor
                ]
             else:
                return detections: List[
                    Dict[
                        boxes: Tensor, K x 4
                        labels: Tensor, K
                        scores: Tensor, K
                    ]
                ]
    """

    def __init__(
        self, backbone: BackboneWithFPN_, rpn: RegionProposalNetwork_, 
        roi_heads: RoIHeads_, transform: GeneralizedRCNNTransform_
    ):
        super().__init__(backbone, rpn, roi_heads, transform)