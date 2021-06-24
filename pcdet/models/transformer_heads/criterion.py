import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import TargetMatcher
from ...utils.common_utils import (is_dist_avail_and_initialized, get_dist_info, nested_tensor_from_tensor_list,
                                   accuracy)

from ...ops.iou3d_nms import iou3d_nms_utils

class SetCriterion(nn.Module):
    """ This class computes the loss for TrPD.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, eos_coef, losses, matcher=TargetMatcher()):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes+1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels(self, batch_dict, indices, num_boxes, log=True):
        """Classification Loss(NLL)"""
        assert 'rcnn_cls' in batch_dict, 'cannot find keyword "rcnn_cls"'
        cls = batch_dict['rcnn_cls']
        idx = self._get_src_permutation_idx(indices) #idx: [batch,[batch_idx, src_idx]]

        gt_box = batch_dict['gt_boxes']
        label = gt_box[:, :, 7] #[B,M]

        target_classes_o = torch.cat([l[J] for l, (_, J) in zip(label, indices)]).long()
        target_classes = torch.full(cls.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=cls.device)

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(cls.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce':loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(cls[idx], target_classes_o)[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, batch_dict, indices, numboxes):
        """ this is just for logging"""
        assert 'rcnn_cls' in batch_dict
        cls = batch_dict['rcnn_cls']
        idx = self._get_src_permutation_idx(indices)
        batch_size = batch_dict['batch_size']
        gt_boxes = batch_dict['gt_boxes']
        label = gt_boxes[:, :, 7]
        target_length = torch.as_tensor([len(label[i]) for i in range(batch_size)], device=label.device)

        card_pred = (cls.argmax(-1) != cls.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), target_length.float())
        losses = {'cardinality_error': card_err}
        return losses

    '''
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    '''
    def loss_boxes(self, batch_dict, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format(center_x, center_y, center_z, l, w, h),
        normalized by the space size"""

        assert 'rcnn_reg' in batch_dict
        idx = self._get_src_permutation_idx(indices)
        batch_size = batch_dict['batch_size']
        src_boxes = batch_dict['rcnn_reg'][idx]
        gt_boxes = batch_dict['gt_boxes']
        boxes = gt_boxes[:, :, :7]
        target_boxes = torch.cat([b[J] for b, (_, J) in zip(boxes, indices)])
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_iou = 1 - torch.diag(iou3d_nms_utils.boxes_iou3d_gpu(src_boxes, target_boxes))
        losses['loss_iou'] = loss_iou.sum() / num_boxes
        return losses

    def loss_masks(self, batch_dict, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
            targets dicts must contain the key "mask" containing a tensor of dim [number_target_boxes, l, w, h]
        """
        raise NotImplementedError

    '''
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    '''
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, indices, num_boxes, **kwargs)

    def forward(self, batch_dict):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            batch_dict:
                    'batch_size'        : B
                    'rcnn_cls'          :(B, number of query, number of classes + 1)
                    'rcnn_reg'          :(B, number of query, 7)[x, y, z, dx, dy, dz, heading]
                    'gt_boxes'          :(N, M, 8) [x, y, z, dx, dy, dz, heading, classes]
        """
        #outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        batch_dict = self.matcher(batch_dict)
        batch_size = batch_dict['batch_size']
        indices = batch_dict['assigned_target_indices'] #list:len = batch_size
        gt_boxes = batch_dict['gt_boxes']

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum([len(gt_boxes[i]) for i in range(batch_size)])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=gt_boxes.device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_gpus = get_dist_info()[1]
        num_boxes = torch.clamp(num_boxes / num_gpus, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, batch_dict, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        '''
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        '''
        batch_dict.update(losses)
        return batch_dict