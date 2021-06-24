import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from ...ops.iou3d_nms import iou3d_nms_utils

class TargetMatcher(nn.Module):
    def __init__(self, cost_cls=1., cost_box=1., cost_giou=1.):
        super().__init__()
        self.cost_box = cost_box
        self.cost_cls = cost_cls
        self.cost_giou = cost_giou

        assert self.cost_box != 0 or self.cost_giou !=0 or self.cost_giou != 0, 'Parameters cannot be all 0'

    @torch.no_grad()
    def forward(self, batch_dict):
        ''':arg
                batch_dict:
                    'batch_size'        : B
                    'rcnn_cls'          :(B, number of query, number of classes + 1)
                    'rcnn_reg'          :(B, number of query, 7)[x, y, z, dx, dy, dz, heading]
                    'gt_boxes'          :(N, M, 8) [x, y, z, dx, dy, dz, heading, classes]
            :return
                batch_dict:
                    'assigned_targets_indices' :(B, min(number_query, number_target))

        '''

        batch_size = batch_dict['batch_size']
        gt_boxes = batch_dict['gt_boxes'] #(B,M,8)
        gt_box = gt_boxes.flatten(0, 1)
        gt = gt_box[:, :7]
        label = gt_box[:, 7]

        cls = batch_dict['rcnn_cls']
        batch_size, number_query = cls.shape[:2]
        cls = cls.flatten(0, 1).softmax(-1) #[Bxnumber of query, classes+1]
        reg = batch_dict['rcnn_reg'].flatten(0, 1) #[Bxnumber of query, 7]

        cost_class = 1-cls[:, label.long()]
        cost_box = torch.cdist(reg, gt, p=1)

        cost_giou = iou3d_nms_utils.boxes_iou3d_gpu(reg, gt)

        C_cls = self.cost_cls * cost_class
        C_reg = self.cost_box * cost_box
        C_giou = self.cost_giou * cost_giou
        #cost matrix:
        C = self.cost_cls * cost_class + self.cost_box * cost_box + self.cost_giou * cost_giou
        C = C.view(batch_size, number_query, -1)
        C = C.cpu()

        sizes = [gt_boxes[i].shape[0] for i in range(batch_size)]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        batch_dict['assigned_target_indices'] = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # i for indices of selected predictions
        # j for indices of selected targets
        return batch_dict



