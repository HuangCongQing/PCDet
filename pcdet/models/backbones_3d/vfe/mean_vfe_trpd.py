import torch
from .vfe_template import VFETemplate


class MeanVFETrPD(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
                gt_box: (Batch_size, N, 8)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        size = 0.1000
        #对已经有的None 长宽高 设置为0.1
        gt = batch_dict['gt_boxes']
        b, m, c = gt.shape
        indices = torch.nonzero(gt[:, :, -1] < 0.99)
        for (batch, n) in indices:
            gt[batch, n, 3: 6] = size
        #补齐100个num query
        temp = torch.zeros((b, 100-m, c), dtype=gt.dtype, device=gt.device)
        temp[: ,:, 3: 6] = size
        gt = torch.cat((gt, temp), dim=1)
        batch_dict['gt_boxes'] = gt

        return batch_dict