import torch
import torch.nn as nn
import spconv

from .position_encoding import PositionEmbedingSpconv
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils

from pcdet.utils.common_utils import NestedTensor

class VoxelEncoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.sampling_points = self.model_cfg.SAMPLINGPOINTS

        self.FPS = pointnet2_stack_utils.FurthestPointSamplingSpconv.apply
        self.NestedTensor = NestedTensor
        self.position = PositionEmbedingSpconv(self.model_cfg.POSITIONENCODER)

        self.pos = torch.Tensor()
        self.out = torch.Tensor()
        self.num_output_feature = self.model_cfg.POSITIONENCODER.NUM_POS_FEATS

    def forward(self, batch_dict):
        ''':param
            batch_dict:
                'encoded_spconv_tensor', (N,C)
        '''
        batch_size = batch_dict['batch_size']
        spconv_feature = batch_dict['encoded_spconv_tensor']
        assert isinstance(spconv_feature, spconv.SparseConvTensor), 'spconv_feature is not a sparse tensor'

        features = spconv_feature.features
        indices = spconv_feature.indices

        for i in range(batch_size):
            idx = indices[:, 0] == i#(N,1)
            cur_cood = indices[idx, 1:].type(torch.float32)#(M,3)
            cur_feature = features[idx]
            feature_idx = self.FPS(cur_cood.contiguous(), self.sampling_points).squeeze(dim=0).long()
            cur_feature = cur_feature[feature_idx].unsqueeze(0)
            cur_pos = self.position(cur_cood[feature_idx]).unsqueeze(0)

            if i == 0:
                self.out = cur_feature
                self.pos = cur_pos
            else:
                self.out = torch.cat((self.out, cur_feature), dim=0)#(B, M, C)
                self.pos = torch.cat((self.pos, cur_pos), dim=0)#(B, M, C)
        bs, m, c = self.out.shape
        mask = torch.zeros((bs, 1, m), dtype=torch.bool, device=self.out.device)
        self.out = self.NestedTensor(self.out, mask)

        batch_dict['encoded_spconv_feature'] = self.out #(B, M, C)
        batch_dict['pos_embed'] = self.pos #(B, M, C)
        return batch_dict

