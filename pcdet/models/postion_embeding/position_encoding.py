# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from pcdet.utils.common_utils import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class PositionEmbeddingSine3D(nn.Module):
    '''
    This is a general form of the postional encoding in point cloud form.
    It is extended from the method that the detr network used.
    '''
    def __init__(self, model_cfg, scale=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_pos_feats = self.model_cfg.NUM_POS_FEATS
        self.temperature = self.model_cfg.TEMPERATURE
        self.normalize = self.model_cfg.NORMALIZE
        self.scale = scale
        if self.scale is not None and self.normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if self.scale is None:
            self.scale = 2 * math.pi
        #self.scale = scale

    def forward(self, tensor:NestedTensor):
        '''
        :param
         tensor:
            NestedTensor(B,H,L,W)
        '''

        encoded_mask = tensor.mask

        encoded_mask = ~encoded_mask
        voxel_embed = encoded_mask.cumsum(axis=0, dtype=torch.float32)
        z_embeded = encoded_mask.cumsum(axis=1, dtype=torch.float32)
        y_embeded = encoded_mask.cumsum(axis=2, dtype=torch.float32)
        x_embeded = encoded_mask.cumsum(axis=3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            voxel_embed = voxel_embed / (voxel_embed[-1, :, :, :] + eps) * self.scale
            z_embeded = z_embeded / (z_embeded[:, -1:, :, :] + eps) * self.scale
            y_embeded = y_embeded / (y_embeded[:, :, -1:, :] + eps) * self.scale
            x_embeded = x_embeded / (x_embeded[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=encoded_mask.device)
        dim_t = self.temperature ** (2*(dim_t//2) / self.num_pos_feats)

        pos_voxel = voxel_embed[:, :, :, :, None] / dim_t
        pos_x = x_embeded[:, :, :, :, None] / dim_t
        pos_y = y_embeded[:, :, :, :, None] / dim_t
        pos_z = z_embeded[:, :, :, :, None] / dim_t

        pos_voxel = torch.stack((pos_voxel[:, :, :, :, 0::2].sin(), pos_voxel[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_voxel, pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)

        return pos


class PositionEmbedingSpconv(nn.Module):
    def __init__(self, model_cfg, scale=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_pos_feats = self.model_cfg.NUM_POS_FEATS
        assert self.num_pos_feats//4, 'number of point position feature should be a multiple of 4'
        self.temperature = self.model_cfg.TEMPERATURE
        self.normalize = self.model_cfg.NORMALIZE
        self.scale = scale
        if self.scale is not None and self.normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if self.scale is None:
            self.scale = 2 * math.pi

    def forward(self, tensor:torch.Tensor):
        '''
        :alg:
            This method is implemented for assignment distance weights of the points clouds.
            This method utilizes 4 dimensions [distance to the origin, x, y, z] to weight the feature
            a point in point cloud.
        :param tensor: positions, (N,3)
        :return: position_embeds
        '''
        shape = tensor.shape
        assert len(shape)==2 and shape[-1]==3, 'input data shape is wrong, the shape should be (N,3) rather than {}'.format(shape)

        #origin = torch.zero_like(tensor.size(), dtype=tensor.dtype, device=tensor.device)
        dis_embed = torch.norm(tensor, dim=1)
        xyz_embed = tensor.cumsum(0, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            xyz_embed = xyz_embed / (xyz_embed[-1,:] + eps) * self.scale
            dis_embed = dis_embed / (dis_embed[-1] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats//4, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2*(dim_t//2) / self.num_pos_feats)

        pos_xyz = xyz_embed[:, :, None] / dim_t
        pos_dis = dis_embed[:, None] / dim_t

        pos_xyz = torch.stack((pos_xyz[:,:, 0::2].sin(), pos_xyz[:,:, 1::2].cos()), dim=3).flatten(2)
        pos_dis = torch.stack((pos_dis[:, 0::2].sin(), pos_dis[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_dis, pos_xyz[:, 0, :], pos_xyz[:, 1, :], pos_xyz[:, 2, :]), dim=1)

        return pos
