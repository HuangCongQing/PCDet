import torch.nn as nn
import MinkowskiEngine as ME

def post_act_block(in_channels,out_channels,kernel_size=1,stride=1,padding=0,dimension=None):
    '''

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param dimension:
    :return:
    '''
    m = nn.Sequential(
        ME.MinkowskiConvolution(in_channels,
                                out_channels,
                                kernel_size,
                                padding=padding,
                                stride=stride,
                                dilation=1,
                                has_bias=False,
                                dimension=dimension),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU()
    )
    return m

class MinkowskiBackbone(ME.MinkowskiNetwork):
    def __init__(self, model_cfg, in_feat, out_feat, D):
        '''

        :param model_cfg: model config
        :param in_feat  : input channels
        :param out_feat : output channels
        :param D        : dimension
        '''
        self.model_cfg = model_cfg
        super().__init__()
        self.sparse_shape = D
        block = post_act_block
        self.conv_input =nn.Sequential(
            block(in_feat, 16, kernel_size=3, padding=1, has_bias=False)
        )
        self.conv1 = nn.Sequential(
            block(16, 16, kernel_size=3, padding=1),
            block(16, 16),
            block(16, 16)
        )

        self.conv2 = nn.Sequential(
            block(16, 32, kernel_size=3, padding=1),
            block(32, 32),
            block(32, 32)
        )

        self.conv3 = nn.Sequential(
            block(32, 64, kernel_size=3, padding=1),
            block(64, 64),
            block(64, 64)
        )

        self.conv4 = nn.Sequential(
            block(64, 128, kernel_size=3, padding=1),
            block(128, out_feat),
            block(out_feat, out_feat)
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4)[batch_idx, z_idx, y_idx, x_idx]
        Returns:
              batch_dict:
                encoded_spconv_tensor: Minkowski Sparse tensor or Pytorch Tensor
        """

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        num_voxels, batches = voxel_coords.shape
        voxel_coords.reshape(num_voxels, batches[0], -1)
        coord_batches = voxel_coords.chunk(chunks=batch_size, dim=1)

        feat, coor = ME.utils.sparse_collate()
        input_sp_tensor = ME.SparseTensor(features=voxel_features,
                                          coordinates=voxel_coords
                                          )

        return batch_dict
