import torch.nn as nn
from .transformer import Transformer


class TransformerWrap(nn.Module):
    '''This module wraps position encoding method and the transformer.
    '''
    def __init__(self, model_cfg):
        super().__init__()
        self.config = model_cfg
        self.query = self.config.NUM_QUERY
        self.inputdim = self.config.INPUTDIM
        self.nhead = self.config.NHEAD
        self.num_encoder_layers = self.config.NUM_ENCODER_LAYERS
        self.num_decoder_layers = self.config.NUM_DECODER_LAYERS
        self.dim_feedforward = self.config.DIM_FEED_FORWARD
        self.dropout = self.config.DROPOUT
        self.activation = self.config.ACTIVATION
        self.normalize_before = self.config.NORMALIZE_BEFORE
        self.return_intermediate_dec = self.config.RETURN_INTERMEDIATE_DEC

        self.Q = nn.Embedding(self.query, self.inputdim)

        self.Transformer = Transformer(d_model=self.inputdim,
                                       nhead=self.nhead,
                                       num_encoder_layers=self.num_encoder_layers,
                                       num_decoder_layers=self.num_decoder_layers,
                                       dim_feedforward=self.dim_feedforward,
                                       dropout=self.dropout,
                                       activation=self.activation,
                                       normalize_before=self.normalize_before,
                                       return_intermediate_dec=self.return_intermediate_dec)

    def forward(self, batch_dict):
        '''
        :param batch_dict:
                    'encoded_nested_tensor':Nested_Tensor, [B, C, L, W, H]
                    'pos_embed': Tensor
                    'query_embed': Tensor
        :return:
                batch_dict:
                    'transformer_memory' [B, C, L, W, H]
                    'transformer_output' [B, number_query, dim_forward] dim_forward=2048(default)
        '''
        batch_dict['query_embed'] = self.Q.weight
        batch_dict = self.Transformer(batch_dict)
        return batch_dict


