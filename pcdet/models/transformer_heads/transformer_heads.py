import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''A very sample implementaion of boundle boxes prediction'''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, X):
        for i, layer in enumerate(self.layers):
            X = F.relu(layer(X)) if i < self.num_layers - 1 else layer(X)

        return X

class transformerHead(nn.Module):
    '''An implementation for detecting'''
    def __init__(self, model_cfg, num_class, cost_cls=1., cost_box=1., cost_giou=1.):
        super().__init__()
        self.config = model_cfg
        self.number_cls = num_class
        self.input_dim = self.config.INPUTDIM
        self.boxes_embed = MLP(self.input_dim, self.input_dim, output_dim=7, num_layers=3)
        self.cls_embed = nn.Linear(self.input_dim, self.number_cls)
        self.shareMLP = MLP(self.input_dim, self.input_dim, output_dim=self.input_dim, num_layers=3)

        self.cost_cls = cost_cls
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.bbox_dim = 7
        self.cls_dim = self.number_cls
        assert self.cost_cls != 0 or self.cost_box != 0 or cost_giou != 0, 'cannot be all 0'


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)


    def forward(self, batch_dict):
        ''':arg
            batch_dict:
                'batch_size'        :batch_size
                'transformer_output':(B, number_query, dim_forward) dim_forward=2048(default)


            :return
            batch_dict:
                'rcnn_cls'          :(B, number of query, number of classes + 1)
                'rcnn_reg'          :(B, number of query, 7)[x, y, z, dx, dy, dz, heading]

        '''

        output = batch_dict['transformer_output'].squeeze(0) # [B, number_query, dim_forward] dim_forward=2048(default)

        cls = self.cls_embed(output)       #(B, number of query, number of classes + 1)
        boxes = self.boxes_embed(output)   #(B, number of query, 7)[x, y, z, dx, dy, dz, heading]

        batch_dict['rcnn_cls'] = cls
        batch_dict['rcnn_reg'] = boxes
        return batch_dict







