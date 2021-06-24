from .detector3d_template import Detector3DTemplate
from .. import transformer_heads

class TrPD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.criterion = transformer_heads.__all__[self.model_cfg.CRITERION.NAME](
            num_classes=len(self.model_cfg.CLASS_NAMES),
            weight_dict=self.model_cfg.CRITERION.WEIGHT_DICT,
            eos_coef=self.model_cfg.CRITERION.NO_OBJECT_WEIGHT,
            losses=self.model_cfg.CRITERION.LOSS
        )

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            batch_dict['batch_box_preds'] = batch_dict['rcnn_reg']
            batch_dict['batch_cls_preds'] = batch_dict['rcnn_cls']
            batch_dict['cls_preds_normalized'] = False
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        batch_dict = self.criterion(batch_dict)
        disp_dict={}
        loss_ce = batch_dict['loss_ce']
        loss_bbox = batch_dict['loss_bbox']
        loss_iou = batch_dict['loss_iou']

        loss = loss_ce + loss_bbox + loss_iou

        tb_dict = {
            'loss': loss.data,
            'loss_ce': loss_ce.data,
            'class_error': batch_dict['class_error'].data,
            'loss_iou': loss_iou.data,
            'loss_bbox': loss_bbox.data,
            'cardinality_error': batch_dict['cardinality_error'].data
        }

        disp_dict.update(tb_dict)
        '''
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        
        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
        '''
        return loss, tb_dict, disp_dict

