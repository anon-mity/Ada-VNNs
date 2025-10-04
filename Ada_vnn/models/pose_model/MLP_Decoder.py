
import torch.nn as nn


'''class MLP_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_pts = cfg.num_point
        self.fc_layers = nn.Sequential(
            nn.Linear(self.num_pts, self.num_pts * 2),
            nn.BatchNorm1d(self.num_pts * 2),
            nn.Tanh(),
            nn.Linear(self.num_pts * 2 , self.num_pts * 3),
        )
        
    def forward(self, feat):
        # inv_feat.shape (b,self.feat_dim ,feat_sim)
        batch_size = feat.shape[0]
        pts = self.fc_layers(feat)
        pts = pts.reshape(batch_size, self.num_pts, 3)
        return pts'''


# 用于点云恢复的解码器
class MLP_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if (cfg.model =='vn_dgcnn' or cfg.model =='vn_pointnet' or cfg.model == 'vn_pointnet_am' or cfg.model == 'vn_ori_dgcnn' or
            cfg.model == 'vn_ori_globa' or cfg.model == 'vn_ori_globa6d'
                or cfg.model == 'vn_ori_globa_nonequ_linear'
                or cfg.model == 'vn_ori_globa_nonequ_linearx1lR'
                or cfg.model == 'vn_ori_globa_nonequ_linearx1lRlR'
                or cfg.model == 'vn_ori_globa_nonequ_linearx1lRlRlR'
                or cfg.model == 'vn_ori_globa_nonequ_linearx1LR'
                or cfg.model == 'vn_ori_globa_nonequ_linearx1LRLR'
                or cfg.model == 'vn_ori_globa_nonequ_linearx1LRLRLR'
                or cfg.model == 'vn_ori_globa9d'
                or cfg.model == 'vn_localori_globa6d'
                or cfg.model == 'vn_ori_globa_LSOG_noequ_linearx1LR'
                or cfg.model == 'vn_ori_globa6d_LRLR'
                or cfg.model == 'vn_ori_globa_LSOG_noequ_linearx1LRLR'
                or cfg.model == 'vn_transformer_am' or cfg.model == 'vn_transformer_amx1'
                or cfg.model == 'vn_transformer_nonequ_linearx1LR'
                or cfg.model == 'vn_transformer_nonequ_linearx1lR'
                or cfg.model == 'vn_transformer_nonequ_linearx1lRlR'
                or cfg.model == 'vn_transformer_nonequ_linearx1lRlRlR'
                or cfg.model == 'vn_transformer_amx3'
                or cfg.model == 'abla_vntrans_wo_rotation'
                or cfg.model == 'abla_vntrans_wo_complex'
                or cfg.model == 'abla_vntrans_wo_aggregation'
                or cfg.model == 'abla_vntrans_eulur'
                or cfg.model == 'abla_vntrans_quat'
                or cfg.model == 'abla_vntrans_axangle'
                or cfg.model == 'vn_dgcnn_linear_wbiasx1lR'
                or cfg.model == 'vn_dgcnn_linear_wbiasx1lRlR'
                or cfg.model == 'vn_dgcnn_linear_wbiasx1lRlRlR'):
            self.dim = cfg.feat_dim *2 *3  # 3072

        elif cfg.model == 'vn_transformer' or cfg.model =='vn_ori_globa6d_res' or cfg.model =='vn_transformer_amx3_res':
            self.dim = cfg.feat_dim * 3


        self.fc_layers = nn.Sequential(
            nn.Linear(self.dim, 2*self.dim),
            nn.LayerNorm(2*self.dim),     # 用 LN 替代 BN
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2*self.dim, 4*self.dim),
            nn.LayerNorm(4 * self.dim),  # 用 LN 替代 BN
            nn.ReLU(),

            nn.Linear(4 * self.dim, 3 * 4096),
            nn.Tanh()               # 保证输出范围 [-1, 1]
        )

    def forward(self, x):
        x_out = self.fc_layers(x).view(-1, 4096, 3)
        return x_out

