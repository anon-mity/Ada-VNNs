import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from models.pose_model.dgcnn_pose import DGCNN
from models.pose_model.vn_dgcnn_pose import VN_DGCNN
from models.pose_model.vn_ori_dgcnn_pose import *
from models.pose_model.pointnet_pose import PointNet
from models.pose_model.point_cloud_transformer import Point_Cloud_Transformer
from models.pose_model.vn_pointnet import VN_PointNet
from models.pose_model.vn_ori_pointnet import VN_PointNet_AM
from models.pose_model.point_transformer import PointTransformer
from models.pose_model.vn_transformer import VN_Transformer
from models.pose_model.vn_ori_transformer import *
from models.pose_model.abla_vn_transformer import VN_Transformer_AMx1_Abla, VN_Transformer_AMx1_Abla_Eulur_Quat
from models.pose_model.MLP_Decoder import MLP_Decoder
from models.pose_model.Regressor import Regressor, VN_Regressor

'''from models.pose_model.vn_ori_dgcnn_pose import (VN_Ori_DGCNN, VN_Ori_Globa, VN_Ori_Globa6D, VN_Ori_Globa9D, VN_Ori_Globa6D_Res,
                                                 VN_Ori_Globa6D_lRlR, VN_Ori_Globa_noequ_linear,VN_Ori_Globa_noequ_linearx1LR,VN_Ori_Globa_noequ_linearx1LRLRLR,
                                                 VN_Ori_Globa_noequ_linearx1LRLR, VN_Ori_Globa6D_LSOG_noequ_linearx1LR,VN_Ori_Globa6D_LSOG_noequ_linearx1LRLR,
                                                 VN_Ori_Globa_noequ_linearx1lR,VN_Ori_Globa_noequ_linearx1lRlR,VN_Ori_Globa_noequ_linearx1lRlRlR,VN_DGCNN_linear_wbiasx1lR,
                                                 VN_DGCNN_linear_wbiasx1lRlRlR,VN_DGCNN_linear_wbiasx1lRlR)'''
#from models.pose_model.vn_ori_transformer import (VN_Transformer_AM,VN_Transformer_AMx1,VN_Transformer_AMx3,VN_Transformer_AMx3_Res,VN_Transformer_noequ_linearx1LR,VN_Transformer_noequ_linearx1lR,
#                                                 VN_Transformer_noequ_linearx1lRlR,VN_Transformer_noequ_linearx1lRlRlR)

def bgdR(Rgts, Rps):
    # input: R Matrix batch * 3 * 3
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(theta)
    return angle

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.init_encoder(cfg)
        self.encoder = torch.nn.DataParallel(self.encoder)

        self.regress = self.init_regress(cfg)
        self.regress = torch.nn.DataParallel(self.regress)

        if cfg.mode == 'recons':
            self.decoder = self.init_decoder(cfg)
            self.decoder = torch.nn.DataParallel(self.decoder)

    def forward(self, pc_aug, pc_norm=None):
        if self.cfg.mode != 'recons':
            # 等变和不变特征提取
            Equ_feat, Inv_feat = self.encoder(pc_aug)  # (b, 2*dim, 3)  # [batch, 2*dim, 1024] / None

            if pc_norm is not None:
                Equ_feat_norm, Inv_feat_norm = self.encoder(pc_norm)
                Equ_feat = torch.cat((Equ_feat, Equ_feat_norm), dim=1)  # (b, 2*dim, 3) -> (b, 4*dim, 3)
            # pose
            pred_r33 = self.regress(Equ_feat)
            return pred_r33
        else:
            # 等变和不变特征提取
            Equ_feat, Inv_feat = self.encoder(pc_aug)  # (b, 2*dim, 3)  # [batch, 2*dim, 1024] / None
            B, C, _ = Equ_feat.shape
            Equ_feat = Equ_feat.view(B, C*_)
            recons_pc = self.decoder(Equ_feat)
            return recons_pc


    def training_step(self, pc_aug, pc_norm, label):
        if self.cfg.model == 'point_transformer' or self.cfg.model == 'point_cloud_transformer':
            src_pc = pc_aug
            tgt_pc = pc_norm
        else:
            # b,1024,3 -> b,3,1024
            src_pc = torch.permute(pc_aug,dims=(0, 2, 1))
            tgt_pc = torch.permute(pc_norm,dims=(0, 2, 1))
        if self.cfg.mode == 'registr':
            pred_r33 = self.forward(src_pc, tgt_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))
            mse_loss = F.mse_loss(label, pred_r33)
            loss = mse_loss + angle_loss #+ sl_loss #

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'mse_loss':mse_loss
                         }
            return loss_dict

        elif self.cfg.mode == 'pose':
            pred_r33 = self.forward(src_pc)
            trans_pc = torch.einsum('bij,bjk->bik',pc_norm, pred_r33)
            sl_loss,_ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            #mse_loss = F.mse_loss(label, pred_r33)
            angle_loss = torch.mean(bgdR(label, pred_r33))
            loss = angle_loss + sl_loss # mse_loss

            loss_dict = {'loss': loss,
                    'sl_loss': sl_loss,
                    #'mse_loss':mse_loss,
                    'angle_loss': angle_loss
             }
            return loss_dict

        elif self.cfg.mode == 'recons':
            recons_pc = self.forward(src_pc)
            recons_loss, _ = chamfer_distance(recons_pc, label, batch_reduction='mean', point_reduction='mean')
            loss = recons_loss
            loss_dict = {'loss': loss,
                         'recons_loss': recons_loss
                         }

            return loss_dict

    def test_step(self, pc_aug, pc_norm, label):
        if self.cfg.model == 'point_transformer' or self.cfg.model == 'point_cloud_transformer':
            src_pc = pc_aug
            tgt_pc = pc_norm
        else:
            # b,1024,3 -> b,3,1024
            src_pc = torch.permute(pc_aug,dims=(0, 2, 1))
            tgt_pc = torch.permute(pc_norm,dims=(0, 2, 1))

        if self.cfg.mode == 'registr':
            pred_r33 = self.forward(src_pc, tgt_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))
            mse_loss = F.mse_loss(label, pred_r33)
            loss =  mse_loss  + angle_loss #+ sl_loss #

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'mse_loss':mse_loss
                         }
            return loss_dict

        elif self.cfg.mode == 'pose':
            pred_r33 = self.forward(src_pc.float())
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = torch.mean(bgdR(label, pred_r33))
            loss = sl_loss + angle_loss

            loss_dict = {'loss': loss,
                         'sl_loss': sl_loss,
                         #'mse_loss':mse_loss,
                         'angle_loss': angle_loss,
                         }
            return loss_dict

        elif self.cfg.mode == 'recons':
            recons_pc = self.forward(src_pc)
            recons_loss, _ = chamfer_distance(recons_pc, label, batch_reduction='mean', point_reduction='mean')
            loss = recons_loss
            loss_dict = {'loss': loss,
                         'recons_loss': recons_loss }
            return loss_dict


    def init_encoder(self,cfg):
        encoder = self.model_choice(self.cfg.model,cfg)

        return encoder

    '''if self.cfg.model1 is not None:
        encoder = self.model_choice(self.cfg.model1,cfg)
        return encoder

    elif self.cfg.model2 is not None:
        encoder = self.model_choice(self.cfg.model2,cfg)
        return encoder'''


    def model_choice(self,cfg_model,cfg):
        if cfg_model == 'pointnet':
            encoder = DGCNN(cfg)
        elif cfg_model == 'vn_dgcnn':
            encoder = VN_DGCNN(cfg)
        elif cfg_model == 'vn_ori_dgcnn':
            encoder = VN_Ori_DGCNN(cfg)
        elif cfg_model == 'vn_ori_globa':
            encoder = VN_Ori_Globa(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linear':
            encoder = VN_Ori_Globa_noequ_linear(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linearx1LRLRLR':
            encoder = VN_Ori_Globa_noequ_linearx1LRLRLR(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linearx1LRLR':
            encoder = VN_Ori_Globa_noequ_linearx1LRLR(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linearx1LR':
            encoder = VN_Ori_Globa_noequ_linearx1LR(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linearx1lRlRlR':
            encoder = VN_Ori_Globa_noequ_linearx1lRlRlR(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linearx1lRlR':
            encoder = VN_Ori_Globa_noequ_linearx1lRlR(cfg)
        elif cfg_model == 'vn_ori_globa_nonequ_linearx1lR':
            encoder = VN_Ori_Globa_noequ_linearx1lR(cfg)
        elif cfg_model == 'vn_dgcnn_linear_wbiasx1lR':
            encoder = VN_DGCNN_linear_wbiasx1lR(cfg)
        elif cfg_model == 'vn_dgcnn_linear_wbiasx1lRlR':
            encoder = VN_DGCNN_linear_wbiasx1lRlR(cfg)
        elif cfg_model == 'vn_dgcnn_linear_wbiasx1lRlRlR':
            encoder = VN_DGCNN_linear_wbiasx1lRlRlR(cfg)

        elif cfg_model == 'vn_ori_globa6d':
            encoder = VN_Ori_Globa6D(cfg)
        elif cfg_model == 'vn_ori_globa6d_lRlR':
            encoder = VN_Ori_Globa6D_lRlR(cfg)
        elif cfg_model == 'vn_ori_globa6d_LRLR':
            encoder = VN_Ori_Globa6D_LRLR(cfg)
        elif cfg_model == 'vn_ori_globa6d_nzt_lRlR':
            encoder = VN_Ori_Globa6D_nzt_lRlR(cfg)
        elif cfg_model == 'vn_ori_globa6d_nzt_lRlRlR':
            encoder = VN_Ori_Globa6D_nzt_lRlRlR(cfg)
        elif cfg_model == 'vn_ori_globa_LSOG_noequ_linearx1LR':
            encoder = VN_Ori_Globa6D_LSOG_noequ_linearx1LR(cfg)
        elif cfg_model == 'vn_ori_globa_LSOG_noequ_linearx1LRLR':
            encoder = VN_Ori_Globa6D_LSOG_noequ_linearx1LRLR(cfg)
        elif cfg_model == 'vn_ori_globa6d_res':
            encoder = VN_Ori_Globa6D_Res(cfg)
        elif cfg_model == 'vn_ori_globa9d':
            encoder = VN_Ori_Globa9D(cfg)
        elif cfg_model == 'pointnet':
            encoder = PointNet(cfg)
        elif cfg_model == 'vn_pointnet':
            encoder = VN_PointNet(cfg)
        elif cfg_model == 'vn_pointnet_am':
            encoder = VN_PointNet_AM(cfg)
        elif cfg_model == 'vn_transformer':
            encoder = VN_Transformer(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_nzt_w_lR':
            encoder = VN_Transformer_NZT_W_lR(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_nzt_w_lRlR':
            encoder = VN_Transformer_NZT_W_lRlR(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_am':
            encoder = VN_Transformer_AM(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_nonequ_linearx1LR':
            encoder = VN_Transformer_noequ_linearx1LR(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_nonequ_linearx1lR':
            encoder = VN_Transformer_noequ_linearx1lR(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_nonequ_linearx1lRlR':
            encoder = VN_Transformer_noequ_linearx1lRlR(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_nonequ_linearx1lRlRlR':
            encoder = VN_Transformer_noequ_linearx1lRlRlR(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_amx1':
            encoder = VN_Transformer_AMx1(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_amx3':
            encoder = VN_Transformer_AMx3(cfg.feat_dim)
        elif cfg_model == 'vn_transformer_amx3_res':
            encoder = VN_Transformer_AMx3_Res(cfg.feat_dim)
        elif cfg_model == 'abla_vntrans_wo_rotation' or cfg_model == 'abla_vntrans_wo_complex' or cfg_model == 'abla_vntrans_wo_aggregation':
            encoder = VN_Transformer_AMx1_Abla(cfg, cfg.feat_dim)
        elif cfg_model == 'abla_vntrans_eulur' or cfg_model == 'abla_vntrans_quat' or cfg_model == 'abla_vntrans_axangle':
            encoder = VN_Transformer_AMx1_Abla_Eulur_Quat(cfg, cfg.feat_dim)
        elif cfg_model == 'point_cloud_transformer':
            encoder = Point_Cloud_Transformer(cfg)
        elif cfg_model == 'point_transformer':
            encoder = PointTransformer(cfg)
        return encoder

    def init_decoder(self,cfg):
        decoder = MLP_Decoder(cfg)
        return decoder

    def init_regress(self,cfg):
        regressor = Regressor(cfg)
        if cfg.regress == 'vn':
            regressor = VN_Regressor(cfg)
        return regressor

    def test_metric(self, pc_aug, pc_norm, label):
        if self.cfg.model == 'point_transformer' or self.cfg.model == 'point_cloud_transformer':
            src_pc = pc_aug
            tgt_pc = pc_norm
        else:
            # b,1024,3 -> b,3,1024
            src_pc = torch.permute(pc_aug, dims=(0, 2, 1))
            tgt_pc = torch.permute(pc_norm, dims=(0, 2, 1))

        if self.cfg.mode == 'registr':
            pred_r33 = self.forward(src_pc, tgt_pc)
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = bgdR(label, pred_r33)
            jiaodu_loss = angle_loss * 180 / math.pi

            loss_dict = {'CD': sl_loss,
                         'RRE_hd': angle_loss,
                         'RRE_jd': jiaodu_loss
                         }
            return loss_dict

        elif self.cfg.mode == 'pose' and not self.cfg.disentangle:
            pred_r33 = self.forward(src_pc.float())
            trans_pc = torch.einsum('bij,bjk->bik', pc_norm, pred_r33)
            sl_loss, _ = chamfer_distance(trans_pc, pc_aug, batch_reduction='mean', point_reduction='mean')
            angle_loss = bgdR(label, pred_r33)
            jiaodu_loss = angle_loss * 180 / math.pi

            loss_dict = {'sl_loss': sl_loss,
                         'angle_loss': angle_loss,
                         'jiaodu': jiaodu_loss
                         }
            return loss_dict

    def test_robust(self, pc_norm):
        if self.cfg.model == 'point_transformer' or self.cfg.model == 'point_cloud_transformer':
            pc_norm = pc_norm
        else:
            # b,1024,3 -> b,3,1024
            pc_norm = torch.permute(pc_norm, dims=(0, 2, 1))

        Equ_feat_norm, _ = self.encoder(pc_norm)  # B,D,3, _
        return Equ_feat_norm

    def latent_test(self, pc_aug):
        pc_aug = torch.permute(pc_aug, dims=(0, 2, 1))

        Equ_feat, Inv_feat = self.encoder(pc_aug)  # (b, 2*dim, 3)  # [batch, 2*dim, 1024] / None

        return Equ_feat