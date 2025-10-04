import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix, axis_angle_to_matrix, quaternion_to_matrix


EPS = 1e-6

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNSimpleLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNSimpleLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out



def get_on_vector(normalized_J):
    '''
    normalized_J: normalized direction vector [batch size, num points, embedding dimension, 3]
    '''

    # calculate normalized U vector that is orthogonal to J
    # Let J = [x, y, z]
    # Then U = [x, y, -(x^2 + y^2) / (z + eps)]

    # get [x, y]
    sub_vec_J = normalized_J[:, :, :, :2]  # b x c x e x 2
    sub_vec_J1 = sub_vec_J.unsqueeze(3)
    sub_vec_J2 = sub_vec_J.unsqueeze(4)
    # calculate (x^2 + y^2)
    U_z1 = torch.einsum("abcik,abckj->abcij",sub_vec_J1, sub_vec_J2)
    U_z = torch.squeeze(torch.squeeze(U_z1, 4), 3)  # b x c x e

    # calculate -(x^2 + y^2) / (z + eps)
    U_z = -U_z / (normalized_J[:, :, :, 2] + EPS)  # b x c x e

    # form [x, y, -(x^2 + y^2) / (z + eps)]
    U = torch.cat((sub_vec_J, U_z.unsqueeze(3)), dim=3)  # b x c x e x 3

    # normalize
    normalized_U = (U.permute(3, 0, 1, 2) / (torch.linalg.norm(U, dim=3) + EPS)).permute(1, 2, 3, 0)  # b x c x e x 3

    return normalized_U

def get_basis(J):
    '''
    J: direction vector [batch size (B), num points (C), embedding dimension (E), 3]
    '''
    J_mochang = torch.linalg.norm(J, dim=3) + EPS
    # normalize J vectors ,将 J 中的每个向量通过除以其模来归一化，得到单位向量
    normalized_J = (J.permute(3, 0, 1, 2) / J_mochang).permute(1, 2, 3, 0)  # b x c x e x 3

    normalized_U = get_on_vector(normalized_J)  # b x c x e x 3

    # calculate V vector that is orthogonal to J and U
    normalized_V = torch.cross(normalized_U, normalized_J, dim=3)  # b x c x e x 3

    # R = (U, V, J)
    R = torch.cat((normalized_U, normalized_V, normalized_J), dim=-1)  # b x c x e x 9
    B, C, E, _ = R.size()
    R = torch.reshape(R, (B, C, E, 3, 3))  # b x c x e x 3 x 3

    return R


def get_rtx(index, RT, X):
    '''
    R: rotation basis [b, n, e, 3, 3]
    X: point features of shape [B, n, E, 3]
    '''

    indexed_RT = RT[:, :, :, index, :]  # b x N x D x 3
    indexed_RT = torch.unsqueeze(indexed_RT, 3) # B x N x D x 1 x 3
    X = torch.unsqueeze(X, 4)                   # B x N x D x 3 x 1
    rtx = torch.einsum("abcik,abckj->abcij",indexed_RT,X)  # B x N x D x 1 x 1
    rtx = torch.squeeze(torch.squeeze(rtx, 4), 3)  # b x c x e

    return rtx

def get_rrtx(index, R, RTX):
    '''
    R: rotation basis [b, N, D, 3, 3]
    RTX: [B, N, D] # 旋转不变表示的标量
    '''

    indexed_R = R[:, :, :, :, index].permute(3, 0, 1, 2)  # 3 x b x c x e
    rrtx = (indexed_R * RTX).permute(1, 2, 3, 0)  # b x c x e x 3

    return rrtx


#
class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexLinear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., 1024, E, 3]
        J: directions of shape [..., 1024, E, 3] (same shape as X)
        '''

        # 1. 基于向量特征J计算一组正交特征
        R = get_basis(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.计算正交基特征矩阵，权重矩阵和投影矩阵的内积
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3

        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y



class ComplexLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(ComplexLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = ComplexLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., C, E, 3]
        J: directions of shape [..., C, E, 3] (same shape as X)
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out



# 6D -> 3x3
def bgs_extended(d6s):
    """
    将6D旋转向量转换为3x3旋转矩阵。

    Args:
        d6s (torch.Tensor): 输入张量，形状为 (b, 1, dim, 3, 2)

    Returns:
        torch.Tensor: 旋转矩阵，形状为 (b, 1, dim, 3, 3)
    """
    # 确保输入形状为 (b, 1, dim, 3, 2)
    assert d6s.dim() == 5 and d6s.shape[-2] == 3 and d6s.shape[-1] == 2, \
        "输入张量形状应为 (b, 1, dim, 3, 2)"

    # 第一步：归一化第一个向量 b1
    # d6s[..., :, 0] 的形状为 (b, 1, dim, 3)
    b1 = F.normalize(d6s[..., :, 0], p=2, dim=-1)  # 形状: (b, 1, dim, 3)

    # 第二步：提取第二个向量 a2
    a2 = d6s[..., :, 1]  # 形状: (b, 1, dim, 3)

    # 第三步：计算 a2 在 b1 上的投影
    # 计算点积 <a2, b1>，形状为 (b, 1, dim, 1)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)

    # 投影向量：<a2, b1> * b1，形状为 (b, 1, dim, 3)
    proj = dot * b1

    # 第四步：计算正交化后的 b2 向量
    b2_unorm = a2 - proj  # 去除在 b1 上的分量，形状: (b, 1, dim, 3)

    # 归一化得到 b2，形状: (b, 1, dim, 3)
    b2 = F.normalize(b2_unorm, p=2, dim=-1)

    # 第五步：计算第三个基向量 b3，通过 b1 和 b2 的外积
    b3 = torch.cross(b1, b2, dim=-1)  # 形状: (b, 1, dim, 3)

    # 第六步：堆叠 b1, b2, b3 生成旋转矩阵
    # 堆叠后形状为 (b, 1, dim, 3, 3)
    rot = torch.stack([b1, b2, b3], dim=-1)

    return rot



# 仿射线性层
class Affine_Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2]
        '''
        # Gram生成的旋转等变的矩阵
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)   #取逆 # b x c x e x 3 x 3

        #1. 使用逆矩阵构造旋转不变的表示 [RT0X,RT1X,RT2X]
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 2.构造从局部坐标系返回全局坐标系构建等变表示（为什么要返回？）
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        # 3.
        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y


# 用于验证等变性，只保留第一步
class Affine_Linear_only1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_only1, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2]
        '''
        # 1.
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        #
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        RT0X = torch.unsqueeze(RT0X, 3)
        RT1X = torch.unsqueeze(RT1X, 3)
        RT2X = torch.unsqueeze(RT2X, 3)
        combined = torch.cat([RT0X, RT1X, RT2X], dim=3)

        return combined

# 用于验证等变性，只保留第一步和第二步
class Affine_Linear_only12(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_only12, self).__init__()
    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2]
        '''
        # 1.
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3
        #
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 2.
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        combined = a_term + b_term + c_term  # b x c x e' x 3

        return combined


# 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
class Affine_Linear_X_YZ(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_X_YZ, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 1. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 2.构造类复数的线性组合。
        a_term = get_rrtx(1, R, RT1X) + get_rrtx(2, R, RT2X)  # b x c x e x 3
        b_term = get_rrtx(2, R, RT1X) - get_rrtx(1, R, RT2X)  # b x c x e x 3
        c_term = get_rrtx(0, R, RT0X)  # b x c x e x 3
        a_term = a_term + b_term

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_c_term  # b x c x e' x 3

        Y = torch.einsum('fe,bcei->bcfi', self.B, Y)  # b x c x e' x 3

        return Y


# 这一版本是使用其他的组合，使用Z和X组合 然后把Y单独拿出来，理论上应该和之前一样。
class Affine_Linear_Y_ZX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Y_ZX, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(2, R, RT2X) + get_rrtx(0, R, RT0X)  # b x c x e x 3
        b_term = get_rrtx(0, R, RT2X) - get_rrtx(2, R, RT0X)  # b x c x e x 3
        c_term = get_rrtx(1, R, RT1X)  # b x c x e x 3

        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y


# 把旋转变成单位矩阵了
class Affine_Linear_Abla_worotation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_worotation, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        #R1 = bgs_extended(J)  # b x c x e x 3 x 3
        B = X.shape[0]
        N = X.shape[1]
        D = X.shape[2]
        identity_matrix = torch.eye(3, device=X.device).view(1, 1, 1, 3, 3)
        R = identity_matrix.expand(B, N, D, 3, 3)

        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

# RT0X  RT1X RT2X 复制三份
class Affine_Linear_Abla_wocomplex(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_wocomplex, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        a_term = torch.cat((RT0X.unsqueeze(-1),RT1X.unsqueeze(-1),RT2X.unsqueeze(-1)),dim=-1)

        # 3.构造类复数的线性组合。
        #a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        #b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        #c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, a_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, a_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

class Affine_Linear_Abla_woaggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_Abla_woaggregation, self).__init__()
        #self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        #self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        #self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3

        #summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        #summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        #summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = a_term + b_term + c_term  # b x c x e' x 3

        return Y


class Affine_Geometric_Module(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Affine_Geometric_Module, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        #self.linear = Affine_Linear(in_channels, out_channels)

        #self.linear = Affine_Linear_only1(in_channels, out_channels)
        #self.linear = Affine_Linear_only12(in_channels, out_channels)
        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
        # 就是使用YZ 轴构建复数空间，然后使用X轴表示额外的自由度。
        self.linear = Affine_Linear_X_YZ(in_channels, out_channels)

        # 这一版本是使用Y_ZX的组合，其中使用ZX构建复数空间，然后使用Y表示单独的自由度。
        #self.linear = Affine_Linear_Y_ZX(in_channels, out_channels)


        # 线性构造的方式改变，全用加法。
        #self.linear = Affine_Linear_add(in_channels, out_channels)

        # 线性构造的方式改变，全用减法。
        #self.linear = Affine_Linear_subtr(in_channels, out_channels)

        # 标量复数构造
        #self.linear = Affine_Linear_hbfushu1(in_channels, out_channels, args)

        # 向量复数构造
        #self.linear = Affine_Linear_hbfushu2(in_channels, out_channels, args)

        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

class Affine_Geometric_Module_only12(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Affine_Geometric_Module_only12, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        #self.linear = Affine_Linear(in_channels, out_channels)
        #self.linear = Affine_Linear_only1(in_channels, out_channels)
        self.linear = Affine_Linear_only12(in_channels, out_channels)
        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。


        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class Affine_Geometric_Module_LSOG_lRlR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Affine_Geometric_Module_LSOG_lRlR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear1 = Affine_Linear(in_channels, out_channels)
        self.linear2 = OnlyNonequi_linear(out_channels, out_channels)

        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.leaky_relu2 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        # LSOG +LR
        x = self.linear1(X, J).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1).permute(0, 2, 3, 1)
        x2 = self.leaky_relu2(x2)
        return x2


class Affine_Geometric_Module_LSOG_LRLR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Affine_Geometric_Module_LSOG_LRLR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear1 = Affine_Linear(in_channels, out_channels)
        self.linear2 = OnlyNonequi_Linear(out_channels, out_channels)

        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.leaky_relu2 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        # LSOG +LR
        x = self.linear1(X, J).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1).permute(0, 2, 3, 1)
        x2 = self.leaky_relu2(x2)
        return x2


class Module_nzt_lRlR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Module_nzt_lRlR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear1 = Affine_Linear_only12(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = Nonequ_Module_lRlR(out_channels, out_channels)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        # LSOG +LR
        x = self.linear1(X, J).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1, device)
        return x2


class Module_nzt_lRlRlR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Module_nzt_lRlRlR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear1 = Affine_Linear_only12(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = Nonequ_Module_lRlRlR(out_channels, out_channels)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        # LSOG +LR
        x = self.linear1(X, J).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1, device)
        return x2

class Module_nzt_W_lR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Module_nzt_W_lR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear1 = Affine_Linear(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = Nonequ_Module_lRlR(out_channels, out_channels)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        # LSOG +LR
        x = self.linear1(X, J).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1, device)
        return x2

class Module_nzt_W_lRlR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Module_nzt_W_lRlR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear1 = Affine_Linear(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = Nonequ_Module_lRlR(out_channels, out_channels)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        # LSOG +LR
        x = self.linear1(X, J).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1, device)
        return x2

class Affine_Geometric_Module_LSOG(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Affine_Geometric_Module_LSOG, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = Affine_Linear_only12(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

    def forward(self, X, J, device):
        '''
        X: point features of shape [B, N, D, 3]
        J: directions of shape [B, N, D, 3,2]
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

class OnlyNonequi_linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OnlyNonequi_linear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X):
        '''
        X: point features of shape [..., N, D, 3]
        '''
        a_term = torch.einsum('fe,bcei->bcfi', self.A, X)  # b x c x e' x 3
        Y = a_term  # b x c x e' x 3

        return Y

class OnlyNonequi_Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OnlyNonequi_Linear, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
    def forward(self, X):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2]
        '''
        a_term = torch.einsum('fe,bcei->bcfi', self.A, X)  # b x c x e' x 3
        b_term = torch.einsum('fe,bcei->bcfi', self.B, X)  # b x c x e' x 3
        c_term = torch.einsum('fe,bcei->bcfi', self.C, X)  # b x c x e' x 3
        Y = a_term + b_term + c_term# b x c x e' x 3

        return Y

class Nonequ_Module_LR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Nonequ_Module_LR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 直接一个线性层
        self.linear = OnlyNonequi_Linear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, device):
        '''
        X: point features of shape [B, N, D, 3]
        '''
        x_out = self.linear(X).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x_out)
        return x_out


class Nonequ_Module_LRLR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Nonequ_Module_LRLR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 直接一个线性层
        self.linear1 = OnlyNonequi_Linear(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = OnlyNonequi_Linear(out_channels, out_channels)
        self.leaky_relu2 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, device):
        '''
        X: point features of shape [B, N, D, 3]
        '''
        # LR
        x = self.linear1(X).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1).permute(0, 2, 3, 1)
        x2 = self.leaky_relu2(x2)
        return x2


class Nonequ_Module_LRLRLR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Nonequ_Module_LRLRLR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 直接一个线性层
        self.linear1 = OnlyNonequi_Linear(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = OnlyNonequi_Linear(out_channels, out_channels)
        self.leaky_relu2 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                  negative_slope=negative_slope)
        self.linear3 = OnlyNonequi_Linear(out_channels, out_channels)
        self.leaky_relu3 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                       negative_slope=negative_slope)
        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, device):
        '''
        X: point features of shape [B, N, D, 3]
        '''
        # LR
        x = self.linear1(X).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1).permute(0, 2, 3, 1)
        x2 = self.leaky_relu2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        # LR
        x3 = self.linear2(x2).permute(0, 2, 3, 1)
        x3 = self.leaky_relu2(x3)
        return x3


# 大写L和小写l是有区别的，大写表示W是三个，小写表示W就一个
class Nonequ_Module_lR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Nonequ_Module_lR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 直接一个线性层
        self.linear = OnlyNonequi_linear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, device):
        '''
        X: point features of shape [B, N, D, 3]
        '''
        x1 = self.linear(X).permute(0, 2, 3, 1)  #leakrelu之前的维度是: B,D,3,N (其中N=1)
        # LeakyReLU
        x_out = self.leaky_relu(x1)
        return x_out


class BiasLinearandRelu(nn.Module):
    def __init__(self, in_channels, out_channels,share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(BiasLinearandRelu, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=True)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        '''
        x: point features of shape [B, Dim, 3, N_samples]
        '''
        x1 = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)  #leakrelu之前的维度需要是: B,D,3,N (其中N=1)
        x_out = self.leaky_relu(x1)
        return x_out

class Nonequ_Module_lRlR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Nonequ_Module_lRlR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 直接一个线性层
        self.linear1 = OnlyNonequi_linear(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = OnlyNonequi_linear(out_channels, out_channels)
        self.leaky_relu2 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, device):
        '''
        X: point features of shape [B, N, D, 3]
        '''
        # LR
        x = self.linear1(X).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1).permute(0, 2, 3, 1)
        x2 = self.leaky_relu2(x2)
        return x2


class Nonequ_Module_lRlRlR(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(Nonequ_Module_lRlRlR, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 直接一个线性层
        self.linear1 = OnlyNonequi_linear(in_channels, out_channels)
        self.leaky_relu1 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)
        self.linear2 = OnlyNonequi_linear(out_channels, out_channels)
        self.leaky_relu2 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                  negative_slope=negative_slope)
        self.linear3 = OnlyNonequi_linear(out_channels, out_channels)
        self.leaky_relu3 = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                       negative_slope=negative_slope)
        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, device):
        '''
        X: point features of shape [B, N, D, 3]
        '''
        # LR
        x = self.linear1(X).permute(0, 2, 3, 1)
        x1 = self.leaky_relu1(x)
        x1 = x1.permute(0, 3, 1, 2)
        # LR
        x2 = self.linear2(x1).permute(0, 2, 3, 1)
        x2 = self.leaky_relu2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        # LR
        x3 = self.linear2(x2).permute(0, 2, 3, 1)
        x3 = self.leaky_relu2(x3)
        return x3

class VN_PointNet_Affine_Module(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None):
        super(VN_PointNet_Affine_Module, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        self.linear = Affine_Linear(in_channels, out_channels)

        # 这一版本是对构造好的复数空间XY-Z，使用单个A学习XY而不是AB两个分开学习A和B 正常版本是上面的。
        #self.linear = Affine_Linear_onlyAC(in_channels, out_channels)

        # 复数空间为XY-Z , 但是全部只用一个A来学，而不是AB学习XY；C学习Z
        #self.linear = Affine_Linear_onlyA(in_channels, out_channels)

        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
        # 就是使用YZ 轴构建复数空间，然后使用X轴表示额外的自由度。
        self.linear = Affine_Linear_X_YZ(in_channels, out_channels)

        # 这一版本是使用Y_ZX的组合，其中使用ZX构建复数空间，然后使用Y表示单独的自由度。
        #self.linear = Affine_Linear_Y_ZX(in_channels, out_channels)

        # 线性构造的方式改变，全用加法。
        #self.linear = Affine_Linear_add(in_channels, out_channels)

        # 线性构造的方式改变，全用减法。
        #self.linear = Affine_Linear_subtr(in_channels, out_channels)

        # 标量复数构造
        #self.linear = Affine_Linear_hbfushu1(in_channels, out_channels, args)

        # 向量复数构造
        #self.linear = Affine_Linear_hbfushu2(in_channels, out_channels, args)

        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3,2] (same shape as X)
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class Affine_Module_Abla(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2, args=None, ablamode=''):
        super(Affine_Module_Abla, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        self.ablamode = ablamode

        # 正常情况：使用XY轴构建复数空间，然后使用Z轴表示额外的自由度。并且XY轴分开使用线性层。
        if self.ablamode == 'rotation':
            self.linear = Affine_Linear_Abla_worotation(in_channels, out_channels)
        elif self.ablamode == 'complex':
            self.linear = Affine_Linear_Abla_wocomplex(in_channels, out_channels)
        elif self.ablamode == 'aggregation':
            self.linear = Affine_Linear_Abla_woaggregation(in_channels, out_channels)

        # 旋转的方式，欧拉，四元数等
        elif self.ablamode == 'eulur':
            self.linear = Affine_Linear_Abla_Eulur(in_channels, out_channels)
        elif self.ablamode == 'quat':
            self.linear = Affine_Linear_Abla_Quat(in_channels, out_channels)
        elif self.ablamode == 'axangle':
            self.linear = Affine_Linear_Abla_Axangle(in_channels, out_channels)
        # 这一版本是使用其他的组合，使用1和2组合 然后把0单独拿出来，理论上应该和之前一样。
        # 就是使用YZ 轴构建复数空间，然后使用X轴表示额外的自由度。
        #self.linear = Affine_Linear_X_YZ(in_channels, out_channels)

        # 这一版本是使用Y_ZX的组合，其中使用ZX构建复数空间，然后使用Y表示单独的自由度。
        #self.linear = Affine_Linear_Y_ZX(in_channels, out_channels)


        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, X, J, device):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3,2] (same shape as X)
        if eulur J:B,N,D,3
        if quat / axangle  J:B,N,D,4
        '''
        x = self.linear(X, J).permute(0, 2, 3, 1)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

# AR  input:B,D,3,N
class Local_Geometric(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(Local_Geometric, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.vnn_lin_1 = VNSimpleLinear(in_channels, out_channels)
        self.vnn_lin_2 = VNSimpleLinear(in_channels, out_channels)
        self.vnn_lin_2_= VNSimpleLinear(in_channels, out_channels)

        self.linear = Affine_Linear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        #self.use_batchnorm = use_batchnorm
        #if use_batchnorm != 'none':
        #    self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, input, device):
        # input (32, 21, 3, 1024)

        v = self.vnn_lin_1(input).permute(0, 3, 1, 2)  # [batch, N, D, 3]
        # self.vnn_lin_2 就是VNLinearLeakyReLU 少了BR ，只有L
        j = self.vnn_lin_2(input)  # [batch, D , 3, N]
        j_ = self.vnn_lin_2_(input)  # [batch, D, 3, N]
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, N, D, 3, 2]

        x = self.linear(v, j_6d).permute(0, 2, 3, 1)
        # LeakyReLU

        x_out = self.leaky_relu(x)
        x_out = x_out.contiguous()
        #y = torch.cat((v.permute(0, 2, 3, 1), x_out), dim=1)
        return x_out


class Affine_Linear_9D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine_Linear_9D, self).__init__()
        self.A = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.B = torch.nn.Parameter(torch.randn((out_channels, in_channels)))
        self.C = torch.nn.Parameter(torch.randn((out_channels, in_channels)))

    def forward(self, X, J, S):
        '''
        X: point features of shape [..., N, D, 3]
        J: directions of shape [..., N, D, 3, 2] (same shape as X)
        S: [..., N, D, 3, 3 ] scaling
        '''
        # 1. 基于向量特征J计算一组正交特征
        R = bgs_extended(J)  # b x c x e x 3 x 3
        R = torch.einsum('bcefi,bceik->bcefk', R, S)
        RT = torch.transpose(R, -1, -2)  # b x c x e x 3 x 3

        # 2. 基于正交特征投影（旋转）X点特征 ， 下面分别是在局部坐标系下的3D点表示。
        RT0X = get_rtx(0, RT, X)  # b x c x e
        RT1X = get_rtx(1, RT, X)  # b x c x e
        RT2X = get_rtx(2, RT, X)  # b x c x e

        # 3.构造类复数的线性组合。
        a_term = get_rrtx(0, R, RT0X) + get_rrtx(1, R, RT1X)  # b x c x e x 3
        b_term = get_rrtx(1, R, RT0X) - get_rrtx(0, R, RT1X)  # b x c x e x 3
        c_term = get_rrtx(2, R, RT2X)  # b x c x e x 3


        summed_a_term = torch.einsum('fe,bcei->bcfi', self.A, a_term)  # b x c x e' x 3
        summed_b_term = torch.einsum('fe,bcei->bcfi', self.B, b_term)  # b x c x e' x 3
        summed_c_term = torch.einsum('fe,bcei->bcfi', self.C, c_term)  # b x c x e' x 3
        Y = summed_a_term + summed_b_term + summed_c_term  # b x c x e' x 3

        return Y

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out




class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0