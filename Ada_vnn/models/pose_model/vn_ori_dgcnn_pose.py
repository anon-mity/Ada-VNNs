import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature



class VN_Ori_DGCNN(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_DGCNN, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(64 // 3 + 64 // 3 + 64 // 3, 512 // 3)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(64 // 3 + 64 // 3 + 64 // 3, 512 // 3)
        self.complex_lin_1 = ComplexLinearAndLeakyReLU(512 // 3, 512 // 3)

        self.vnn_lin_3 = VNSimpleLinear(2 * (512 // 3), 512 // 3)
        self.vnn_lin_4 = VNSimpleLinear(2 * (512 // 3), 512 // 3)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(512 // 3, 512 // 3)
        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(2 * (512 // 3), self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)
        l = x[:, 0, 0:16].reshape(batch_size, 1, 16)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        # >>>>>>>>> orientation-Aware Mechanism
        # first
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        x = self.vnn_lin_1(x123).permute(0, 3, 1, 2)  # (b, 1024, dim/3, 3)
        # self.vnn_lin_2 就是VNLinearLeakyReLU 少了BR ，只有L
        j = self.vnn_lin_2(x123).permute(0, 3, 1, 2)  # (b, 1024, dim/3, 3)
        # f_complexlinear: y = R(j) @ Z(A,B,C) @ R(j).T @ x
        y = self.complex_lin_1(x, j, device)  # B x dim/3 x 3 x 1024
        comb = torch.cat((x.permute(0, 2, 3, 1), y), dim=1)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        j_2 = self.vnn_lin_4(comb).permute(0, 3, 1, 2)
        y_2 = self.complex_lin_2(x, j_2, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)

        x = self.conv6(comb) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]
        x1 = None


        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# VN_Ori_Globa有提升的  3D，使用3D生成旋转基，被证明性能不好，可能因为描述的旋转不明确。
class VN_Ori_Globa(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_1 = ComplexLinearAndLeakyReLU(self.feat_dim, self.feat_dim)

        self.vnn_lin_3 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.complex_lin_2 = ComplexLinearAndLeakyReLU(self.feat_dim, self.feat_dim)
        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)
        l = x[:, 0, 0:16].reshape(batch_size, 1, 16)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3]
        # self.vnn_lin_2 就是VNLinearLeakyReLU 少了BR ，只有L
        j = self.vnn_lin_2(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3]
        # f_complexlinear: y = R(j) @ Z(A,B,C) @ R(j).T @ x
        y = self.complex_lin_1(v, j, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        j_2 = self.vnn_lin_4(comb).permute(0, 3, 1, 2)
        y_2 = self.complex_lin_2(x, j_2, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# VN_Ori_Globa 使用6d生成旋转基，提升非常大。直接在最后的全局特征使用，局部不动
# 这里进行的测试是看仿射模块叠了几层。
# 对构造好的复数空间，使用单个A学习而不是AB两个
# 尝试a_term b_term c_term的别的线性组合。XY-Z ; X-YZ; Y-ZX
class VN_Ori_Globa6D(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa6D, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_1 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=cfg)

        self.vnn_lin_3 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_2 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=cfg)

        self.vnn_lin_5 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_6 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_6_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_3 = Affine_Geometric_Module(self.feat_dim, self.feat_dim)
        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j_ = self.vnn_lin_2_(x) # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.complex_lin_1(v, j_6d, device)  # B x dim * 2 x 3

        #comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)   # 这里v是等变的，y中LSOG第一步构造的是旋转不变的。经过第二步是旋转等变的。
        comb = v.permute(0, 2, 3, 1) + y

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        # 使用linear从先前的特征中学习6D参数。
        j_2 = self.vnn_lin_4(comb)
        j_2_ = self.vnn_lin_4_(comb)
        j_2 = j_2.permute(0, 3, 1, 2)
        j_2_ = j_2_.permute(0, 3, 1, 2)
        j_2_6d = torch.stack([j_2, j_2_], dim=-1)  # [batch,N, dim , 3, 2]
        y_2 = self.complex_lin_2(x, j_2_6d, device)

        #comb2 = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)
        comb2 = x.permute(0, 2, 3, 1) + y_2

        # third
        x = self.vnn_lin_5(comb2).permute(0, 3, 1, 2)
        j_3 = self.vnn_lin_6(comb2)
        j_3_ = self.vnn_lin_6_(comb2)
        j_3_6d = torch.stack([j_3.permute(0, 3, 1, 2), j_3_.permute(0, 3, 1, 2)], dim=-1)  # [batch, dim * 2, 3, 2]
        y_3 = self.complex_lin_3(x, j_3_6d, device)

        #comb3 = torch.cat((x.permute(0, 2, 3, 1), y_3), dim=1)
        comb3 = x.permute(0, 2, 3, 1) + y_3

        x = comb3.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


class VN_Ori_Globa6D_lRlR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa6D_lRlR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_1 = Affine_Geometric_Module_LSOG_lRlR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]

        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j_ = self.vnn_lin_2_(x) # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.complex_lin_1(v, j_6d, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)   # 这里v是等变的，y中LSOG第一步构造的是旋转不变的。经过第二步是旋转等变的。

        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


class VN_Ori_Globa6D_LRLR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa6D_LRLR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_1 = Affine_Geometric_Module_LSOG_LRLR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]

        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j_ = self.vnn_lin_2_(x) # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.complex_lin_1(v, j_6d, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)   # 这里v是等变的，y中LSOG第一步构造的是旋转不变的。经过第二步是旋转等变的。

        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]

class VN_Ori_Globa6D_LSOG_noequ_linearx1LR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa6D_LSOG_noequ_linearx1LR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.LSOG = Affine_Geometric_Module_LSOG(self.feat_dim, self.feat_dim, args=cfg)

        self.vnn_lin_3 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin = Nonequ_Module_LR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # LSOG
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, N, dim * 2, 3] 这里v是等变特征

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j_ = self.vnn_lin_2_(x) # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.LSOG(v, j_6d, device)  # B x dim * 2 x 3
        #comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)   # 这里v是等变的，y中LSOG第一步构造的是旋转不变的。经过第二步是旋转等变的。
        x = self.vnn_lin_3(y).permute(0, 3, 1, 2)

        # LR
        y_2 = self.complex_lin(x, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)


        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


class VN_Ori_Globa6D_LSOG_noequ_linearx1LRLR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa6D_LSOG_noequ_linearx1LRLR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.LSOG = Affine_Geometric_Module_LSOG(self.feat_dim, self.feat_dim, args=cfg)

        self.vnn_lin_3 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin = Nonequ_Module_LRLR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # LSOG
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, N, dim * 2, 3] 这里v是等变特征

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j_ = self.vnn_lin_2_(x) # [batch, dim, 3, 1]  #线性层生成的6D向量j是等变的
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.LSOG(v, j_6d, device)  # B x dim * 2 x 3
        #comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)   # 这里v是等变的，y中LSOG第一步构造的是旋转不变的。经过第二步是旋转等变的。
        x = self.vnn_lin_3(y).permute(0, 3, 1, 2)

        # LRLR
        y_2 = self.complex_lin(x, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)

        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


class VN_Ori_Globa_noequ_linear(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linear, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_LR(self.feat_dim, self.feat_dim, args=cfg)


        self.vnn_lin_3 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.complex_lin_2 = Nonequ_Module_LR(self.feat_dim, self.feat_dim, args=cfg)

        '''self.vnn_lin_5 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_3 = Affine_Geometric_Module(self.feat_dim, self.feat_dim)'''
        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        y2 = self.complex_lin_1(x, device)  # B x dim * 2 x 3
        comb2 = torch.cat((x.permute(0, 2, 3, 1), y2), dim=1)

        x = comb2.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+l（单个W）R
class VN_Ori_Globa_noequ_linearx1lR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linearx1lR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_lR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        comb_res = v.permute(0, 2, 3, 1) + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+lRlR
class VN_Ori_Globa_noequ_linearx1lRlR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linearx1lRlR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_lRlR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, 1, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        comb_res = v.permute(0, 2, 3, 1) + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+lRlRlR
class VN_Ori_Globa_noequ_linearx1lRlRlR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linearx1lRlRlR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_lRlRlR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        comb_res = v.permute(0, 2, 3, 1) + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+（W+bias）R
class VN_DGCNN_linear_wbiasx1lR(nn.Module):
    def __init__(self, cfg):
        super(VN_DGCNN_linear_wbiasx1lR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = BiasLinearandRelu(self.feat_dim, self.feat_dim)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> 带有bias的linear
        # first
        x = x.unsqueeze(3)  # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x)  # [batch, dim * 2, 3, 1]
        y = self.complex_lin_1(v)  # [batch, dim * 2, 3, 1]
        comb = torch.cat((v, y), dim=1)
        comb_res = v + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+（W+bias）R（W+bias）R
class VN_DGCNN_linear_wbiasx1lRlR(nn.Module):
    def __init__(self, cfg):
        super(VN_DGCNN_linear_wbiasx1lRlR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = BiasLinearandRelu(self.feat_dim, self.feat_dim)
        self.complex_lin_2 = BiasLinearandRelu(self.feat_dim, self.feat_dim)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> 带有bias的linear
        # first
        x = x.unsqueeze(3)  # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x)  # [batch, dim * 2, 3, 1]
        y = self.complex_lin_1(v)  # [batch, dim * 2, 3, 1]
        y = self.complex_lin_2(y)
        comb = torch.cat((v, y), dim=1)
        comb_res = v + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+（W+bias）R（W+bias）R（W+bias）R
class VN_DGCNN_linear_wbiasx1lRlRlR(nn.Module):
    def __init__(self, cfg):
        super(VN_DGCNN_linear_wbiasx1lRlRlR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = BiasLinearandRelu(self.feat_dim, self.feat_dim)
        self.complex_lin_2 = BiasLinearandRelu(self.feat_dim, self.feat_dim)
        self.complex_lin_3 = BiasLinearandRelu(self.feat_dim, self.feat_dim)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> 带有bias的linear
        # first
        x = x.unsqueeze(3)  # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x)  # [batch, dim * 2, 3, 1]
        y = self.complex_lin_1(v)  # [batch, dim * 2, 3, 1]
        y = self.complex_lin_2(y)
        y = self.complex_lin_3(y)
        comb = torch.cat((v, y), dim=1)
        comb_res = v + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


# vndgcnn+L（三个W）R
class VN_Ori_Globa_noequ_linearx1LR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linearx1LR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_LR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        #comb_res = v.permute(0, 2, 3, 1) + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


class VN_Ori_Globa_noequ_linearx1LRLR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linearx1LRLR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_LRLR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        #comb_res = v.permute(0, 2, 3, 1) + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]

class VN_Ori_Globa_noequ_linearx1LRLRLR(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa_noequ_linearx1LRLRLR, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.complex_lin_1 = Nonequ_Module_LRLRLR(self.feat_dim, self.feat_dim, args=cfg)

        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3] 这里v是等变特征
        y = self.complex_lin_1(v, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        #comb_res = v.permute(0, 2, 3, 1) + y
        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]



class VN_Ori_Globa9D(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa9D, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        #self.vnn_lin_s1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.s1relu = nn.ReLU()
        self.complex_lin_1 = Affine_Geometric_Module_9D(self.feat_dim, self.feat_dim)

        self.vnn_lin_3 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(2 *self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        #self.vnn_lin_s2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.s2relu = nn.ReLU()
        self.complex_lin_2 = Affine_Geometric_Module_9D(self.feat_dim, self.feat_dim)

        '''self.vnn_lin_5 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_3 = Affine_Geometric_Module(self.feat_dim, self.feat_dim)'''
        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)
        l = x[:, 0, 0:16].reshape(batch_size, 1, 16)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)
        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3]
        # self.vnn_lin_2 就是VNLinearLeakyReLU 少了BR ，只有L
        j = self.vnn_lin_2(x)  # [batch, dim * 2, 3, 1]
        j_ = self.vnn_lin_2_(x) # [batch, dim * 2, 3, 1]
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        #s1 = self.vnn_lin_s1(x).permute(0, 3, 1, 2)
        s1 = nn.Parameter(torch.randn(batch_size, 1, self.feat_dim, 3)).to(device)
        s1 = self.s1relu(s1)
        s1 = torch.diag_embed(s1)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim * 2, 3, 2]
        # f_complexlinear: y = R(j) @ Z(A,B,C) @ R(j).T @ x
        y = self.complex_lin_1(v, j_6d, s1, device)  # B x dim * 2 x 3
        comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)
        j_2 = self.vnn_lin_4(comb)
        j_2_ = self.vnn_lin_4_(comb)
        #s2 = self.vnn_lin_s2(comb).permute(0, 3, 1, 2)
        s2 = nn.Parameter(torch.randn(batch_size, 1, self.feat_dim, 3)).to(device)
        s2 = self.s2relu(s2)
        s2 = torch.diag_embed(s2)
        j_2_6d = torch.stack([j_2.permute(0, 3, 1, 2), j_2_.permute(0, 3, 1, 2)], dim=-1)  # [batch, dim * 2, 3, 2]
        y_2 = self.complex_lin_2(x, j_2_6d, s2, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)

        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]


class VN_Ori_Globa6D_Res(nn.Module):
    def __init__(self, cfg):
        super(VN_Ori_Globa6D_Res, self).__init__()
        self.cfg = cfg
        self.n_knn = 20
        self.feat_dim = cfg.feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        self.vnn_lin_1 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)  # 就是VNLinear
        self.vnn_lin_2 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_2_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_1 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=cfg)

        self.vnn_lin_3 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_4 = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.vnn_lin_4_ = VNSimpleLinear(self.feat_dim, self.feat_dim)
        self.complex_lin_2 = Affine_Geometric_Module(self.feat_dim, self.feat_dim, args=cfg)

        '''self.vnn_lin_5 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6 = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.vnn_lin_6_ = VNSimpleLinear(2 * self.feat_dim, self.feat_dim)
        self.complex_lin_3 = Affine_Geometric_Module(self.feat_dim, self.feat_dim)'''
        self.VnInv = VNStdFeature(2 * self.feat_dim, dim=3, normalize_frame=False)


        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(self.feat_dim * 2)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, self.feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 1024)

    def forward(self, x):
        device = torch.device("cuda")
        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)

        batch_size = x.size(0)
        num_points = x.size(2)
        l = x[:, 0, 0:16].reshape(batch_size, 1, 16)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123) # b, dim, 3, 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, dim * 2, 3]


        # >>>>>>>>> orientation-Aware Mechanism
        # first
        x = x.unsqueeze(3) # [batch, dim * 2, 3, 1]
        # self.vnn_lin_1 就是VNLinearLeakyReLU 少了BR ，只有L
        v = self.vnn_lin_1(x).permute(0, 3, 1, 2)  # [batch, dim * 2, 3]

        # 使用linear从先前的特征中学习6D参数。
        j = self.vnn_lin_2(x)  # [batch, dim, 3, 1]
        j_ = self.vnn_lin_2_(x) # [batch, dim, 3, 1]
        j = j.permute(0, 3, 1, 2)
        j_ = j_.permute(0, 3, 1, 2)
        j_6d = torch.stack([j, j_], dim=-1)  # [batch, dim, 3, 2]
        y = self.complex_lin_1(v, j_6d, device)  # B x dim * 2 x 3

        #comb = torch.cat((v.permute(0, 2, 3, 1), y), dim=1)
        v = v.permute(0, 2, 3, 1)
        comb = v + y

        # second
        x = self.vnn_lin_3(comb).permute(0, 3, 1, 2)

        # 使用linear从先前的特征中学习6D参数。
        j_2 = self.vnn_lin_4(comb)
        j_2_ = self.vnn_lin_4_(comb)
        j_2 = j_2.permute(0, 3, 1, 2)
        j_2_ = j_2_.permute(0, 3, 1, 2)
        j_2_6d = torch.stack([j_2, j_2_], dim=-1)  # [batch,N, dim , 3, 2]
        y_2 = self.complex_lin_2(x, j_2_6d, device)

        #comb = torch.cat((x.permute(0, 2, 3, 1), y_2), dim=1)
        comb = x.permute(0, 2, 3, 1) + y_2
        # third
        '''x = self.vnn_lin_5(comb).permute(0, 3, 1, 2)
        j_3 = self.vnn_lin_6(comb)
        j_3_ = self.vnn_lin_6_(comb)
        j_3_6d = torch.stack([j_3.permute(0, 3, 1, 2), j_3_.permute(0, 3, 1, 2)], dim=-1)  # [batch, dim * 2, 3, 2]
        y_3 = self.complex_lin_3(x, j_3_6d, device)
        comb = torch.cat((x.permute(0, 2, 3, 1), y_3), dim=1)'''

        x = comb.squeeze(3)

        x1 = None
        if self.cfg.disentangle:
            x1, z0 = self.VnInv(x)
            x1 = self.linear0(x1)  # [batch, 2*dim, 1024]

        # x = x.view(batch_size, -1)  # b, dim*2 *3
        return x, x1   # [b, dim*2, 3]   [batch, 2*dim, 1024]