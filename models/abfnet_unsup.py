import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
# from .EUCB import EUCB
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)



def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # 构造激活函数层，根据传入的名称返回相应激活模块
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('未实现的激活函数层 [%s]' % act)
    return layer

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int=16, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        # 若卷积核尺寸为1，则强制设置组数为1以保持计算一致性
        if kernel_size == 1:
            groups = 1
        # 构建处理全局信息的卷积及归一化模块
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 构建处理局部信息的卷积及归一化模块
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 定义计算注意力权重的模块，由1x1卷积、批归一化和Sigmoid激活构成
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # 利用前面定义的函数构造激活层
        self.activation = act_layer(activation, inplace=True)
        # ai缝合大王

    def forward(self, g, x):
        # 对全局输入进行卷积处理
        g1 = self.W_g(g)
        # 对局部输入进行卷积处理
        x1 = self.W_x(x)
        # 将全局和局部特征相加后，先通过激活函数再计算注意力权重
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        # 返回局部特征与计算出的注意力权重逐元素相乘的结果
        return x * psi

class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        """
            小卷积核策略：采用尺寸较小的卷积核（如3x3）可以更精细地捕获图像中的纹理与细微结构，
            这些细节构成了高频信息的核心部分。
        """
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):

        """
            密集与残差连接机制：通过在各层之间构建密集连接，网络能够融合初级和高级特征，
            从而增强对图像细节的识别；同时，残差连接使网络直接学习输入与输出间的差分，
            专注于那些需要强化的细微部分。
        """

        # 输入张量形状示例：[1, 32, 64, 64]
        x1 = self.conv1(x)      # 输出尺寸仍为 [1, 32, 64, 64]
        x1 = self.gelu(x1 + x)  # 结合输入形成残差，保持尺寸不变

        x2 = self.conv2(x1)     # 经过第二个卷积层后尺寸：[1, 32, 64, 64]
        x2 = self.gelu(x2 + x1 + x)  # 密集连接，将前几层信息叠加

        x3 = self.conv3(x2)             # 第三层卷积输出：[1, 32, 64, 64]
        x3 = self.gelu(x3 + x2 + x1 + x) # 叠加前面所有层的特征

        x4 = self.conv4(x3)             # 第四层卷积处理后尺寸不变
        x4 = self.gelu(x4 + x3 + x2 + x1 + x)

        x5 = self.conv5(x4)                 # 第五层卷积，输出尺寸依然为 [1, 32, 64, 64]
        x5 = self.gelu(x5 + x4 + x3 + x2 + x1 + x)

        x6 = self.conv6(x5)                 # 最后一层卷积
        x6 = self.gelu(x6 + x5 + x4 + x3 + x2 + x1 + x)  # 最终融合所有层的信息
        # ai缝合大王

        return x6

class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()  # 初始化父类模块
        self.num_heads = num_heads  # 指定多头注意力中的头数
        # self.fea = ConvBlock(8, dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 初始化温度参数，用于缩放点积

        # 定义用于生成查询、键和值的 1x1x1 卷积（3D卷积）
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # 对 qkv 结果应用 3D 深度可分离卷积以捕捉局部空间关系
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1,
                                     groups=dim * 3, bias=bias)
        # 输出投影层，将多头注意力结果映射回原始通道数
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

        # 全连接层用于调整通道数，将通道数从 3*num_heads 调整到 9
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        # 深度卷积层用于提取局部细节特征，其分组数使得每个组只处理部分通道
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
                                  bias=True, groups=dim // self.num_heads, padding=1)
        # self.out = nn.Conv2d(dim, 8, 1)
        # self.nihe = nn.Conv2d(8, 1, kernel_size=1)
        # ai缝合大王

    def forward(self, x):
        # x = self.fea(x)
        b, c, h, w = x.shape  # x 的形状为 (B, C, H, W)
        x = x.unsqueeze(2)  # 在第三个维度插入单一深度维度，形状变为 (B, C, 1, H, W)
        qkv = self.qkv_dwconv(self.qkv(x))  # 首先计算 qkv，然后通过深度卷积进一步提取特征
        qkv = qkv.squeeze(2)  # 移除深度维度，恢复至 (B, C, H, W)

        # ========= 局部特征分支 =========
        # 将张量转换为 (B, H, W, C) 以便后续全连接层处理
        f_conv = qkv.permute(0, 2, 3, 1)
        # 重塑张量以组织多头特征，通道维度分为 3*num_heads 和剩余特征维度
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))  # 应用全连接层调整特征通道
        f_all = f_all.squeeze(2)
        # 调整维度以适应局部卷积层的输入
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # 利用深度卷积提取局部细节信息
        out_conv = out_conv.squeeze(2)

        # ========= 全局特征分支 =========
        # 将 qkv 分成查询、键和值三个部分
        q, k, v = qkv.chunk(3, dim=1)
        # 重排张量形状，转换为 (B, head, c, (H*W)) 以便于多头注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 对查询和键进行 L2 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算缩放后的点积注意力，并归一化
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 用注意力权重加权值向量
        out = (attn @ v)
        # 将多头输出重排回原始空间尺寸 (B, C, H, W)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)  # 进行输出投影
        out = out.squeeze(2)

        # 将局部与全局分支结果相加，融合多尺度特征
        output = out + out_conv
        # output = self.out(output)
        # nihe = self.nihe(output)
        return output

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class SRCModule(nn.Module):
    def __init__(self, channels, kernel=-1):
        super(SRCModule, self).__init__()

        mid_channels = channels // 4
        self.reduction_pan = nn.Conv2d(channels, mid_channels*mid_channels, 1)
        self.reduction_ms = nn.Conv2d(channels, mid_channels, 1)


        self.expand_ms = nn.Conv2d(mid_channels, channels, 1)
        self.mid_channels = mid_channels

    def forward(self, xpan, xms):
        """

        Args:
            xpan: bn, dim, h, w
            xms: bn,  dim
        Returns:

        """
        bn, c, h, w = xpan.shape
        kernel = self.reduction_pan(xpan).view(bn, self.mid_channels, self.mid_channels, h, w) # b c h w
        xms = self.reduction_ms(xms)

        d = torch.rsqrt((kernel ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-10)
        kernel = kernel * d

        out = torch.einsum('n c h w, n c d h w -> n d h w', xms, kernel) #+ xms

        out = self.expand_ms(out)

        # 1 nb*channels, h, w
        return out


class SCModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCModule, self).__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU()
        )
        self.a_head = nn.Conv2d(in_channels, in_channels, 1)
        self.b_head = nn.Conv2d(in_channels, in_channels, 1)

        self.norm = nn.InstanceNorm2d(in_channels)

    def forward(self, xpan, xms):
        """

        Args:
            xpan: bn, dim, h, w
            xms: bn, ns, dim
        Returns:

        """
        nb, c, h, w = xpan.shape
        xpan = self.norm(xpan)
        out = self.neck(xms)
        gamma = self.a_head(out)
        bias = self.b_head(out)
        out = xpan * gamma + bias

        return out

class SpatialDiscriminator(nn.Module):
    def __init__(self):
        super(SpatialDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # 第一层
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 第二层
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 第三层
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 第四层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 第五层
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 输出层
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 全局平均池化替代原来的卷积
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 8, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, x):
        # 如果输入是多通道的，取平均值
        # if x.size(1) > 1:
        #     x = x.mean(dim=1, keepdim=True)
        return self.model(x)



class ABFNet(nn.Module):
    def __init__(self, dim=32, band=8):
        super(ABFNet, self).__init__()

        dims = [dim] * 4

        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.SpectralModule = nn.ModuleList()
        self.SpatialModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
            self.SpectralModule.append(SCModule(dim, dim))
            self.SpatialModule.append(SRCModule(dim))

        self.out = nn.Conv2d(dims[-1]*2, band, 1)
        self.nihe = nn.Conv2d(band,1, kernel_size=1)



    def forward(self, X_MS, X_PAN):

        nb, c, h, w = X_PAN.shape
        X_MS = F.interpolate(X_MS, size=(h, w), mode='bicubic')

        xms = X_MS
        xpan = X_PAN

        for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule, self.SpatialModule):
            xms_t = ms_cb(xms)
            xpan_t = pan_cb(xpan)
            xpan = sc_module(xpan_t, xms_t)
            xms = src_module(xpan, xms_t)

        out = torch.cat((xms, xpan), 1)

        pr = self.out(out) + X_MS
        I_nihe = self.nihe(pr)
        return pr, I_nihe


# class ABFNet_eucb(nn.Module):
#     def __init__(self, dim=32, band=8):
#         super(ABFNet_eucb, self).__init__()
#
#         dims = [dim] * 4
#
#         self.PanModule = nn.ModuleList()
#
#         self.MSModule = nn.ModuleList()
#
#         self.SpectralModule = nn.ModuleList()
#         self.SpatialModule = nn.ModuleList()
#         self.upms = EUCB(band, band)
#         for i, dim in enumerate(dims):
#             if i == 0:
#                 self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
#                 self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
#             else:
#                 self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
#                 self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
#             self.SpectralModule.append(SCModule(dim, dim))
#             self.SpatialModule.append(SRCModule(dim))
#
#         self.out = nn.Conv2d(dims[-1]*2, band, 1)
#         self.nihe = nn.Conv2d(band,1, kernel_size=1)
#
#
#
#     def forward(self, X_MS, X_PAN):
#
#         nb, c, h, w = X_PAN.shape
#         X_MS = self.upms(X_MS)
#
#         xms = X_MS
#         xpan = X_PAN
#
#         for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule, self.SpatialModule):
#             xms_t = ms_cb(xms)
#             xpan_t = pan_cb(xpan)
#             xpan = sc_module(xpan_t, xms_t)
#             xms = src_module(xpan, xms_t)
#
#         out = torch.cat((xms, xpan), 1)
#
#         pr = self.out(out) + X_MS
#         I_nihe = self.nihe(pr)
#         return pr, I_nihe




class ABFNet_high(nn.Module):
    def __init__(self, dim=32, band=4):
        super(ABFNet_high, self).__init__()

        dims = [dim] * 4

        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.SpectralModule = nn.ModuleList()
        self.SpatialModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
                self.PanModule.append(Dense(dim))
                self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
            self.SpectralModule.append(SCModule(dim, dim))
            self.SpatialModule.append(SRCModule(dim))
        # self.catblock = TFF(dim, 8)
        self.nihe = nn.Conv2d(band, 1, kernel_size=1)
        self.out = nn.Conv2d(dims[-1]*2, band, 1)

    def forward(self, X_MS, X_PAN):

        nb, c, h, w = X_PAN.shape
        X_MS = F.interpolate(X_MS, size=(h, w), mode='bicubic')

        xms = X_MS
        xpan = X_PAN

        for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule,
                                                        self.SpatialModule):
            xms_t = ms_cb(xms)
            xpan_t = pan_cb(xpan)
            xpan = sc_module(xpan_t, xms_t)
            xms = src_module(xpan, xms_t)

        out = torch.cat((xms, xpan), 1)

        pr = self.out(out) + X_MS
        I_nihe = self.nihe(pr)
        return pr, I_nihe


class ABFNet_lgag(nn.Module):
    def __init__(self, dim=32, band=8):
        super(ABFNet_lgag, self).__init__()

        dims = [dim] * 4

        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.SpectralModule = nn.ModuleList()
        self.SpatialModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
            self.SpectralModule.append(SCModule(dim, dim))
            self.SpatialModule.append(SRCModule(dim))

        self.ronghe = LGAG(dim, dim)
        self.out = nn.Conv2d(dim, band, 1)
        # self.nihe = nn.Conv2d(band, 1, kernel_size=1)



    def forward(self, X_MS, X_PAN):

        nb, c, h, w = X_PAN.shape
        X_MS = F.interpolate(X_MS, size=(h, w), mode='bicubic')

        xms = X_MS
        xpan = X_PAN

        for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule, self.SpatialModule):
            xms_t = ms_cb(xms)
            xpan_t = pan_cb(xpan)
            xpan = sc_module(xpan_t, xms_t)
            xms = src_module(xpan, xms_t)

        out = self.ronghe(xms, xpan)

        pr = self.out(out) + X_MS
        # I_nihe = self.nihe(pr)
        return pr

class ABFNet_lgag_unsp(nn.Module):
    def __init__(self, dim=32, band=8):
        super(ABFNet_lgag, self).__init__()

        dims = [dim] * 4

        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.SpectralModule = nn.ModuleList()
        self.SpatialModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
            self.SpectralModule.append(SCModule(dim, dim))
            self.SpatialModule.append(SRCModule(dim))

        self.ronghe = LGAG(dim, dim)
        self.out = nn.Conv2d(dim, band, 1)
        self.nihe = nn.Conv2d(band, 1, kernel_size=1)



    def forward(self, X_MS, X_PAN):

        nb, c, h, w = X_PAN.shape
        X_MS = F.interpolate(X_MS, size=(h, w), mode='bicubic')

        xms = X_MS
        xpan = X_PAN

        for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule, self.SpatialModule):
            xms_t = ms_cb(xms)
            xpan_t = pan_cb(xpan)
            xpan = sc_module(xpan_t, xms_t)
            xms = src_module(xpan, xms_t)

        out = self.ronghe(xms, xpan)

        pr = self.out(out) + X_MS
        I_nihe = self.nihe(pr)
        return pr,  I_nihe



class ABFNet_all(nn.Module):
    def __init__(self, dim=32, band=8):
        super(ABFNet_all, self).__init__()

        dims = [dim] * 4
        # self.upms = EUCB(in_channels=8, out_channels=8)
        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.SpectralModule = nn.ModuleList()
        self.SpatialModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
                self.PanModule.append(Dense(dim))
                self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
            self.SpectralModule.append(SCModule(dim, dim))
            self.SpatialModule.append(SRCModule(dim))

        self.ronghe = LGAG(dim, dim)
        self.cafm = CAFM(dim=32)
        self.out = nn.Conv2d(dim, band, 1)
        self.nihe = nn.Conv2d(band, 1, kernel_size=1)



    def forward(self, X_MS, X_PAN):

        nb, c, h, w = X_PAN.shape
        X_MS = F.interpolate(X_MS, size=(h, w), mode='bicubic')

        xms = X_MS
        xpan = X_PAN

        for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule, self.SpatialModule):
            xms_t = ms_cb(xms)
            xpan_t = pan_cb(xpan)
            xpan = sc_module(xpan_t, xms_t)
            xms = src_module(xpan, xms_t)

        out = self.ronghe(xms, xpan) #(1,32,256,256)
        out = self.cafm(out)
        # out = self.out(out)
        pr = self.out(out) + X_MS
        I_nihe = self.nihe(pr)
        return pr, I_nihe


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net = ABFNet_all()
    lms = torch.randn(1, 8, 64, 64)
    pan = torch.randn(1, 1, 256, 256)
    flops = FlopCountAnalysis(net, (lms, pan))
    print(flop_count_table(flops))