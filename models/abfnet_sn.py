import torch
import torch.nn.functional as F
from torch import nn


class SpectrallyAdaptiveDegradation(nn.Module):
    def __init__(self, num_bands, m=1, init_weights=None):
        """
        SNet 实现
        参数:
            num_bands (int): 多光谱波段数 (B)
            m (int): 分组数，将波段分为m组 (默认2)
            init_weights (list): 根据SRF计算的初始权重 (B维列表)
        """
        super().__init__()
        self.num_bands = num_bands
        self.m = m

        # 验证参数合理性
        assert num_bands % m == 0, "num_bands必须能被m整除"
        groups = m
        in_channels_per_group = num_bands // m

        # P-P Conv层 (分组卷积)
        self.pixel_wise_conv = nn.Conv2d(
            in_channels=num_bands,
            out_channels=1,
            kernel_size=(1, 1),
            groups=groups,  # 关键分组设置
            bias=False
        )

        # 初始化卷积核权重
        if init_weights is not None:
            assert len(init_weights) == num_bands
            # 转换为分组形式 (m组, 每组 in_channels_per_group 通道)
            grouped_weights = torch.tensor(init_weights).view(m, in_channels_per_group)
            grouped_weights = grouped_weights.unsqueeze(-1).unsqueeze(-1)  # [m, C, 1, 1]
            self.pixel_wise_conv.weight = nn.Parameter(grouped_weights)

        # 全连接层部分
        self.fc = nn.Sequential(
            nn.Linear(1, 16),  # 压缩特征
            nn.ReLU(),
            nn.Linear(16, 64),  # 中间层
            nn.ReLU(),
            nn.Linear(64, 1)  # 恢复特征
        )

    def forward(self, F_n):
        """
        输入:
            F_n (Tensor): HRMS图像 [B, C, H, W]
        输出:
            P_hat (Tensor): 伪PAN图像 [B, 1, H, W]
        """
        B, C, H, W = F_n.shape

        # 步骤1: Permute + Reshape -> [m, (C/m), H*W]
        D1 = F_n.permute(0, 2, 3, 1)  # [B, H, W, C]
        D1 = D1.reshape(B, H * W, C)  # [B, H*W, C]
        D1 = D1.permute(0, 2, 1)  # [B, C, H*W]
        D1 = D1.view(B, self.m, C // self.m, H * W)  # [B, m, C/m, H*W]

        # 步骤2: Pixel-by-Pixel分组卷积
        # 输入形状适应: [B, C, H, W]
        D2 = self.pixel_wise_conv(F_n)  # [B, 1, H, W]
        D2 = D2.view(B, 1, H * W)  # [B, 1, H*W]

        # 步骤3: 全连接层处理每个像素
        # 扩展通道维度
        D2 = D2.permute(0, 2, 1)  # [B, H*W, 1]
        processed = self.fc(D2)  # [B, H*W, 1]
        processed = processed.permute(0, 2, 1)  # [B, 1, H*W]

        # 步骤4: Reshape回图像尺寸
        P_hat = processed.view(B, 1, H, W)

        return P_hat


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


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
        return pr


class SSPC(nn.Module):
    def __init__(self, dim=32, band=8):
        super(SSPC, self).__init__()
        self.band = band
        self.snet = SpectrallyAdaptiveDegradation(num_bands=band, m=1, init_weights=None)
        self.abf = ABFNet(dim, band)

    def forward(self, HRPAN, HRMS, LRMS):
        pseudopan = self.snet(HRMS)
        Pres = F.relu(HRPAN - pseudopan)

        nb, c, h, w = LRMS.shape
        fnr = F.interpolate(HRMS, size=(h, w), mode='bicubic')

        pr = self.abf(fnr, Pres)

        return pr + HRMS






# class ABFSNet(nn.Module):
#     def __init__(self, dim=32, band=8):
#         super(ABFSNet, self).__init__()
#
#         self.abf = ABFNet(dim, band)
#         self.sspc1 = SSPC(dim, band)
#         self.sspc2 = SSPC(dim, band)
#         self.sspc3 = SSPC(dim, band)
#
#     def forward(self, LRMS, PAN):
#         f = self.abf(LRMS, PAN)
#         f1 = self.sspc1(PAN, f, LRMS)
#         f2 = self.sspc2(PAN, f1, LRMS)
#         f3 = self.sspc3(PAN, f2, LRMS)
#         return f3, f


class ABFSNet(nn.Module):
    def __init__(self, dim=32, band=8):
        super(ABFSNet, self).__init__()

        self.abf = ABFNet(dim, band)
        self.sspc1 = SSPC(dim, band)
        self.sspc2 = SSPC(dim, band)
        self.sspc3 = SSPC(dim, band)
        self.nihe = nn.Conv2d(band, 1, kernel_size=1)

    def forward(self, LRMS, PAN):
        f = self.abf(LRMS, PAN)

        if self.training:
            # 训练阶段：执行所有SSPC模块
            f1 = self.sspc1(PAN, f, LRMS)
            f2 = self.sspc2(PAN, f1, LRMS)
            f3 = self.sspc3(PAN, f2, LRMS)
            nihe = self.nihe(f3)
            return f3, f, nihe
        else:
            # 验证/测试阶段：仅返回ABFNet的结果
            # 保持返回结构为元组，第二个元素为f，第一个元素可为空或占位符
            return f  # 或直接 return (f, )

class ABFSNet_all(nn.Module):
    def __init__(self, dim=32, band=8):
        super(ABFSNet_all, self).__init__()

        self.abf = ABFNet(dim, band)
        self.sspc1 = SSPC(dim, band)
        self.sspc2 = SSPC(dim, band)
        self.sspc3 = SSPC(dim, band)
        self.nihe = nn.Conv2d(band, 1, kernel_size=1)

    def forward(self, LRMS, PAN):
        f = self.abf(LRMS, PAN)
        f1 = self.sspc1(PAN, f, LRMS)
        f2 = self.sspc2(PAN, f1, LRMS)
        f3 = self.sspc3(PAN, f2, LRMS)
        nihe = self.nihe(f3)
        return f3, f, nihe



if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net = ABFSNet()
    lms = torch.randn(1, 8, 16, 16)
    pan = torch.randn(1, 1, 64, 64)
    # 验证输出一致性
    net.train() # 测试模式输出
    f3, f = net(lms, pan)  # 直接调用子模块

    net.eval()
    with torch.no_grad():
        abf_output = net.abf(lms, pan)
    print(f3, f)
    print(abf_output)
    # assert torch.allclose(test_output, abf_output), "输出不一致！"
    # out = net(lms, pan)
    flops = FlopCountAnalysis(net, (lms, pan))
    print(flop_count_table(flops))