import torch
import torch.nn.functional as F
from torch import nn


class SCModule(nn.Module):
    def __init__(self, in_channels):
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

class DaconvAtten3(nn.Module):
    def     __init__(self, dim):
        super().__init__()
        self.conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # 应用padding使输入输出shape保持一致
        self.da_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        # self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_m = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # x: (B,C,H,W)


        attn1 = self.conv_3(x)  # 应用第一个卷积层: (B,C,H,W)--> (B,C,H,W)
        attn2 = self.da_conv(x)  # 应用第二个卷积层: (B,C,H,W)--> (B,C,H,W)

        attn = attn1 + attn2

        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # 应用全局平均池化: (B,C,H,W)-->(B,1,H,W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # 应用全局最大池化: (B,C,H,W)-->(B,1,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均池化和最大池化特征进行拼接: (B,2,H,W)
        sig = self.conv_squeeze(agg).sigmoid()  # 将2个通道映射为N个通道, N是尺度的个数, 并通过sigmoid函数得到每个尺度对应的权重表示: (B,N,H,W), 在这里N==2
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(
            1)  # 对多个尺度的信息进行加权求和: (B,C/2,H,W)
        attn = self.conv_m(attn)  # 将通道恢复为原通道数量: (B,C/2,H,W)-->(B,C,H,W)
        return x * attn  # 最后与输入特征执行逐元素乘法


class DaconvAtten5(nn.Module):
    def     __init__(self, dim):
        super().__init__()
        self.conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # 应用padding使输入输出shape保持一致
        self.da_conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, dilation=2)
        # self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        # self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_m = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # x: (B,C,H,W)


        attn1 = self.conv_5(x)  # 应用第一个卷积层: (B,C,H,W)--> (B,C,H,W)
        attn2 = self.da_conv(x)  # 应用第二个卷积层: (B,C,H,W)--> (B,C,H,W)

        attn = attn1 + attn2

        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # 应用全局平均池化: (B,C,H,W)-->(B,1,H,W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # 应用全局最大池化: (B,C,H,W)-->(B,1,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均池化和最大池化特征进行拼接: (B,2,H,W)
        sig = self.conv_squeeze(agg).sigmoid()  # 将2个通道映射为N个通道, N是尺度的个数, 并通过sigmoid函数得到每个尺度对应的权重表示: (B,N,H,W), 在这里N==2
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(
            1)  # 对多个尺度的信息进行加权求和: (B,C/2,H,W)
        attn = self.conv_m(attn)  # 将通道恢复为原通道数量: (B,C/2,H,W)-->(B,C,H,W)
        return x * attn  # 最后与输入特征执行逐元素乘法


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()

        ### 坐标注意力模块  ###
        group_x = x.reshape(b * self.groups, -1, h, w)  # 在通道方向上将输入分为G组: (B,C,H,W)-->(B*G,C/G,H,W)
        x_h = self.pool_h(group_x) # 使用全局平均池化压缩水平空间方向: (B*G,C/G,H,W)-->(B*G,C/G,H,1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # 使用全局平均池化压缩垂直空间方向: (B*G,C/G,H,W)-->(B*G,C/G,1,W)-->(B*G,C/G,W,1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# 将水平方向和垂直方向的全局特征进行拼接: (B*G,C/G,H+W,1), 然后通过1×1Conv进行变换,来编码空间水平和垂直方向上的特征
        x_h, x_w = torch.split(hw, [h, w], dim=2) # 沿着空间方向将其分割为两个矩阵表示: x_h:(B*G,C/G,H,1); x_w:(B*G,C/G,W,1)

        ### 1×1分支和3×3分支的输出表示  ###
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) # 通过水平方向权重和垂直方向权重调整输入,得到1×1分支的输出: (B*G,C/G,H,W) * (B*G,C/G,H,1) * (B*G,C/G,1,W)=(B*G,C/G,H,W)
        x2 = self.conv3x3(group_x) # 通过3×3卷积提取局部上下文信息: (B*G,C/G,H,W)-->(B*G,C/G,H,W)

        ### 跨空间学习 ###
        ## 1×1分支生成通道描述符来调整3×3分支的输出
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对1×1分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将3×3分支的输出进行变换,以便与1×1分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y1 = torch.matmul(x11, x12) # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        ## 3×3分支生成通道描述符来调整1×1分支的输出
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对3×3分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  # 将1×1分支的输出进行变换,以便与3×3分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y2 = torch.matmul(x21, x22)  # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        # 聚合两种尺度的空间位置信息, 通过sigmoid生成空间权重, 从而再次调整输入表示
        weights = (y1+y2).reshape(b * self.groups, 1, h, w)  # 将两种尺度下的空间位置信息进行聚合: (B*G,1,H*W)-->reshape-->(B*G,1,H,W)
        weights_ =  weights.sigmoid() # 通过sigmoid生成权重表示: (B*G,1,H,W)
        out = (group_x * weights_).reshape(b, c, h, w) # 通过空间权重再次校准输入: (B*G,C/G,H,W)*(B*G,1,H,W)==(B*G,C/G,H,W)-->reshape(B,C,H,W)
        return out

class SEAttention(nn.Module):

    def __init__(self, channel=64, reduction=2):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(channel,channel // reduction,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        out = self.conv(out)
        return out

class SEAttention64(nn.Module):

    def __init__(self, channel=64, reduction=2):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        out = self.conv(out)
        return out


class SSCAConv(nn.Module):
    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(SSCAConv, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        self.attention = nn.Sequential(
            nn.Conv2d(in_planes, in_planes * (kernel_size ** 2), kernel_size, stride, padding, dilation,
                      groups=in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes * (kernel_size ** 2), in_planes * (kernel_size ** 2), 1, 1, 0, groups=in_planes),
            nn.Tanh()
        )  # b,1,H,W 全通道像素级通道注意力

        if use_bias == True:
            conv1 = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
            self.bias = conv1.bias

    def forward(self, x):
        (b, n, H, W) = x.shape
        k = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)

        atw = self.attention(x).reshape(b, n, k * k, n_H, n_W)  # b,n,k*k,n_H*n_W
        unf_x = F.unfold(x, kernel_size=k, dilation=self.dilation, padding=self.padding, stride=self.stride).reshape(b,
                                                                                                                     n,
                                                                                                                     k * k,
                                                                                                                     n_H,
                                                                                                                     n_W)  # b,n*k*k,n_H*n_W
        unf_y = unf_x * atw  # b,n,k*k,n_H,n_W
        y = torch.sum(unf_y, dim=2, keepdim=False)  # b,n,n_H,n_W

        if self.use_bias == True:
            y = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(y) + y

        return y

class SENet(nn.Module):

    def __init__(self, channel=64, reduction=2):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        outchannels = 32

        self.conv = nn.Sequential(
            SSCAConv(channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            SSCAConv(channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.conv(x) + x


class FIAFN(nn.Module):
    def __init__(self, cons_dim=32):
        super(FIAFN, self).__init__()
        dim_ms = 8
        dim_pan = 1

        self.conv_ms = nn.Conv2d(dim_ms, cons_dim, 3, 1, 1)
        self.conv_pan = nn.Conv2d(dim_pan, cons_dim, 3, 1, 1)
        self.conv_1_1 = nn.Conv2d(cons_dim,64, kernel_size=1, stride=1, padding=0)
        self.conv_64_32 = nn.Conv2d(2*cons_dim,cons_dim,kernel_size=1,stride=1,padding=0)
        self.sc = SCModule(cons_dim)
        self.dalsk3 = DaconvAtten3(cons_dim)
        self.dalsk5 = DaconvAtten5(2*cons_dim)
        self.ematten_pan = EMA(2*cons_dim)
        self.ematten_pan_1 = EMA(2*cons_dim)
        self.sea_atten1 = SEAttention64()
        self.sea_atten2 = SEAttention()

        self.res = Residual_Block(2 * cons_dim)
        self.senet = SENet(channel=2 * cons_dim)
        self.out = nn.Conv2d(2 * cons_dim, dim_ms, 1)

    def forward(self, pan, ms):
        x_ms = ms
        x_pan = pan

        nb, c, h, w = x_pan.shape
        X_MS = F.interpolate(x_ms, size=(h, w), mode='bicubic', align_corners=True)

        xms_t = self.conv_ms(X_MS)
        xpan_t = self.conv_pan(x_pan)

        xpan = self.sc(xpan_t, xms_t)
        xpan = self.conv_1_1(xpan) #调整通道数为64
        xpan_em = self.ematten_pan(xpan)
        xms = self.dalsk3(xms_t)
        xms = self.conv_1_1(xms) # #调整通道数为64
        ms_pan_1 = xpan_em + xms
        ms_pan_1 = self.sea_atten1(ms_pan_1)

        xms_1 = self.dalsk5(xms)
        xpan_em_1 = self.ematten_pan_1(ms_pan_1)
        ms_pan_2 = xpan_em_1 + xms_1
        ms_pan_32 = self.sea_atten2(ms_pan_2)
        xms_2 = self.conv_64_32(xms_1)

        out = torch.cat((xms_2, ms_pan_32), 1)
        out = self.res(out)
        out = self.senet(out)

        pr = self.out(out) + X_MS
        return pr

    # def loss(self, rec, databatch):
    #     # _, _, h, w = rec.shape
    #     gt = databatch['GT']
    #     com_loss = nn.L1Loss()
    #     return com_loss(rec, gt)

if __name__ == '__main__':
    xms = torch.randn(1, 8, 64, 64)
    pan = torch.randn(1, 1, 256, 256)
    Model = FIAFN()
    # summary(Model, input_size=[(pan.shape),(xms.shape)], device='cpu')
    x = Model(pan, xms)
    print(x.shape)
    # for nam, param in Model.named_modules():
    #     print(f"Layer:{nam} | Size:{param.size()}")
    total_params = sum(p.numel() for p in Model.parameters())
    trainable_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')