import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from typing import Optional, Sequence


# ==================== LayerNorm 定义 ====================
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'WithBias':
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        return x



# ==================== SMFA 模块 ====================
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = 8
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


# ==================== InceptionBottleneck 模块 ====================
def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Calculate the padding size based on kernel size and dilation."""
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2
    return padding


def make_divisible(value: int, divisor: int = 8) -> int:
    """Make a value divisible by a certain divisor."""
    return int((value + divisor // 2) // divisor * divisor)


class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                dilation=dilation, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1),
                                  eps=norm_cfg.get('eps', 1e-5))
        elif norm_cfg['type'] == 'InstanceNorm2d':
            return nn.InstanceNorm2d(num_features)
        else:
            raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        elif act_cfg['type'] == 'LeakyReLU':
            return nn.LeakyReLU(negative_slope=act_cfg.get('negative_slope', 0.01), inplace=True)
        else:
            raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")


class InceptionBottleneck(nn.Module):
    """Bottleneck with Inception module for multi-scale feature extraction"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7),
            dilations: Sequence[int] = (1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList()
        for i, ksize in enumerate(kernel_sizes):
            self.conv_branches.append(
                ConvModule(hidden_channels, hidden_channels, ksize, 1,
                           autopad(ksize, None, dilations[i]), dilation=dilations[i],
                           groups=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )

        self.pw_conv = ConvModule(hidden_channels * len(kernel_sizes), out_channels, 1, 1, 0,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.add_identity = add_identity and in_channels == out_channels
        if self.add_identity and in_channels != out_channels:
            self.identity_conv = ConvModule(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        identity = x

        x = self.pre_conv(x)

        # 多尺度特征融合
        branch_outputs = []
        for branch in self.conv_branches:
            branch_outputs.append(branch(x))

        x = torch.cat(branch_outputs, dim=1)
        x = self.pw_conv(x)

        if self.add_identity:
            if hasattr(self, 'identity_conv'):
                identity = self.identity_conv(identity)
            x = x + identity

        return x


class SSCAConv(nn.Module):
    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(SSCAConv, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        self.attention=nn.Sequential(
            nn.Conv2d(in_planes,in_planes*(kernel_size**2),kernel_size,stride,padding,dilation,groups=in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes*(kernel_size**2),in_planes*(kernel_size**2),1,1,0,groups=in_planes),
            nn.Tanh()
        ) # b,1,H,W 全通道像素级通道注意力

        if use_bias==True:
            conv1 = nn.Conv2d(in_planes, in_planes, 1,1,0)
            self.bias=conv1.bias

    def forward(self,x):
        (b, n, H, W) = x.shape
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)

        atw=self.attention(x).reshape(b,n,k*k,n_H,n_W) #b,n,k*k,n_H*n_W
        unf_x=F.unfold(x,kernel_size=k,dilation=self.dilation,padding=self.padding,stride=self.stride).reshape(b,n,k*k,n_H,n_W) #b,n*k*k,n_H*n_W
        unf_y=unf_x*atw #b,n,k*k,n_H,n_W
        y=torch.sum(unf_y,dim=2,keepdim=False)#b,n,n_H,n_W

        if self.use_bias==True:
            y = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(y) + y

        return y



class Residual_Block(nn.Module):
    def __init__(self, channels, isd):
        super(Residual_Block, self).__init__()
        if isd==True:
            self.conv = nn.Sequential(
                SSCAConv(channels,3,1,1),
                nn.Conv2d(channels,channels,1,1,0),
                nn.ReLU(inplace=True),
                SSCAConv(channels,3,1,1),
                nn.Conv2d(channels, channels, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1,groups=channels),
                nn.Conv2d(channels, channels, 1, 1, 0),
            )

    def forward(self, x):
        return self.conv(x) + x

# ==================== 增强的MS支路 ====================
class EnhancedMSBranch(nn.Module):
    def __init__(self, channels, block_num):
        super(EnhancedMSBranch, self).__init__()

        # 创建MS支路模块序列
        modules = []
        for i in range(block_num):
            # 添加SMFA模块
            modules.append((f"smfa_{i}", SMFA(dim=channels)))
            # 添加残差块
            modules.append((f"residual_ms_{i}", Residual_Block(channels, isd=False)))

        self.ms_branch = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.ms_branch(x)


# ==================== 增强的PAN支路 ====================
class EnhancedPANBranch(nn.Module):
    def __init__(self, channels, block_num):
        super(EnhancedPANBranch, self).__init__()

        # 创建PAN支路模块序列
        modules = []
        for i in range(block_num):
            # 添加InceptionBottleneck模块
            modules.append((f"inception_{i}", InceptionBottleneck(
                in_channels=channels,
                out_channels=channels,
                kernel_sizes=(3, 5, 7),
                expansion=1.0
            )))
            # 添加残差块
            modules.append((f"residual_pan_{i}", Residual_Block(channels, isd=False)))

        self.pan_branch = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.pan_branch(x)



# ==================== 增强的SSCANet ====================
class EnhancedSSCANet(nn.Module):
    def __init__(self, channels=32, block_num=4, inchannel=4, in2=1, isd=True):
        super(EnhancedSSCANet, self).__init__()

        # 头部卷积
        self.head_conv_ms = nn.Sequential(
            nn.Conv2d(inchannel, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.head_conv_pan = nn.Sequential(
            nn.Conv2d(in2, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )


        # 增强的MS支路
        self.enhanced_ms_branch = EnhancedMSBranch(channels, block_num)

        # 增强的PAN支路
        self.enhanced_pan_branch = EnhancedPANBranch(channels, block_num)
        self.rbs = self._make_resblocks(channels, block_num, isd)
        self.ronghe = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1)
        self.tail_conv = nn.Conv2d(channels, inchannel, 3, 1, 1)


    def forward(self, lms, pan):
        nb, c, h, w = pan.shape
        lms = F.interpolate(lms, size=(h, w), mode='bicubic')

        # 头部卷积
        # x = torch.cat([pan, lms], dim=1)
        xms = self.head_conv_ms(lms)
        lpan = self.head_conv_pan(pan)
        # 分别通过增强的支路
        pan_feat = self.enhanced_pan_branch(lpan)
        ms_feat = self.enhanced_ms_branch(xms)

        # 特征融合
        fused_feat = torch.cat([pan_feat, ms_feat], dim=1)
        fused_feat = self.ronghe(fused_feat)
        # 公共残差块处理
        x = self.rbs(fused_feat)

        # 尾部卷积
        x = self.tail_conv(x)

        return x + lms

    def _make_resblocks(self, channels, block_num, isd):
        blocks = []
        for i in range(block_num):
            blocks.append((f"resblock_{i}", Residual_Block(channels, isd)))
        return nn.Sequential(OrderedDict(blocks))


# ==================== 工具函数 ====================
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# 需要添加的重排函数
def rearrange(tensor, pattern, **kwargs):
    """简单的张量重排函数"""
    b, c, h, w = tensor.shape
    if 'head' in kwargs:
        heads = kwargs['head']
        if pattern == 'b (head c) h w -> b head h (w c)':
            c_per_head = c // heads
            tensor = tensor.view(b, heads, c_per_head, h, w)
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
            tensor = tensor.view(b, heads, h, w * c_per_head)
            return tensor
        elif pattern == 'b head h (w c) -> b (head c) h w':
            w_new = kwargs['w']
            c_per_head = c // heads
            tensor = tensor.view(b, heads, h, w_new, c_per_head)
            tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
            tensor = tensor.view(b, c, h, w_new)
            return tensor
    return tensor


# 测试代码
if __name__ == '__main__':
    # from torchsummary import summary

    # 创建增强版模型
    model = EnhancedSSCANet(channels=32, block_num=4, inchannel=4, in2=1)

    # 打印模型参数
    print("Enhanced SSCANet 参数统计:")
    param_stats = get_parameter_number(model)
    print(f"总参数量: {param_stats['Total']:,}")
    print(f"可训练参数量: {param_stats['Trainable']:,}")

    # 测试前向传播
    pan = torch.randn(1, 1, 64, 64)
    lms = torch.randn(1, 4, 64, 64)

    with torch.no_grad():
        output = model(lms,pan)
        print(f"输入PAN形状: {pan.shape}")
        print(f"输入LMS形状: {lms.shape}")
        print(f"输出形状: {output.shape}")