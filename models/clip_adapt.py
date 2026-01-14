import torch
import torch.nn as nn
from torch.nn import functional as F



class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class FuseAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(FuseAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * c_in, 2 * c_in // reduction, bias=False),  # Double input channels
            nn.ReLU(inplace=True),
            nn.Linear(2 * c_in // reduction, c_in, bias=False),  # Output original channels
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: first input feature [batch_size, c_in]
            x2: second input feature [batch_size, c_in]
        Returns:
            fused feature [batch_size, c_in]
        """
        # Concatenate features along channel dimension
        x = torch.cat([x1, x2], dim=-1)  # [batch_size, 2*c_in]
        x = self.fc(x)  # [batch_size, c_in]
        return x


class CLIP_Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(CLIP_Adapter, self).__init__()
        self.adapt_text_ms = Adapter(c_in, reduction)
        self.adapt_text_pan = Adapter(c_in, reduction)
        self.text_fuse = FuseAdapter(c_in, reduction)
        # self.adapt_text_out = Adapter(c_in, reduction)
        self.adapt_ms = Adapter(c_in, reduction)
        self.adapt_pan = Adapter(c_in, reduction)
        self.image_fuse = FuseAdapter(c_in, reduction)
        # self.adapt_out = Adapter(c_in, reduction)

    def forward(self, text_ms, text_pan, ms, pan):
        ratio = 0.2
        a = 0.5
        # 图像编码
        ms_features = self.adapt_ms(ms)
        ms_features = ratio * ms_features + (1 - ratio) * ms

        pan_features = self.adapt_pan(pan)
        pan_features = ratio * pan_features + (1 - ratio) * pan

        # 文本编码
        text_features_ms = self.adapt_text_ms(text_ms)
        text_features_ms = a * text_features_ms + (1 - a) * text_ms

        text_features_pan = self.adapt_text_pan(text_pan)
        text_features_pan = a * text_features_pan + (1 - a) * text_pan
        # 融合
        # text_fuse = torch.cat([text_features_ms, text_features_pan], dim=1)
        text_fuse = self.text_fuse(text_features_ms, text_features_pan)

        # image_fuse = torch.cat([ms_features, pan_features], dim=1)
        image_fuse = self.image_fuse(ms_features, pan_features)
        return text_fuse, image_fuse, ms_features, pan_features, text_features_ms, text_features_pan

if __name__ == '__main__':
    xms = torch.randn(32, 512)
    pan = torch.randn(32, 512)
    text_ms = torch.randn(32, 512)
    text_pan = torch.randn(32, 512)
    Model = CLIP_Adapter(c_in=512)
    # summary(Model, input_size=[(pan.shape),(xms.shape)], device='cpu')
    text, image = Model(text_ms, text_pan, xms, pan)
    print(image.shape)
    # for nam, param in Model.named_modules():
    #     print(f"Layer:{nam} | Size:{param.size()}")
    total_params = sum(p.numel() for p in Model.parameters())
    trainable_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')