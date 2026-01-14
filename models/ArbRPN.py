#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ArbRPN.py
@Contact :   lihuichen@126.com
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
xxxxxxxxxx        LihuiChen      1.0         None
'''

# import lib

import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inFe, outFe):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inFe, outFe, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outFe, outFe, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


class Net(nn.Module):
    def __init__(self, opt=None):
        super(Net, self).__init__()
        hid_dim = 64
        input_dim = 64
        num_resblock = 3
        self.num_cycle = 5

        self.wrapper = nn.Conv2d(1, hid_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(1, input_dim, 3, 1, 1)

        self.hidden_unit_forward_list = nn.ModuleList()
        self.hidden_unit_backward_list = nn.ModuleList()
        
        for _ in range(self.num_cycle):
            compress_1 = (nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_1 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_forward_list.append(nn.Sequential(compress_1, resblock_1))

            compress_2 = (nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_2 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_backward_list.append(nn.Sequential(compress_2, resblock_2))

        self.conv2 = nn.Conv2d(hid_dim, 1, 3, 1, 1)
        self.nihe = nn.Conv2d(8, 1, kernel_size=1)
    def forward(self, ms, pan, mask=None, is_cat_out=True):
        '''
        :param ms: LR ms images
        :param pan: pan images
        :param mask: mask to record the batch size of each band
        :return:
            HR_ms: a list of HR ms images,
        '''

        if mask is None:
            mask = [1 for _ in range(ms.shape[1])]
            is_cat_out = True

        ms = ms.split(1, dim=1)
        pan_state = self.wrapper(pan)
        hidden_state = pan_state
        blur_ms_list = []


        backward_hidden = []
        for idx, band in enumerate(ms):
            band = F.interpolate(band[:mask[idx]], scale_factor=4, mode='bicubic', align_corners=False)
            blur_ms_list.append(band)
            backward_hidden.append(self.conv1(band))

        backward_hidden = backward_hidden[::-1]
        for idx_cycle in range(self.num_cycle):
            ## forward recurrence
            forward_hidden = []
            for idx in range(len(blur_ms_list)):
                hidden_state = hidden_state[:mask[idx]]
                band = torch.cat((backward_hidden[-(idx+1)], hidden_state, pan_state[:mask[idx]]), dim=1)
                hidden_state = self.hidden_unit_forward_list[idx_cycle](band)
                forward_hidden.append(hidden_state)
            ## backward recurrence
            backward_hidden = []
            for idx in range(len(blur_ms_list)):
                start_pan_stat = hidden_state.shape[0]
                hidden_state = torch.cat((hidden_state, pan_state[start_pan_stat:mask[-(idx+1)]]),dim=0)
                band = torch.cat((forward_hidden[-(idx + 1)], hidden_state, pan_state[:mask[-(idx+1)]]), dim=1)
                hidden_state = self.hidden_unit_backward_list[idx_cycle](band)
                backward_hidden.append(hidden_state)

        HR_ms = []
        for idx in range(len(blur_ms_list)):
            band = self.conv2(backward_hidden[-(idx+1)])
            band = band + blur_ms_list[idx]
            HR_ms.append(band)
        nihe = self.nihe(torch.cat(HR_ms, dim=1))
        return HR_ms if not is_cat_out else (HR_ms, torch.cat(HR_ms, dim=1)), nihe


class Net1(nn.Module):
    def __init__(self, opt=None):
        super(Net1, self).__init__()
        hid_dim = 64
        input_dim = 64
        num_resblock = 3
        self.num_cycle = 5

        self.wrapper = nn.Conv2d(1, hid_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(1, input_dim, 3, 1, 1)

        self.hidden_unit_forward_list = nn.ModuleList()
        self.hidden_unit_backward_list = nn.ModuleList()

        for _ in range(self.num_cycle):
            compress_1 = (nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_1 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_forward_list.append(nn.Sequential(compress_1, resblock_1))

            compress_2 = (nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_2 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_backward_list.append(nn.Sequential(compress_2, resblock_2))

        self.conv2 = nn.Conv2d(hid_dim, 1, 3, 1, 1)
        self.nihe = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, ms, pan, mask=None, is_cat_out=True):
        '''
        :param ms: LR ms images (B, C, H, W)
        :param pan: pan images (B, 1, H*4, W*4)
        :param mask: mask to record the batch size of each band
        :return:
            HR_ms: a list of HR ms images,
        '''
        batch_size = ms.shape[0]
        num_bands = ms.shape[1]

        if mask is None:
            mask = [batch_size for _ in range(num_bands)]
            is_cat_out = True

        ms = ms.split(1, dim=1)
        pan_state = self.wrapper(pan)  # (B, hid_dim, H*4, W*4)
        hidden_state = pan_state
        blur_ms_list = []

        # Process each band
        backward_hidden = []
        for idx, band in enumerate(ms):
            # band shape: (B, 1, H, W)
            band = F.interpolate(band, scale_factor=4, mode='bicubic', align_corners=False)
            blur_ms_list.append(band)  # (B, 1, H*4, W*4)
            backward_hidden.append(self.conv1(band))  # (B, input_dim, H*4, W*4)

        backward_hidden = backward_hidden[::-1]

        for idx_cycle in range(self.num_cycle):
            ## forward recurrence
            forward_hidden = []
            for idx in range(len(blur_ms_list)):
                # Use all samples (no slicing with mask)
                band = torch.cat((backward_hidden[-(idx + 1)], hidden_state, pan_state), dim=1)
                hidden_state = self.hidden_unit_forward_list[idx_cycle](band)
                forward_hidden.append(hidden_state)

            ## backward recurrence
            backward_hidden = []
            for idx in range(len(blur_ms_list)):
                # Use all samples (no slicing with mask)
                band = torch.cat((forward_hidden[-(idx + 1)], hidden_state, pan_state), dim=1)
                hidden_state = self.hidden_unit_backward_list[idx_cycle](band)
                backward_hidden.append(hidden_state)

        HR_ms = []
        for idx in range(len(blur_ms_list)):
            band = self.conv2(backward_hidden[-(idx + 1)])  # (B, 1, H*4, W*4)
            band = band + blur_ms_list[idx]
            HR_ms.append(band)

        nihe = self.nihe(torch.cat(HR_ms, dim=1))  # (B, 1, H*4, W*4)

        return HR_ms if not is_cat_out else (HR_ms, torch.cat(HR_ms, dim=1)), nihe


class Net3(nn.Module):
    def __init__(self, opt=None):
        super(Net3, self).__init__()
        hid_dim = 64
        input_dim = 64
        num_resblock = 3
        self.num_cycle = 5

        self.wrapper = nn.Conv2d(1, hid_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(1, input_dim, 3, 1, 1)

        self.hidden_unit_forward_list = nn.ModuleList()
        self.hidden_unit_backward_list = nn.ModuleList()

        for _ in range(self.num_cycle):
            compress_1 = nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0)
            resblock_1 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_forward_list.append(nn.Sequential(compress_1, resblock_1))

            compress_2 = nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0)
            resblock_2 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_backward_list.append(nn.Sequential(compress_2, resblock_2))

        self.conv2 = nn.Conv2d(hid_dim, 1, 3, 1, 1)
        self.nihe = nn.Conv2d(8, 1, kernel_size=1)
    def forward(self, ms, pan, is_cat_out=True):
        '''
        :param ms: LR ms images [B, C, H, W]
        :param pan: pan images [B, 1, H*4, W*4]
        :return:
            HR_ms: a list of HR ms images (if not is_cat_out)
                   or concatenated HR ms images [B, C, H*4, W*4]
        '''
        B = ms.size(0)  # batch size
        num_bands = ms.size(1)

        # Upsample each band and prepare initial states
        blur_ms_list = []
        backward_hidden = []
        pan_state = self.wrapper(pan)  # [B, hid_dim, H*4, W*4]
        hidden_state = pan_state.clone()

        for band_idx in range(num_bands):
            band = ms[:, band_idx:band_idx + 1]  # [B, 1, H, W]
            band = F.interpolate(band, scale_factor=4, mode='bicubic', align_corners=False)
            blur_ms_list.append(band)  # [B, 1, H*4, W*4]
            backward_hidden.append(self.conv1(band))  # [B, input_dim, H*4, W*4]

        backward_hidden = backward_hidden[::-1]  # reverse for backward pass

        # Main cycle loop
        for idx_cycle in range(self.num_cycle):
            # Forward recurrence
            forward_hidden = []
            for idx in range(num_bands):
                band = torch.cat((backward_hidden[-(idx + 1)], hidden_state, pan_state), dim=1)
                hidden_state = self.hidden_unit_forward_list[idx_cycle](band)
                forward_hidden.append(hidden_state)

            # Backward recurrence
            backward_hidden = []
            for idx in range(num_bands):
                band = torch.cat((forward_hidden[-(idx + 1)], hidden_state, pan_state), dim=1)
                hidden_state = self.hidden_unit_backward_list[idx_cycle](band)
                backward_hidden.append(hidden_state)

        # Generate output
        HR_ms = []
        for idx in range(num_bands):
            band = self.conv2(backward_hidden[-(idx + 1)])  # [B, 1, H*4, W*4]
            band = band + blur_ms_list[idx]  # residual connection
            HR_ms.append(band)

        if is_cat_out:
            hrms = torch.cat(HR_ms, dim=1)
            nihe = self.nihe(hrms)
            return hrms, nihe # [B, C, H*4, W*4]
        else:
            return HR_ms  # list of [B, 1, H*4, W*4]

class myloss(nn.Module):
    def __init__(self, opt=None):
        super(myloss, self).__init__()

    def forward(self, ms, HR, mask=None):
        diff = 0
        count = 0
        HR = torch.split(HR, 1, dim=1)
        if mask is None:
            mask = [1 for _ in range(len(HR))]
            ms = ms[0]
        for idx, (band, hr) in enumerate(zip(ms, HR)):
            b, t, h, w = band.shape
            count += b * t * h * w
            diff += torch.sum(torch.abs(band - hr[:mask[idx]]))
        return diff / count

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net = Net3()
    lms = torch.randn(1, 8, 64, 64)
    pan = torch.randn(1, 1, 256, 256)
    out, nihe = net(lms, pan)
    # print(out.shape)
    flops = FlopCountAnalysis(net, (lms, pan))
    print(flop_count_table(flops))