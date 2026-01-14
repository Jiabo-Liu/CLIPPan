import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d


def BDSD(lrms, pan, ratio=4, S=16, sensor='WV3'):
    """
    Final corrected version that handles all dimensions properly
    """
    B, C, H, W = lrms.shape
    assert pan.shape == (B, 1, H * ratio, W * ratio)

    # 1. Upsample MS
    ms_up = F.interpolate(lrms, scale_factor=ratio, mode='bicubic', align_corners=False)

    # 2. Generate low-pass versions
    pan_lp = MTF_PAN_torch(pan, sensor, ratio)
    pan_lp_d = pan_lp[:, :, ::ratio, ::ratio]
    ms_lp_d = MTF_torch(lrms, sensor, ratio)

    # 3. Estimate gamma (B, C+1, C, H, W)
    in3 = torch.cat([ms_lp_d, lrms, pan_lp_d], dim=1)
    gamma = estimate_gamma_cube_torch(in3, S, ratio)

    # 4. Prepare gamma for fusion - reshape to (B, (C+1)*C, H, W) and upsample
    gamma_reshaped = gamma.reshape(B, (C + 1) * C, H, W)
    gamma_upsampled = F.interpolate(gamma_reshaped, scale_factor=ratio, mode='nearest')

    # 5. Concat all inputs for fusion
    # ms_up: (B,C,64,64), pan: (B,1,64,64), gamma_upsampled: (B,(C+1)*C,64,64)
    in3_full = torch.cat([ms_up, pan, gamma_upsampled], dim=1)

    # 6. Apply fusion
    fused = compH_inject_torch(in3_full, C, S)
    return fused


def compH_inject_torch(in3, C, S):
    """
    Corrected fusion function with proper gamma handling
    in3: (B, C + 1 + (C+1)*C, H, W) = (B, C + 1 + 72, 64, 64) when C=8
    """
    B, _, H, W = in3.shape
    fused = torch.zeros(B, C, H, W, device=in3.device)

    # Split inputs
    ms_up = in3[:, :C]  # (B,C,H,W)
    pan = in3[:, C:C + 1]  # (B,1,H,W)
    gamma_all = in3[:, C + 1:]  # (B,(C+1)*C,H,W)

    # Process in blocks
    for i in range(0, H, S):
        for j in range(0, W, S):
            # Get current block data
            ms_block = ms_up[:, :, i:i + S, j:j + S]  # (B,C,S,S)
            pan_block = pan[:, :, i:i + S, j:j + S]  # (B,1,S,S)
            gamma_block = gamma_all[:, :, i:i + S, j:j + S]  # (B,(C+1)*C,S,S)

            # Prepare H matrix (B, S*S, C+1)
            H = torch.zeros(B, S * S, C + 1, device=in3.device)
            H[:, :, :C] = ms_block.permute(0, 2, 3, 1).reshape(B, S * S, C)
            H[:, :, C] = pan_block.reshape(B, S * S)

            # Reshape gamma to (B, C+1, C)
            gamma = gamma_block.reshape(B, C + 1, C, S * S).mean(dim=-1)

            # Compute injection (B, C, S*S)
            injection = torch.bmm(H, gamma).permute(0, 2, 1)
            fused[:, :, i:i + S, j:j + S] = ms_block + injection.reshape(B, C, S, S)

    return fused


def estimate_gamma_cube_torch(in3, S, ratio):
    """保持不变，使用之前的实现"""
    B, _, H, W = in3.shape
    C = (in3.shape[1] - 1) // 2
    block_size = S // ratio
    gamma = torch.zeros(B, C + 1, C, H, W, device=in3.device)

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = in3[:, :, i:i + block_size, j:j + block_size]

            Hd = torch.zeros(B, block_size ** 2, C + 1, device=in3.device)
            Hd[:, :, :C] = block[:, :C].reshape(B, C, -1).permute(0, 2, 1)
            Hd[:, :, C] = block[:, -1].reshape(B, -1)

            diff = (block[:, C:2 * C] - block[:, :C]).reshape(B, C, -1)
            gamma_block = torch.bmm(
                torch.linalg.pinv(Hd.transpose(1, 2) @ Hd) @ Hd.transpose(1, 2),
                diff.permute(0, 2, 1)
            )
            gamma[:, :, :, i:i + block_size, j:j + block_size] = gamma_block.unsqueeze(-1).unsqueeze(-1)

    return gamma


def MTF_PAN_torch(pan, sensor, ratio):
    """Batch PAN filtering"""
    kernel = get_mtf_kernel(sensor, ratio, pan.device)
    return conv2d(pan, kernel, padding=kernel.size(-1) // 2)


def MTF_torch(ms, sensor, ratio):
    """Batch MS filtering with fixed kernel dimensions"""
    kernel = get_mtf_kernel(sensor, ratio, ms.device)

    # Expand kernel to match input channels
    # Original kernel shape: (1,1,kH,kW)
    # For groups=ms.size(1), we need (out_channels, in_channels/groups, kH, kW)
    # Since in_channels = out_channels = ms.size(1) and groups=ms.size(1)
    # The kernel should be (C,1,kH,kW)
    kernel = kernel.repeat(ms.size(1), 1, 1, 1)

    return F.conv2d(
        ms,
        kernel,
        padding=kernel.size(-1) // 2,
        groups=ms.size(1)
    )


def get_mtf_kernel(sensor, ratio, device):
    """Generate MTF kernel with proper dimensions"""
    kernel_size = 5
    sigma = ratio / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    x = coords.view(-1, 1)
    y = coords.view(1, -1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)  # (1,1,kH,kW)

