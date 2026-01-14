import torch
import torch.nn.functional as F


def QIndex_torch(a, b):
    r"""
    PyTorch version of QIndex calculation

    Args:
        a (torch.Tensor): one-channel image, shape like [B, H, W]
        b (torch.Tensor): one-channel image, shape like [B, H, W]
    Returns:
        torch.Tensor: Q index values for each batch, shape [B]
    """
    B = a.shape[0]
    a = a.reshape(B, -1)  # [B, H*W]
    b = b.reshape(B, -1)  # [B, H*W]

    # Calculate covariance
    a_mean = a.mean(dim=1, keepdim=True)  # [B, 1]
    b_mean = b.mean(dim=1, keepdim=True)  # [B, 1]
    a_centered = a - a_mean
    b_centered = b - b_mean

    cov = (a_centered * b_centered).mean(dim=1)  # [B]
    d1 = (a_centered  **  2).mean(dim=1)  # [B]
    d2 = (b_centered  **  2).mean(dim=1)  # [B]

    m1 = a_mean.squeeze(1)  # [B]
    m2 = b_mean.squeeze(1)  # [B]

    Q = 4 * cov * m1 * m2 / (d1 + d2) / (m1  **  2 + m2  **  2)
    return Q


def D_lambda_torch(l_ms, ps):
    r"""
    PyTorch version of D_lambda calculation

    Args:
        l_ms (torch.Tensor): LR MS image, shape like [B, C, H, W]
        ps (torch.Tensor): pan-sharpened image, shape like [B, C, H, W]
    Returns:
        torch.Tensor: D_lambda values for each batch, shape [B]
    """
    B, L, H, W = ps.shape
    sum_D = torch.zeros(B, device=ps.device)

    for i in range(L):
        for j in range(L):
            if j != i:
                q_ps = QIndex_torch(ps[:, i], ps[:, j])
                q_lms = QIndex_torch(l_ms[:, i], l_ms[:, j])
                sum_D += torch.abs(q_ps - q_lms)

    return sum_D / (L * (L - 1))


def D_s_torch(l_ms, pan, ps):
    r"""
    PyTorch version of D_s calculation

    Args:
        l_ms (torch.Tensor): LR MS image, shape like [B, C, H, W]
        pan (torch.Tensor): pan image, shape like [B, 1, H, W]
        ps (torch.Tensor): pan-sharpened image, shape like [B, C, H, W]
    Returns:
        torch.Tensor: D_s values for each batch, shape [B]
    """
    B, L, H, W = ps.shape

    # Downsample pan image using average pooling (similar to pyrDown)
    l_pan = F.avg_pool2d(pan, kernel_size=2)  # first downsampling
    l_pan = F.avg_pool2d(l_pan, kernel_size=2)  # second downsampling

    sum_D = torch.zeros(B, device=ps.device)

    for i in range(L):
        q_ps = QIndex_torch(ps[:, i], pan.squeeze(1))  # remove channel dim for pan
        q_lms = QIndex_torch(l_ms[:, i], l_pan.squeeze(1))
        sum_D += torch.abs(q_ps - q_lms)

    return sum_D / L


def QNR_torch(fusion, ms, pan, alpha=1, beta=1):
    r"""
    PyTorch version of QNR calculation

    Args:
        fusion (torch.Tensor): pan-sharpened image, shape like [B, C, H, W]
        ms (torch.Tensor): LR MS image, shape like [B, C, H, W]
        pan (torch.Tensor): pan image, shape like [B, 1, H, W]
        alpha (int): weight for D_lambda
        beta (int): weight for D_s
    Returns:
        tuple: (D_lambda, D_s, QNR) all with shape [B]
    """
    D_lambda = D_lambda_torch(ms, fusion)
    D_s = D_s_torch(ms, pan, fusion)
    QNR = ((1 - D_lambda)  **  alpha) * ((1 - D_s)  **  beta)

    return D_lambda, D_s, QNR