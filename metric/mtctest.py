import numpy as np
import math
import cv2
import os
#from skimage.measure import compare_ssim
import skimage.measure
import metrics as mtc
#from scipy.stats import pearsonr

#from data import common



##数据集评估####

# original_path = r"F:\stx python\MMFN\MMFN-mainstx\results\BI\MFTHREE\MyImage\x4_output"
original_path = r"F:\stx python\FAFNet-main\FAFNet-main\results\8.22_wv4_npy"
#loss1_path = r"F:\MMFN\MMFN-mainstx\results\BI\MFTHREE\MyImage\loss1"
gt_path = r"F:\stx python\PCNN-stx\PCNN\PNN-master\WV4_500samples\test\GT_output"
ref_path = r"F:\stx python\PCNN-stx\PCNN\PNN-master\WV4_500samples\test\REF_output"
lr_path = r"F:\stx python\PCNN-stx\PCNN\PNN-master\WV4_500samples\test\LR_output"
#ref_fr_path = r"F:/MMFN/MMFN-mainstx/data_stx/test/LR_npy"
length = len(os.listdir(gt_path))
CC1 = 0
MPSNR1= 0
SSIM1 = 0
ERGAS1 = 0
RMSE1 = 0
SAM1 = 0
SCC1 = 0
FCC1 = 0
D_lambda1 = 0
D_s1 = 0
SF1 = 0
SD1 = 0
# for file in os.listdir(original_path):
#     original_img = os.path.join(original_path,file)
#     img_fused = np.load(original_img)
#
#     # loss1_img = os.path.join(loss1_path,file)
#     # loss1 = np.load(loss1_img)
#
#     gt_img = os.path.join(gt_path, file)
#     gt = np.load(gt_img)
#
#     ref_img = os.path.join(ref_path, file)
#     ref = np.load(ref_img)
#
#     lr_img = os.path.join(lr_path, file)
#     lr = np.load(lr_img)
# 获取两个文件夹中的文件列表
files1 = os.listdir(original_path)
files2 = os.listdir(gt_path)
files3 = os.listdir(ref_path)
files4 = os.listdir(lr_path)
# 遍历每个文件夹中的文件，假设文件名相同但在不同的文件夹中
for file1, file2, file3, file4 in zip(files1, files2, files3, files4):
    file1_path = os.path.join(original_path, file1)
    file2_path = os.path.join(gt_path, file2)
    file3_path = os.path.join(ref_path, file3)
    file4_path = os.path.join(lr_path, file4)
        # 加载对应的文件并进行处理
    img_fused = np.load(file1_path)
    gt = np.load(file2_path)
    ref = np.load(file3_path)
    lr = np.load(file4_path)
        # 在这里可以进行你想要的处理操作
        # 例如，你可以将 data1 和 data2 进行组合、计算、合并等等
    CC2 = mtc.CC_numpy(gt, img_fused)
    MPSNR2 = mtc.MPSNR_numpy(gt, img_fused, 2047)
    SSIM2 = mtc.SSIM_numpy(gt, img_fused, 2047, sewar=False)
    ERGAS2 = mtc.ERGAS_numpy(gt, img_fused, ratio=0.25, sewar=False)
    RMSE2 = mtc.RMSE_numpy(gt, img_fused, sewar=False)
    SAM2 = mtc.SAM_numpy(gt, img_fused, sewar=False)
    SCC2 = mtc.SCC_numpy(gt, img_fused, sewar=False)

    FCC2 = mtc.FCC_numpy(ref, img_fused)
    D_lambda2 = mtc.D_lambda_numpy(lr, img_fused, sewar=False)
    D_s2 = mtc.D_s_numpy(lr, ref, img_fused, sewar=False)
    SF2 = mtc.SF_numpy(img_fused)
    SD2 = mtc.SD_numpy(img_fused)

    CC1 = CC1 + CC2
    MPSNR1 = MPSNR1 + MPSNR2
    SSIM1 = SSIM1 + SSIM2
    ERGAS1 = ERGAS1 + ERGAS2
    RMSE1 = RMSE1 + RMSE2
    SAM1 = SAM1 + SAM2
    SCC1 = SCC1 + SCC2
    FCC1 = FCC1 + FCC2
    D_lambda1 = D_lambda1 + D_lambda2
    D_s1 = D_s1 + D_s2
    SF1 = SF1 + SF2
    SD1 = SD1 + SD2

CC = CC1 / length
MPSNR = MPSNR1 / length
SSIM = SSIM1 / length
ERGAS = ERGAS1 / length
RMSE = RMSE1 / length
SAM = SAM1 / length
SCC = SCC1 / length
FCC = FCC1 / length
D_lambda = D_lambda1 / length
D_s = D_s1 / length
SF = SF1 / length
SD = SD1 / length

print(CC)
print(MPSNR)
print(SSIM)
print(ERGAS)
print(RMSE)
print(SAM)
print(SCC)
print(FCC)
print(D_lambda)
print(D_s)
print(SF)
print(SD)


