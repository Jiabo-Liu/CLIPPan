import torch
import clip
import scipy.io
import numpy as np
from PIL import Image
import os
from datetime import datetime
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path

# 自定义模块导入
from data import Datain
from quality_assessment import calc_psnr, calc_rmse, calc_ergas, calc_sam
from args_parser import args_parser
from models.clip_adapt import CLIP_Adapter
from models.highorder import Net
from models.Panform import CrossSwinTransformer
from pytorch_msssim import ssim
from torchvision.transforms import GaussianBlur


class MultispectralImageFusionTrainer:
    """多光谱图像融合训练器优化版"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_directories()
        self.setup_models()
        self.setup_optimizers()
        self.scaler = GradScaler()  # 混合精度训练

    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """创建必要的目录"""
        Path(self.args.model_path).mkdir(parents=True, exist_ok=True)
        Path(self.args.result_path).mkdir(parents=True, exist_ok=True)

    def setup_models(self):
        """初始化所有模型组件"""
        # CLIP模型
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # 特征降维模块
        self.ms_reducer = BottleneckReduction(in_channels=8).to(self.device)
        self.out_reducer = BottleneckReduction(in_channels=8).to(self.device)

        # CLIP适配器
        self.clip_adapter = CLIP_Adapter(c_in=512, reduction=4).to(self.device)

        # 主融合模型
        self.sup_model = Net().to(self.device)
        self.fusion_model = CrossSwinTransformer().to(self.device)

        # 加载预训练权重
        self.load_pretrained_weights()

        # 冻结预训练模型参数
        self.freeze_pretrained_models()

    def load_pretrained_weights(self):
        """加载预训练权重"""
        pretrained_paths = {
            'ship': "./model_WV3crop_ship_605/model_epoch999.pth",
            'clip_adapter': "./model_WV3_CLIP_adapter_421_conloss_fenduan_bl0.4+0.4+0.2/model_epoch299.pth",
            'ms_reducer': "./ms_reducer_622_conloss_fenduan421/ms_reducer_epoch999.pth",
            'out_reducer': "./out_reducer_622_conloss_fenduan421/out_reducer_epoch999.pth"
        }

        for name, path in pretrained_paths.items():
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    getattr(self, self.get_model_attribute_name(name)).load_state_dict(checkpoint)
                    self.logger.info(f"成功加载 {name} 的预训练权重")
                except Exception as e:
                    self.logger.warning(f"加载 {name} 失败: {e}")
            else:
                self.logger.warning(f"预训练文件不存在: {path}")

    def get_model_attribute_name(self, model_name):
        """获取模型属性名称"""
        mapping = {
            'ship': 'sup_model',
            'clip_adapter': 'clip_adapter',
            'ms_reducer': 'ms_reducer',
            'out_reducer': 'out_reducer'
        }
        return mapping.get(model_name, model_name)

    def freeze_pretrained_models(self):
        """冻结预训练模型参数"""
        for param in self.sup_model.parameters():
            param.requires_grad = False
        self.logger.info("预训练模型参数已冻结")

    def setup_optimizers(self):
        """设置优化器"""
        self.optimizer = Adam(
            list(self.fusion_model.parameters()) +
            list(self.clip_adapter.parameters()),
            lr=self.args.learning_rate,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs
        )

        self.criterion = nn.L1Loss().to(self.device)

    def create_text_descriptions(self, batch_size):
        """创建批量文本描述"""
        ms_descriptions = [f"A multispectral image with rich spectral information" for _ in range(batch_size)]
        pan_descriptions = [f"A panchromatic image with high spatial details" for _ in range(batch_size)]
        out_descriptions = [
            "High-quality fused image preserving both spectral and spatial information"
            for _ in range(batch_size)
        ]
        return ms_descriptions, pan_descriptions, out_descriptions

    def encode_text_features(self, descriptions):
        """编码文本特征"""
        tokens = clip.tokenize(descriptions).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def encode_image_features(self, images):
        """编码图像特征"""
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True)

    def compute_losses(self, out, ms, pan, ref, nihe, out_ship):
        """计算所有损失函数"""
        # 图像预处理
        resize_transform = transforms.Resize((224, 224), antialias=True)

        with torch.no_grad():
            ms_pca = self.ms_reducer(ms)
            ms_pca_resize = resize_transform(ms_pca)
            pan_resize = resize_transform(pan)
            out_resize = resize_transform(out)
            pan_3channel = pan_resize.repeat(1, 3, 1, 1)
            out_3channel = self.out_reducer(out_resize)

        # 文本和图像特征编码
        ms_descriptions, pan_descriptions, out_descriptions = self.create_text_descriptions(ms.size(0))

        text_features_ms = self.encode_text_features(ms_descriptions)
        text_features_pan = self.encode_text_features(pan_descriptions)
        text_features_out = self.encode_text_features(out_descriptions)

        image_features_ms = self.encode_image_features(ms_pca_resize)
        image_features_pan = self.encode_image_features(pan_3channel)
        image_features_out = self.encode_image_features(out_3channel)

        # CLIP适配器输出
        with autocast():
            adapted_text, adapted_image = self.clip_adapter(
                text_features_ms, text_features_pan, image_features_ms, image_features_pan
            )

        # 各项损失计算
        loss_lg = self.compute_direction_loss(text_features_out, adapted_text, image_features_out, adapted_image)
        loss_unsup = self.unsupervised_loss(out, ms, pan, nihe)
        loss_recon = self.criterion(out, out_ship)

        # 加权总损失
        total_loss = (
                self.args.lambda_lg * loss_lg +
                self.args.lambda_unsup * loss_unsup +
                self.args.lambda_recon * loss_recon
        )

        return total_loss, {
            'loss_lg': loss_lg.item(),
            'loss_unsup': loss_unsup.item(),
            'loss_recon': loss_recon.item(),
            'total_loss': total_loss.item()
        }

    def compute_direction_loss(self, text_out, text_ref, out, ref):
        """计算方向一致性损失"""

        def cosine_similarity(a, b):
            return F.cosine_similarity(a, b, dim=1)

        cos_sim_ms = cosine_similarity(text_out, out)
        cos_sim_pan = cosine_similarity(text_ref, ref)

        direction_loss = 1 - 0.5 * (torch.mean(cos_sim_ms) + torch.mean(cos_sim_pan))
        return direction_loss

    def unsupervised_loss(self, out, ms, pan, nihe):
        """无监督损失"""
        spectral_loss = self.spectral_loss(out, ms)
        spatial_loss = self.spatial_loss(nihe, pan)
        return spectral_loss + spatial_loss

    def spectral_loss(self, hr_ms, lr_ms):
        """光谱损失"""
        ms_down = downsample_ms_tensor_pytorch(hr_ms, ratio=4)
        loss_mse = nn.MSELoss()(ms_down, lr_ms)
        loss_ssim = 1 - ssim(ms_down, lr_ms)
        return loss_mse + loss_ssim

    def spatial_loss(self, I_nihe, pan):
        """空间损失"""
        loss_mse = nn.MSELoss()(I_nihe, pan)
        loss_ssim = 1 - ssim(I_nihe, pan)
        return loss_mse + loss_ssim

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.fusion_model.train()
        self.clip_adapter.train()

        total_loss = 0
        loss_dict = {'loss_lg': 0, 'loss_unsup': 0, 'loss_recon': 0}

        for batch_idx, (ref, pan, ms) in enumerate(train_loader):
            # 数据转移到设备
            ref = ref.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            with autocast():
                with torch.no_grad():
                    out_ship = self.sup_model(ms, pan)
                out, nihe = self.fusion_model(pan, ms)

                loss, batch_loss_dict = self.compute_losses(out, ms, pan, ref, nihe, out_ship)

            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 损失记录
            total_loss += loss.item()
            for k in loss_dict:
                loss_dict[k] += batch_loss_dict[k]

            if batch_idx % self.args.log_interval == 0:
                self.logger.info(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                    f'Loss: {loss.item():.6f}'
                )

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        for k in loss_dict:
            loss_dict[k] /= len(train_loader)

        return avg_loss, loss_dict

    def validate(self, val_loader, epoch):
        """验证模型"""
        self.fusion_model.eval()

        metrics = {'rmse': [], 'psnr': [], 'ergas': [], 'sam': []}

        with torch.no_grad():
            for ref, pan, ms in val_loader:
                ref = ref.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)

                out, _ = self.fusion_model(pan, ms)

                # 转换为numpy计算指标
                ref_np = ref.detach().cpu().numpy()
                out_np = out.detach().cpu().numpy()

                metrics['rmse'].append(calc_rmse(ref_np, out_np))
                metrics['psnr'].append(calc_psnr(ref_np, out_np))
                metrics['ergas'].append(calc_ergas(ref_np, out_np))
                metrics['sam'].append(calc_sam(ref_np, out_np))

        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        self.logger.info(
            f'Validation Epoch: {epoch} - '
            f'PSNR: {avg_metrics["psnr"]:.4f}, RMSE: {avg_metrics["rmse"]:.4f}, '
            f'ERGAS: {avg_metrics["ergas"]:.4f}, SAM: {avg_metrics["sam"]:.4f}'
        )

        return avg_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'fusion_model_state_dict': self.fusion_model.state_dict(),
            'clip_adapter_state_dict': self.clip_adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }

        checkpoint_path = Path(self.args.model_path) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = Path(self.args.model_path) / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        self.logger.info("开始训练...")

        best_psnr = 0
        writer = SummaryWriter("logs/training")

        for epoch in range(self.args.epochs):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            self.logger.info(f"{'=' * 60}")

            # 训练阶段
            train_loss, train_loss_dict = self.train_epoch(train_loader, epoch)

            # 验证阶段
            val_metrics = self.validate(val_loader, epoch)

            # 学习率调整
            self.scheduler.step()

            # TensorBoard记录
            writer.add_scalar('Loss/train', train_loss, epoch)
            for k, v in train_loss_dict.items():
                writer.add_scalar(f'Loss/{k}', v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, epoch)
            writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.logger.info(f"新的最佳模型已保存，PSNR: {best_psnr:.4f}")

            # 定期保存检查点
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics)

        writer.close()
        self.logger.info("训练完成！")


def downsample_ms_tensor_pytorch(img_ms_tensor, ratio=4, sigma=2):
    """优化的降采样函数"""
    blur = GaussianBlur(kernel_size=5, sigma=2.1)
    blurred_ms = blur(img_ms_tensor)

    downsampled = F.interpolate(
        blurred_ms,
        size=(img_ms_tensor.shape[2] // ratio, img_ms_tensor.shape[3] // ratio),
        mode='area'
    )
    return downsampled


class BottleneckReduction(nn.Module):
    """优化的特征降维模块"""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(in_channels // reduction, 64)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 3, kernel_size=1)
        )

        self.shortcut = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        out += residual
        return F.relu(out)


def main():
    """主函数"""
    args = args_parser()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化训练器
    trainer = MultispectralImageFusionTrainer(args)

    # 数据加载
    train_dataset = Datain(args.data_path_mat_train, args.scale_ratio)
    val_dataset = Datain(args.data_path_mat_val, args.scale_ratio)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()