
import torch
import clip
import scipy.io
import numpy as np
from PIL import Image
import os
from datetime import datetime
import torch
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import scipy.io as scio
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import logging
from typing import Tuple, Dict, List, Optional

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置参数类"""

    def __init__(self):
        self.batch_size = 8
        self.epochs = 200
        self.learning_rate = 1e-4
        self.scale_ratio = 4
        self.data_path_mat_train = "./data/train"
        self.model_path = "./models"
        self.result_path = "./results"
        self.ms_reducer = "./ms_reducer"
        self.out_reducer = "./out_reducer"
        self.use_amp = True  # 启用混合精度训练

        # 损失权重系数
        self.loss_weights = {
            'intra_modal': 1.0,
            'inter_modal': 1.0,
            'fusion': 1.0,
            'direction': 1.0
        }


class BottleneckReduction(nn.Module):
    """
    高效的瓶颈结构残差降维模块
    采用先降维再升维的策略，减少参数量同时保持性能
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.hidden_dim = in_channels // reduction_ratio

        self.conv1 = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, 3, kernel_size=1)  # 输出3通道RGB图像

        self.shortcut = nn.Conv2d(in_channels, 3, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        out += residual
        return self.relu(out)


class MultimodalContrastiveLoss:
    """多模态对比损失计算器"""

    def __init__(self, temp_img: float = 0.07, temp_text: float = 0.07):
        self.temp_img = temp_img
        self.temp_text = temp_text
        self.cross_entropy = nn.CrossEntropyLoss()

    def image_contrast(self, features: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        """图像模态内对比损失"""
        batch_size = features.size(0)
        labels = torch.arange(batch_size, device=features.device)

        # 计算相似度矩阵
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t()) / temp

        return self.cross_entropy(similarity_matrix, labels)

    def cross_modal_contrast(self, img_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """跨模态对比损失（图像-文本）"""
        batch_size = img_features.size(0)
        labels = torch.arange(batch_size, device=img_features.device)

        # 归一化特征
        img_features_norm = F.normalize(img_features, p=2, dim=1)
        text_features_norm = F.normalize(text_features, p=2, dim=1)

        # 计算相似度
        similarity = torch.mm(img_features_norm, text_features_norm.t()) / ((self.temp_img + self.temp_text) / 2)

        return self.cross_entropy(similarity, labels)

    def intra_modal_loss(self, ms_img: torch.Tensor, pan_img: torch.Tensor, ref_img: torch.Tensor) -> torch.Tensor:
        """计算图像模态内损失"""
        batch_size = ms_img.size(0)
        combined_features = torch.cat([ms_img, pan_img, ref_img], dim=0)
        return self.image_contrast(combined_features, self.temp_img)

    def inter_modal_loss(self, ms_img: torch.Tensor, pan_img: torch.Tensor, ref_img: torch.Tensor,
                         ms_text: torch.Tensor, pan_text: torch.Tensor, ref_text: torch.Tensor) -> torch.Tensor:
        """计算所有跨模态损失"""
        loss_ms = self.cross_modal_contrast(ms_img, ms_text)
        loss_pan = self.cross_modal_contrast(pan_img, pan_text)
        loss_ref = self.cross_modal_contrast(ref_img, ref_text)

        return (loss_ms + loss_pan + loss_ref) / 3


class DirectionConsistencyLoss:
    """方向一致性损失模块"""

    @staticmethod
    def compute(features_ms: torch.Tensor, features_pan: torch.Tensor, features_out: torch.Tensor,
                text_ms: torch.Tensor, text_pan: torch.Tensor, text_out: torch.Tensor) -> torch.Tensor:
        """计算特征空间中的方向一致性损失[1](@ref)"""

        # 计算图像特征空间中的转移向量
        delta_img_ms = features_out - features_ms
        delta_img_pan = features_out - features_pan

        # 计算文本特征空间中的转移向量
        delta_text_ms = text_out - text_ms
        delta_text_pan = text_out - text_pan

        # 计算余弦相似度
        def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return F.cosine_similarity(a, b, dim=1)

        cos_sim_ms = cosine_similarity(delta_img_ms, delta_text_ms)
        cos_sim_pan = cosine_similarity(delta_img_pan, delta_text_pan)

        # 计算方向损失
        direction_loss = 1 - 0.5 * (torch.mean(cos_sim_ms) + torch.mean(cos_sim_pan))
        return direction_loss


class CLIPMultimodalTrainer:
    """CLIP多模态训练器主类"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        self.setup_training_tools()

        # 创建目录
        self.create_directories()

        logger.info(f"训练器初始化完成，设备: {self.device}")

    def setup_models(self):
        """初始化所有模型组件"""
        try:
            # 加载CLIP模型
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

            # 初始化适配器模型（需要从models.clip_adapt导入CLIP_Adapter）
            from models.clip_adapt import CLIP_Adapter
            self.adapter_model = CLIP_Adapter(c_in=512, reduction=4).to(self.device)

            # 初始化降维模块
            self.ms_reducer = BottleneckReduction(in_channels=4).to(self.device)
            self.out_reducer = BottleneckReduction(in_channels=4).to(self.device)

            # 冻结CLIP模型参数
            for param in self.clip_model.parameters():
                param.requires_grad = False

            logger.info("模型初始化成功")

        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

    def setup_training_tools(self):
        """设置优化器、损失函数等训练工具"""
        # 优化器
        self.optimizer = Adam([
            {'params': self.adapter_model.parameters()},
            {'params': self.ms_reducer.parameters()},
            {'params': self.out_reducer.parameters()}
        ], lr=self.config.learning_rate)

        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.contrastive_loss = MultimodalContrastiveLoss()
        self.direction_loss = DirectionConsistencyLoss()

        # 混合精度训练
        self.scaler = GradScaler(enabled=self.config.use_amp)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.8)

    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config.model_path, exist_ok=True)
        os.makedirs(self.config.result_path, exist_ok=True)
        os.makedirs(self.config.ms_reducer, exist_ok=True)
        os.makedirs(self.config.out_reducer, exist_ok=True)

    def generate_descriptions(self, batch_size: int) -> Tuple[List[str], List[str], List[str]]:
        """生成图像描述文本[3](@ref)"""
        ms_descriptions = ["A multispectral satellite image" for _ in range(batch_size)]
        pan_descriptions = ["A panchromatic high-resolution image" for _ in range(batch_size)]
        ref_descriptions = [
            "High-quality reference image with spectral consistency and spatial sharpness following Wald's protocol"
            for _ in range(batch_size)
        ]
        return ms_descriptions, pan_descriptions, ref_descriptions

    def extract_multimodal_features(self, ms_pca, pan, ref, batch_size: int) -> Dict[str, torch.Tensor]:
        """提取多模态特征（图像+文本）"""
        # 图像预处理
        resize_transform = transforms.Resize((224, 224))

        ms_resized = resize_transform(ms_pca)
        pan_resized = resize_transform(pan)
        ref_resized = resize_transform(ref)

        # 通道调整（PAN图像复制为3通道）
        pan_3channel = pan_resized.repeat(1, 3, 1, 1)
        ref_3channel = self.out_reducer(ref_resized)

        # 生成文本描述并编码
        ms_descriptions, pan_descriptions, ref_descriptions = self.generate_descriptions(batch_size)

        with torch.no_grad():
            # 文本特征提取
            text_tokens_ms = clip.tokenize(ms_descriptions).to(self.device)
            text_tokens_pan = clip.tokenize(pan_descriptions).to(self.device)
            text_tokens_ref = clip.tokenize(ref_descriptions).to(self.device)

            text_features_ms = self.clip_model.encode_text(text_tokens_ms)
            text_features_pan = self.clip_model.encode_text(text_tokens_pan)
            text_features_ref = self.clip_model.encode_text(text_tokens_ref)

            # 归一化文本特征
            text_features_ms = F.normalize(text_features_ms, p=2, dim=-1)
            text_features_pan = F.normalize(text_features_pan, p=2, dim=-1)
            text_features_ref = F.normalize(text_features_ref, p=2, dim=-1)

            # 图像特征提取
            ms_features = self.clip_model.encode_image(ms_resized)
            pan_features = self.clip_model.encode_image(pan_3channel)
            ref_features = self.clip_model.encode_image(ref_3channel)

            # 归一化图像特征
            ms_features = F.normalize(ms_features, p=2, dim=-1)
            pan_features = F.normalize(pan_features, p=2, dim=-1)
            ref_features = F.normalize(ref_features, p=2, dim=-1)

        return {
            'ms_img': ms_features, 'pan_img': pan_features, 'ref_img': ref_features,
            'ms_text': text_features_ms, 'pan_text': text_features_pan, 'ref_text': text_features_ref
        }

    def compute_total_loss(self, features: Dict[str, torch.Tensor], text_fuse: torch.Tensor,
                           image_fuse: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失函数"""
        # 对比损失
        intra_loss = self.contrastive_loss.intra_modal_loss(
            features['ms_img'], features['pan_img'], features['ref_img'])

        inter_loss = self.contrastive_loss.inter_modal_loss(
            features['ms_img'], features['pan_img'], features['ref_img'],
            features['ms_text'], features['pan_text'], features['ref_text'])

        # 融合损失
        fusion_loss = self.l1_loss(image_fuse, features['ref_img']) + \
                      self.l1_loss(text_fuse, features['ref_text'])

        # 方向一致性损失
        direction_loss = self.direction_loss.compute(
            features['ms_img'], features['pan_img'], features['ref_img'],
            features['ms_text'], features['pan_text'], features['ref_text'])

        # 加权总损失
        total_loss = (
                self.config.loss_weights['intra_modal'] * intra_loss +
                self.config.loss_weights['inter_modal'] * inter_loss +
                self.config.loss_weights['fusion'] * fusion_loss +
                self.config.loss_weights['direction'] * direction_loss
        )

        loss_details = {
            'intra_modal': intra_loss.item(),
            'inter_modal': inter_loss.item(),
            'fusion': fusion_loss.item(),
            'direction': direction_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_details

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.adapter_model.train()
        self.ms_reducer.train()
        self.out_reducer.train()

        epoch_losses = {key: 0.0 for key in ['intra_modal', 'inter_modal', 'fusion', 'direction', 'total']}

        for batch_idx, (ref, pan, ms, ms_pca) in enumerate(train_loader):
            # 数据转移到设备
            ref = ref.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)
            ms_pca = self.ms_reducer(ms)  # 多光谱降维

            # 混合精度训练前向传播
            with autocast(enabled=self.config.use_amp):
                # 提取多模态特征
                features = self.extract_multimodal_features(ms_pca, pan, ref, ref.size(0))

                # 适配器前向传播
                text_fuse, image_fuse, _, _, _, _ = self.adapter_model(
                    features['ms_text'], features['pan_text'],
                    features['ms_img'], features['pan_img']
                )

                # 计算损失
                total_loss, loss_details = self.compute_total_loss(features, text_fuse, image_fuse)

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.adapter_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.ms_reducer.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.out_reducer.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 记录损失
            for key in epoch_losses:
                epoch_losses[key] += loss_details[key]

            # 每10个batch打印一次日志
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx} - "
                    f"Intra: {loss_details['intra_modal']:.4f}, "
                    f"Inter: {loss_details['inter_modal']:.4f}, "
                    f"Fusion: {loss_details['fusion']:.4f}, "
                    f"Direction: {loss_details['direction']:.4f}, "
                    f"Total: {loss_details['total']:.4f}"
                )

        # 计算平均损失
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def save_checkpoint(self, epoch: int, losses: Dict[str, float]):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'adapter_model_state_dict': self.adapter_model.state_dict(),
            'ms_reducer_state_dict': self.ms_reducer.state_dict(),
            'out_reducer_state_dict': self.out_reducer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses
        }

        checkpoint_path = os.path.join(self.config.model_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # 同时保存独立的模型文件
        torch.save(self.adapter_model.state_dict(),
                   os.path.join(self.config.model_path, f'adapter_epoch_{epoch}.pth'))
        torch.save(self.ms_reducer.state_dict(),
                   os.path.join(self.config.ms_reducer, f'ms_reducer_epoch_{epoch}.pth'))
        torch.save(self.out_reducer.state_dict(),
                   os.path.join(self.config.out_reducer, f'out_reducer_epoch_{epoch}.pth'))

        logger.info(f"检查点已保存: {checkpoint_path}")

    def train(self, train_loader):
        """主训练循环"""
        logger.info("开始训练...")
        writer = SummaryWriter("logs/training")

        best_loss = float('inf')

        for epoch in range(self.config.epochs):
            logger.info(f"开始Epoch {epoch}/{self.config.epochs}")

            epoch_start_time = datetime.now()

            # 训练一个epoch
            epoch_losses = self.train_epoch(train_loader, epoch)

            # 更新学习率
            self.scheduler.step()

            # 记录到TensorBoard
            for loss_name, loss_value in epoch_losses.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value, epoch)

            # 保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, epoch_losses)

            # 保存最佳模型
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                self.save_checkpoint(epoch, epoch_losses)

            epoch_duration = datetime.now() - epoch_start_time

            logger.info(
                f"Epoch {epoch} 完成, 耗时: {epoch_duration}\n"
                f"损失 - 模态内: {epoch_losses['intra_modal']:.4f}, "
                f"模态间: {epoch_losses['inter_modal']:.4f}, "
                f"融合: {epoch_losses['fusion']:.4f}, "
                f"方向: {epoch_losses['direction']:.4f}, "
                f"总计: {epoch_losses['total']:.4f}"
            )

        writer.close()
        logger.info("训练完成!")


def main():
    """主函数"""
    try:
        # 初始化配置
        config = TrainingConfig()

        # 初始化训练器
        trainer = CLIPMultimodalTrainer(config)

        # 加载数据（需要实现Datain类）
        from data import Datain
        data_train = Datain(config.data_path_mat_train, config.scale_ratio, train=None)
        train_loader = torch.utils.data.DataLoader(
            data_train, batch_size=config.batch_size, shuffle=True, num_workers=2
        )

        # 开始训练
        trainer.train(train_loader)

    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        raise


if __name__ == '__main__':
    main()