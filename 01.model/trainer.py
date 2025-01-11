from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import logging
import json
from tqdm import tqdm
from scipy import stats
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class Trainer:
    def __init__(self,
                 model, 
                 train_loader, 
                 test_loader, 
                 regression_criterion, 
                 classification_criterion,
                 optimizer, 
                 device, 
                 log_dir, 
                 save_dir,
                 patience=30):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.regression_criterion = regression_criterion
        self.classification_criterion = classification_criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir
        self.patience = patience
        
       # 创建日志和模型保存目录
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置TensorBoard
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(self.log_dir / current_time)
        
        # 设置日志记录器
        self.setup_logger(current_time)
        
        
        # 初始化指标
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.validation_losses = []
        self.current_epoch = 0

        # 记录配置
        self.save_config({
            'patience': patience,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        })

    def setup_logger(self, current_time):
        """设置日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / f'training_{current_time}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_config(self, config):
        """保存训练配置"""
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch in progress_bar:
            loss = self._process_batch(batch, is_training=True)
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型性能"""
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        class_labels = []  # 新增: 收集分类标签
        class_true = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Validating'):
                total_loss, fev1_pred, class_prob, class_label, fev1_targets, disease_targets = self._process_batch(batch, is_training=False)
                val_loss += total_loss.item()
                predictions.extend(fev1_pred.cpu().numpy())
                targets.extend(fev1_targets.cpu().numpy())
                class_labels.extend(class_label.cpu().numpy())  # 新增: 收集分类标签
                class_true.extend(disease_targets.cpu().numpy())  # 新增: 收集分类标签

                
        val_loss = val_loss / len(self.test_loader)
        metrics = self._calculate_metrics(predictions, targets)
        return val_loss, metrics, predictions, targets, class_labels, class_true  # 新增: 返回分类标签
    
    def _process_batch(self, batch, is_training=True):
        """处理一个批次的数据"""
        # 获取数据
        expr_tensor = batch['expr'].to(self.device)
        expr_mask = batch['mask'].to(self.device)
        disease_tensor = batch['label'].unsqueeze(1).to(self.device)  # 回归目标
        targets = batch['target'].unsqueeze(1).to(self.device)  # 二分类目标
        meta_tensor = batch['meta'].to(self.device)

        if is_training:
            self.optimizer.zero_grad()
            
        # 模型输出
        fev1_pred, class_pred, combined_embed = self.model(expr_tensor, meta_tensor, expr_mask)
        # MSE loss 
        regression_loss = self.regression_criterion(fev1_pred, disease_tensor)
        # BCE loss
        classification_loss = self.classification_criterion(class_pred, targets)
        
        # 合并损失
        total_loss = 1*regression_loss + 1*classification_loss  # 可以加权，例如：a*regression_loss + b*classification_loss

        if is_training:
            total_loss.backward()
            self.optimizer.step()
            return total_loss
        else:
            # 推理阶段，计算分类概率和标签
            class_prob = torch.sigmoid(class_pred)  # 转换为概率
            class_label = (class_prob > 0.5).long()  # 根据概率计算分类标签
            return total_loss, fev1_pred, class_prob, class_label, disease_tensor, targets
    
    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        return {
            'pearson': pearsonr(targets, predictions)[0],
            'spearman': spearmanr(targets, predictions)[0],
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions))
        }
    
    # def _save_checkpoint(self, epoch, loss, metrics, is_best=False):
    #     """保存检查点"""
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'loss': loss,
    #         'metrics': metrics,
    #         'best_val_loss': self.best_val_loss
    #     }
        
    #     # 保存最新检查点
    #     latest_path = self.save_dir / 'latest_checkpoint.pth'
    #     torch.save(checkpoint, latest_path)
        
    #     if is_best:
    #         best_path = self.save_dir / 'best_model.pth'
    #         torch.save(checkpoint, best_path)
    #         self.logger.info(f'Saved best model checkpoint to {best_path}')

    def _save_checkpoint(self, epoch, loss, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存模型权重
        weights_path = self.save_dir / 'latest_weights.pth'
        torch.save(self.model.state_dict(), weights_path)
        
        # 保存最新检查点（不包括模型本身，只是元数据）
        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        if is_best:
            # 保存最佳模型权重
            best_weights_path = self.save_dir / 'best_weights.pth'
            torch.save(self.model.state_dict(), best_weights_path)
            self.logger.info(f'Saved best model weights to {best_weights_path}')
    
    def train(self, num_epochs):
        """训练模型"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.best_epoch = 0
        self.best_cor= 0
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # 训练epoch
                train_loss = self.train_epoch()
                
                # 验证
                val_loss, metrics, predictions, targets, class_labels, class_true = self.validate()
                
                # 记录损失和指标
                self._log_metrics(epoch, train_loss, val_loss, metrics)
                
                # 可视化
                if (epoch + 1) % 2 == 0:  # 每5个epoch可视化一次
                    self._visualize(epoch, predictions, targets, class_labels, class_true)
                
                # 记录cor
                pearson_corr, pearson_p = stats.pearsonr(targets, predictions)
                spearman_corr, spearman_p = stats.spearmanr(targets, predictions)
                # 记录ACC
                class_true = np.concatenate(class_true).ravel()  # 从批次中获取真实标签
                class_predictions = np.concatenate(class_labels).ravel()  # 从模型输出中获取预测标签
                accuracy = (class_predictions == class_true).mean()  # 计算准确率

                if pearson_corr > spearman_corr:
                    cor = pearson_corr
                else:
                    cor = spearman_corr
                if cor > self.best_cor:
                    self.best_cor = cor
                    self.best_epoch = epoch+1
                    self.acc = accuracy


                # 早停检查
                if self._check_early_stopping(epoch, val_loss, metrics):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
        finally:
            self.writer.close()
            self.logger.info("Training completed")
        return (self.best_cor, self.best_epoch, self.acc)
    
    def _log_metrics(self, epoch, train_loss, val_loss, metrics):
        """记录指标"""
        # 更新历史记录
        self.train_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        
        # TensorBoard记录
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # 记录其他指标
        for name, value in metrics.items():
            self.writer.add_scalar(f'Metrics/{name}', value, epoch)
        
        # 日志输出
        self.logger.info(
            f"Epoch [{epoch + 1}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Metrics: {metrics}"
        )
        
    def _check_early_stopping(self, epoch, val_loss, metrics):
        """检查是否需要早停"""
        is_best = val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self._save_checkpoint(epoch, val_loss, metrics, is_best=True)
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                return True
                
        # 保存最新检查点
        self._save_checkpoint(epoch, val_loss, metrics, is_best=False)
        return False

    def _visualize(self, epoch, predictions, targets, class_labels, class_ture):
        """可视化训练过程"""
        # 1. 损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title(f'Loss Curves (Epoch {epoch + 1})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        self.writer.add_figure('Loss Curves', plt.gcf(), epoch)
        plt.close()
        
        # 2. 预测散点图
        predictions = np.concatenate(predictions).ravel()
        targets = np.concatenate(targets).ravel()
        pearson_corr, pearson_p = stats.pearsonr(targets, predictions)
        spearman_corr, spearman_p = stats.spearmanr(targets, predictions)

        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], 
                [min(targets), max(targets)], 
                'r--', label='Perfect Prediction')
        plt.title(f'Predictions vs Targets (Epoch {epoch + 1})\nPearson r: {pearson_corr:.4f}, Spearman r: {spearman_corr:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.legend()
        plt.grid(True)
        self.writer.add_figure('Predictions vs Targets', plt.gcf(), epoch)
        plt.close()
        

        # 3. 分类任务准确率
        class_ture = np.concatenate(class_ture).ravel()  # 从批次中获取真实标签
        class_predictions = np.concatenate(class_labels).ravel()  # 从模型输出中获取预测标签
        accuracy = (class_predictions == class_ture).mean()  # 计算准确率

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')  # 训练损失
        plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Accuracy: {accuracy:.4f}')  # 准确率水平线
        plt.title(f'Classification Accuracy (Epoch {epoch + 1})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        self.writer.add_figure('Classification Accuracy', plt.gcf(), epoch)
        plt.close()

        # 4. 参数分布
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/{name}', 
                                        param.data.cpu().numpy(), 
                                        epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', 
                                            param.grad.cpu().numpy(), 
                                            epoch)
