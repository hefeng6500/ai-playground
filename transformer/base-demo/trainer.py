"""训练器模块 - 负责模型训练、验证和保存"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import Config
from transformer_model import Transformer, count_parameters
from data_processor import Vocabulary

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数 - 提高模型泛化能力"""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑损失
        
        参数:
            pred: 预测logits，形状为 [batch_size * seq_len, vocab_size]
            target: 目标标签，形状为 [batch_size * seq_len]
        """
        # 创建平滑标签
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # 排除pad和目标词
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 忽略填充标记
        mask = (target != self.ignore_index)
        true_dist = true_dist * mask.unsqueeze(1).float()
        
        # 计算KL散度
        kl_div = torch.sum(true_dist * (torch.log(true_dist + 1e-12) - 
                                       torch.log_softmax(pred, dim=1)), dim=1)
        
        return torch.mean(kl_div * mask.float())

class NoamOptimizer:
    """Noam 学习率调度器 - Transformer 论文中使用的学习率调度"""
    
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        """更新学习率并执行优化步骤"""
        self.step_num += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
    
    def _get_lr(self) -> float:
        """计算当前学习率"""
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

class Trainer:
    """Transformer 模型训练器"""
    
    def __init__(self, model: Transformer, train_loader: DataLoader, 
                 val_loader: DataLoader, src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = Config.DEVICE
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = LabelSmoothingLoss(
            vocab_size=len(tgt_vocab),
            smoothing=Config.LABEL_SMOOTHING,
            ignore_index=tgt_vocab.pad_idx
        )
        
        # 优化器（添加权重衰减）
        self.base_optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-4
        )
        
        # Noam 学习率调度器
        self.optimizer = NoamOptimizer(
            self.base_optimizer,
            d_model=Config.D_MODEL,
            warmup_steps=Config.WARMUP_STEPS
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"模型参数数量: {count_parameters(self.model):,}")
        print(f"训练设备: {self.device}")
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="训练中")
        
        for batch_idx, (src, tgt_input, tgt_output) in enumerate(progress_bar):
            # 移动数据到设备
            src = src.to(self.device)
            tgt_input = tgt_input.to(self.device)
            tgt_output = tgt_output.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt_input, 
                              src_pad_idx=self.src_vocab.pad_idx,
                              tgt_pad_idx=self.tgt_vocab.pad_idx)
            
            # 计算损失
            # 重塑输出和目标以计算损失
            output = output.view(-1, output.size(-1))  # [batch_size * seq_len, vocab_size]
            tgt_output = tgt_output.view(-1)  # [batch_size * seq_len]
            
            loss = self.criterion(output, tgt_output)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP)
            
            # 优化器步骤
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            current_lr = self.optimizer.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # 记录学习率
            if batch_idx % Config.LOG_INTERVAL == 0:
                self.learning_rates.append(current_lr)
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="验证中")
            
            for src, tgt_input, tgt_output in progress_bar:
                # 移动数据到设备
                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_output = tgt_output.to(self.device)
                
                # 前向传播
                output = self.model(src, tgt_input,
                                  src_pad_idx=self.src_vocab.pad_idx,
                                  tgt_pad_idx=self.tgt_vocab.pad_idx)
                
                # 计算损失
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.view(-1)
                
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
                
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int = Config.NUM_EPOCHS):
        """完整训练流程（带早停机制）"""
        print("开始训练...")
        print("=" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 计算困惑度
            train_ppl = math.exp(min(train_loss, 10))  # 防止溢出
            val_ppl = math.exp(min(val_loss, 10))
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.4f} (困惑度: {train_ppl:.2f})")
            print(f"  验证损失: {val_loss:.4f} (困惑度: {val_ppl:.2f})")
            print(f"  用时: {epoch_time:.2f}秒")
            
            # 检查是否为最佳模型
            if val_loss < best_val_loss - Config.MIN_DELTA:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch + 1}.pth")
                print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  验证损失未改善 ({patience_counter}/{Config.EARLY_STOP_PATIENCE})")
            
            # 早停检查
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"\n早停触发！最佳模型在第 {best_epoch} 轮，验证损失: {best_val_loss:.4f}")
                break
            
            print("-" * 60)
        
        print("训练完成！")
        print(f"最终最佳验证损失: {best_val_loss:.4f} (第 {best_epoch} 轮)")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename: str):
        """保存模型"""
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
        
        save_path = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.base_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'src_vocab_size': self.model.src_vocab_size,
                'tgt_vocab_size': self.model.tgt_vocab_size,
                'd_model': Config.D_MODEL,
                'n_heads': Config.N_HEADS,
                'n_layers': Config.N_LAYERS,
                'd_ff': Config.D_FF,
                'max_seq_len': Config.MAX_SEQ_LEN,
                'dropout': Config.DROPOUT
            }
        }, save_path)
    
    def load_model(self, filename: str):
        """加载模型"""
        load_path = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), filename)
        
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            
            print(f"模型已从 {load_path} 加载")
        else:
            print(f"模型文件 {load_path} 不存在")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_losses or not self.val_losses:
            print("没有训练历史数据可绘制")
            return
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='训练损失')
        plt.plot(epochs, self.val_losses, 'r-', label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 困惑度曲线
        plt.subplot(1, 3, 2)
        train_ppl = [math.exp(min(loss, 10)) for loss in self.train_losses]
        val_ppl = [math.exp(min(loss, 10)) for loss in self.val_losses]
        plt.plot(epochs, train_ppl, 'b-', label='训练困惑度')
        plt.plot(epochs, val_ppl, 'r-', label='验证困惑度')
        plt.title('训练和验证困惑度')
        plt.xlabel('Epoch')
        plt.ylabel('困惑度')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线
        if self.learning_rates:
            plt.subplot(1, 3, 3)
            plt.plot(self.learning_rates)
            plt.title('学习率变化')
            plt.xlabel('步骤')
            plt.ylabel('学习率')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练曲线已保存为 training_curves.png")
    
    def translate_sentence(self, sentence: str, max_len: int = 100) -> str:
        """翻译单个句子"""
        self.model.eval()
        
        # 预处理输入句子
        words = sentence.lower().split()
        src_indices = [self.src_vocab.bos_idx] + \
                     [self.src_vocab.get_idx(word) for word in words] + \
                     [self.src_vocab.eos_idx]
        
        # 填充到固定长度
        if len(src_indices) > Config.MAX_SEQ_LEN:
            src_indices = src_indices[:Config.MAX_SEQ_LEN]
        else:
            src_indices += [self.src_vocab.pad_idx] * (Config.MAX_SEQ_LEN - len(src_indices))
        
        # 转换为张量
        src = torch.tensor([src_indices]).to(self.device)
        
        # 生成翻译
        with torch.no_grad():
            generated = self.model.generate(
                src,
                src_pad_idx=self.src_vocab.pad_idx,
                tgt_bos_idx=self.tgt_vocab.bos_idx,
                tgt_eos_idx=self.tgt_vocab.eos_idx,
                max_len=max_len
            )
        
        # 解码生成的序列
        generated_indices = generated[0].cpu().tolist()
        translated_chars = []
        
        for idx in generated_indices[1:]:  # 跳过BOS标记
            if idx == self.tgt_vocab.eos_idx:  # 遇到EOS停止
                break
            if idx != self.tgt_vocab.pad_idx:  # 跳过填充标记
                translated_chars.append(self.tgt_vocab.get_word(idx))
        
        return ''.join(translated_chars)

def main():
    """测试训练器"""
    from data_processor import DataProcessor
    
    # 加载数据
    processor = DataProcessor()
    en_sentences, zh_sentences = processor.load_data()
    
    # 构建词汇表
    processor.build_vocabularies(en_sentences, zh_sentences)
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset = processor.create_datasets(en_sentences, zh_sentences)
    train_loader, val_loader = processor.create_dataloaders(train_dataset, val_dataset)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(processor.en_vocab),
        tgt_vocab_size=len(processor.zh_vocab)
    )
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, processor.en_vocab, processor.zh_vocab)
    
    # 开始训练
    trainer.train(num_epochs=2)  # 测试用较少的epoch
    
    # 测试翻译
    test_sentence = "hello world"
    translation = trainer.translate_sentence(test_sentence)
    print(f"\n翻译测试:")
    print(f"输入: {test_sentence}")
    print(f"输出: {translation}")

if __name__ == "__main__":
    main()