# Transformer 中英文翻译 Demo

这是一个完整的 Transformer 模型实现，用于中英文翻译任务。本项目专为教学设计，代码结构清晰，注释详尽，展示了 Transformer 架构的核心特征。

## 🎯 项目特点

### Transformer 核心特征展示
- ✅ **多头注意力机制** (Multi-Head Attention)
- ✅ **位置编码** (Positional Encoding)
- ✅ **编码器-解码器架构** (Encoder-Decoder)
- ✅ **残差连接** (Residual Connections)
- ✅ **层归一化** (Layer Normalization)
- ✅ **前馈神经网络** (Feed-Forward Networks)
- ✅ **掩码机制** (Masking)
- ✅ **标签平滑** (Label Smoothing)
- ✅ **学习率调度** (Learning Rate Scheduling)

### 教学友好设计
- 📚 详尽的代码注释，解释每个组件的作用
- 🏗️ 模块化设计，便于理解和修改
- 📊 训练过程可视化
- 🔧 完整的配置管理
- 🚀 一键启动训练

## 📁 项目结构

```
transformer/base-demo/
├── data/
│   └── raw/
│       ├── computer_en_26k.jsonl    # 英文数据集
│       └── computer_zh_26k.jsonl    # 中文数据集
├── models/                          # 模型保存目录
├── config.py                        # 配置文件
├── data_processor.py               # 数据处理模块
├── transformer_model.py            # Transformer 模型实现
├── trainer.py                      # 训练器模块
├── main.py                         # 主训练脚本
├── requirements.txt                # 依赖包列表
└── README.md                       # 项目说明
```

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于 GPU 加速)

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 数据准备
确保数据文件位于正确位置：
- `data/raw/computer_en_26k.jsonl`
- `data/raw/computer_zh_26k.jsonl`

### 2. 开始训练
```bash
python main.py
```

### 3. 自定义训练参数
```bash
# 指定训练轮数
python main.py --epochs 20

# 指定批次大小
python main.py --batch-size 64

# 指定学习率
python main.py --learning-rate 0.0001

# 从检查点恢复训练
python main.py --resume best_model_epoch_10.pth

# 仅测试模式
python main.py --test-only --resume best_model_epoch_10.pth
```

## 🏗️ 架构详解

### 1. Transformer 整体架构

```
输入序列 → 编码器 → 编码器输出
                      ↓
目标序列 → 解码器 ← 编码器输出 → 输出概率分布
```

### 2. 编码器 (Encoder)

每个编码器层包含：
1. **多头自注意力** - 捕获序列内部依赖关系
2. **残差连接 + 层归一化** - 稳定训练过程
3. **前馈神经网络** - 非线性变换
4. **残差连接 + 层归一化** - 再次稳定训练

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 3. 解码器 (Decoder)

每个解码器层包含：
1. **掩码自注意力** - 防止看到未来信息
2. **编码器-解码器注意力** - 关注源序列信息
3. **前馈神经网络** - 非线性变换

```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # 掩码自注意力
        self_attn_output = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 编码器-解码器注意力
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

### 4. 多头注意力机制

注意力机制的核心公式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

多头注意力将输入投影到多个子空间：
```python
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### 5. 位置编码

由于 Transformer 没有循环结构，需要位置编码来提供位置信息：

```python
# 偶数位置使用 sin，奇数位置使用 cos
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

## 📊 训练监控

训练过程中会显示：
- 训练损失和验证损失
- 困惑度 (Perplexity)
- 学习率变化
- 训练时间

训练完成后会生成训练曲线图：`training_curves.png`

## 🎛️ 配置说明

主要超参数在 `config.py` 中配置：

```python
class Config:
    # 模型超参数
    D_MODEL = 512          # 模型维度
    N_HEADS = 8            # 注意力头数
    N_LAYERS = 6           # 编码器/解码器层数
    D_FF = 2048            # 前馈网络维度
    DROPOUT = 0.1          # Dropout 概率
    MAX_SEQ_LEN = 256      # 最大序列长度
    
    # 训练配置
    BATCH_SIZE = 32        # 批次大小
    LEARNING_RATE = 1e-4   # 学习率
    NUM_EPOCHS = 10        # 训练轮数
    WARMUP_STEPS = 4000    # 学习率预热步数
    LABEL_SMOOTHING = 0.1  # 标签平滑
```

## 🔧 高级特性

### 1. 标签平滑 (Label Smoothing)
减少过拟合，提高泛化能力：
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
```

### 2. Noam 学习率调度
遵循原始 Transformer 论文的学习率调度策略：
```python
lr = (d_model ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)
```

### 3. 梯度裁剪
防止梯度爆炸：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 📈 性能优化建议

1. **GPU 加速**：确保安装了 CUDA 版本的 PyTorch
2. **批次大小**：根据 GPU 内存调整 `BATCH_SIZE`
3. **序列长度**：根据数据特点调整 `MAX_SEQ_LEN`
4. **模型大小**：可以调整 `D_MODEL`、`N_HEADS`、`N_LAYERS` 等参数

## 🐛 常见问题

### Q: 训练过程中出现 CUDA 内存不足
A: 减小 `BATCH_SIZE` 或 `MAX_SEQ_LEN`

### Q: 训练损失不下降
A: 检查学习率设置，可能需要调整 `LEARNING_RATE` 或 `WARMUP_STEPS`

### Q: 翻译质量不好
A: 增加训练轮数，或者使用更大的模型（增加 `D_MODEL`、`N_LAYERS`）

### Q: 数据加载失败
A: 确保数据文件路径正确，文件格式为 JSONL

## 📚 学习资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化解释
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - 官方教程

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个教学项目！

## 📄 许可证

本项目采用 MIT 许可证。

---

**注意**：这是一个教学用的简化实现，主要目的是展示 Transformer 的核心概念。在生产环境中，建议使用更成熟的框架如 Hugging Face Transformers。