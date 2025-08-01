# Transformer 中英文翻译项目

这是一个基于 PyTorch 实现的 Transformer 模型中英文翻译教学项目。项目从零开始构建了一个完整的神经机器翻译系统，包括数据预处理、模型训练、评估和预测功能。

## 项目特点

- 🚀 **从零实现**: 完整实现 Transformer 架构，包括位置编码、多头注意力机制等
- 📚 **教学导向**: 代码结构清晰，注释详细，适合学习 Transformer 原理
- 🔧 **完整流程**: 涵盖数据预处理、模型训练、评估和推理的完整机器学习流程
- 💡 **可扩展性**: 模块化设计，易于修改和扩展

## 项目结构

```
translation-transformer/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   │   └── cmn.txt        # 中英文平行语料
│   └── processed/         # 处理后的数据
│       ├── zh_vocab.txt   # 中文词表
│       ├── en_vocab.txt   # 英文词表
│       ├── indexed_train.jsonl  # 训练集
│       └── indexed_test.jsonl   # 测试集
├── src/                    # 源代码
│   ├── config.py          # 配置文件
│   ├── tokenizer.py       # 分词器
│   ├── dataset.py         # 数据集类
│   ├── model.py           # Transformer模型
│   ├── process.py         # 数据预处理
│   ├── train.py           # 训练脚本
│   ├── predict.py         # 预测脚本
│   └── evaluate.py        # 评估脚本
├── models/                 # 模型保存目录
├── logs/                   # 训练日志
└── README.md              # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA (可选，用于GPU加速)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

项目使用中英文平行语料进行训练。请将数据文件 `cmn.txt` 放置在 `data/raw/` 目录下。

数据格式：每行包含英文句子和中文句子，用制表符分隔。

## 使用方法

### 1. 数据预处理

首先需要对原始数据进行预处理，构建词表并将文本转换为索引序列：

```bash
cd src
python process.py
```

这一步会：
- 读取原始数据并清洗
- 划分训练集和测试集
- 构建中英文词表
- 将文本转换为索引序列

### 2. 模型训练

运行训练脚本开始训练模型：

```bash
cd src
python train.py
```

训练过程中会：
- 自动选择可用的设备（GPU/CPU）
- 使用 TensorBoard 记录训练过程
- 自动保存最佳模型

### 3. 模型预测

训练完成后，可以使用模型进行翻译：

```bash
cd src
python predict.py
```

进入交互式翻译模式，输入中文句子即可获得英文翻译结果。

### 4. 查看训练日志

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir=logs
```

## 模型配置

可以在 `src/config.py` 中修改模型和训练参数：

```python
# 模型参数
DIM_MODEL = 128          # 模型维度
NUM_HEADS = 4            # 注意力头数
NUM_ENCODER_LAYERS = 2   # 编码器层数
NUM_DECODER_LAYERS = 2   # 解码器层数

# 训练参数
SEQ_LEN = 32            # 序列长度
BATCH_SIZE = 128        # 批次大小
LEARNING_RATE = 1e-3    # 学习率
EPOCHS = 30             # 训练轮数
```

## 核心组件说明

### Tokenizer
- **ChineseTokenizer**: 中文字符级分词器
- **EnglishTokenizer**: 英文单词级分词器，基于 NLTK

### Model
- **PositionEncoding**: 位置编码层
- **TranslationModel**: 完整的 Transformer 翻译模型

### Dataset
- **TranslationDataset**: 自定义数据集类，支持批量加载

## 注意事项

1. **NLTK 数据**: 首次运行可能需要下载 NLTK 数据包：
   ```python
   import nltk
   nltk.download('punkt_tab')
   ```

2. **内存使用**: 根据可用内存调整 `BATCH_SIZE` 参数

3. **训练时间**: 在 CPU 上训练可能需要较长时间，建议使用 GPU

## 扩展建议

- 添加 BLEU 评估指标
- 实现 Beam Search 解码
- 支持更多语言对
- 添加注意力可视化
- 实现模型量化和优化

## 许可证

本项目仅用于教学目的，请遵循相关开源协议。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！