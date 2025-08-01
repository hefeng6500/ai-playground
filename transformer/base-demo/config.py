"""配置文件 - 定义模型超参数和训练配置"""

import torch

class Config:
    """Transformer 翻译模型配置类"""
    
    # 数据相关配置
    DATA_DIR = "data/raw"
    EN_FILE = "computer_en_26k.jsonl"
    ZH_FILE = "computer_zh_26k.jsonl"
    PROCESSED_DIR = "data/processed"
    
    # 模型超参数（调整为更小的模型以便快速训练）
    D_MODEL = 256          # 模型维度
    N_HEADS = 4            # 多头注意力头数
    N_LAYERS = 3           # 编码器/解码器层数
    D_FF = 512             # 前馈网络维度
    DROPOUT = 0.1          # Dropout 概率
    MAX_SEQ_LEN = 128      # 最大序列长度
    
    # 词汇表配置
    VOCAB_SIZE = 32000     # 词汇表大小
    PAD_TOKEN = "<pad>"    # 填充标记
    UNK_TOKEN = "<unk>"    # 未知词标记
    BOS_TOKEN = "<bos>"    # 句子开始标记
    EOS_TOKEN = "<eos>"    # 句子结束标记
    
    # 训练超参数（优化收敛性）
    BATCH_SIZE = 32        # 增大批次大小提升训练稳定性
    LEARNING_RATE = 2e-4   # 提高学习率加快收敛
    NUM_EPOCHS = 15        # 增加训练轮数确保充分收敛
    WARMUP_STEPS = 2000    # 减少预热步数更快达到最优学习率
    LABEL_SMOOTHING = 0.1  # 标签平滑
    WEIGHT_DECAY = 1e-4    # 添加权重衰减防止过拟合
    GRAD_CLIP = 1.0        # 梯度裁剪防止梯度爆炸
    EARLY_STOP_PATIENCE = 5  # 早停耐心值（验证损失不改善的epoch数）
    MIN_DELTA = 1e-4       # 最小改善阈值
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存路径
    MODEL_SAVE_PATH = "models/transformer_translator.pth"
    VOCAB_SAVE_PATH = "models/vocab.pkl"
    
    # 日志配置
    LOG_INTERVAL = 100     # 日志打印间隔
    SAVE_INTERVAL = 1000   # 模型保存间隔
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("Transformer 翻译模型配置")
        print("=" * 50)
        print(f"设备: {cls.DEVICE}")
        print(f"模型维度: {cls.D_MODEL}")
        print(f"注意力头数: {cls.N_HEADS}")
        print(f"编码器/解码器层数: {cls.N_LAYERS}")
        print(f"前馈网络维度: {cls.D_FF}")
        print(f"最大序列长度: {cls.MAX_SEQ_LEN}")
        print(f"词汇表大小: {cls.VOCAB_SIZE}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print("=" * 50)