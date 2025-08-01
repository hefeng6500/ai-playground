"""主训练脚本 - 中英文翻译 Transformer 模型"""

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from transformer_model import Transformer
from trainer import Trainer

def setup_environment():
    """设置训练环境"""
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("环境设置完成")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")

def prepare_data():
    """准备训练数据"""
    print("\n" + "="*60)
    print("数据准备阶段")
    print("="*60)
    
    processor = DataProcessor()
    
    # 检查是否已有处理好的词汇表
    if os.path.exists(Config.VOCAB_SAVE_PATH):
        print("发现已存在的词汇表，正在加载...")
        try:
            processor.load_vocabularies()
            print("词汇表加载成功")
        except Exception as e:
            print(f"词汇表加载失败: {e}")
            print("重新构建词汇表...")
            processor = DataProcessor()  # 重新初始化
    
    # 加载原始数据
    try:
        en_sentences, zh_sentences = processor.load_data()
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        print("请确保以下文件存在:")
        print(f"  - {os.path.join(Config.DATA_DIR, Config.EN_FILE)}")
        print(f"  - {os.path.join(Config.DATA_DIR, Config.ZH_FILE)}")
        sys.exit(1)
    except Exception as e:
        print(f"数据加载错误: {e}")
        sys.exit(1)
    
    # 如果词汇表不存在，构建新的词汇表
    if not os.path.exists(Config.VOCAB_SAVE_PATH):
        print("构建词汇表...")
        processor.build_vocabularies(en_sentences, zh_sentences)
        processor.save_vocabularies()
    
    # 创建数据集
    print("创建训练和验证数据集...")
    train_dataset, val_dataset = processor.create_datasets(en_sentences, zh_sentences)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = processor.create_dataloaders(train_dataset, val_dataset)
    
    print(f"数据准备完成:")
    print(f"  - 英文词汇表大小: {len(processor.en_vocab)}")
    print(f"  - 中文词汇表大小: {len(processor.zh_vocab)}")
    print(f"  - 训练样本数: {len(train_dataset)}")
    print(f"  - 验证样本数: {len(val_dataset)}")
    
    return processor, train_loader, val_loader

def create_model(src_vocab_size, tgt_vocab_size):
    """创建 Transformer 模型"""
    print("\n" + "="*60)
    print("模型创建阶段")
    print("="*60)
    
    # 打印配置信息
    Config.print_config()
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.DROPOUT
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型创建完成:")
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 可训练参数数: {trainable_params:,}")
    print(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model

def train_model(model, train_loader, val_loader, src_vocab, tgt_vocab, resume_from=None):
    """训练模型"""
    print("\n" + "="*60)
    print("模型训练阶段")
    print("="*60)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, src_vocab, tgt_vocab)
    
    # 如果指定了恢复训练的检查点
    if resume_from:
        print(f"从检查点恢复训练: {resume_from}")
        trainer.load_model(resume_from)
    
    # 开始训练
    try:
        trainer.train(num_epochs=Config.NUM_EPOCHS)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.save_model("interrupted_model.pth")
        print("模型已保存为 interrupted_model.pth")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        trainer.save_model("error_model.pth")
        print("模型已保存为 error_model.pth")
        raise
    
    return trainer

def test_translation(trainer):
    """测试翻译功能"""
    print("\n" + "="*60)
    print("翻译测试阶段")
    print("="*60)
    
    # 测试句子
    test_sentences = [
        "hello world",
        "how are you",
        "what is your name",
        "i love programming",
        "artificial intelligence is amazing"
    ]
    
    print("翻译测试结果:")
    print("-" * 40)
    
    for sentence in test_sentences:
        try:
            translation = trainer.translate_sentence(sentence)
            print(f"EN: {sentence}")
            print(f"ZH: {translation}")
            print("-" * 40)
        except Exception as e:
            print(f"翻译 '{sentence}' 时出错: {e}")
            print("-" * 40)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Transformer 中英文翻译模型训练')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--test-only', action='store_true', help='仅进行测试，不训练')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=Config.LEARNING_RATE, help='学习率')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.epochs != Config.NUM_EPOCHS:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size != Config.BATCH_SIZE:
        Config.BATCH_SIZE = args.batch_size
    if args.learning_rate != Config.LEARNING_RATE:
        Config.LEARNING_RATE = args.learning_rate
    
    print("Transformer 中英文翻译模型")
    print("=" * 60)
    print("这是一个教学用的 Transformer 实现，展示了:")
    print("  ✓ 完整的编码器-解码器架构")
    print("  ✓ 多头注意力机制")
    print("  ✓ 位置编码")
    print("  ✓ 残差连接和层归一化")
    print("  ✓ 标签平滑和学习率调度")
    print("  ✓ 中英文翻译任务")
    print("=" * 60)
    
    try:
        # 1. 环境设置
        setup_environment()
        
        # 2. 数据准备
        processor, train_loader, val_loader = prepare_data()
        
        # 3. 模型创建
        model = create_model(
            src_vocab_size=len(processor.en_vocab),
            tgt_vocab_size=len(processor.zh_vocab)
        )
        
        # 4. 训练或测试
        if args.test_only:
            # 仅测试模式
            if args.resume:
                trainer = Trainer(model, train_loader, val_loader, 
                                processor.en_vocab, processor.zh_vocab)
                trainer.load_model(args.resume)
                test_translation(trainer)
            else:
                print("测试模式需要指定 --resume 参数来加载模型")
        else:
            # 训练模式
            trainer = train_model(model, train_loader, val_loader,
                                processor.en_vocab, processor.zh_vocab, args.resume)
            
            # 训练完成后进行测试
            test_translation(trainer)
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"\n程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()