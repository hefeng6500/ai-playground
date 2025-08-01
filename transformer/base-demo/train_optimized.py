#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的训练脚本 - 使用改进的超参数和早停机制
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

from config import Config
from data_processor import DataProcessor, TranslationDataset
from transformer_model import Transformer, count_parameters
from trainer import Trainer

def main():
    """主训练函数"""
    print("=" * 70)
    print("Transformer 英中翻译模型 - 优化训练")
    print("=" * 70)
    
    # 打印配置信息
    Config.print_config()
    
    # 检查设备
    print(f"\n使用设备: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 数据处理
    print("\n正在处理数据...")
    processor = DataProcessor()
    
    # 加载数据
    print("加载训练数据...")
    en_sentences, zh_sentences = processor.load_data()
    
    # 加载或构建词汇表
    if os.path.exists(Config.VOCAB_SAVE_PATH):
        print("加载现有词汇表...")
        processor.load_vocabularies()
    else:
        print("构建新词汇表...")
        processor.build_vocabularies(en_sentences, zh_sentences)
        processor.save_vocabularies()
    
    print(f"词汇表大小 - 英文: {len(processor.en_vocab)}, 中文: {len(processor.zh_vocab)}")
    
    # 准备数据集
    print("\n准备训练和验证数据集...")
    train_dataset, val_dataset = processor.create_datasets(en_sentences, zh_sentences)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Windows 兼容性
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    print("\n创建 Transformer 模型...")
    model = Transformer(
        src_vocab_size=len(processor.en_vocab),
        tgt_vocab_size=len(processor.zh_vocab),
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.DROPOUT
    )
    
    # 打印模型信息
    total_params = count_parameters(model)
    print(f"模型参数总数: {total_params:,}")
    print(f"模型大小: {total_params * 4 / 1024**2:.1f} MB (float32)")
    
    # 创建训练器
    print("\n初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab=processor.en_vocab,
        tgt_vocab=processor.zh_vocab
    )
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始优化训练过程...")
    print(f"最大训练轮数: {Config.NUM_EPOCHS}")
    print(f"早停耐心值: {Config.EARLY_STOP_PATIENCE}")
    print(f"学习率: {Config.LEARNING_RATE}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print(f"权重衰减: {Config.WEIGHT_DECAY}")
    print(f"梯度裁剪: {Config.GRAD_CLIP}")
    print("=" * 70)
    
    try:
        trainer.train(Config.NUM_EPOCHS)
        print("\n训练成功完成！")
        
        # 测试翻译
        print("\n测试翻译功能...")
        test_sentences = [
            "hello world",
            "how are you",
            "good morning",
            "thank you"
        ]
        
        for sentence in test_sentences:
            try:
                translation = trainer.translate_sentence(sentence)
                print(f"英文: {sentence}")
                print(f"中文: {translation}")
                print("-" * 30)
            except Exception as e:
                print(f"翻译 '{sentence}' 时出错: {e}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n程序结束")

if __name__ == "__main__":
    main()