#!/usr/bin/env python3
"""调试推理过程中的张量形状问题"""

import torch
from config import Config
from data_processor import DataProcessor
from transformer_model import Transformer

def debug_inference():
    """调试推理过程"""
    print("开始调试推理过程...")
    
    # 加载数据处理器
    processor = DataProcessor()
    processor.load_vocabularies()
    
    print(f"词汇表大小 - 英文: {len(processor.en_vocab)}, 中文: {len(processor.zh_vocab)}")
    
    # 加载模型
    checkpoint = torch.load('models/best_model_epoch_3.pth', map_location='cpu')
    config = checkpoint['config']
    
    print(f"模型配置: {config}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 准备测试输入
    text = "hello world"
    words = text.lower().strip().split()
    
    # 转换为索引序列
    src_indices = [processor.en_vocab.bos_idx]
    for word in words:
        src_indices.append(processor.en_vocab.get_idx(word))
    src_indices.append(processor.en_vocab.eos_idx)
    
    print(f"原始序列长度: {len(src_indices)}")
    print(f"原始序列: {src_indices}")
    
    # 填充到固定长度
    if len(src_indices) > Config.MAX_SEQ_LEN:
        src_indices = src_indices[:Config.MAX_SEQ_LEN]
    else:
        src_indices += [processor.en_vocab.pad_idx] * (Config.MAX_SEQ_LEN - len(src_indices))
    
    print(f"填充后序列长度: {len(src_indices)}")
    
    # 转换为张量
    src = torch.tensor([src_indices])
    print(f"输入张量形状: {src.shape}")
    
    # 测试编码器
    with torch.no_grad():
        try:
            print("\n测试编码器...")
            src_mask = model.create_padding_mask(src, processor.en_vocab.pad_idx)
            print(f"源掩码形状: {src_mask.shape}")
            
            encoder_output = model.encode(src, src_mask)
            print(f"编码器输出形状: {encoder_output.shape}")
            
            print("\n测试解码器...")
            # 初始化目标序列
            tgt = torch.tensor([[processor.zh_vocab.bos_idx]])
            print(f"初始目标序列形状: {tgt.shape}")
            
            # 创建目标掩码
            tgt_len = tgt.size(1)
            tgt_mask = model.create_causal_mask(tgt_len)
            print(f"目标掩码形状: {tgt_mask.shape}")
            
            # 测试解码
            decoder_output = model.decode(tgt, encoder_output, tgt_mask, src_mask)
            print(f"解码器输出形状: {decoder_output.shape}")
            
            print("\n调试成功！")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_inference()