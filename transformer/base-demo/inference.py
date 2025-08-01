"""推理脚本 - 使用训练好的模型进行翻译"""

import os
import torch
import argparse
from config import Config
from data_processor import DataProcessor
from transformer_model import Transformer
from trainer import Trainer

class TranslatorInference:
    """翻译推理类"""
    
    def __init__(self, model_path: str):
        self.device = Config.DEVICE
        self.model_path = model_path
        
        # 加载词汇表
        print("正在加载词汇表...")
        self.processor = DataProcessor()
        
        if not os.path.exists(Config.VOCAB_SAVE_PATH):
            raise FileNotFoundError(f"词汇表文件不存在: {Config.VOCAB_SAVE_PATH}")
        
        self.processor.load_vocabularies()
        print(f"词汇表加载完成 - 英文: {len(self.processor.en_vocab)}, 中文: {len(self.processor.zh_vocab)}")
        
        # 加载模型
        print("正在加载模型...")
        self.model = self._load_model()
        print("模型加载完成")
    
    def _load_model(self) -> Transformer:
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 从检查点获取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
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
        else:
            # 使用默认配置
            model = Transformer(
                src_vocab_size=len(self.processor.en_vocab),
                tgt_vocab_size=len(self.processor.zh_vocab)
            )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def translate(self, text: str, max_length: int = 100) -> str:
        """翻译单个句子"""
        # 预处理输入
        words = text.lower().strip().split()
        if not words:
            return ""
        
        # 转换为索引序列
        src_indices = [self.processor.en_vocab.bos_idx]
        for word in words:
            src_indices.append(self.processor.en_vocab.get_idx(word))
        src_indices.append(self.processor.en_vocab.eos_idx)
        
        # 填充到固定长度
        if len(src_indices) > Config.MAX_SEQ_LEN:
            src_indices = src_indices[:Config.MAX_SEQ_LEN]
        else:
            src_indices += [self.processor.en_vocab.pad_idx] * (Config.MAX_SEQ_LEN - len(src_indices))
        
        # 转换为张量
        src = torch.tensor([src_indices]).to(self.device)
        
        # 生成翻译
        with torch.no_grad():
            generated = self.model.generate(
                src,
                src_pad_idx=self.processor.en_vocab.pad_idx,
                tgt_bos_idx=self.processor.zh_vocab.bos_idx,
                tgt_eos_idx=self.processor.zh_vocab.eos_idx,
                max_len=max_length
            )
        
        # 解码生成的序列
        generated_indices = generated[0].cpu().tolist()
        translated_chars = []
        
        for idx in generated_indices[1:]:  # 跳过BOS标记
            if idx == self.processor.zh_vocab.eos_idx:  # 遇到EOS停止
                break
            if idx not in [self.processor.zh_vocab.pad_idx, self.processor.zh_vocab.unk_idx]:
                char = self.processor.zh_vocab.get_word(idx)
                if char not in ['<pad>', '<unk>', '<bos>', '<eos>']:
                    translated_chars.append(char)
        
        return ''.join(translated_chars)
    
    def translate_batch(self, texts: list, max_length: int = 100) -> list:
        """批量翻译"""
        results = []
        for text in texts:
            try:
                translation = self.translate(text, max_length)
                results.append(translation)
            except Exception as e:
                print(f"翻译 '{text}' 时出错: {e}")
                results.append("")
        return results
    
    def interactive_translate(self):
        """交互式翻译"""
        print("\n" + "="*60)
        print("交互式翻译模式")
        print("输入英文句子，按回车获得中文翻译")
        print("输入 'quit' 或 'exit' 退出")
        print("="*60)
        
        while True:
            try:
                text = input("\n英文: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not text:
                    continue
                
                print("翻译中...")
                translation = self.translate(text)
                
                if translation:
                    print(f"中文: {translation}")
                else:
                    print("翻译失败，请尝试其他句子")
                    
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='Transformer 翻译推理')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--text', type=str, help='要翻译的文本')
    parser.add_argument('--interactive', action='store_true', help='交互式翻译模式')
    parser.add_argument('--batch-file', type=str, help='批量翻译文件（每行一个句子）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--max-length', type=int, default=100, help='最大生成长度')
    
    args = parser.parse_args()
    
    try:
        # 创建翻译器
        translator = TranslatorInference(args.model)
        
        if args.interactive:
            # 交互式模式
            translator.interactive_translate()
            
        elif args.text:
            # 单句翻译
            print(f"输入: {args.text}")
            translation = translator.translate(args.text, args.max_length)
            print(f"输出: {translation}")
            
        elif args.batch_file:
            # 批量翻译
            if not os.path.exists(args.batch_file):
                print(f"文件不存在: {args.batch_file}")
                return
            
            print(f"正在从 {args.batch_file} 读取句子...")
            
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"共 {len(texts)} 个句子，开始翻译...")
            
            translations = translator.translate_batch(texts, args.max_length)
            
            # 输出结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for text, translation in zip(texts, translations):
                        f.write(f"{text}\t{translation}\n")
                print(f"翻译结果已保存到: {args.output}")
            else:
                print("\n翻译结果:")
                print("-" * 60)
                for text, translation in zip(texts, translations):
                    print(f"EN: {text}")
                    print(f"ZH: {translation}")
                    print("-" * 60)
        
        else:
            # 默认测试句子
            test_sentences = [
                "hello world",
                "how are you today",
                "what is your name",
                "i love programming",
                "artificial intelligence is amazing",
                "the weather is nice today",
                "can you help me",
                "thank you very much"
            ]
            
            print("使用默认测试句子:")
            print("=" * 60)
            
            for sentence in test_sentences:
                translation = translator.translate(sentence, args.max_length)
                print(f"EN: {sentence}")
                print(f"ZH: {translation}")
                print("-" * 40)
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()