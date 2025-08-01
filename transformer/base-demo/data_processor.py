"""数据预处理模块 - 处理中英文对话数据"""

import json
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class Vocabulary:
    """词汇表类 - 管理词汇到索引的映射"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # 添加特殊标记
        self.add_word(Config.PAD_TOKEN)
        self.add_word(Config.UNK_TOKEN)
        self.add_word(Config.BOS_TOKEN)
        self.add_word(Config.EOS_TOKEN)
        
        # 特殊标记的索引
        self.pad_idx = self.word2idx[Config.PAD_TOKEN]
        self.unk_idx = self.word2idx[Config.UNK_TOKEN]
        self.bos_idx = self.word2idx[Config.BOS_TOKEN]
        self.eos_idx = self.word2idx[Config.EOS_TOKEN]
    
    def add_word(self, word: str) -> int:
        """添加单词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
        return self.word2idx[word]
    
    def get_idx(self, word: str) -> int:
        """获取单词的索引"""
        return self.word2idx.get(word, self.unk_idx)
    
    def get_word(self, idx: int) -> str:
        """获取索引对应的单词"""
        return self.idx2word.get(idx, Config.UNK_TOKEN)
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def build_vocab(self, sentences: List[str], max_vocab_size: int = None):
        """从句子列表构建词汇表"""
        # 统计词频
        for sentence in sentences:
            for word in sentence.split():
                self.word_count[word] += 1
        
        # 按词频排序，保留高频词
        if max_vocab_size:
            most_common = self.word_count.most_common(max_vocab_size - 4)  # 减去特殊标记
            for word, _ in most_common:
                if word not in self.word2idx:
                    self.add_word(word)

class TranslationDataset(Dataset):
    """翻译数据集类"""
    
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str], 
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int = Config.MAX_SEQ_LEN):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.src_sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取一个样本"""
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # 编码源语言句子
        src_tokens = [self.src_vocab.bos_idx] + \
                    [self.src_vocab.get_idx(word) for word in src_sentence.split()] + \
                    [self.src_vocab.eos_idx]
        
        # 编码目标语言句子
        tgt_tokens = [self.tgt_vocab.bos_idx] + \
                    [self.tgt_vocab.get_idx(word) for word in tgt_sentence.split()] + \
                    [self.tgt_vocab.eos_idx]
        
        # 截断或填充到固定长度
        src_tokens = self._pad_sequence(src_tokens, self.src_vocab.pad_idx)
        tgt_input = self._pad_sequence(tgt_tokens[:-1], self.tgt_vocab.pad_idx)  # 解码器输入
        tgt_output = self._pad_sequence(tgt_tokens[1:], self.tgt_vocab.pad_idx)  # 解码器目标
        
        return torch.tensor(src_tokens), torch.tensor(tgt_input), torch.tensor(tgt_output)
    
    def _pad_sequence(self, tokens: List[int], pad_idx: int) -> List[int]:
        """填充序列到固定长度"""
        if len(tokens) > self.max_len:
            return tokens[:self.max_len]
        else:
            return tokens + [pad_idx] * (self.max_len - len(tokens))

class DataProcessor:
    """数据处理器 - 负责数据加载、预处理和词汇表构建"""
    
    def __init__(self):
        self.en_vocab = Vocabulary()
        self.zh_vocab = Vocabulary()
    
    def load_data(self) -> Tuple[List[str], List[str]]:
        """加载中英文对话数据"""
        en_file = os.path.join(Config.DATA_DIR, Config.EN_FILE)
        zh_file = os.path.join(Config.DATA_DIR, Config.ZH_FILE)
        
        print(f"正在加载数据: {en_file}, {zh_file}")
        
        en_sentences = []
        zh_sentences = []
        
        # 读取英文数据
        with open(en_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                conversation = data['conversation']
                for turn in conversation:
                    if 'human' in turn:
                        en_sentences.append(self._preprocess_text(turn['human']))
                    if 'assistant' in turn:
                        en_sentences.append(self._preprocess_text(turn['assistant']))
        
        # 读取中文数据
        with open(zh_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                conversation = data['conversation']
                for turn in conversation:
                    if 'human' in turn:
                        zh_sentences.append(self._preprocess_text(turn['human']))
                    if 'assistant' in turn:
                        zh_sentences.append(self._preprocess_text(turn['assistant']))
        
        # 确保数据对齐
        min_len = min(len(en_sentences), len(zh_sentences))
        en_sentences = en_sentences[:min_len]
        zh_sentences = zh_sentences[:min_len]
        
        print(f"加载完成，共 {len(en_sentences)} 个句子对")
        return en_sentences, zh_sentences
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 简单的文本清理
        text = text.strip().lower()
        # 移除多余的空格
        text = ' '.join(text.split())
        return text
    
    def build_vocabularies(self, en_sentences: List[str], zh_sentences: List[str]):
        """构建中英文词汇表"""
        print("正在构建词汇表...")
        
        # 构建英文词汇表
        self.en_vocab.build_vocab(en_sentences, Config.VOCAB_SIZE // 2)
        print(f"英文词汇表大小: {len(self.en_vocab)}")
        
        # 构建中文词汇表（字符级别）
        zh_chars = []
        for sentence in zh_sentences:
            zh_chars.extend(list(sentence.replace(' ', '')))
        
        char_counter = Counter(zh_chars)
        most_common_chars = char_counter.most_common(Config.VOCAB_SIZE // 2 - 4)
        
        for char, _ in most_common_chars:
            self.zh_vocab.add_word(char)
        
        print(f"中文词汇表大小: {len(self.zh_vocab)}")
    
    def create_datasets(self, en_sentences: List[str], zh_sentences: List[str], 
                       train_ratio: float = 0.8) -> Tuple[TranslationDataset, TranslationDataset]:
        """创建训练和验证数据集"""
        # 分割数据
        split_idx = int(len(en_sentences) * train_ratio)
        
        train_en = en_sentences[:split_idx]
        train_zh = [' '.join(list(s.replace(' ', ''))) for s in zh_sentences[:split_idx]]  # 字符级别
        
        val_en = en_sentences[split_idx:]
        val_zh = [' '.join(list(s.replace(' ', ''))) for s in zh_sentences[split_idx:]]  # 字符级别
        
        # 创建数据集
        train_dataset = TranslationDataset(train_en, train_zh, self.en_vocab, self.zh_vocab)
        val_dataset = TranslationDataset(val_en, val_zh, self.en_vocab, self.zh_vocab)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: TranslationDataset, 
                          val_dataset: TranslationDataset) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0  # Windows 上设置为 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def save_vocabularies(self):
        """保存词汇表"""
        os.makedirs(os.path.dirname(Config.VOCAB_SAVE_PATH), exist_ok=True)
        
        vocab_data = {
            'en_vocab': self.en_vocab,
            'zh_vocab': self.zh_vocab
        }
        
        with open(Config.VOCAB_SAVE_PATH, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"词汇表已保存到: {Config.VOCAB_SAVE_PATH}")
    
    def load_vocabularies(self):
        """加载词汇表"""
        with open(Config.VOCAB_SAVE_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.en_vocab = vocab_data['en_vocab']
        self.zh_vocab = vocab_data['zh_vocab']
        
        print(f"词汇表已从 {Config.VOCAB_SAVE_PATH} 加载")

def main():
    """测试数据处理功能"""
    processor = DataProcessor()
    
    # 加载数据
    en_sentences, zh_sentences = processor.load_data()
    
    # 构建词汇表
    processor.build_vocabularies(en_sentences, zh_sentences)
    
    # 创建数据集
    train_dataset, val_dataset = processor.create_datasets(en_sentences, zh_sentences)
    
    # 创建数据加载器
    train_loader, val_loader = processor.create_dataloaders(train_dataset, val_dataset)
    
    # 保存词汇表
    processor.save_vocabularies()
    
    # 测试数据加载
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"源语言形状: {src.shape}")
        print(f"目标输入形状: {tgt_input.shape}")
        print(f"目标输出形状: {tgt_output.shape}")
        if batch_idx >= 2:  # 只显示前3个批次
            break

if __name__ == "__main__":
    main()