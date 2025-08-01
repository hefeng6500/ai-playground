"""Transformer 模型实现 - 包含完整的编码器-解码器架构"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class PositionalEncoding(nn.Module):
    """位置编码 - 为序列中的每个位置添加位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        
        # 添加批次维度并注册为缓冲区（不参与梯度更新）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 形状为 [seq_len, batch_size, d_model] 的张量
        返回:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制 - Transformer 的核心组件"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性变换层：用于生成 Q、K、V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        缩放点积注意力
        
        参数:
            Q, K, V: 查询、键、值矩阵，形状为 [batch_size, n_heads, seq_len, d_k]
            mask: 注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权值
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 查询张量，形状为 [batch_size, query_len, d_model]
            key: 键张量，形状为 [batch_size, key_len, d_model]
            value: 值张量，形状为 [batch_size, value_len, d_model]
            mask: 注意力掩码
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)
        
        # 1. 线性变换生成 Q、K、V
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 2. 重塑为多头形式：[batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, value_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 应用缩放点积注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 连接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        # 5. 最终线性变换
        output = self.w_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    """前馈神经网络 - Transformer 中的位置级前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：x -> Linear -> ReLU -> Dropout -> Linear
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """编码器层 - 包含多头注意力和前馈网络"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        编码器层前向传播
        使用残差连接和层归一化
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """解码器层 - 包含掩码自注意力、编码器-解码器注意力和前馈网络"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                self_mask: torch.Tensor = None, cross_mask: torch.Tensor = None) -> torch.Tensor:
        """
        解码器层前向传播
        
        参数:
            x: 解码器输入
            encoder_output: 编码器输出
            self_mask: 自注意力掩码（因果掩码）
            cross_mask: 交叉注意力掩码
        """
        # 1. 掩码自注意力 + 残差连接 + 层归一化
        self_attn_output = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. 编码器-解码器注意力 + 残差连接 + 层归一化
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    """完整的 Transformer 模型 - 编码器-解码器架构"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int = Config.D_MODEL, n_heads: int = Config.N_HEADS,
                 n_layers: int = Config.N_LAYERS, d_ff: int = Config.D_FF,
                 max_seq_len: int = Config.MAX_SEQ_LEN, dropout: float = Config.DROPOUT):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """
        创建填充掩码
        
        参数:
            seq: 输入序列，形状为 [batch_size, seq_len]
            pad_idx: 填充标记的索引
        
        返回:
            掩码张量，形状为 [batch_size, 1, 1, seq_len]
        """
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """
        创建因果掩码（下三角掩码）
        确保解码器只能看到当前位置之前的信息
        
        参数:
            size: 序列长度
        
        返回:
            因果掩码，形状为 [1, 1, size, size]
        """
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        编码器前向传播
        
        参数:
            src: 源序列，形状为 [batch_size, src_len]
            src_mask: 源序列掩码
        
        返回:
            编码器输出，形状为 [batch_size, src_len, d_model]
        """
        # 词嵌入 + 位置编码
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb.transpose(0, 1)  # [src_len, batch_size, d_model]
        src_emb = self.pos_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [batch_size, src_len, d_model]
        src_emb = self.dropout(src_emb)
        
        # 通过编码器层
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        return encoder_output
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        解码器前向传播
        
        参数:
            tgt: 目标序列，形状为 [batch_size, tgt_len]
            encoder_output: 编码器输出
            tgt_mask: 目标序列掩码
            src_mask: 源序列掩码
        
        返回:
            解码器输出，形状为 [batch_size, tgt_len, d_model]
        """
        # 词嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, batch_size, d_model]
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [batch_size, tgt_len, d_model]
        tgt_emb = self.dropout(tgt_emb)
        
        # 通过解码器层
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
        
        return decoder_output
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        """
        模型前向传播
        
        参数:
            src: 源序列，形状为 [batch_size, src_len]
            tgt: 目标序列，形状为 [batch_size, tgt_len]
            src_pad_idx: 源序列填充索引
            tgt_pad_idx: 目标序列填充索引
        
        返回:
            输出logits，形状为 [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建掩码
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_len = tgt.size(1)
        tgt_padding_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        tgt_causal_mask = self.create_causal_mask(tgt_len).to(tgt.device)
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate(self, src: torch.Tensor, src_pad_idx: int, tgt_bos_idx: int, 
                 tgt_eos_idx: int, max_len: int = 100) -> torch.Tensor:
        """
        生成翻译序列（贪心解码）
        
        参数:
            src: 源序列，形状为 [1, src_len]
            src_pad_idx: 源序列填充索引
            tgt_bos_idx: 目标序列开始标记索引
            tgt_eos_idx: 目标序列结束标记索引
            max_len: 最大生成长度
        
        返回:
            生成的目标序列
        """
        self.eval()
        device = src.device
        
        # 编码源序列
        src_mask = self.create_padding_mask(src, src_pad_idx)
        encoder_output = self.encode(src, src_mask)
        
        # 初始化目标序列
        tgt = torch.tensor([[tgt_bos_idx]], device=device)
        
        for _ in range(max_len):
            # 创建目标掩码
            tgt_len = tgt.size(1)
            tgt_mask = self.create_causal_mask(tgt_len).to(device)
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # 预测下一个词
            next_token_logits = self.output_projection(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果生成了结束标记，停止生成
            if next_token.item() == tgt_eos_idx:
                break
        
        return tgt

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """测试 Transformer 模型"""
    # 创建模型
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=8000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048
    )
    
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size, src_len, tgt_len = 2, 20, 15
    src = torch.randint(0, 10000, (batch_size, src_len))
    tgt = torch.randint(0, 8000, (batch_size, tgt_len))
    
    output = model(src, tgt)
    print(f"输出形状: {output.shape}")
    
    # 测试生成
    src_test = torch.randint(0, 10000, (1, 10))
    generated = model.generate(src_test, src_pad_idx=0, tgt_bos_idx=1, tgt_eos_idx=2)
    print(f"生成序列形状: {generated.shape}")

if __name__ == "__main__":
    main()