import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    位置编码：给序列的每个位置添加唯一的固定编码，以弥补Transformer缺乏序列顺序信息的缺陷。
    论文中用的是正余弦函数实现的位置编码。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，方便后续广播
        self.register_buffer('pe', pe)  # 注册为buffer，不作为模型参数，但会保存到state_dict

    def forward(self, x):
        """
        输入:
            x: (batch_size, seq_len, d_model)
        输出:
            x + 位置编码
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    将输入映射成Query, Key, Value，分别计算注意力得分并加权聚合。
    多头是为了让模型可以在不同子空间学习不同的注意力。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层，将d_model映射到多个头的Q,K,V空间
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)  # 最后输出的线性层
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)  # 缩放因子，避免内积数值过大

    def forward(self, query, key, value, mask=None):
        """
        输入:
            query, key, value: (batch_size, seq_len, d_model)
            mask: 注意力掩码，shape: (batch_size, 1, 1, seq_len) 或 (batch_size, num_heads, seq_len, seq_len)
        输出:
            output: 多头注意力的结果，(batch_size, seq_len, d_model)
            attn: 注意力权重矩阵，(batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # 线性变换并reshape成多头格式，(batch_size, num_heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # 对mask=0的地方打负无穷

        attn = torch.softmax(scores, dim=-1)  # 归一化注意力权重
        attn = self.dropout(attn)  # dropout防止过拟合

        # 根据注意力权重加权Value
        context = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, d_k)

        # 拼回多头的结果
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        output = self.fc(context)  # 线性变换输出
        return output, attn

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络，论文中的FFN结构
    两个线性层，中间ReLU激活
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含自注意力 + 残差连接 + 层归一化 + 前馈网络
    """
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力子层
        _src, _ = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout(_src)  # 残差连接
        src = self.norm1(src)  # 层归一化

        # 前馈网络子层
        _src = self.ff(src)
        src = src + self.dropout(_src)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含掩码自注意力 + 编码器-解码器注意力 + 前馈网络
    """
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 掩码自注意力，保证解码时只能看到当前位置之前的token
        _tgt, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout(_tgt)
        tgt = self.norm1(tgt)

        # 编码器-解码器注意力，查询来自解码器，键值来自编码器输出
        _tgt, _ = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout(_tgt)
        tgt = self.norm2(tgt)

        # 前馈网络
        _tgt = self.ff(tgt)
        tgt = tgt + self.dropout(_tgt)
        tgt = self.norm3(tgt)

        return tgt

class Transformer(nn.Module):
    """
    Transformer整体模型结构
    包括词嵌入、位置编码、编码器堆叠、解码器堆叠、输出层
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()

        self.d_model = d_model

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        # 解码器层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出线性层，投影到词汇表大小，用于预测下一个词
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        """
        创建编码器输入掩码，padding位置为0，非padding为1
        输入:
            src: (batch_size, src_len)
        输出:
            mask: (batch_size, 1, 1, src_len)，便于广播
        """
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        """
        创建解码器掩码，结合padding掩码和未来位置掩码（防止模型看到未来词）
        输入:
            tgt: (batch_size, tgt_len)
        输出:
            mask: (batch_size, 1, tgt_len, tgt_len)
        """
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # padding mask
        tgt_len = tgt.size(1)
        # 下三角矩阵，确保位置i只能看到i及之前位置
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        """
        Transformer前向传播
        输入:
            src: (batch_size, src_len) 编码器输入token id序列
            tgt: (batch_size, tgt_len) 解码器输入token id序列（通常是目标序列右移）
        输出:
            logits: (batch_size, tgt_len, vocab_size) 每个位置预测下一个词的概率分布
        """
        src_mask = self.make_src_mask(src)  # 编码器padding掩码
        tgt_mask = self.make_tgt_mask(tgt)  # 解码器padding + future mask

        # 词嵌入 + 位置编码 + dropout
        src_emb = self.dropout(self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model)))

        # 编码器堆叠
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        # 解码器堆叠
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, src_mask)

        logits = self.fc_out(output)  # 输出投影到词表大小
        return logits

    def encode(self, src):
        src_mask = self.make_src_mask(src)  # 生成编码器掩码
        src_emb = self.dropout(self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model)))
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
        return memory

    def decode(self, memory, tgt):
        tgt_mask = self.make_tgt_mask(tgt)  # 生成解码器掩码
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model)))
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, None)
        logits = self.fc_out(output)
        return logits
