import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        # pred: [batch*seq, vocab_size], target: [batch*seq]
        with torch.no_grad():
            # 全部初始化为均匀分布
            true_dist = pred.new_full((target.size(0), self.vocab_size), self.smoothing / (self.vocab_size - 2))
            # 正确标签位置的概率
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            # padding token 置 0
            true_dist[:, self.padding_idx] = 0
            mask = target == self.padding_idx
            true_dist[mask] = 0
        return self.criterion(pred.log_softmax(dim=-1), true_dist)


def get_loss_function(ignore_index=0, vocab_size=32000, smoothing=0.15):
    """
    创建带 Label Smoothing 的损失函数，忽略 pad 部分。
    vocab_size 必须传入。
    """
    if vocab_size is None:
        raise ValueError("必须传入 vocab_size 用于 Label Smoothing")
    return LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=ignore_index, smoothing=smoothing)
