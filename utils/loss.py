import torch.nn as nn

def get_loss_function(ignore_index=0):
    """
    创建用于序列模型的交叉熵损失函数，忽略填充（pad）部分。
    """
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
