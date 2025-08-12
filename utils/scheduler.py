import torch


class NoamOpt:
    """
    自定义学习率调度器，来自 "Attention is All You Need" 论文
    学习率公式：d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """

    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self._rate = 0

    def step(self):
        """
        更新步数，并调整学习率
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        """
        根据当前步数计算学习率
        """
        if step is None:
            step = self._step
        return self.factor * (
                self.d_model ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
