import time
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from pathlib import Path
import sentencepiece as spm

# 推荐使用 torch.amp（替代 torch.cuda.amp 的未来接口）
from torch.amp import autocast, GradScaler

from model.transformer import Transformer
from tokenizer.dataset import get_dataloader
# 注意：在本文件中实现了一个不执行 optimizer.step() 的 NoamLR，
# 因此不依赖外部 utils.scheduler.NoamOpt 的 "封装 step" 行为
from utils.loss import get_loss_function
from utils.evaluate import evaluate_loss, calculate_bleu
import torch.nn.utils as nn_utils


# ====================== 简单的 Noam learning-rate scheduler（不做 optimizer.step） ======================
class NoamLR:
    """
    Noam learning rate schedule (和论文相同形式)，但 **不在 step() 中调用 optimizer.step()**。
    这样能把 optimizer 的 step (通过 GradScaler) 与 lr 更新分开调用，方便 AMP 使用。
    """

    def __init__(self, optimizer, d_model, factor=1, warmup=4000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model

    def step(self):
        self._step += 1
        lr = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def rate(self, step=None):
        if step is None:
            step = max(self._step, 1)
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))


# ====================== 数据采样工具函数 ======================
def sample_dataset(src_path, tgt_path, ratio=1.0, seed=42):
    """
    从平行语料中随机采样一定比例的句对
    用于 warmup 调试阶段减少训练集规模，加快训练速度
    """
    assert 0 < ratio <= 1.0, "ratio 必须在 (0, 1] 之间"

    with open(src_path, "r", encoding="utf-8") as f_src, \
            open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    assert len(src_lines) == len(tgt_lines), "源语言与目标语言句对数量不一致"

    total = len(src_lines)
    if ratio < 1.0:
        random.seed(seed)  # 保证可复现
        indices = random.sample(range(total), int(total * ratio))
        src_lines = [src_lines[i] for i in indices]
        tgt_lines = [tgt_lines[i] for i in indices]

    return src_lines, tgt_lines


def write_temp_dataset(src_lines, tgt_lines, src_out, tgt_out):
    """将采样后的数据写入临时文件"""
    with open(src_out, "w", encoding="utf-8") as f_src, \
            open(tgt_out, "w", encoding="utf-8") as f_tgt:
        f_src.writelines(src_lines)
        f_tgt.writelines(tgt_lines)


# ====================== 训练一个 epoch ======================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                writer=None, epoch=0, use_amp=True, grad_clip=None):
    """
    执行一个 epoch 的训练
    - optimizer: torch.optim.Optimizer（例如 Adam）
    - scheduler: 一个只更新 lr 的对象（如上面的 NoamLR），**不应在 step() 内执行 optimizer.step()**
    - use_amp: 是否使用混合精度
    - grad_clip: 如果需要梯度裁剪，传入一个 float（如 1.0），否则传入 None
    """
    model.train()
    total_loss = 0.0

    # GradScaler：负责 FP16 的梯度缩放与数值稳定性
    scaler = GradScaler(enabled=use_amp)

    # 选择 autocast 的 device_type（如果用 cpu，就设为 'cpu'）
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for i, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        decoder_input = tgt[:, :-1]  # 输入序列
        target = tgt[:, 1:]  # 目标序列（对齐）

        # 前向（在 autocast 上下文中自动混合精度）
        with autocast(device_type=device_type, enabled=use_amp):
            logits = model(src, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # 反向 + step（使用 scaler）
        scaler.scale(loss).backward()

        # 在调用 scaler.step 之前，先做 unscale（便于梯度裁剪）
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            nn_utils.clip_grad_norm_(model.parameters(), grad_clip)

        # 使用实际的 optimizer（例如 Adam）进行一次 step（由 GradScaler 调用）
        scaler.step(optimizer)
        scaler.update()

        # 更新 lr（注意：这个 scheduler.step() 只更新 lr，不要在 scheduler.step 中再执行 optimizer.step()）
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Batch {i + 1}/{len(dataloader)}: Loss={loss.item():.4f}")
            if writer:
                writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(dataloader) + i)

    return total_loss / len(dataloader)


# ====================== 主入口 ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    sp_model_path = Path("data/bpe/wmt14_bpe.model")
    train_src_path = Path("data/bpe/train.bpe.en")
    train_tgt_path = Path("data/bpe/train.bpe.de")
    valid_src = Path("data/bpe/valid.bpe.en")
    valid_tgt = Path("data/bpe/valid.bpe.de")

    # ===== 数据采样比例（调试可设为 0.1，全量设为 1.0） =====
    train_ratio = 1
    temp_train_src = Path("data/bpe/train_sample.bpe.en")
    temp_train_tgt = Path("data/bpe/train_sample.bpe.de")

    if train_ratio < 1.0:
        src_lines, tgt_lines = sample_dataset(train_src_path, train_tgt_path, ratio=train_ratio)
        write_temp_dataset(src_lines, tgt_lines, temp_train_src, temp_train_tgt)
        train_src = temp_train_src
        train_tgt = temp_train_tgt
        print(f"已采样 {train_ratio * 100:.1f}% 训练集，共 {len(src_lines)} 句对")
    else:
        train_src = train_src_path
        train_tgt = train_tgt_path

    # 超参数
    batch_size = 64
    max_len = 100
    vocab_size = 32000
    num_epochs = 50
    model_save_path = "model/en2de_model.pth"
    grad_clip = 1.0  # 可选：梯度裁剪，防止梯度爆炸（None 表示不裁剪）

    # 加载 SentencePiece 模型
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    # 数据加载器
    train_loader = get_dataloader(train_src, train_tgt, sp_model_path, batch_size, max_len, shuffle=True)
    valid_loader = get_dataloader(valid_src, valid_tgt, sp_model_path, batch_size, max_len, shuffle=False)

    # 模型、优化器、损失函数、调度器
    model = Transformer(vocab_size=vocab_size).to(device)
    optimizer = Adam(model.parameters(), weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # 使用上面实现的 NoamLR（注意：这个 NoamLR 不会在 step() 执行 optimizer.step()）
    scheduler = NoamLR(optimizer, d_model=model.d_model, factor=1, warmup=4000)

    criterion = get_loss_function(ignore_index=0, vocab_size=vocab_size)

    # TensorBoard 日志
    writer = SummaryWriter(log_dir="runs/transformer")

    best_bleu = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        start = time.time()

        # 使用混合精度训练（传入 optimizer 与 scheduler）
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion,
                                 device, writer, epoch, use_amp=True, grad_clip=grad_clip)

        # 验证损失（不用混合精度，保证评估稳定）
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)

        # 验证 BLEU（beam search 会比较慢，max_len 调低加速）
        bleu_score = calculate_bleu(model, valid_loader, sp, device, beam_size=1, max_len=60)

        end = time.time()

        print(f"Epoch {epoch + 1} completed in {end - start:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | BLEU: {bleu_score:.2f}")

        # 记录 TensorBoard
        writer.add_scalar("Train/Epoch_Loss", train_loss, epoch)
        writer.add_scalar("Valid/Epoch_Loss", valid_loss, epoch)
        writer.add_scalar("Valid/BLEU", bleu_score, epoch)

        # 保存 BLEU 最优模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with BLEU: {bleu_score:.2f}")
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), "final_model.pth")
            print(f"final model saved with BLEU: {bleu_score:.2f}")
    writer.close()
    print(f"训练结束。Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
