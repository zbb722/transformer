import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from pathlib import Path
import sentencepiece as spm

from model.transformer import Transformer
from tokenizer.dataset import get_dataloader
from utils.scheduler import NoamOpt
from utils.loss import get_loss_function
from utils.evaluate import evaluate_loss, calculate_bleu


def train_epoch(model, dataloader, optimizer, criterion, device, writer=None, epoch=0):
    model.train()
    total_loss = 0

    for i, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]

        logits = model(src, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Batch {i+1}/{len(dataloader)}: Loss={loss.item():.4f}")
            if writer:
                writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(dataloader) + i)

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    sp_model_path = Path("data/bpe/wmt14_bpe.model")
    train_src = Path("data/bpe/train.bpe.en")
    train_tgt = Path("data/bpe/train.bpe.de")
    valid_src = Path("data/bpe/valid.bpe.en")
    valid_tgt = Path("data/bpe/valid.bpe.de")

    # 超参数
    batch_size = 64
    max_len = 100
    vocab_size = 32000
    num_epochs = 10
    model_save_path = "model/transformer_best.pth"

    # 加载 SentencePiece 模型一次
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    # 数据加载器
    train_loader = get_dataloader(train_src, train_tgt, sp_model_path, batch_size, max_len, shuffle=True)
    valid_loader = get_dataloader(valid_src, valid_tgt, sp_model_path, batch_size, max_len, shuffle=False)

    # 模型、优化器、损失函数、调度器
    model = Transformer(vocab_size=vocab_size).to(device)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(model.d_model, factor=1, warmup=4000, optimizer=optimizer)
    criterion = get_loss_function(ignore_index=0)

    # TensorBoard 日志
    writer = SummaryWriter(log_dir="runs/transformer")

    best_bleu = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start = time.time()

        train_loss = train_epoch(model, train_loader, scheduler, criterion, device, writer, epoch)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)

        # 验证时降低beam_size和max_len加速
        bleu_score = calculate_bleu(model, valid_loader, sp, device, beam_size=2, max_len=60)

        end = time.time()

        print(f"Epoch {epoch+1} completed in {end - start:.1f}s")
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

    writer.close()
    print(f"训练结束。Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
