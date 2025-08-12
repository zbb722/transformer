import torch
import torch.nn.functional as F
from tqdm import tqdm
import sacrebleu
import sentencepiece as spm


def evaluate_loss(model, dataloader, criterion, device):
    """
    计算验证集 loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            logits = model(src, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def beam_search_decode(
    model,
    src,
    beam_size=4,
    max_len=60,
    device="cpu",
    start_symbol=2,  # <sos>
    end_symbol=3,    # <eos>
    pad_symbol=0,    # <pad>
    repetition_penalty=1.2,
    length_penalty=0.6
):
    """
    Beam Search 解码
    - length_penalty < 1.0 会鼓励生成更长句子（Google Transformer 论文用 0.6）
    - repetition_penalty > 1.0 会惩罚重复
    - 屏蔽 <sos> / <pad>
    """
    model.eval()
    src = src.unsqueeze(0).to(device)  # [1, src_len]

    sequences = [[torch.tensor([start_symbol], device=device), 0.0]]  # (token_seq, log_prob)

    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1].item() == end_symbol:
                    # 已完成句子，分数加 length_penalty
                    lp = ((5 + len(seq)) / 6) ** length_penalty
                    all_candidates.append((seq, score / lp))
                    continue

                tgt_input = seq.unsqueeze(0)  # [1, seq_len]
                out = model(src, tgt_input)   # [1, seq_len, vocab_size]
                probs = torch.log_softmax(out[:, -1], dim=-1).squeeze(0)  # [vocab_size]

                # 屏蔽 <sos> / <pad>
                probs[start_symbol] = -float("inf")
                probs[pad_symbol] = -float("inf")

                # 重复惩罚
                for token_id in set(seq.tolist()):
                    probs[token_id] -= repetition_penalty

                # 取 top-k
                topk_probs, topk_indices = torch.topk(probs, beam_size)
                for i in range(beam_size):
                    next_token = topk_indices[i].unsqueeze(0)
                    new_seq = torch.cat([seq, next_token])
                    new_score = score + topk_probs[i].item()
                    all_candidates.append((new_seq, new_score))

            # 排序并保留前 beam_size 个
            sequences = sorted(
                all_candidates,
                key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** length_penalty),
                reverse=True
            )[:beam_size]

            # 全部完成就提前停止
            if all(seq[-1].item() == end_symbol for seq, _ in sequences):
                break

        # 选最佳序列
        best_seq = max(
            sequences,
            key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** length_penalty)
        )[0]

        # 去掉 <eos>
        best_seq = best_seq.tolist()
        if end_symbol in best_seq:
            best_seq = best_seq[:best_seq.index(end_symbol)]

        return torch.tensor(best_seq, device=device)


def calculate_bleu(model, dataloader, sp, device, beam_size=4, max_len=60):
    """
    用 Beam Search 解码并计算 corpus BLEU
    """
    model.eval()
    refs = []
    hyps = []

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in tqdm(dataloader, desc="Evaluating BLEU"):
            for i in range(src.size(0)):
                src_sent = src[i]
                tgt_sent = tgt[i]

                pred_ids = beam_search_decode(
                    model,
                    src_sent,
                    beam_size=beam_size,
                    max_len=max_len,
                    device=device
                )

                pred_text = sp.decode(pred_ids.tolist())

                # 目标截断到 <eos>
                tgt_ids = tgt_sent.tolist()
                if 3 in tgt_ids:
                    tgt_ids = tgt_ids[:tgt_ids.index(3)]
                tgt_text = sp.decode(tgt_ids)

                hyps.append(pred_text)
                refs.append([tgt_text])  # sacrebleu 要求每条参考是 list

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score
