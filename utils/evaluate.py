import torch
import torch.nn.functional as F
from tqdm import tqdm
import sacrebleu
import sentencepiece as spm


def evaluate_loss(model, dataloader, criterion, device):
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


# def beam_search_decode(model, src, beam_size=2, max_len=60, device="cpu", start_symbol=2, end_symbol=3,
#                        repetition_penalty=1.2):
#     model.eval()
#     src = src.unsqueeze(0).to(device)  # [1, src_len]
#
#     sequences = [[torch.tensor([start_symbol], device=device), 0.0]]  # 初始 beam：[(序列, 累积 log 概率)]
#
#     with torch.no_grad():
#         for _ in range(max_len):
#             all_candidates = []
#             for seq, score in sequences:
#                 if seq[-1].item() == end_symbol:
#                     all_candidates.append((seq, score))
#                     continue
#
#                 tgt_input = seq.unsqueeze(0)  # [1, seq_len]
#                 out = model(src, tgt_input)  # [1, seq_len, vocab_size]
#                 probs = torch.log_softmax(out[:, -1], dim=-1).squeeze(0)  # [vocab_size]
#
#                 # 对已经出现的token做重复惩罚
#                 for token_id in set(seq.tolist()):
#                     probs[token_id] -= repetition_penalty
#
#                 topk_probs, topk_indices = torch.topk(probs, beam_size)
#                 for i in range(beam_size):
#                     next_token = topk_indices[i].unsqueeze(0)
#                     new_seq = torch.cat([seq, next_token])
#                     new_score = score + topk_probs[i].item()
#                     all_candidates.append((new_seq, new_score))
#
#             # 选取 top beam_size 个
#             sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
#
#             if all(seq[-1].item() == end_symbol for seq, _ in sequences):
#                 break
#
#         for seq, score in sequences:
#             if seq[-1].item() == end_symbol:
#                 return seq
#         return sequences[0][0]
#
#
# def calculate_bleu(model, dataloader, sp, device, beam_size=2, max_len=60):
#     model.eval()
#     refs = []
#     hyps = []
#
#     with torch.no_grad():
#         for src, tgt, src_lens, tgt_lens in tqdm(dataloader, desc="Evaluating BLEU"):
#             for i in range(src.size(0)):
#                 src_sent = src[i]
#                 tgt_sent = tgt[i]
#
#                 pred_ids = beam_search_decode(model, src_sent, beam_size=beam_size, max_len=max_len, device=device)
#                 pred_text = sp.decode(pred_ids.tolist())
#
#                 tgt_ids = tgt_sent.tolist()
#                 if 3 in tgt_ids:
#                     tgt_ids = tgt_ids[:tgt_ids.index(3)]  # 截断至 <eos>
#                 tgt_text = sp.decode(tgt_ids)
#
#                 hyps.append(pred_text)
#                 refs.append([tgt_text])  # sacrebleu 需要每条是 list
#
#     bleu = sacrebleu.corpus_bleu(hyps, refs)
#     return bleu.score


def batch_beam_search_decode(model, src_batch, beam_size=2, max_len=60, device="cpu",
                             start_symbol=2, end_symbol=3, repetition_penalty=1.2):
    """
    批量 Beam Search 解码
    src_batch: [batch_size, src_len]
    """
    model.eval()
    batch_size = src_batch.size(0)
    src_batch = src_batch.to(device)

    # 初始化每个句子的 beam
    sequences = [
        [(torch.tensor([start_symbol], device=device), 0.0)]
        for _ in range(batch_size)
    ]
    finished_flags = [False] * batch_size

    with torch.no_grad():
        for _ in range(max_len):
            for b in range(batch_size):
                if finished_flags[b]:
                    continue  # 这个句子已经生成完毕

                all_candidates = []
                for seq, score in sequences[b]:
                    if seq[-1].item() == end_symbol:
                        all_candidates.append((seq, score))
                        continue

                    tgt_input = seq.unsqueeze(0)
                    out = model(src_batch[b].unsqueeze(0), tgt_input)
                    probs = torch.log_softmax(out[:, -1], dim=-1).squeeze(0)

                    # 惩罚重复 token
                    for token_id in set(seq.tolist()):
                        probs[token_id] /= repetition_penalty

                    topk_probs, topk_indices = torch.topk(probs, beam_size)
                    for i in range(beam_size):
                        next_token = topk_indices[i].unsqueeze(0)
                        new_seq = torch.cat([seq, next_token])
                        new_score = score + topk_probs[i].item()
                        all_candidates.append((new_seq, new_score))

                sequences[b] = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                # 如果所有 beam 都到 EOS，则提前结束
                if all(seq[-1].item() == end_symbol for seq, _ in sequences[b]):
                    finished_flags[b] = True

            # 如果整个 batch 都完成，退出循环
            if all(finished_flags):
                break

    # 从每个 beam 的第一名截取到 EOS
    results = []
    for b in range(batch_size):
        best_seq = sequences[b][0][0]
        if end_symbol in best_seq.tolist():
            best_seq = best_seq[:best_seq.tolist().index(end_symbol) + 1]
        results.append(best_seq)

    return results


@torch.no_grad()
def calculate_bleu_batch(model, dataloader, sp, device, beam_size=4, max_len=60):
    model.eval()
    refs, hyps = [], []

    for src, tgt, src_lens, tgt_lens in dataloader:
        pred_ids_list = batch_beam_search_decode(
            model, src, beam_size=beam_size, max_len=max_len, device=device
        )

        for pred_ids, tgt_sent in zip(pred_ids_list, tgt):
            pred_text = sp.decode(pred_ids.tolist())
            tgt_ids = tgt_sent.tolist()
            if 3 in tgt_ids:
                tgt_ids = tgt_ids[:tgt_ids.index(3)]
            tgt_text = sp.decode(tgt_ids)

            hyps.append(pred_text)
            refs.append([tgt_text])

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score
