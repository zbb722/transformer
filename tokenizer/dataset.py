import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from pathlib import Path

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, sp_model_path, max_len=100):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))

        self.src_lines = open(src_file, encoding='utf-8').read().strip().split('\n')
        self.tgt_lines = open(tgt_file, encoding='utf-8').read().strip().split('\n')
        assert len(self.src_lines) == len(self.tgt_lines), "源和目标行数不匹配"

        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.sp.encode(self.src_lines[idx], out_type=int)
        tgt = self.sp.encode(self.tgt_lines[idx], out_type=int)

        # 加入 <bos> 和 <eos>
        bos_id = 2
        eos_id = 3
        src = src[:self.max_len]
        tgt = tgt[:self.max_len-2]  # 留空间给 bos 和 eos

        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor([bos_id] + tgt + [eos_id], dtype=torch.long)

        return src, tgt

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    return src_padded, tgt_padded, torch.tensor(src_lens), torch.tensor(tgt_lens)

def get_dataloader(src_file, tgt_file, sp_model_path, batch_size=64, max_len=100, shuffle=True):
    dataset = TranslationDataset(src_file, tgt_file, sp_model_path, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

if __name__ == "__main__":
    sp_model = Path("../data/bpe/wmt14_bpe.model")
    src_file = Path("../data/bpe/train.bpe.en")
    tgt_file = Path("../data/bpe/train.bpe.de")

    dataloader = get_dataloader(src_file, tgt_file, sp_model, batch_size=8)

    for batch_idx, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print("src shape:", src.shape)
        print("tgt shape:", tgt.shape)
        break
