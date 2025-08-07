import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm

def apply_bpe(model_path, input_path, output_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    with open(input_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc=f"Tokenizing {input_path.name}"):
            pieces = sp.encode(line.strip(), out_type=str)
            fout.write(" ".join(pieces) + "\n")

if __name__ == "__main__":
    bpe_model = Path("../data/bpe/wmt14_bpe.model")
    raw_dir = Path("../data/wmt14_raw")
    bpe_dir = Path("../data/bpe")
    bpe_dir.mkdir(exist_ok=True)

    splits = ["train", "valid"]
    langs = ["en", "de"]

    for split in splits:
        for lang in langs:
            in_path = raw_dir / f"{split}.{lang}"
            out_path = bpe_dir / f"{split}.bpe.{lang}"
            apply_bpe(bpe_model, in_path, out_path)

    print("✅ BPE tokenization finished.")
