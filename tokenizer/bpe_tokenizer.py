import os
import sentencepiece as spm
from pathlib import Path

class BPETokenizer:
    def __init__(self, model_prefix="bpe", vocab_size=32000, model_type="bpe"):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp = None

    def train(self, input_files, model_dir="../data/bpe"):
        os.makedirs(model_dir, exist_ok=True)
        input_str = ",".join([str(f) for f in input_files])
        model_path = os.path.join(model_dir, self.model_prefix)

        spm.SentencePieceTrainer.train(
            input=input_str,
            model_prefix=model_path,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        print(f"Model saved to {model_path}.model")

    def load(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

if __name__ == "__main__":
    # 项目根目录下运行，raw数据路径
    raw_path = Path("../") / "data" / "wmt14_raw"
    en_file = raw_path / "train.en"
    de_file = raw_path / "train.de"

    print(f"Training BPE tokenizer on files:\n  {en_file}\n  {de_file}")

    tokenizer = BPETokenizer(model_prefix="wmt14_bpe")
    tokenizer.train([en_file, de_file])
