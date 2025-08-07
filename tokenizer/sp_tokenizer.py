import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(model_path))

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
