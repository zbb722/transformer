# test.py
import os
from tokenizer.bpe_tokenizer import BPETokenizer

def main():
    # 模型路径（根据你 bpe_tokenizer.py 的保存路径调整）
    model_path = "data/bpe/wmt14_bpe.model"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}，请先运行 bpe_tokenizer.py 训练分词器。")

    # 加载分词模型
    tokenizer = BPETokenizer()
    tokenizer.load(model_path)

    # 测试样例
    test_sentences = [
        "Hello world!",
        "I love machine translation.",
        "Das ist ein Test."
    ]

    for text in test_sentences:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print("=" * 40)
        print(f"原文本: {text}")
        print(f"编码ID: {ids}")
        print(f"解码文本: {decoded}")

if __name__ == "__main__":
    main()
