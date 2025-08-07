import torch
import sentencepiece as spm
from model.transformer import Transformer

def load_model(model_path, sp_model_path, device, vocab_size=32000):
    # 加载分词器
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    # 初始化模型结构
    model = Transformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, sp

def preprocess(text, sp):
    # 分词成id序列
    ids = sp.encode(text, out_type=int)
    return ids

def postprocess(ids, sp):
    # 转回文本，忽略特殊token
    # 这里假设pad=0, sos=1, eos=2或3，根据你词表修改
    if 3 in ids:
        ids = ids[:ids.index(3)]
    text = sp.decode(ids)
    return text

def translate_sentence(model, sp, sentence, device, max_len=100, beam_size=5):
    model.eval()
    src_ids = preprocess(sentence, sp)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)

    # 调用模型的beam_search_decode函数，返回预测的token id序列
    pred_ids = model.beam_search_decode(src_tensor, max_len=max_len, beam_size=beam_size,
                                       sos_id=1, eos_id=3)  # 注意根据你词表修改sos/eos id

    translated_text = postprocess(pred_ids, sp)
    return translated_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "model/transformer_wmt14.pth"
    sp_model_path = "data/bpe/wmt14_bpe.model"

    model, sp = load_model(model_path, sp_model_path, device)

    print("请输入英文句子（输入 exit 退出）：")
    while True:
        sentence = input(">> ").strip()
        if sentence.lower() == "exit":
            break
        if not sentence:
            continue

        translation = translate_sentence(model, sp, sentence, device)
        print("翻译结果：", translation)

if __name__ == "__main__":
    main()
