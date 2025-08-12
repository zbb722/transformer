import torch
import sentencepiece as spm
from model.transformer import Transformer
from utils.evaluate import beam_search_decode

# ====== 常量 ======
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
VOCAB_SIZE = 32000
MAX_LEN = 100
BEAM_SIZE = 5


def load_model(model_path, sp_model_path, device):
    """加载模型和分词器"""
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    model = Transformer(vocab_size=VOCAB_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, sp


def preprocess(text, sp):
    """分词成ID，并加SOS/EOS"""
    ids = sp.encode(text, out_type=int)
    return [SOS_ID] + ids + [EOS_ID]


def postprocess(ids, sp):
    """将ID转为文本，去掉特殊符号"""
    if torch.is_tensor(ids):
        ids = ids.tolist()

    # 截断到EOS
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]

    # 去除SOS/PAD
    ids = [i for i in ids if i not in {SOS_ID, PAD_ID}]

    # 去除UNK
    ids = [i for i in ids if i != UNK_ID]

    return sp.decode(ids)


@torch.no_grad()
def translate_sentence(model, sp, sentence, device, max_len=MAX_LEN, beam_size=BEAM_SIZE):
    """单句翻译"""
    src_ids = preprocess(sentence, sp)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device)  # 注意：不加batch维

    pred_ids = beam_search_decode(
        model,
        src_tensor,  # 你的 beam_search_decode 内部会自己 unsqueeze(0)
        beam_size=beam_size,
        max_len=max_len,
        device=device,
        start_symbol=SOS_ID,
        end_symbol=EOS_ID
    )

    return postprocess(pred_ids, sp)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "model/en2de_model.pth"
    sp_model_path = "data/bpe/wmt14_bpe.model"

    model, sp = load_model(model_path, sp_model_path, device)

    print(f"模型加载完成（设备：{device}），开始翻译（输入 exit 退出）")
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
