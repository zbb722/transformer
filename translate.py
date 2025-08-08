import torch
import sentencepiece as spm
from model.transformer import Transformer
from utils.evaluate import batch_beam_search_decode

# ======== 常量定义 ========
PAD_ID = 0
SOS_ID = 2
EOS_ID = 3
VOCAB_SIZE = 32000
MAX_LEN = 60
BEAM_SIZE = 2

# ======== 加载模型与分词器 ========
def load_model(model_path, sp_model_path, device):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    model = Transformer(vocab_size=VOCAB_SIZE)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, sp

# ======== 文本处理 ========
def preprocess(text, sp):
    text = text.strip()
    if not text:
        return []
    ids = sp.encode(text, out_type=int)
    return ids

def postprocess(ids, sp):
    if not ids:
        return ""
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    return sp.decode(ids)

# ======== 翻译函数 ========
@torch.no_grad()
def translate_sentence(model, sp, sentence, device):
    src_ids = preprocess(sentence, sp)
    print("[调试] 输入ID：", src_ids)
    if not src_ids:
        return "[空输入]"

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # batch=1

    pred_ids_list = batch_beam_search_decode(
        model,
        src_tensor,
        beam_size=BEAM_SIZE,
        max_len=MAX_LEN,
        device=device,
        start_symbol=SOS_ID,
        end_symbol=EOS_ID
    )

    # 取第一个元素
    pred_ids = pred_ids_list[0]
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.tolist()

    print("[调试] 输出ID：", pred_ids)
    return postprocess(pred_ids, sp)

# ======== 主程序 ========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "model/en2de_model.pth"
    sp_model_path = "data/bpe/wmt14_bpe.model"

    print(f"加载模型中（设备：{device}）...")
    model, sp = load_model(model_path, sp_model_path, device)
    print("模型加载完成，开始翻译（输入 exit 退出）")

    while True:
        sentence = input(">> ").strip()
        if sentence.lower() == "exit":
            break
        translation = translate_sentence(model, sp, sentence, device)
        print("翻译结果：", translation)

if __name__ == "__main__":
    main()
