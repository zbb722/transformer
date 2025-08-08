import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/bpe/wmt14_bpe.model")

print("ID=0 对应 token:", sp.id_to_piece(0))  # 应该是 <pad>
print("ID=1 对应 token:", sp.id_to_piece(1))  # 应该是 <unk>
print("ID=2 对应 token:", sp.id_to_piece(2))  # 应该是 <s>
print("ID=3 对应 token:", sp.id_to_piece(3))  # 应该是 </s>

print("<s> 的 ID:", sp.piece_to_id("<s>"))    # 2
print("</s> 的 ID:", sp.piece_to_id("</s>"))  # 3
