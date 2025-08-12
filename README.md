# Transformer - From Scratch

这是一个基于 PyTorch 实现的原始 Transformer 模型复现项目，参考论文：[Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762)。支持从原始文本训练翻译模型、使用 BPE 分词、Beam Search 解码以及 BLEU 评估。

## 📌 特性

- 原生实现 Transformer 编码器-解码器结构
- 支持 SentencePiece 和 BPE 分词
- 支持 Beam Search 解码
- BLEU 打分评估翻译质量
- 支持 TensorBoard 可视化训练过程
- 模型训练中保存 best checkpoint（基于验证集 loss）

---

## 📁 项目结构
<pre> ```transformer
├─data                          # 数据目录
│  ├─bpe                        # BPE 分词后的训练/验证/测试集
│  └─wmt14_raw                  # 原始 WMT14 英德数据集
├─model                         # 模型定义与保存路径
│  ├─en2de_model.pth            # 训练过程中表现最好的模型权重
│  └─transformer.py             # Transformer 模型结构定义
├─runs                          # 日志和可视化相关
│  ├─transformer                # TensorBoard 日志目录
│  └─plot_tensorboard_logs.py   # 日志可视化脚本
├─tokenizer                     # 分词器相关模块
│  ├─bpe_tokenizer.py           # 自定义 BPE 分词器
│  ├─dataset.py                 # 数据加载与 Dataloader 封装
│  ├─sp_tokenizer.py            # 使用 SentencePiece 的分词器
│  └─apply_bpe.py               # 批量应用 BPE 分词的脚本
└─utils                         # 工具模块
│  ├─evaluate.py                # BLEU 评估等评测函数
│  ├─scheduler.py               # 学习率调度器（含 warm-up）
│  └─loss.py                    # Label Smoothing 等损失函数
├─train.py                      # 训练主脚本
└─translate.py                  # 推理/翻译主脚本，支持 Beam Search``` </pre>

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2️⃣ 准备数据

* 下载 WMT14 英德翻译数据集并放入目录 `data/wmt14_raw/`
* 使用 SentencePiece 或 BPE 进行分词（建议使用 BPE）：

```bash
python tokenizer/apply_bpe.py
```

> 分词后的结果将保存在 `data/bpe/` 目录下。

---

### 3️⃣ 开始训练

```bash
python train.py \
    --epochs 10 \
    --batch_size 64 \
    --device cuda
```

训练过程中将自动保存验证集表现最优的模型至：

```
model/transformer_best.pth
```

---

### 4️⃣ 翻译句子（推理）

```bash
python translate.py \
    --checkpoint model/transformer_best.pth \
    --input "Hello, how are you?" \
    --beam_size 5
```

---

### 📊 可视化训练过程

使用 TensorBoard 查看训练过程中的 loss 和指标变化：

```bash
tensorboard --logdir runs/transformer
```

浏览器中访问 [http://localhost:6006](http://localhost:6006) 查看图表。

---

### 📈 示例翻译结果

| 输入句子                  | 模型输出翻译  |
| --------------------- | ------- |
| Hello, how are you?   |  |
| I love deep learning. |  |

---

### 🧪 评估 BLEU 分数

使用如下命令对测试集进行 BLEU 评估（支持 Beam Search 解码）：

```bash
python -c "
from utils.evaluate import calculate_bleu
from model.transformer import Transformer
from tokenizer.dataset import get_dataloader
import sentencepiece as spm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(vocab_size=XXX)
model.load_state_dict(torch.load('model/transformer_best.pth', map_location=device))
model.to(device)

sp = spm.SentencePieceProcessor(model_file='tokenizer/bpe.model')
test_loader = get_dataloader('data/bpe/test.src', 'data/bpe/test.tgt', batch_size=1)

bleu = calculate_bleu(model, test_loader, sp, device=device, beam_size=5)
print(f'BLEU score: {bleu:.2f}')
"
```

> ✅ 注：你需要根据实际情况替换 `vocab_size=XXX` 和 `bpe.model` 路径。

---

### 说明

* 默认使用 `beam_size=5` 进行 Beam Search 解码；
* 每条参考答案在 `refs` 中是一个列表，以适配 `sacrebleu`；
* `calculate_bleu()` 内部会自动截断 `<eos>`；
* 支持 `sentencepiece` 分词器还原文本。

---

---
