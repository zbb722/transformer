# download_wmt14.py

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm


def save_to_file(pairs, src_path, tgt_path):
    with open(src_path, 'w', encoding='utf-8') as src_f, \
            open(tgt_path, 'w', encoding='utf-8') as tgt_f:
        for pair in tqdm(pairs):
            src_f.write(pair['translation']['en'].strip() + '\n')
            tgt_f.write(pair['translation']['de'].strip() + '\n')


def download_wmt14(subset_size=100000):
    print("Downloading WMT14 En-De...")
    dataset = load_dataset("wmt14", "de-en", split='train')

    print(f"Total samples: {len(dataset)}")

    if subset_size:
        dataset = dataset.select(range(subset_size))  # 选前subset_size个

    save_dir = Path("wmt14_raw")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_to_file(dataset,
                 src_path=save_dir / "train.en",
                 tgt_path=save_dir / "train.de")

    # 下载验证集
    valid_set = load_dataset("wmt14", "de-en", split="validation")
    save_to_file(valid_set,
                 src_path=save_dir / "valid.en",
                 tgt_path=save_dir / "valid.de")


if __name__ == "__main__":
    download_wmt14(subset_size=500_000)  # 选前50万样本
