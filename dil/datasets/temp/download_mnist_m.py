# 필요한 패키지 설치
# pip install datasets pillow tqdm

from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm

root = Path("root/dataset/DIL/digit_five")      # 원하는 루트
out  = root / "mnist_m"
train_dir = out / "mnist_m_train"
test_dir  = out / "mnist_m_test"
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# HF MNIST-M 로드
ds = load_dataset("Mike0307/MNIST-M")  # splits train test 제공
# :contentReference[oaicite:3]{index=3}

# 라벨 파일 준비
train_lab = open(out / "mnist_m_train_labels.txt", "w")
test_lab  = open(out / "mnist_m_test_labels.txt", "w")

# 변환 함수
def export_split(split_name, split_ds, out_dir, out_lab):
    for i, eg in enumerate(tqdm(split_ds, desc=split_name)):
        img = eg["image"]          # PIL.Image
        y   = int(eg["label"])
        fname = f"img_{i:05d}.png"
        img.convert("RGB").save(out_dir / fname)
        out_lab.write(f"{fname} {y}\n")

export_split("train", ds["train"], train_dir, train_lab)
export_split("test",  ds["test"],  test_dir,  test_lab)

train_lab.close()
test_lab.close()
print("완료 경로", out.resolve())
