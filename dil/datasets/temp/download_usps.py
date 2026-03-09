from pathlib import Path
from torchvision.datasets import USPS
import os

root = Path("/root/dataset/DIL/digit_five")  # USPS의 상위 폴더
raw = root / "USPS" / "raw"
proc = root / "USPS" / "processed"
raw.mkdir(parents=True, exist_ok=True)

# 잘못 둔 파일을 제자리로 이동
for name in ["usps.bz2", "usps.t.bz2"]:
    if (root / name).exists():
        (root / name).replace(raw / name)

# 처리 파일 생성 트리거
USPS(root=str(root), train=True,  download=True)
USPS(root=str(root), train=False, download=True)

# 검증
print("raw 존재 여부", raw.exists())
print("processed 목록", list(proc.iterdir()) if proc.exists() else "없음")
