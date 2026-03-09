from pathlib import Path

from torchvision.datasets import MNIST, USPS, SVHN
from pytorch_adapt.datasets import MNISTM
from salad.datasets.digits.synth import Synth  # SynthDigits

root = Path("/root/dataset/DIL/digit_five").resolve()
root.mkdir(parents=True, exist_ok=True)

# # 1) MNIST
# MNIST(root=str(root), train=True,  download=True)
# MNIST(root=str(root), train=False, download=True)   # :contentReference[oaicite:4]{index=4}

# # 2) USPS
# USPS(root=str(root)+'/USPS', train=True,  download=True)
# USPS(root=str(root)+'/USPS', train=False, download=True)    # :contentReference[oaicite:5]{index=5}

# 3) SVHN (train/test, extra는 선택)
# SVHN(root=str(root)+'/SVHN', split="train", download=True)
# SVHN(root=str(root)+'/SVHN', split="test",  download=True)
# SVHN(root=str(root), split="extra", download=True)  # 필요 시
# 참고: torchvision은 SVHN의 '0' 라벨을 0으로 매핑해줍니다. :contentReference[oaicite:6]{index=6}

# # 4) MNIST-M (train/test)
MNISTM(root=str(root), train=True,  download=True)
MNISTM(root=str(root), train=False, download=True)  # Cornell Box 미러, md5 포함. :contentReference[oaicite:7]{index=7}

# # 5) SynthDigits (train/test)
# Synth(root=str(root)+'/Synth', split="train", download=True)
# Synth(root=str(root)+'/Synth', split="test",  download=True) # salad가 GitHub 미러에서 .mat 파일 수급. :contentReference[oaicite:8]{index=8}

print("Done. Downloaded to:", root)
