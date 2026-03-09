import os
import copy
import glob
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion, transform_domainnet, transform_core50s, transform_pacs
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from model_dist import define_model
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
from m3dloss import M3DLoss

from dil.datasets import ContinualDataset, get_dataset

def compute_average_forgetting(task_accuracy_dict):
    """
    Chaudhry et al. 2018 정의에 따른 Average Forgetting 계산
    입력
      task_accuracy_dict  키는 task index  값은 해당 task까지 학습 후 각 테스트 태스크 정확도 리스트
    출력
      F_final  마지막 태스크 학습이 끝난 시점의 평균 포게팅
      forgetting_curve  각 시점 t에서의 평균 포게팅 값 리스트
      A  정확도 행렬  shape T x T  A[t][k]는 t시점 학습 후 k태스크 정확도
    """
    import numpy as np

    # dict를 일관된 순서의 행렬로 변환
    T = len(task_accuracy_dict)
    max_len = max(len(v) for v in task_accuracy_dict.values())
    A = np.full((T, max_len), np.nan, dtype=float)
    for t in range(T):
        row = np.asarray(task_accuracy_dict[t], dtype=float)
        A[t, :len(row)] = row

    # 각 시점의 평균 포게팅 계산
    forgetting_curve = []
    for t in range(T):
        if t == 0:
            forgetting_curve.append(0.0)
            continue
        fks = []
        for k in range(t):  # 이미 학습이 끝난 과거 태스크만 고려
            past = A[k:t + 1, k]  # 시점 k부터 t까지의 k열 성능
            max_past = np.nanmax(past)
            curr = A[t, k]
            if np.isnan(curr):
                continue
            fks.append(max(0.0, float(max_past - curr)))
        F_t = float(np.mean(fks)) if fks else 0.0
        forgetting_curve.append(F_t)

    # 마지막 시점의 평균 포게팅
    final_fks = []
    t = T - 1
    for k in range(T - 1):
        past = A[k:T, k]
        max_past = np.nanmax(past)
        curr = A[t, k]
        if np.isnan(curr):
            continue
        final_fks.append(max(0.0, float(max_past - curr)))
    F_final = float(np.mean(final_fks)) if final_fks else 0.0

    return F_final, forgetting_curve, A


def to_hwc_uint8(img: torch.Tensor):
    # img shape can be H W or C H W or B C H W or B H W
    if img.dim() == 2:
        x = img.unsqueeze(0)  # 1 H W
    elif img.dim() == 3:
        if img.shape[0] in [1, 3]:  # C H W
            x = img
        else:  # H W C assumed
            x = img.permute(2, 0, 1)
    elif img.dim() == 4:
        raise ValueError("Batch tensor passed to _to_hwc_uint8. Call this on a single image.")
    else:
        raise ValueError(f"Unsupported tensor shape {img.shape}")

    # normalize to 0..1
    x = x.detach().cpu().float()
    vmin = x.min().item()
    vmax = x.max().item()
    if vmin >= 0.0 and vmax <= 1.0:
        x01 = x
    elif vmin >= -1.0 and vmax <= 1.0:
        x01 = (x + 1.0) / 2.0
    else:
        # per image min max normalization
        x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # to H W C
    if x01.shape[0] == 1:
        x01 = x01.repeat(3, 1, 1)  # make it 3 channel for consistent logging
    x_hwc = x01.permute(1, 2, 0).clamp(0, 1)
    x_uint8 = (x_hwc * 255.0).round().to(torch.uint8)
    return x_uint8

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            idx = torch.randperm(data.shape[0], device=data.device)[:max_size]
            data = data[idx]
            target = target[idx]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
    
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        
        self.data_dec = data
        self.target_dec = target
        
        return self.data_dec, self.target_dec

    def loader(self, args, augment=True, verbose=True):
        if args.dataset in ['imagenet', 'imagenette'] or args.dataset == 'seq-core50':
            train_transform, _ = transform_imagenet(
                augment=augment,
                from_tensor=True,
                size=0,
                rrc=args.rrc,
                rrc_size=self.size[0]
            )
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'domain-net':
            train_transform, _ = transform_domainnet(augment=augment, from_tensor=True)
        elif args.dataset == 'seq-core50-s':
            train_transform, _ = transform_core50s(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'perm-mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True, normalize=False)
        elif args.dataset == 'rot-mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True, normalize=False)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)
        elif args.dataset == 'pacs':
            train_transform, _ = transform_pacs(augment=augment, from_tensor=True)

        # GPU에서 decode 수행

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            data = self.data[self.targets == c].detach()       # GPU 텐서
            target = self.targets[self.targets == c].detach()  # GPU 텐서

            data, target = self.decode(data, target)           # GPU decode
            data_dec.append(data)
            target_dec.append(target)

        # GPU에서 concat
        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        # CPU로 한 번만 이동
        data_dec = data_dec.cpu()
        target_dec = target_dec.cpu()

        if verbose:
            print("Decode condensed data", data_dec.shape)

        # Dataset은 반드시 CPU 텐서 기반
        train_dataset = TensorDataset(data_dec, target_dec, train_transform)

        # DataLoader 생성
        nw = 0 #if not augment else args.workers
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0)
        )

        return train_loader


    def test(self, args, val_loader, logger):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        result, model = test_data(args, loader, val_loader, test_resnet=False, logger=logger)

        return result, model

def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    return aug_batch

import os
import math
import torch
import torchvision
import numpy as np
import wandb
from typing import Optional, Callable, Dict

# -------------------- utility: per-tile reduction --------------------
def _tile_weighted_mean(x: torch.Tensor, ids_all: torch.Tensor, n_instance: int):
    """
    Compute per-tile mean from a [B,C,H,W] tensor using ids_all (Long) in [B,1,H,W].
    We average over channels first to stabilize statistics.
    Returns: [n_instance] on same device.
    """
    assert x.dim() == 4 and ids_all.shape[:1] == x.shape[:1]
    # reduce over channels -> [B,H,W]
    val = x.abs().mean(dim=1)  # [B,H,W]
    ids = ids_all.view(-1)
    weights = val.reshape(-1)
    # bincount supports CUDA; minlength ensures all tiles covered
    sums   = torch.bincount(ids, weights=weights, minlength=n_instance)
    counts = torch.bincount(ids, minlength=n_instance).clamp_min_(1)
    return sums / counts  # [n_instance]

def _pearsonr(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Pearson correlation for 1D tensors a,b on same device. Returns float (CPU).
    """
    a = a.float()
    b = b.float()
    am = a.mean()
    bm = b.mean()
    num = ((a - am) * (b - bm)).sum()
    den = (a.var(unbiased=False) * b.var(unbiased=False)).clamp_min(1e-12).sqrt()
    return float((num / den).detach().cpu())

def _entropy01(v: torch.Tensor, eps=1e-12) -> float:
    """
    Entropy of values normalized to [0,1] histogram. Cheap proxy:
    build 64-bin histogram over [0,1] and compute Shannon entropy.
    """
    vc = v.detach().clamp(0, 1).float().cpu().numpy()
    hist, _ = np.histogram(vc, bins=64, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    Z = p.sum()
    if Z <= 0:
        return 0.0
    p /= Z
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def _jsd01(p: torch.Tensor, q: torch.Tensor, bins=64, eps=1e-12) -> float:
    """
    Jensen-Shannon divergence between two [0,1] vectors via histograms (symmetric).
    """
    pc = p.detach().clamp(0,1).float().cpu().numpy()
    qc = q.detach().clamp(0,1).float().cpu().numpy()
    ph, _ = np.histogram(pc, bins=bins, range=(0,1), density=True)
    qh, _ = np.histogram(qc, bins=bins, range=(0,1), density=True)
    ph = np.clip(ph, eps, None); ph /= ph.sum()
    qh = np.clip(qh, eps, None); qh /= qh.sum()
    m = 0.5*(ph+qh)
    def _kl(a,b): return float(np.sum(a*np.log(a/b)))
    return 0.5*_kl(ph,m) + 0.5*_kl(qh,m)

def _decode_tile_id(tile_id: int, f: int):
    """
    Decode a flat tile id into (b, r, c) with fxf tiles per image.
    """
    tiles_per_img = f * f
    b  = tile_id // tiles_per_img
    t  = tile_id % tiles_per_img
    r  = t // f
    c  = t %  f
    return int(b), int(r), int(c)
def should_log_meta(args, it: int) -> bool:
    if args.log_meta == "off":
        return False
    if args.log_meta_every <= 0:
        return False
    return (it % args.log_meta_every) == 0

@torch.no_grad()
def log_alpha_beta(
    *,
    it: int,
    t_task: int,
    synset,
    ids_all: torch.Tensor,      # [B,1,H,W]
    f: int,
    alpha_vec_fn,               # callable or None
    beta_vec_fn,                # callable or None
    g_plastic: torch.Tensor=None,   # [B,C,H,W] or None
    g_stable: torch.Tensor=None,    # [B,C,H,W] or None
    args=None,
    tag: str = "meta"
):
    """Configurable (fast vs rich) logging for alpha/beta."""
    if args.log_meta == "off":
        return

    device = synset.data.device
    B, C, H, W = synset.data.shape
    tiles_per_img = f * f
    n_instance = B * tiles_per_img

    # bounds
    lo_a = getattr(args, "alpha_lo", getattr(args, "uni_lo", 0.0))
    hi_a = getattr(args, "alpha_hi", getattr(args, "uni_hi", 1.0))
    lo_b = getattr(args, "beta_lo",  getattr(args, "uni_lo", 0.0))
    hi_b = getattr(args, "beta_hi",  getattr(args, "uni_hi", 1.0))

    level_rich = (args.log_meta == "rich")

    def _stats_only(name, vec, lo, hi):
        vc = vec.detach().float().cpu()
        d = {
            f"{tag}/{name}/mean": float(vc.mean()),
            f"{tag}/{name}/std":  float(vc.std(unbiased=False)),
            f"{tag}/{name}/min":  float(vc.min()),
            f"{tag}/{name}/max":  float(vc.max()),
        }
        # saturation near bounds
        rng = max(1e-6, (hi - lo))
        eps = 0.02 * rng
        d.update({
            f"{tag}/{name}/frac_low":  float(((vc - lo) <= eps).float().mean()),
            f"{tag}/{name}/frac_high": float(((hi - vc) <= eps).float().mean()),
        })
        wandb.log(d, step=it)

    def _maybe_hist(name, vec):
        if not level_rich:
            return
        bins = max(8, int(getattr(args, "log_meta_bins", 64)))
        wandb.log({f"{tag}/{name}/hist": wandb.Histogram(vec.detach().cpu().numpy(), num_bins=bins)}, step=it)

    def _maybe_grid(name, vec, lo, hi):
        if not level_rich or not getattr(args, "log_meta_grid", False):
            return
        # per-pixel map (normalize 0~1 for rendering)
        m = vec[ids_all].detach()                                # [B,1,H,W]
        m = ((m - lo) / max(1e-12, (hi - lo))).clamp_(0, 1)
        # subsample batch for speed
        maxB = max(1, int(getattr(args, "log_meta_grid_maxB", 16)))
        if B > maxB:
            m = m[:maxB]
        grid = torchvision.utils.make_grid(m.expand(-1,3,-1,-1), nrow=min(8, m.shape[0]), normalize=False)
        wandb.log({f"{tag}/{name}/tiles_grid": wandb.Image((grid*255).byte().cpu().permute(1,2,0).numpy(),
                                                           caption=f"{name} tiles (task {t_task}, iter {it})")},
                  step=it)

    def _maybe_per_class(name, vec):
        if not level_rich or not getattr(args, "log_meta_per_class", False):
            return
        targets = synset.targets.detach().cpu()
        nclass = int(targets.max()) + 1 if targets.numel() > 0 else 0
        for c in range(nclass):
            idx = torch.nonzero(synset.targets == c, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            base = (idx.view(-1,1) * tiles_per_img).to(vec.device)            # [B_c,1]
            offs = torch.arange(tiles_per_img, device=vec.device).view(1,-1)  # [1, f*f]
            ids_c = (base + offs).view(-1)
            vc = vec[ids_c].detach().float().cpu()
            wandb.log({
                f"{tag}/{name}/class_{c}/mean": float(vc.mean()),
                f"{tag}/{name}/class_{c}/std":  float(vc.std(unbiased=False)),
                f"{tag}/{name}/class_{c}/min":  float(vc.min()),
                f"{tag}/{name}/class_{c}/max":  float(vc.max()),
                # hist is heavy, include only in rich mode:
                f"{tag}/{name}/class_{c}/hist": wandb.Histogram(vc.numpy(), num_bins=getattr(args, "log_meta_bins", 64)),
            }, step=it)

    def _maybe_corr(name, vec, grad, grad_name):
        if not level_rich or not getattr(args, "log_meta_corr", False) or grad is None:
            return
        # per-tile mean |grad|
        g = grad.detach().abs().mean(dim=1)                 # [B,H,W]
        ids = ids_all.view(-1)
        g_tile = torch.bincount(ids, weights=g.reshape(-1), minlength=n_instance)
        cnts   = torch.bincount(ids, minlength=n_instance).clamp_min(1)
        g_tile = g_tile / cnts
        # pearson corr
        a = vec.detach().float()
        b = g_tile.detach().float().to(a.device)
        am, bm = a.mean(), b.mean()
        num = ((a-am)*(b-bm)).sum()
        den = (a.var(unbiased=False)*b.var(unbiased=False)).clamp_min(1e-12).sqrt()
        corr = float((num/den).detach().cpu())
        wandb.log({f"{tag}/{name}_vs_{grad_name}_corr": corr}, step=it)

    def _maybe_topk(name, vec):
        k = int(getattr(args, "log_meta_topk", 0))
        if not level_rich or k <= 0:
            return
        vcpu = vec.detach().cpu()
        k = min(k, vcpu.numel())
        topv, topi = torch.topk(vcpu, k=k, largest=True)
        botv, boti = torch.topk(vcpu, k=k, largest=False)
        rows = []
        for rank, (val, tid) in enumerate(zip(topv.tolist(), topi.tolist()), 1):
            b  = tid // tiles_per_img
            t  = tid % tiles_per_img
            r  = t // f
            c_ = t %  f
            rows.append(["top", rank, int(tid), int(b), int(r), int(c_), float(val)])
        for rank, (val, tid) in enumerate(zip(botv.tolist(), boti.tolist()), 1):
            b  = tid // tiles_per_img
            t  = tid % tiles_per_img
            r  = t // f
            c_ = t %  f
            rows.append(["bottom", rank, int(tid), int(b), int(r), int(c_), float(val)])
        table = wandb.Table(columns=["which","rank","tile_id","b","row","col","value"], data=rows)
        wandb.log({f"{tag}/{name}/topk_tiles": table}, step=it)

    # ---- log alpha
    if alpha_vec_fn is not None:
        a_vec = alpha_vec_fn()  # [n_instance]
        _stats_only("alpha", a_vec, lo_a, hi_a)
        if level_rich:
            _maybe_hist("alpha", a_vec)
            _maybe_grid("alpha", a_vec, lo_a, hi_a)
            _maybe_per_class("alpha", a_vec)
            _maybe_corr("alpha", a_vec, g_stable, "g_stable")
            _maybe_topk("alpha", a_vec)

    # ---- log beta
    if beta_vec_fn is not None:
        b_vec = beta_vec_fn()
        _stats_only("beta", b_vec, lo_b, hi_b)
        if level_rich:
            _maybe_hist("beta", b_vec)
            _maybe_grid("beta", b_vec, lo_b, hi_b)
            _maybe_per_class("beta", b_vec)
            _maybe_corr("beta", b_vec, g_plastic, "g_plastic")
            _maybe_topk("beta", b_vec)