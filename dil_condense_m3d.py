import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import wandb

from data import ClassDataLoader, ClassMemDataLoader, save_img
from dil.datasets import get_dataset
from m3dloss import M3DLoss
from model_dist import define_model
from util import compute_average_forgetting, Synthesizer, diffaug, to_hwc_uint8

DATASET = {
    'R-MNIST': 'rot-mnist',
    'CORe50': 'seq-core50',
    'CORe50-S': 'seq-core50-s',
    'PACS': 'pacs',
}
NCLASS = {
    'R-MNIST': 10,
    'CORe50': 50,
    'CORe50-S': 50,
    'PACS': 7,
}

def apply_augmentation(args, aug, img, img_syn, img_syn_old, target_old, t, c):
    nr = img.shape[0]
    nd = img_syn.shape[0]

    if img_syn_old is not None:
        data_old_c = img_syn_old[target_old == c]
        img_aug = aug(torch.cat([img, img_syn, data_old_c]))
        img_aug_r = img_aug[:nr]
        img_aug_d = img_aug[nr:nr+nd]
        img_aug_d_old = img_aug[nr+nd:]
    else:
        img_aug = aug(torch.cat([img, img_syn]))
        img_aug_r = img_aug[:nr]
        img_aug_d = img_aug[nr:nr+nd]
        img_aug_d_old = None

    return img_aug_r, img_aug_d, img_aug_d_old

def do_distill(
    args,
    nclass,
    synset,
    loader_real,
    aug,
    t,
    device,
    JOINT,
    N_TASKS,
    it_log=1500,
    synset_old=None,
    model_old=None,
):
    # -------------------- Iterations & Losses --------------------
    n_iter = args.niter * (N_TASKS if JOINT else 1)
    m3d_criterion = M3DLoss(kernel_type=args.kernel)

    # -------------------- Decode old synthetic --------------------
    if synset_old is not None:
        img_syn_old_all, target_old_all = synset_old.decode(synset_old.data, synset_old.targets)
        img_syn_old_all = img_syn_old_all.detach()
        target_old_all = target_old_all.detach()
    else:
        img_syn_old_all, target_old_all = None, None

    # -------------------- Build global tile ids --------------------
    B_all, C, H, W = synset.data.shape
    f  = max(1, args.factor)
    sH = max(1, H // f)
    sW = max(1, W // f)

    ih = (torch.arange(H, device=synset.data.device) // sH).clamp(max=f-1)  # [H]
    iw = (torch.arange(W, device=synset.data.device) // sW).clamp(max=f-1)  # [W]
    ids_in_img = (ih.view(H, 1) * f + iw.view(1, W)).to(torch.long)         # [H, W] in [0 .. f*f-1]

    ids_all = ids_in_img.view(1, 1, H, W).repeat(B_all, 1, 1, 1)            # [B_all,1,H,W]
    ids_all += torch.arange(B_all, device=synset.data.device).view(B_all, 1, 1, 1) * (f * f)
    n_instance_all = B_all * f * f

    # -------------------- Meta-parameters (alpha/beta) --------------------
    lo_a = args.alpha_lo
    hi_a = args.alpha_hi
    lo_b = args.beta_lo
    hi_b = args.beta_hi

    lr_alpha = args.lr_alpha
    lr_beta  = args.lr_beta

    meta_param_groups = []
    alpha_param = None
    beta_param  = None

    if args.asym:
        alpha_param = nn.Parameter(torch.zeros(n_instance_all, device=synset.data.device))
        meta_param_groups.append({"params": [alpha_param], "lr": lr_alpha})
        def alpha_vec():
            return lo_a + (hi_a - lo_a) * torch.sigmoid(alpha_param)

        beta_param = nn.Parameter(torch.zeros(n_instance_all, device=synset.data.device))
        meta_param_groups.append({"params": [beta_param], "lr": lr_beta})
        def beta_vec():
            return lo_b + (hi_b - lo_b) * torch.sigmoid(beta_param)

    else:
        def alpha_vec():
            return torch.ones(n_instance_all, device=synset.data.device, dtype=synset.data.dtype)
        def beta_vec():
            return torch.ones(n_instance_all, device=synset.data.device, dtype=synset.data.dtype)

    opt_meta = torch.optim.Adam(meta_param_groups, betas=(0.9, 0.999)) if meta_param_groups else None

    # -------------------- Image optimizer --------------------
    opt_img = torch.optim.SGD([p for p in synset.parameters()], lr=args.lr_img, momentum=args.mom_img)

    for it in range(n_iter):
        model = define_model(args, nclass).to(device)
        model.train()
        # No parameter grads needed for computing input grads (saves memory).
        for p in model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            synset.data.clamp_(0.0, 1.0)

        loss_total = 0.0

        # -------------------- Class-wise updates --------------------
        for c in range(nclass):

            # === 1) Sample real/synthetic for class c ===
            img_real, _ = loader_real.class_sample(c)                         # [N_r,C,H,W]
            img_syn_c, target_syn_c = synset.sample(c, max_size=args.batch_syn_max)  # [N_d,C,H,W]

            # === 2) Augmentation ===
            img_aug_r, img_aug_d, img_aug_d_old = apply_augmentation(
                args, aug, img_real, img_syn_c, img_syn_old_all, target_old_all, t, c)

            # === 3) Compute plasticity loss (match real) ===
            with torch.no_grad():
                _, feat_tg = model(img_aug_r, return_features=True)          # target features
            _, feat = model(img_aug_d, return_features=True)
            loss_plastic = m3d_criterion(feat, feat_tg)

            # === 4) Compute stability loss (match old synthetic), if available ===
            loss_stable = torch.tensor(0.0, device=device)
            use_stability = (t > 0) and (img_aug_d_old is not None) and (args.asym)
            if use_stability:
                with torch.no_grad():
                    _, feat_old_tg = model(img_aug_d_old, return_features=True)
                loss_stable = m3d_criterion(feat, feat_old_tg) * args.lambda_reg

            # === 5) Get gradients at X_pre (current synset.data) ===
            g_plastic = torch.autograd.grad(loss_plastic, synset.data, retain_graph=True)[0]   # [B_all,C,H,W]
            if use_stability:
                g_stable_tmp = torch.autograd.grad(loss_stable, synset.data, retain_graph=False, allow_unused=True)[0]
                g_stable = torch.zeros_like(g_plastic) if g_stable_tmp is None else g_stable_tmp
            else:
                g_stable = torch.zeros_like(g_plastic)

            # === 6) Meta-step: update alpha/beta ===
            if (t>0) and (opt_meta is not None) and (args.asym):
                # indices in synset corresponding to class c
                idx_c = torch.nonzero(synset.targets == c, as_tuple=True)[0]        # [B_c]
                if idx_c.numel() > 0:
                    # Gather class-specific slices to reduce memory
                    a_map_all = alpha_vec()[ids_all]                                # [B_all,1,H,W]
                    b_map_all = beta_vec()[ids_all]                                 # [B_all,1,H,W]
                    # Channel-broadcast & index to class c
                    a_map_c = a_map_all[idx_c].expand(-1, C, -1, -1)               # [B_c,C,H,W]
                    b_map_c = b_map_all[idx_c].expand(-1, C, -1, -1)               # [B_c,C,H,W]

                    # Fixed “pre-step” point and detached grads (avoid second-order on grads)
                    X_pre_c   = synset.data.detach()[idx_c]                         # [B_c,C,H,W]
                    gP_c      = g_plastic.detach()[idx_c]                           # [B_c,C,H,W]
                    gS_c      = g_stable.detach()[idx_c]                            # [B_c,C,H,W]

                    # Look-ahead one step with current alpha/beta (only alpha/beta need grads)
                    X_prime_c = X_pre_c - args.lr_img * (b_map_c * gP_c + a_map_c * gS_c)   # alpha/beta in-graph
                    # --- Meta objectives ---
                    meta_terms = []

                    img_prime_c, _ = synset.decode(X_prime_c, synset.targets[idx_c])
                    img_aug_r_meta, img_aug_d_meta, img_aug_d_old_meta = apply_augmentation(
                        args, aug, img_real, img_prime_c, img_syn_old_all, target_old_all, t, c)
                    
                    # Stability meta-loss
                    _, feat_prime_c = model(img_aug_d_meta, return_features=True)
                    if args.asym:
                        with torch.no_grad():
                            _, feat_old_meta = model(img_aug_d_old_meta, return_features=True)
                        loss_stab_meta = m3d_criterion(feat_prime_c, feat_old_meta) * args.lambda_reg
                        meta_terms.append(loss_stab_meta)
                        # alpha sum penalty
                        penalty_alpha = alpha_vec().mean() * args.lambda_alpha
                        meta_terms.append(penalty_alpha)

                        # Plasticity meta-loss (push current tiles to better match current real targets)
                        with torch.no_grad():
                            _, feat_tg_meta = model(img_aug_r_meta, return_features=True)
                        loss_plas_meta = m3d_criterion(feat_prime_c, feat_tg_meta)
                        meta_terms.append(loss_plas_meta)
                        # beta sum penalty
                        penalty_beta = beta_vec().mean() * args.lambda_beta
                        meta_terms.append(penalty_beta)

                    if meta_terms:
                        opt_meta.zero_grad()
                        loss_meta = torch.stack([m if torch.is_tensor(m) else torch.tensor(m, device=device)
                                                for m in meta_terms]).sum()
                        # backprop to alpha/beta only
                        loss_meta.backward()
                        opt_meta.step()
                        
                    wandb.log({
                        'alpha': alpha_vec().mean().item(),
                        'beta': beta_vec().mean().item(),
                        'meta_loss_plastic': loss_plastic.item() if args.asym else 0.0,
                        'meta_loss_stable': loss_stable.item() if args.asym else 0.0,
                        'penalty_alpha': penalty_alpha.item() if args.asym else 0.0,
                        'penalty_beta': penalty_beta.item() if args.asym else 0.0,
                    })

            # === 7) Real image update WITH UPDATED alpha/beta ===
            opt_img.zero_grad()
            if args.asym:
                a_map = alpha_vec()[ids_all].expand(-1, C, -1, -1)   # [B_all,C,H,W]
                b_map = beta_vec()[ids_all].expand(-1, C, -1, -1)    # [B_all,C,H,W]
                grad_apply = b_map * g_plastic + a_map * g_stable
            else:
                grad_apply = g_plastic + g_stable

            with torch.no_grad():
                synset.data.grad = grad_apply
            opt_img.step()

            # === 8) Logging (per class) ===
            total_now = float(loss_plastic.item() + loss_stable.item())
            loss_total += total_now
            wandb.log({
                'train_loss_plastic': loss_plastic.item(),
                'train_loss_stable': loss_stable.item(),
                'train_loss_total': total_now,
            })

        # Pretty scaling for dashboards
        if args.kernel == 'gaussian':
            loss_total *= 1000.0
        elif args.kernel == 'linear':
            loss_total *= 100.0

        if it % it_log == 0:
            logger(f"[Task {t:2d}] (Iter {it:3d}) loss {loss_total/nclass:.2f}")


def condense(args, logger, device='cuda', logger_time=None):
    """Optimize condensed data."""
    torch.cuda.reset_peak_memory_stats()

    # --------------- Initialization ---------------
    nclass = NCLASS[args.dataset]
    FLAG = args.flag
    JOINT = 'joint' in FLAG
    FINETUNE = 'finetune' in FLAG
    args.dataset = DATASET[args.dataset]
    args.batch_size = args.batch_real
    args.num_workers = 0

    wandb.init(project="DIDD", entity="GDDD", name=f"{args.dataset}_IPC{args.ipc}_{args.flag}")
    
    trg_datasets = get_dataset(args, distill=True)
    N_TASKS = trg_datasets.N_TASKS
    if JOINT:
        trg_datasets.set_joint()
        print("Setting dataset to joint mode")
        print(f"N Tasks: {trg_datasets.N_TASKS}")
        print(f"Total number of training images: {len(trg_datasets.train_loaders[0].dataset)}")

    if 'domain-net' in args.dataset:
        trainsets = [loader.dataset.dataset for loader in trg_datasets.train_loaders]
    else:
        trainsets = [loader.dataset for loader in trg_datasets.train_loaders]
    testloaders = trg_datasets.test_loaders
    task_accuracy = {t: [] for t in range(len(trainsets))}

    # --------------- Main Loop (Task) ---------------
    model_old = None
    for t, trainset in enumerate(trainsets):
        if args.load_memory:
            loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real, joint=JOINT)
        else:
            loader_real = ClassDataLoader(trainset,
                                        batch_size=args.batch_real,
                                        num_workers=args.workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True)
        
        nch, hs, ws = trainset[0][0].shape
        if t > 0:
            synset_old = Synthesizer(args, nclass, nch, hs, ws)
            synset_old.data = synset.data.detach().clone()
            synset_old.targets = synset.targets.detach().clone()
        else:
            synset_old = None

        # Define syn dataset
        synset = Synthesizer(args, nclass, nch, hs, ws)

        if t == 0:
            synset.init(loader_real, init_type=args.init)
        else:
            synset.data = nn.Parameter(synset_old.data.detach().clone().to(synset.device))

        save_img(os.path.join(args.save_dir, f'task{t}_init.png'),
                    synset.data,
                    unnormalize=False,
                    dataname=args.dataset)

        aug = diffaug(args)

        # --------------- Distillation ---------------
        logger(f"\n [Task {t}] M3D: Start condensing with {args.kernel} kernel for {args.niter} iteration")

        start = time.time()
        do_distill(args, nclass, synset, loader_real, aug, t, device, JOINT, N_TASKS,
                   synset_old=synset_old,
                   model_old=model_old)

        # ------------------- After Task T --------------------
        img_to_save = synset.data.clone().detach().cpu()
        if img_to_save.dim() == 4:
            if img_to_save.shape[1] in [1, 3]:        # B C H W
                batch = img_to_save
            elif img_to_save.shape[-1] in [1, 3]:     # B H W C
                batch = img_to_save.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Ambiguous tensor shape {tuple(img_to_save.shape)}. Expected channels to be 1 or 3.")

            images_to_log = []
            for i in range(batch.shape[0]):
                img_uint8 = to_hwc_uint8(batch[i])
                images_to_log.append(wandb.Image(img_uint8.numpy(), caption=f"Image {i}"))

            grid_bchw = torchvision.utils.make_grid(
                batch.detach().cpu().float(),
                nrow=min(8, batch.shape[0]),
                normalize=True,
            )
            grid_uint8 = to_hwc_uint8(grid_bchw)

            wandb.log({
                f"Task_{t}_Condensed_Images": images_to_log,
                f"Task_{t}_Condensed_Grid": wandb.Image(grid_uint8.numpy(), caption="Grid"),
            })

        torch.save(
            [synset.data.detach().cpu(), synset.targets.cpu()],
            os.path.join(args.save_dir, f'task{t}_data{args.niter + 1}.pt'))
        logger("img and data saved!")

        # --------------- Evaluation ---------------
        if 'debug' in FLAG:
            logger("Debug mode: skipping evaluation")
        else:
            for test_idx, testloader in enumerate(testloaders):
                if not JOINT:
                    if test_idx > t:
                        task_accuracy[t].append(0.0)
                        continue
                conv_result, model_old = synset.test(args, testloader, logger)
                task_accuracy[t].append(conv_result)
                logger(f"->->->->->->->->->->->->-> [Task {t}] {test_idx}-th Task Result: {conv_result:.2f}")

            for acc in task_accuracy[t]:
                wandb.log({f'accuracy_after_{t}': acc})

        end = time.time()
        logger_time(f"[Task {t}] Finished condensing in {end - start:.2f} seconds")

    # --------------- After All Tasks ---------------
    save_img(
        os.path.join(args.save_dir, 'last.png'),
        synset.data,
        unnormalize=False,
        dataname=args.dataset,
    )

    # save task_accuracy
    with open(os.path.join(args.save_dir, 'task_accuracy.json'), 'w') as f:
        json.dump(task_accuracy, f, indent=4)

    final_acc = sum(task_accuracy[t]) / len(task_accuracy[t])
    wandb.log({"Final_Average_Accuracy": final_acc})
    logger(f"Final Average Accuracy  {final_acc:.4f}")
    for t, acc in task_accuracy.items():
        mean_acc = sum(acc) / len(acc)
        logger(f"Average Accuracy after task {t}  {mean_acc:.4f}")
        wandb.log({f"Average_Accuracy_after_task_{t}": mean_acc})
    
    # compute and save average forgetting
    F_final, forgetting_curve, A = compute_average_forgetting(task_accuracy)
    logger(f"Final Average Forgetting  {F_final:.4f}")
    for t, Ft in enumerate(forgetting_curve):
        logger(f"Average Forgetting after task {t}  {Ft:.4f}")

    # save forgetting artifacts
    np.save(os.path.join(args.save_dir, 'accuracy_matrix.npy'), A)
    with open(os.path.join(args.save_dir, 'forgetting_curve.json'), 'w') as f:
        json.dump({ "final_average_forgetting": float(F_final),
                    "forgetting_curve": [float(x) for x in forgetting_curve]
                  }, f, indent=4)

    # wandb logging
    try:
        wandb.log({"Final_Average_Forgetting": float(F_final)})
        for t, Ft in enumerate(forgetting_curve):
            wandb.log({f"Average_Forgetting_after_task_{t}": float(Ft)})
    except Exception as e:
        logger(f"WandB logging skipped: {e}")

    wandb.finish()


if __name__ == '__main__':
    import argparse

    import torch.backends.cudnn as cudnn

    from misc.cfg import CFG as cfg
    from misc.utils import Logger

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument("--dataset", type=str, default="PACS",
                        help='R-MNIST, CORe50-S, PACS')
    parser.add_argument("--ipc", type=int, default=1)
    parser.add_argument("--f_niter", type=int, default=3000, help='force niter')
    parser.add_argument("--flag", type=str, default=None, help='joint, finetune')

    # ------------- Regularization ----------------
    parser.add_argument("--reg", action='store_true', help='distillation from old synthetic dataset')
    parser.add_argument("--lambda_reg", type=float, default=1.0, help='weight for distillation from old synthetic dataset')

    # ------------- Asymmetric Update ----------------
    parser.add_argument("--asym", action='store_true', help='asymmetric update per image')

    # Stability (alpha)
    parser.add_argument("--alpha_lo", type=float, default=0.0, help='alpha lower bound')
    parser.add_argument("--alpha_hi", type=float, default=2.0, help='alpha upper bound')
    parser.add_argument("--lr_alpha", type=float, default=1e-2, help='alpha learning rate')
    parser.add_argument("--lambda_alpha", type=float, default=1e-4, help='alpha sum penalty weight')

    # Plasticity (beta)
    parser.add_argument("--beta_lo", type=float, default=0.0, help='beta lower bound')
    parser.add_argument("--beta_hi", type=float, default=2.0, help='beta upper bound')
    parser.add_argument("--lr_beta", type=float, default=1e-2, help='beta learning rate')
    parser.add_argument("--lambda_beta", type=float, default=1e-4, help='beta sum penalty weight')

    args = parser.parse_args()

    file_cfg = f"configs/{args.dataset}/IPC{args.ipc}.yaml"
    cfg.merge_from_file(file_cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        if parser._option_string_actions.get(arg_name) is not None:
            continue
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()

    print(f"CUDNN STATUS: {cudnn.enabled}")
    if args.seed > 0:
        print(f"Seed fixed: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    DATANAME = args.dataset if args.dataset != 'imagenet' else f"{args.dataset}{args.nclass}"

    if args.flag:
        args.save_dir = os.path.join(args.results_path, DATANAME, f"IPC{args.ipc}", args.flag)
    else:
        args.save_dir = os.path.join(args.results_path, DATANAME, f"IPC{args.ipc}")

    if args.f_niter:
        args.niter = args.f_niter
    
    os.makedirs(args.save_dir, exist_ok=True)

    logger = Logger(args.save_dir)
    logger_time = Logger(args.save_dir, time=True)
    logger(f"Save dir: {args.save_dir}")

    with open(os.path.join(args.save_dir, 'args.log'), 'w') as f:
        json.dump(args.__dict__, f, indent=3)

    start = time.time()
    condense(args, logger, logger_time=logger_time)
    end = time.time()
    logger_time(f"Total running time: {end - start:.2f} seconds")
