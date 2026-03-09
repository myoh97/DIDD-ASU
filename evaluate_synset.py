import os
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset
from model_dist import define_model
from train import train
from data import TensorDataset, ImageFolder, MultiEpochsDataLoader
from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
import models.resnet as RN
import models.densenet_cifar as DN
from efficientnet_pytorch import EfficientNet
import models.convnet as CN
import models.resnet_ap as RNAP

def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#######################################################################################
#                           the models to evaluate                                    #
#######################################################################################

def convnet3(args, nclass, logger=None):
    width = int(128)
    model = CN.ConvNet(nclass,
                        net_norm='instance',
                        net_depth=3,
                        net_width=width,
                        channel=args.nch,
                        im_size=(args.size, args.size))
    if logger is not None:
        logger(f"=> creating model convnet-3, norm: instance")
    return model


def resnetap10_in(args, nclass, logger=None):
    model = RNAP.ResNetAP(args.dataset,
                              10,
                              nclass,
                              width=1.0,
                              norm_type='instance',
                              size=args.size,
                              nch=args.nch)
    if logger is not None:
        logger(f"=> creating model rensetap-10, norm: instance")
    return model


def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model


def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model


def resnet18_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 18, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-18, norm: batch")
    return model


def densenet(args, nclass, logger=None):
    if 'cifar' == args.dataset[:5]:
        model = DN.densenet_cifar(nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating DenseNet")
    return model


def efficientnet(args, nclass, logger=None):
    if args.dataset == 'imagenet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating EfficientNet")
    return model

#######################################################################################
#                              decode functions                                       #
#######################################################################################

def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

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
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)

def decode_fn(data, target, factor, decode_type):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target

def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    # save_img('./test_results/test_dec.png', data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec


def load_data_path(args):
    """Load condensed data from the given path
    """

    if args.dataset in ['imagenet', 'imagenette']:
        valdir = os.path.join(args.data_dir, 'val')
        train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)
        # Load condensed dataset
        if args.nclass == 10:
            data, target = torch.load(args.syn_data_dir)

        else:
            data_all = []
            target_all = []
            for idx in range(args.nclass // args.nclass_sub):
                path = f'{args.syn_data_dir}/phase_{idx}/data_best.pt'
                data, target = torch.load(path)
                data_all.append(data)
                target_all.append(target)
                print(f"Load data from {path}")

            data = torch.cat(data_all)
            target = torch.cat(target_all)

        print("Load condensed data ", data.shape, args.syn_data_dir)

        if args.factor > 1:
            data, target = decode(args, data, target)
        train_transform, _ = transform_imagenet(augment=args.augment,
                                                from_tensor=True,
                                                size=args.size,
                                                rrc=args.rrc)
        train_dataset = TensorDataset(data, target, train_transform)

        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)

    else:
        if args.dataset[:5] == 'cifar':
            transform_fn = transform_cifar
        elif args.dataset == 'svhn':
            transform_fn = transform_svhn
        elif args.dataset == 'mnist':
            transform_fn = transform_mnist
        elif args.dataset == 'fashion':
            transform_fn = transform_fashion
        train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)


        data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
        print("Load condensed data ", args.save_dir, data.shape)
        # This does not make difference to the performance
        # data = torch.clamp(data, min=0., max=1.)
        if args.factor > 1:
            data, target = decode(args, data, target)

        train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
        train_dataset = TensorDataset(data, target, train_transform)

        # Test dataset
        if args.dataset == 'cifar10':
            val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                       train=False,
                                                       transform=test_transform)
        elif args.dataset == 'cifar100':
            val_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                        train=False,
                                                        transform=test_transform)
        elif args.dataset == 'svhn':
            val_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'SVHN'),
                                                    split='test',
                                                    transform=test_transform)
        elif args.dataset == 'mnist':
            val_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                     train=False,
                                                     transform=test_transform)
        elif args.dataset == 'fashion':
            val_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                            train=False,
                                                            transform=test_transform)

    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)
    print()

    return train_dataset, val_dataset




def test_data(args,
              train_loader,
              val_loader,
              model_fn=None,
              repeat=1,
              logger=print):
    """Train neural networks on condensed data
    """
    if model_fn is None:
        model_fn_ls = [define_model]
    else:
        model_fn_ls = [model_fn]

    for model_fn in model_fn_ls:
        best_acc_l = []
        acc_l = []
        for _ in range(repeat):
            model = model_fn(args, args.nclass, logger=logger)
            best_acc, acc = train(args, model, train_loader, val_loader, logger=logger)
            best_acc_l.append(best_acc)
            acc_l.append(acc)
        logger(
            f'Repeat {repeat} => Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}\n')



if __name__ == '__main__':

    import torch.backends.cudnn as cudnn
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dseed', default=0, type=int, help='seed for class sampling')
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument('--decode_type',
                    type=str,
                    default='single',
                    choices=['single', 'multi', 'bound'],
                    help='multi-formation type')
    parser.add_argument("--nch", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--syn_data_dir", type=str, default="")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument('--dsa',
                    type=str2bool,
                    default=True,
                    help='Use DSA augmentation for evaluation or not')
    parser.add_argument('--rrc',
                    type=str2bool,
                    default=True,
                    help='use random resize crop for ImageNet')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument("--epoch_print_freq", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument('-l',
                    '--load_memory',
                    type=str2bool,
                    default=True,
                    help='load training images on the memory')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size for training')
    parser.add_argument('--mixup',
                    default='cut',
                    type=str,
                    choices=('vanilla', 'cut'),
                    help='mixup choice for evaluation')

    parser.add_argument('--beta', default=1.0, type=float, help='mixup beta distribution')
    parser.add_argument('--mix_p', default=0.5, type=float, help='mixup probability')
    parser.add_argument('--epoch_eval_interval',
                    default=500,
                    type=int)
    args = parser.parse_args()

    if args.dsa:
        args.augment = False
        print("DSA strategy: ", args.dsa_strategy)
    else:
        args.augment = True

    cudnn.benchmark = True


    train_dataset, val_dataset = load_data_path(args)

    train_loader = MultiEpochsDataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers if args.augment else 0,
                                            persistent_workers=args.augment > 0)
    val_loader = MultiEpochsDataLoader(val_dataset,
                                        batch_size=args.batch_size // 2,
                                        shuffle=False,
                                        persistent_workers=True,
                                        num_workers=4)

    test_data(args, train_loader, val_loader, repeat=1, model_fn=resnetap10_in)



