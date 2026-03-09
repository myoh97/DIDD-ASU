import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
from efficientnet_pytorch import EfficientNet

import timm

import torch.nn as nn

def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset,
                        args.depth,
                        nclass,
                        norm_type=args.norm_type,
                        size=size,
                        nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.dataset,
                            args.depth,
                            nclass,
                            width=args.width,
                            norm_type=args.norm_type,
                            size=size,
                            nch=args.nch)
    elif args.net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        model = CN.ConvNet(nclass,
                        net_norm=args.norm_type,
                        net_depth=args.depth,
                        net_width=width,
                        channel=args.nch,
                        im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")
        # logger('# model parameters: {:.1f}M'.format(
        #     sum([p.data.nelement() for p in model.parameters()]) / 10**6))

    return model