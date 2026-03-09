# GPU-optimized version of DiffAug
# All operations are performed on the same device as input tensor
import torch
import torch.nn.functional as F


class DiffAug():
    def __init__(self,
                 strategy='color_crop_cutout_flip_scale_rotate',
                 batch=False,
                 ratio_cutout=0.5,
                 single=False):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

        self.batch = batch

        self.aug = True
        if strategy == '' or strategy.lower() == 'none':
            self.aug = False
        else:
            self.strategy = []
            self.flip = False
            self.color = False
            self.cutout = False
            for aug in strategy.lower().split('_'):
                if aug == 'flip' and single == False:
                    self.flip = True
                elif aug == 'color' and single == False:
                    self.color = True
                elif aug == 'cutout' and single == False:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            'color': [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            'crop': [self.crop_fn],
            'cutout': [self.cutout_fn],
            'flip': [self.flip_fn],
            'scale': [self.scale_fn],
            'rotate': [self.rotate_fn],
            'translate': [self.translate_fn],
        }

    def __call__(self, x, single_aug=True, seed=-1):
        if not self.aug:
            return x
        else:
            if self.flip:
                self.set_seed(seed, x.device)
                x = self.flip_fn(x, self.batch)
            if self.color:
                for f in self.aug_fn['color']:
                    self.set_seed(seed, x.device)
                    x = f(x, self.batch)
            if len(self.strategy) > 0:
                if single_aug:
                    # single
                    self.set_seed(seed, x.device)
                    idx = torch.randint(len(self.strategy), (1,), device=x.device).item()
                    p = self.strategy[idx]
                    for f in self.aug_fn[p]:
                        self.set_seed(seed, x.device)
                        x = f(x, self.batch)
                else:
                    # multiple
                    for p in self.strategy:
                        for f in self.aug_fn[p]:
                            self.set_seed(seed, x.device)
                            x = f(x, self.batch)
            if self.cutout:
                self.set_seed(seed, x.device)
                x = self.cutout_fn(x, self.batch)

            x = x.contiguous()
            return x

    def set_seed(self, seed, device):
        if seed > 0:
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed)

    def scale_fn(self, x, batch=True):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale
        device = x.device
        dtype = x.dtype

        if batch:
            # Generate random values directly on device
            rand_vals = torch.rand(2, device=device, dtype=dtype)
            sx = rand_vals[0] * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = rand_vals[1] * (ratio - 1.0 / ratio) + 1.0 / ratio
            
            theta = torch.tensor([[sx, 0, 0], [0, sy, 0]], dtype=dtype, device=device)
            theta = theta.unsqueeze(0).expand(x.shape[0], 2, 3)
        else:
            rand_vals = torch.rand(x.shape[0], 2, device=device, dtype=dtype)
            sx = rand_vals[:, 0] * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = rand_vals[:, 1] * (ratio - 1.0 / ratio) + 1.0 / ratio
            
            theta = torch.zeros(x.shape[0], 2, 3, dtype=dtype, device=device)
            theta[:, 0, 0] = sx
            theta[:, 1, 1] = sy

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def rotate_fn(self, x, batch=True):
        # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.ratio_rotate
        device = x.device
        dtype = x.dtype
        pi = torch.tensor(3.14159265359, device=device, dtype=dtype)

        if batch:
            # Generate random angle directly on device
            rand_val = torch.rand(1, device=device, dtype=dtype)
            angle = (rand_val - 0.5) * 2 * ratio / 180 * pi
            
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            
            theta = torch.tensor([[cos_theta, -sin_theta, 0],
                                 [sin_theta, cos_theta, 0]], dtype=dtype, device=device)
            theta = theta.unsqueeze(0).expand(x.shape[0], 2, 3)
        else:
            rand_vals = torch.rand(x.shape[0], device=device, dtype=dtype)
            angles = (rand_vals - 0.5) * 2 * ratio / 180 * pi
            
            cos_theta = torch.cos(angles)
            sin_theta = torch.sin(angles)
            
            theta = torch.zeros(x.shape[0], 2, 3, dtype=dtype, device=device)
            theta[:, 0, 0] = cos_theta
            theta[:, 0, 1] = -sin_theta
            theta[:, 1, 0] = sin_theta
            theta[:, 1, 1] = cos_theta

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def flip_fn(self, x, batch=True):
        prob = self.prob_flip

        if batch:
            coin = torch.rand(1, device=x.device, dtype=x.dtype)
            if coin < prob:
                return x.flip(3)
            else:
                return x
        else:
            randf = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
            return torch.where(randf < prob, x.flip(3), x)

    def brightness_fn(self, x, batch=True):
        # mean
        ratio = self.brightness

        if batch:
            randb = torch.rand(1, device=x.device, dtype=x.dtype)
        else:
            randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5) * ratio
        return x

    def saturation_fn(self, x, batch=True):
        # channel concentration
        ratio = self.saturation

        x_mean = x.mean(dim=1, keepdim=True)
        if batch:
            rands = torch.rand(1, device=x.device, dtype=x.dtype)
        else:
            rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def contrast_fn(self, x, batch=True):
        # spatially concentrating
        ratio = self.contrast

        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        if batch:
            randc = torch.rand(1, device=x.device, dtype=x.dtype)
        else:
            randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def translate_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad
        device = x.device

        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_y = torch.randint(-shift_y, shift_y + 1, (1,), device=device)
        else:
            translation_y = torch.randint(-shift_y,
                                          shift_y + 1,
                                          size=[x.size(0), 1, 1],
                                          device=device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=device),
            torch.arange(x.size(2), dtype=torch.long, device=device),
            torch.arange(x.size(3), dtype=torch.long, device=device),
            indexing='ij'
        )
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def crop_fn(self, x, batch=True):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad
        device = x.device

        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        if batch:
            translation_x = torch.randint(-shift_x, shift_x + 1, (1,), device=device)
            translation_y = torch.randint(-shift_y, shift_y + 1, (1,), device=device)
        else:
            translation_x = torch.randint(-shift_x,
                                          shift_x + 1,
                                          size=[x.size(0), 1, 1],
                                          device=device)

            translation_y = torch.randint(-shift_y,
                                          shift_y + 1,
                                          size=[x.size(0), 1, 1],
                                          device=device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=device),
            torch.arange(x.size(2), dtype=torch.long, device=device),
            torch.arange(x.size(3), dtype=torch.long, device=device),
            indexing='ij'
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1, 1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def cutout_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        device = x.device
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), (1,), device=device)
            offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), (1,), device=device)
        else:
            offset_x = torch.randint(0,
                                     x.size(2) + (1 - cutout_size[0] % 2),
                                     size=[x.size(0), 1, 1],
                                     device=device)

            offset_y = torch.randint(0,
                                     x.size(3) + (1 - cutout_size[1] % 2),
                                     size=[x.size(0), 1, 1],
                                     device=device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=device),
            torch.arange(cutout_size[0], dtype=torch.long, device=device),
            torch.arange(cutout_size[1], dtype=torch.long, device=device),
            indexing='ij'
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    def cutout_inv_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        device = x.device
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = torch.randint(0, x.size(2) - cutout_size[0], (1,), device=device)
            offset_y = torch.randint(0, x.size(3) - cutout_size[1], (1,), device=device)
        else:
            offset_x = torch.randint(0,
                                     x.size(2) - cutout_size[0],
                                     size=[x.size(0), 1, 1],
                                     device=device)
            offset_y = torch.randint(0,
                                     x.size(3) - cutout_size[1],
                                     size=[x.size(0), 1, 1],
                                     device=device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=device),
            torch.arange(cutout_size[0], dtype=torch.long, device=device),
            torch.arange(cutout_size[1], dtype=torch.long, device=device),
            indexing='ij'
        )
        grid_x = torch.clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
        mask = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=device)
        mask[grid_batch, grid_x, grid_y] = 1.
        x = x * mask.unsqueeze(1)
        return x