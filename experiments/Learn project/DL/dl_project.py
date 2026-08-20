# %% [markdown]
# # C0.0 WHICH DATASET — set this first, before running anything else
# `DATASET` is read by Cell 1.4 via `os.environ`, so it must be set BEFORE that cell
# executes. On Kaggle the simplest place is right here, in the first cell.
#
# It is ordinary configuration, NOT a Kaggle Secret — secrets are for credentials only
# (the wandb API key). Nothing here is sensitive.
#
# Changing it re-namespaces everything automatically: run ids, the checkpoint directory,
# and the wandb artifact names. RSNA keeps its historical identifiers untouched, so
# switching to another dataset and back cannot disturb results already stored.
# Cell 1.7 asserts all of that; if it does not print PASS, stop.

# %% [CELL 0.0]  Dataset selector

import os
os.environ['DATASET'] = 'RSNA'      # <-- RSNA | VinCXR | LAG | BrainTumor | BraTS2021

# Epochs follow the dataset automatically (MedIAnomaly options.py): 250 for RSNA,
# VinCXR and LAG; 600 for BrainTumor. You do not set them by hand.
print(f"DATASET set to {os.environ['DATASET']}")


# %% [markdown]
# # C1.1 General used support functions
# %% [CELL 1.1]  General used support functions
import subprocess, sys
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

#  This function checks if a package is installed and meets the minimum version requirement. 
#  If not, it installs or upgrades the package using pip.
def check_import(pkg, install_name=None, min_version=None):
    """
    pkg          : the name you 'import' in code (e.g. 'sklearn', 'skimage')
    install_name : pip package name, if it differs from the import name
                   (e.g. import sklearn -> pip install scikit-learn)
    min_version  : minimum acceptable version, e.g. '2.0.0'. None = any version ok.
    """
    name = install_name or pkg
    try:
        __import__(pkg)
        if min_version is not None:
            try:
                installed = version(name)
            except PackageNotFoundError:
                installed = None
            if installed is None or Version(installed) < Version(min_version):
                print(f"  ⚠ {pkg} version {installed} < required {min_version} — upgrading...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                        f'{name}>={min_version}', '-q'])
            else:
                print(f"  ✓ {pkg} ({installed})")
        else:
            print(f"  ✓ {pkg}")
    except ImportError:
        target = f"{name}>={min_version}" if min_version else name
        print(f"  ✗ {pkg} — installing {target}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', target, '-q'])


# %% [markdown]
# # C1.2 Define required packages and check/install them
# %% [CELL 1.2]  define required packages and check/install them
# --- SINGLE SOURCE OF TRUTH ---
# To add a package: just add one line here. Nothing else in this cell changes.
REQUIRED_PACKAGES = {
    'torch':      {'install_name': None,           'min_version': '2.0.0'},
    'sklearn':    {'install_name': 'scikit-learn',  'min_version': '1.2.0'},
    'numpy':      {'install_name': None,            'min_version': '1.24.0'},
    'matplotlib': {'install_name': None,            'min_version': None},
    'pandas':     {'install_name': None,            'min_version': None},
    'seaborn':    {'install_name': None,            'min_version': None}
}

for pkg, spec in REQUIRED_PACKAGES.items():
    check_import(pkg, install_name=spec['install_name'], min_version=spec['min_version'])



# %% [markdown]
# # C1.3 Import the required packages for the project
# %% [CELL 1.3] import the required packages for the project
import os, time, json, random, warnings
import glob as _glob
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torchvision.models as tv_models
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve,
)
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 12,
    'axes.titlesize'   : 14,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'legend.framealpha': 0.9,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

PAL = {   # per-method colours (populated as methods are added)
    'ae': '#4878CF', 'aeu': '#F5A623', 'ae_pl': '#7B68EE',
    'dae': '#2ECC71', 'vae': '#95A5A6', 'ganomaly': '#E84C3D',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == 'cuda':
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True





# %% [markdown]
# ---
# ## Configuration
# Hyperparameters follow the MedIAnomaly reference implementation exactly
# (see CELL 1.4). We reproduce before we tune: any gap to their Table 6 must be
# attributable to our code, not to a different operating point.

# %% [CELL 1.3b]  Network definitions
# UNet + its GroupNorm/Swish/weight-standardised-conv building blocks, ported from
# MedIAnomaly (reconstruction/networks/). This is the architecture their DAE uses --
# the best pixel-level method in their benchmark (~20% Dice over 2nd place).
# Kept inline deliberately: a separate package needed a build step whose Makefile
# prerequisites silently broke on the space in 'Learn project', producing stale
# notebooks. One file = one source of truth.
from math import sqrt

# ---- inlined from networks/base_units/swish.py ------------------------------
# medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


# ---- inlined from networks/base_units/ws_conv.py ------------------------------
from torch.nn import functional as F


# From https://github.com/joe-siyuan-qiao/WeightStandardization
class WNConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# ---- inlined from networks/unet.py ------------------------------
from math import sqrt



def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=True,
            norm="group",
            up_mode='upconv'):
        """
        A modified U-Net implementation [1].

        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward_down(self, x):

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)

        return x, blocks

    def forward_up_without_last(self, x, blocks):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)

        return x

    def forward_without_last(self, x):
        x, blocks = self.forward_down(x)
        x = self.forward_up_without_last(x, blocks)
        return x

    def forward(self, x):
        x = self.get_features(x)
        # return self.last(x)
        return {'x_hat': self.last(x)}

    def get_features(self, x):
        return self.forward_without_last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == '__main__':
    model = UNet()



# ---- MedIAnomaly reconstruction backbone -------------------------------------
# AE / AE-U and their building blocks, ported UNCHANGED so our reproduction of
# Table 6 is not confounded by an architecture difference. AE-U subclasses AE and
# splits the final layer into (x_hat, log_var) for uncertainty-weighted scoring.


# ---- ported verbatim from MedIAnomaly/reconstruction/networks/base_units/conv_layers.py --------
def down_conv(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)


def up_conv(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

# ---- ported verbatim from MedIAnomaly/reconstruction/networks/base_units/blocks.py --------------------
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, num_layers, downsample=False, upsample=False, last_layer=False):
        super(BasicBlock, self).__init__()
        assert not (downsample and upsample)
        layers = []
        if downsample:
            layers.append(down_conv(inplanes, planes))
        elif upsample:
            layers.append(up_conv(inplanes, planes))
        else:
            layers.append(conv3x3(inplanes, planes))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))

        # Deeper block
        if upsample:
            for _ in range(1, num_layers):
                add_layer = [conv3x3(inplanes, inplanes),
                             nn.BatchNorm2d(inplanes),
                             nn.ReLU(inplace=True)]

                layers = add_layer + layers
        else:
            for _ in range(1, num_layers):
                add_layer = [conv3x3(planes, planes),
                             nn.BatchNorm2d(planes),
                             nn.ReLU(inplace=True)]

                layers = layers + add_layer

        if last_layer:
            layers = layers[:-2]  # remove the BN and ReLU for the output layer.

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, num_layers, downsample=False, upsample=False, last_layer=False):
        super(ResBlock, self).__init__()
        assert not (downsample and upsample)
        self.last_layer = last_layer
        self.relu = nn.ReLU(inplace=True)

        layers = []
        if downsample:
            layers.append(down_conv(inplanes, planes))
            self.skip = nn.Sequential(
                down_conv(inplanes, planes),
                nn.BatchNorm2d(planes)
            )
        elif upsample:
            layers.append(up_conv(inplanes, planes))
            self.skip = nn.Sequential(
                up_conv(inplanes, planes),
                nn.BatchNorm2d(planes)
            )
        else:
            layers.append(conv3x3(inplanes, planes))
            self.skip = nn.Identity()
        layers.append(nn.BatchNorm2d(planes))

        # Deeper block
        if upsample:
            for _ in range(1, num_layers):
                add_layer = [conv3x3(inplanes, inplanes),
                             nn.BatchNorm2d(inplanes),
                             nn.ReLU(inplace=True)]

                layers = add_layer + layers
        else:
            for _ in range(1, num_layers):
                add_layer = [nn.ReLU(inplace=True),
                             conv3x3(planes, planes),
                             nn.BatchNorm2d(planes)]

                layers = layers + add_layer

        if last_layer:  # remove the BN for the output layer.
            layers = layers[:-1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        out = self.model(x)

        identity = self.skip(identity)
        out += identity

        if not self.last_layer:  # remove the relu for the output layer.
            out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(BottleNeck, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.linear_enc = nn.Sequential(
            nn.Linear(in_planes * feature_size * feature_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, latent_size))

        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, in_planes * feature_size * feature_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)
        out = self.linear_dec(z)

        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)

        return {'out': out, 'z': z}


class SpatialBottleNeck(nn.Module):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(SpatialBottleNeck, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.linear_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=mid_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_num),
            nn.ReLU(True),
            nn.Conv2d(in_channels=mid_num, out_channels=latent_size, kernel_size=1, stride=1, padding=0, bias=False))

        self.linear_dec = nn.Sequential(
            nn.Conv2d(in_channels=latent_size, out_channels=mid_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_num),
            nn.ReLU(True),
            nn.Conv2d(in_channels=mid_num, out_channels=in_planes, kernel_size=1, stride=1, padding=0, bias=False),)

    def forward(self, x):
        z = self.linear_enc(x)
        out = self.linear_dec(z)

        return {'out': out, 'z': z}


class MemBottleNeck(BottleNeck):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16, mem_size=25, shrink_thres=0.0025):
        super(MemBottleNeck, self).__init__(in_planes, feature_size, mid_num, latent_size)
        self.memory_module = MemModule(mem_dim=mem_size, fea_dim=latent_size, shrink_thres=shrink_thres)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)

        mem_out = self.memory_module(z)
        z_hat, att = mem_out['output'], mem_out['att']

        out = self.linear_dec(z_hat)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)

        return {'out': out, 'att': att, 'z': z, 'z_hat': z_hat}


class VaeBottleNeck(BottleNeck):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(VaeBottleNeck, self).__init__(in_planes, feature_size, mid_num, latent_size)
        self.linear_enc = nn.Sequential(
            nn.Linear(in_planes * feature_size * feature_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, 2 * latent_size))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)

        mu, log_var = z.chunk(2, dim=1)
        z_hat = self.reparameterize(mu, log_var)

        out = self.linear_dec(z_hat)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)
        return {'out': out, 'mu': mu, 'log_var': log_var}

# ---- ported verbatim from MedIAnomaly/reconstruction/networks/ae.py --------------------
class AE(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1, spatial=False):
        super(AE, self).__init__()

        bottleneck = SpatialBottleNeck if spatial else BottleNeck

        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)

        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)

        self.bottle_neck = bottleneck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                      latent_size=latent_size)

        self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return {'x_hat': x_hat, 'z': z, 'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}

# ---- ported verbatim from MedIAnomaly/reconstruction/networks/aeu.py --------------------
class AEU(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None):
        super(AEU, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                  de_num_layers)

        self.de_block4 = BasicBlock(1 * base_width * expansion,  2 * in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat, log_var = self.de_block4(de1).chunk(2, 1)

        return {'x_hat': x_hat, 'log_var': log_var, 'z': z,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}


# %% [CELL 1.4]  Configuration — MedIAnomaly reference hyperparameters
# All values below are taken from the benchmark's own defaults
# (MedIAnomaly/reconstruction/options.py + data_utils.get_transform), NOT tuned by us.
# Rationale: our first goal is to REPRODUCE their numbers so ours are comparable to
# Table 6. Hyperparameter search (manual ablation / Optuna) comes later, on top of a
# verified baseline -- tuning before reproducing makes any gap uninterpretable.
SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))

# Which dataset this run of the notebook is for. Everything below namespaces itself by
# this, so two datasets can never share a run id, a checkpoint directory or a wandb
# artifact name. This mirrors how the benchmark itself does it — MedIAnomaly puts the
# dataset at the TOP of its output path (options.py:68,
# result_dir = ~/Experiment/MedIAnomaly/{dataset}, then {model}/fold_{fold}) rather
# than branching its code per dataset.
DATASET = os.environ.get('DATASET', 'RSNA')

# Epochs are a property of the DATASET in the reference implementation
# (options.py:23 self.epochs), not a global. Brain Tumor is the odd one out at 600.
DATASET_EPOCHS = {'RSNA': 250, 'VinCXR': 250, 'LAG': 250,
                  'BrainTumor': 600, 'BraTS2021': 250}
assert DATASET in DATASET_EPOCHS, f'unknown DATASET {DATASET!r}'

# RSNA KEEPS ITS ORIGINAL RUN_VERSION VERBATIM. This is not cosmetic: RUN_VERSION names
# the checkpoint directory AND the wandb artifact (save_run builds f'{WANDB_GROUP}-{rid}'
# and WANDB_GROUP == RUN_VERSION). Changing it for RSNA would orphan every artifact
# already uploaded and silently retrain the entire project.
RUN_VERSION    = 'dl-v1' if DATASET == 'RSNA' else f'dl-v1-{DATASET.lower()}'
SKIP_COMPLETED = True
WANDB_PROJECT  = 'MedIAnomaly-DL'
WANDB_GROUP    = f'{RUN_VERSION}'

OUTPUT_DIR = ('/kaggle/working/results_dl' if os.path.isdir('/kaggle/working')
              else 'results_dl') + ('' if not SAMPLE_MODE else '_sample')
CKPT_DIR   = f'{OUTPUT_DIR}/ckpt_{RUN_VERSION}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ── model / optimisation: MedIAnomaly options.py defaults ───────────────────────
IMAGE_SIZE    = 64          # options.py --input-size ; they show 64 ~= 128 in performance
LATENT_DIM    = 16          # options.py --latent-size
HIDDEN_NUM    = 1024        # options.py --hidden-num (bottleneck FC width)
BASE_WIDTH    = 16          # options.py --base-width
EN_DEPTH      = 1           # options.py --en-depth
DE_DEPTH      = 1           # options.py --de-depth

# DAE's UNet is a separate architecture with its own defaults (networks/unet.py), not a
# variant of the AE above. See the note in build_net about Table 6's params column.
DAE_UNET_DEPTH = 5
DAE_UNET_WF    = 6          # first layer has 2**wf channels
EPOCHS        = DATASET_EPOCHS[DATASET] if not SAMPLE_MODE else 2   # options.py epochs[...]
BATCH_SIZE    = 64  if not SAMPLE_MODE else 4      # options.py --train-batch-size
LR            = 1e-3                                # options.py --train-lr
WEIGHT_DECAY  = 0.0                                 # options.py --train-weight-decay
EPS           = 1e-8

# Images are normalised to [-1, 1], matching data_utils.get_transform:
#   transforms.Normalize((0.5,), (0.5,))  applied after ToTensor().
# This matters: their SSIM loss does ((x+1)/2) internally, and any loss/noise ported
# from their code assumes this range. Do NOT silently switch to [0, 1].
PIXEL_RANGE = (-1.0, 1.0)

# ── seeds ───────────────────────────────────────────────────────────────────────
# The train/test split now comes from data.json (their split), so SPLIT_SEED no longer
# controls it -- only model init / batch order / augmentation.
TRAIN_SEED = int(os.environ.get('TRAIN_SEED', '42'))
SEED       = TRAIN_SEED
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"DATASET     : {DATASET}")
print(f"SAMPLE_MODE : {SAMPLE_MODE}")
print(f"RUN_VERSION : {RUN_VERSION}  (SKIP_COMPLETED={SKIP_COMPLETED})")
print(f"image {IMAGE_SIZE}px | latent {LATENT_DIM} | epochs {EPOCHS} | bs {BATCH_SIZE} | lr {LR}")
print(f"OUTPUT_DIR  : {OUTPUT_DIR}")

# Name of the Kaggle Secret holding your wandb API key.
# Kaggle: Notebook -> Add-ons -> Secrets -> add a secret with THIS label.
# Locally: set WANDB_API_KEY, or just run `wandb login` once.
WANDB_SECRET_NAME = os.environ.get('WANDB_SECRET_NAME', 'REATTN_KEY')

USE_WANDB = False


# %% [markdown]
# # C1.5 Checkpoint save/load helpers

# %% [CELL 1.5]  Run storage — one record per (method, params, seed)
# Replaces the previous save_ckpt/load_ckpt, which were built for a different study and
# caused two real bugs: a positional `disc_scores` slot that later got reused to smuggle
# pixel maps, and artifact names that save/restore built differently. This design makes
# both mistakes structurally impossible:
#   * every payload is NAMED (arrays={'scores':..., 'pixel_maps':...}) — no positional slots
#   * one deterministic run_id derived from (method, params, seed) — no hand-built names
#   * a MANIFEST records the global config fingerprint; loading verifies it and REFUSES
#     to hand back a record produced under different settings (that is the parameter-mixing
#     guard: a result trained at 64px/lr1e-3 can never silently be reused at 128px/lr1e-4)

import hashlib, datetime

all_results  = {}    # run_id -> metrics dict
loss_history = {}    # run_id -> list of per-epoch losses

# Global settings that MUST match for a stored run to be reusable. Anything that changes
# the numbers belongs here; anything cosmetic must not (or every run invalidates).
def config_fingerprint():
    return {
        'image_size': IMAGE_SIZE, 'latent_dim': LATENT_DIM, 'hidden_num': HIDDEN_NUM,
        'base_width': BASE_WIDTH, 'en_depth': EN_DEPTH, 'de_depth': DE_DEPTH,
        'dae_unet_depth': DAE_UNET_DEPTH, 'dae_unet_wf': DAE_UNET_WF,
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'pixel_range': list(PIXEL_RANGE),
        'sample_mode': SAMPLE_MODE, 'run_version': RUN_VERSION,
        'dataset': DATASET,
    }


# Runs stored before DATASET existed have no 'dataset' key in their manifest, and every
# one of them was RSNA. Defaulting the MISSING key at comparison time (never by rewriting
# a stored manifest) keeps those runs valid while still making a genuine cross-dataset
# mix-up raise.
_LEGACY_FINGERPRINT_DEFAULTS = {'dataset': 'RSNA'}


def _slug(v):
    """Compact, filesystem- and wandb-safe rendering of a parameter value."""
    if isinstance(v, float):
        return f'{v:g}'.replace('.', 'p').replace('-', 'm')
    return str(v).replace('.', 'p').replace('/', '-').replace(' ', '')


def run_id(method, seed, **params):
    """Deterministic id: method[_ds-X][_k-v...]_sN. Same inputs -> same id, always.

    The dataset tag is OMITTED for RSNA on purpose. Every RSNA run already stored locally
    and on wandb was written under the un-tagged id, and adding a tag unconditionally
    would orphan all of them. RSNA is the historical default; every other dataset is
    tagged, so a collision across datasets is impossible in either direction."""
    parts = [method]
    if DATASET != 'RSNA':
        parts.append(f'ds-{_slug(DATASET)}')
    parts += [f'{k}-{_slug(v)}' for k, v in sorted(params.items())] + [f's{seed}']
    rid = '_'.join(parts)
    return rid if len(rid) <= 100 else f'{method}_{hashlib.md5(rid.encode()).hexdigest()[:12]}_s{seed}'


def _run_dir(rid):
    return os.path.join(CKPT_DIR, rid)


def _manifest_path(rid):
    return os.path.join(_run_dir(rid), 'manifest.json')


def run_exists(rid):
    return SKIP_COMPLETED and os.path.isfile(_manifest_path(rid))


def save_run(rid, *, method, seed, params, metrics, epoch_loss=None,
             arrays=None, weights=None, extra=None):
    """Persist one run: manifest + named arrays + named weights, then upload as ONE
    wandb artifact. Arrays and weights are written in the same call, so a record can
    never contain arrays from one model and weights from another."""
    d = _run_dir(rid); os.makedirs(d, exist_ok=True)
    files = []
    for name, arr in (arrays or {}).items():
        fp = os.path.join(d, f'{name}.npy'); np.save(fp, np.asarray(arr)); files.append(f'{name}.npy')
    for name, sd in (weights or {}).items():
        fp = os.path.join(d, f'{name}.pth'); torch.save(sd, fp); files.append(f'{name}.pth')

    manifest = {
        'run_id': rid, 'method': method, 'seed': seed, 'params': params,
        'metrics': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in metrics.items()},
        'epoch_loss': [float(v) for v in (epoch_loss or [])],
        'config': config_fingerprint(),
        'arrays': list((arrays or {}).keys()), 'weights': list((weights or {}).keys()),
        'files': files, 'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'extra': extra or {},
    }
    with open(_manifest_path(rid), 'w') as f:
        json.dump(manifest, f, indent=2)

    all_results[rid] = manifest['metrics']
    if epoch_loss:
        loss_history[rid] = manifest['epoch_loss']

    if USE_WANDB and wandb.run is not None:
        wandb.log({f'{method}/{k}': v for k, v in manifest['metrics'].items()
                   if isinstance(v, (int, float))} | {'run_id': rid})
        try:
            art = wandb.Artifact(f'{WANDB_GROUP}-{rid}'.lower().replace('.', 'p'),
                                 type='run', metadata=manifest)
            art.add_dir(d)
            art = wandb.log_artifact(art); art.wait()
            print(f'  [{rid}] artifact -> {art.name}')
        except Exception as e:
            print(f'  [{rid}] artifact upload failed: {e}')
    print(f'  [{rid}] saved ({len(files)} files) -> {d}')
    return manifest


def load_run(rid, models=None, strict=True):
    """Load a stored run. Returns (manifest, arrays_dict). Verifies the stored config
    fingerprint against the CURRENT one and refuses on mismatch when strict — this is
    what stops a run trained under different hyperparameters being silently reused."""
    with open(_manifest_path(rid)) as f:
        man = json.load(f)
    cur, old = config_fingerprint(), man.get('config', {})
    _d = _LEGACY_FINGERPRINT_DEFAULTS
    diff = {k: (old.get(k, _d.get(k)), cur[k])
            for k in cur if old.get(k, _d.get(k)) != cur[k]}
    if diff:
        msg = (f"[{rid}] CONFIG MISMATCH — stored run used different settings:\n" +
               '\n'.join(f'    {k}: stored={a!r} current={b!r}' for k, (a, b) in diff.items()))
        if strict:
            raise RuntimeError(msg + '\n  Delete the run dir to retrain, or set strict=False '
                                     'if you deliberately want to mix.')
        print('  WARNING ' + msg)

    d = _run_dir(rid)
    arrays = {n: np.load(os.path.join(d, f'{n}.npy')) for n in man['arrays']}
    for name, model in (models or {}).items():
        fp = os.path.join(d, f'{name}.pth')
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"[{rid}] weights {name!r} missing at {fp}")
        model.load_state_dict(torch.load(fp, map_location=device))
        model.eval()          # reload path is always inference; train-mode BatchNorm
                              # would use batch stats AND mutate running stats
    all_results[rid] = man['metrics']
    if man.get('epoch_loss'):
        loss_history[rid] = man['epoch_loss']
    print(f"  [{rid}] loaded (saved {man['saved_at']})")
    return man, arrays


def fetch_run(rid, run_version=None, entity=None):
    """Pull a run's artifact from wandb into CKPT_DIR if it is not already local."""
    if os.path.isfile(_manifest_path(rid)):
        return True
    try:
        api = wandb.Api()
        name = f'{run_version or RUN_VERSION}-{rid}'.lower().replace('.', 'p')
        art = api.artifact(f'{entity or api.default_entity}/{WANDB_PROJECT}/{name}:latest')
        art.download(root=_run_dir(rid))
        print(f'  [{rid}] restored from wandb')
        return True
    except Exception as e:
        # Print the MESSAGE, not just the class. wandb raises CommError both for
        # 'artifact does not exist yet' (benign, the normal first-run path) and for a
        # genuine network/auth failure (not benign — it silently retrains everything on
        # a fresh session). Those two are indistinguishable from the class name alone.
        msg = str(e).replace('\n', ' ')[:200]
        benign = 'not found' in msg.lower() or 'does not exist' in msg.lower()
        tag = 'absent' if benign else 'FETCH FAILED'
        print(f'  [{rid}] {tag} ({type(e).__name__}: {msg}) — will train fresh')
        return False


def completed_runs():
    """Every run stored under the current CKPT_DIR, as a DataFrame."""
    rows = []
    for mp in sorted(_glob.glob(os.path.join(CKPT_DIR, '*', 'manifest.json'))):
        with open(mp) as f:
            m = json.load(f)
        rows.append({'run_id': m['run_id'], 'method': m['method'], 'seed': m['seed'],
                     **m['params'], **m['metrics']})
    return pd.DataFrame(rows)


# %% [markdown]
# # C1.6 Wandb setup and login

# %% [CELL 1.6]  Wandb setup and login
# USE_WANDB is the flag every later cell should check before calling wandb.*
# — that's what makes wandb fully optional (see save_ckpt above).
try:
    import wandb
    # Three login paths, tried in order: explicit env var -> Kaggle Secrets -> netrc.
    # Each failure is reported SPECIFICALLY: a missing secret, a wrong secret NAME and
    # a rejected key all used to look identical ("wandb unavailable"), which made this
    # impossible to debug from the notebook output.
    if os.environ.get('WANDB_API_KEY'):
        wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
        print('WandB: logged in via WANDB_API_KEY')
    elif os.path.isdir('/kaggle/working'):
        from kaggle_secrets import UserSecretsClient
        try:
            _key = UserSecretsClient().get_secret(WANDB_SECRET_NAME)
        except Exception as _se:
            raise RuntimeError(
                f"Kaggle Secret {WANDB_SECRET_NAME!r} not readable ({_se}). "
                f"Add-ons -> Secrets: create a secret labelled {WANDB_SECRET_NAME!r} "
                f"AND tick 'attach to notebook', or set WANDB_SECRET_NAME to its label."
            ) from None
        if not _key:
            raise RuntimeError(f"Kaggle Secret {WANDB_SECRET_NAME!r} is empty.")
        wandb.login(key=_key, relogin=True)
        print(f'WandB: logged in via Kaggle Secret {WANDB_SECRET_NAME!r}')
    else:
        wandb.login()
        print('WandB: logged in via netrc / interactive')
    USE_WANDB = True
    # id=RUN_VERSION + resume='allow' means re-running this cell
    # (e.g. after a Kaggle session reset) reattaches to the SAME wandb run
    # instead of creating a new one, so metrics keep appending to one history.
    wandb.init(project=WANDB_PROJECT,
               group=WANDB_GROUP,
               name=f'{RUN_VERSION}',
               config=dict(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM,
                           hidden_num=HIDDEN_NUM, base_width=BASE_WIDTH,
                           epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                           weight_decay=WEIGHT_DECAY, pixel_range=PIXEL_RANGE,
                           run_version=RUN_VERSION),
               tags=['medianomaly', 'dl', RUN_VERSION],
               resume='allow', id=f'{RUN_VERSION}',
               settings=wandb.Settings(init_timeout=120))
    print(f'WandB ready  project={WANDB_PROJECT}  version={RUN_VERSION}')
except Exception as _e:
    # Any failure here (no internet, no key, user declines login, etc.)
    # falls back to USE_WANDB=False so the rest of the notebook still runs.
    USE_WANDB = False
    print(f'WandB unavailable ({_e}) — continuing without.')

# Datasets this project needs: image-level classification only -> RSNA is enough
# ONE dataset per notebook run. find_data_root returns the first root containing EVERY
# name in `required`, so asking for two at once fails outright when they live under
# different roots (a Kaggle mount and a local download, say) with an error that points at
# the wrong problem.
REQUIRED_DATASETS = [DATASET]


# %% [markdown]
# ---
# ## **Cell 1.7** — Dataset-isolation self-test (run this before anything else)
# The run-storage layer was built to stop *hyperparameter* mixing. It did not stop
# **dataset** mixing: `run_id` had no dataset field and `config_fingerprint` had no
# dataset key, so a second dataset's `ae_s42` would have been the same id, the same
# directory and the same wandb artifact as RSNA's. `run_exists` would return True,
# the fingerprint would show no difference, and `load_run` would hand back RSNA's
# scores to be evaluated against the new dataset's labels.
#
# That failure is **silent** for exactly the datasets we most want to run: VinCXR and LAG
# both train for 250 epochs, so the fingerprint stays bit-identical to RSNA's, and
# VinCXR's test set is also 2000 images, so `roc_auc_score` would not even raise on a
# length mismatch. The result would be a complete, plausible, entirely fictitious table.
#
# So this cell does not merely *apply* the fix — it **asserts** it. Two properties matter
# and they pull in opposite directions:
#
# 1. **RSNA ids must be byte-identical to what is already stored**, or every artifact
#    already on wandb is orphaned and the whole project silently retrains.
# 2. **Every other dataset must be unreachable from RSNA's namespace**, in both
#    directions, and a cross-dataset load must RAISE rather than return.

# %% [CELL 1.7]  Dataset-isolation self-test

def dataset_isolation_selftest(verbose=True):
    """Asserts the dataset namespacing is correct. Cheap, and the only thing standing
    between a second dataset and a fictitious results table."""
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    # --- 1. RSNA's historical ids and namespaces are untouched ----------------------
    if DATASET == 'RSNA':
        check("RSNA run id unchanged            run_id('ae',42) == 'ae_s42'",
              run_id('ae', 42) == 'ae_s42')
        check("RSNA param id unchanged          ...w-2_s42",
              run_id('ae-posthoc-u', 42, w='2') == 'ae-posthoc-u_w-2_s42')
        check("RSNA RUN_VERSION unchanged       'dl-v1'", RUN_VERSION == 'dl-v1')
        check("RSNA wandb artifact name unchanged",
              f'{WANDB_GROUP}-{run_id("ae", 42)}'.lower() == 'dl-v1-ae_s42')
        # a legacy manifest (no 'dataset' key) must still validate under the shim
        cur = config_fingerprint()
        legacy = {k: v for k, v in cur.items() if k != 'dataset'}
        d = _LEGACY_FINGERPRINT_DEFAULTS
        check("legacy manifest (no dataset key) still validates",
              not {k for k in cur if legacy.get(k, d.get(k)) != cur[k]})
    else:
        check(f"{DATASET} id is tagged           {run_id('ae', 42)}",
              run_id('ae', 42) == f'ae_ds-{_slug(DATASET)}_s42')
        check(f"{DATASET} id differs from RSNA's", run_id('ae', 42) != 'ae_s42')
        check(f"{DATASET} RUN_VERSION namespaced {RUN_VERSION}",
              RUN_VERSION == f'dl-v1-{DATASET.lower()}')
        check(f"{DATASET} CKPT_DIR namespaced", RUN_VERSION in CKPT_DIR)
        # an RSNA manifest loaded under this dataset MUST raise, not return
        rsna_rid = 'ae_s42'
        legacy_path = os.path.join(OUTPUT_DIR, 'ckpt_dl-v1', rsna_rid, 'manifest.json')
        if os.path.isfile(legacy_path):
            with open(legacy_path) as f:
                stored = json.load(f).get('config', {})
            d = _LEGACY_FINGERPRINT_DEFAULTS
            cur = config_fingerprint()
            check("an RSNA-stored run is REJECTED under this dataset",
                  bool({k for k in cur if stored.get(k, d.get(k)) != cur[k]}))
        else:
            print('  SKIP  no local RSNA manifest to cross-check against')

    # --- 2. params always separate ids ---------------------------------------------
    check("head widths do not collide",
          run_id('ae-posthoc-u', 42, w='2') != run_id('ae-posthoc-u', 42, w='8'))
    check("an epochs override separates ids",
          run_id('ae', 42, ep=50) != run_id('ae', 42))

    # --- 3. the ensemble member resolver carries params -----------------------------
    # A3_MEMBERS is defined much later (Cell 5.2), so this part only runs when the
    # self-test is re-invoked after the whole notebook has been executed. Skipping it on
    # the first pass is correct, not a hole: nothing before Cell 5.2 can use it.
    if 'A3_MEMBERS' in globals():
        check("A3 members carry params (3-tuples)",
              all(len(m) == 3 for m in A3_MEMBERS) and len(A3_MEMBERS) == 4)
        check("the extended set adds the post-hoc head with w='2'",
              ('ssim+head', 'ae-ssim-posthoc-u', {'w': '2'}) in A3_MEMBERS_EXT)
    else:
        print('  SKIP  A3 member checks (A3_MEMBERS not defined until Cell 5.2)')

    print(f"\n  dataset-isolation self-test: {'PASS' if ok else 'FAIL'}   "
          f"(DATASET={DATASET}, RUN_VERSION={RUN_VERSION})")
    if not ok:
        raise RuntimeError('dataset isolation is NOT safe — do not run experiments')
    return ok


dataset_isolation_selftest()

# BEFORE the first run on a NEW dataset, back the checkpoint store up. Ten seconds, and
# it is the only thing that makes any of the above reversible:
#     cp -r results_dl/ckpt_dl-v1 results_dl/ckpt_dl-v1.bak


# %% [markdown]
# ---
# ## **Cell 2.0** — Dataset acquisition (MedIAnomaly preprocessed data)
# We use the benchmark's OWN preprocessed data rather than raw Kaggle DICOMs, because
# `data.json` encodes their exact train/test split. That is what makes our numbers
# directly comparable to MedIAnomaly Table 6 instead of "similar setup, different split".
#
# Expected layout (from https://zenodo.org/records/12677223):
#   <DATA_ROOT>/RSNA/{images/, data.json}        image-level  (3851 train / 1000+1000 test)
#   <DATA_ROOT>/VinCXR/{images/, data.json}      image-level
#   <DATA_ROOT>/BraTS2021/train/                 pixel-level  (the ONLY dataset with masks)
#                        /test/{normal,tumor,annotation}/
#
# The cell searches several roots so the same notebook works on Kaggle (dataset mounted
# under /kaggle/input) and locally, and only downloads if nothing is found.

# %% [CELL 2.0]  Locate / download / verify datasets

import glob as _glob, tarfile, urllib.request

ZENODO = "https://zenodo.org/records/12677223/files/{}.tar.gz?download=1"

# Searched in order. Kaggle Datasets land in /kaggle/input/<slug>/ and the slug is
# user-chosen, so we glob for any directory that contains the expected dataset folders.
def _discover_roots(base='/kaggle/input', max_depth=4):
    """Bounded walk of the Kaggle mount, returning every directory that could be a root.

    Fixed glob patterns are not enough: Kaggle mounts inputs at DIFFERENT DEPTHS
    depending on how the notebook was created. The classic layout is
    /kaggle/input/<slug>/, but the namespaced one is
    /kaggle/input/datasets/<owner>/<slug>/ — two levels deeper, which a
    '/kaggle/input/*' glob never reaches. Rather than enumerate a pattern per layout,
    walk a few levels and let _looks_like decide which directory is real.

    Image folders are pruned: the RSNA competition input alone holds ~27k DICOMs, and
    descending into it would cost seconds for directories that can never be a root."""
    out = []
    if not os.path.isdir(base):
        return out
    base_depth = base.rstrip('/').count(os.sep)
    for root, dirs, _ in os.walk(base):
        out.append(root)
        if root.count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs
                   if d not in ('images', 'train', 'test', 'annotation', 'normal', 'tumor')]
    return out


DATA_ROOT_CANDIDATES = [
    os.environ.get('MEDIANOMALY_DATA', ''),
    os.path.expanduser('~/MedIAnomaly-Data'),
    '/kaggle/working/MedIAnomaly-Data',
] + _discover_roots()


def _looks_like(root, name):
    """True if <root>/<name> has the structure the MedIAnomaly loaders expect."""
    d = os.path.join(root, name)
    if name == 'BraTS2021':
        return all(os.path.isdir(os.path.join(d, p))
                   for p in ['train', 'test/normal', 'test/tumor', 'test/annotation'])
    return os.path.isdir(os.path.join(d, 'images')) and os.path.isfile(os.path.join(d, 'data.json'))


def find_data_root(required):
    """Return the first candidate root that contains every dataset in `required`."""
    for root in DATA_ROOT_CANDIDATES:
        if root and os.path.isdir(root) and all(_looks_like(root, n) for n in required):
            return root
    return None


def download_datasets(names, root=None, force=False):
    """Fetch + extract from Zenodo. NOT called automatically — these archives are large
    and Kaggle sessions have limited disk, so downloading is an explicit decision."""
    root = root or os.path.expanduser('~/MedIAnomaly-Data')
    os.makedirs(root, exist_ok=True)
    for name in names:
        if _looks_like(root, name) and not force:
            print(f'  {name}: already present, skipping'); continue
        tgz = os.path.join(root, f'{name}.tar.gz')
        if not os.path.exists(tgz) or force:
            print(f'  {name}: downloading …')
            urllib.request.urlretrieve(ZENODO.format(name), tgz)
        print(f'  {name}: extracting …')
        with tarfile.open(tgz, 'r:gz') as t:
            t.extractall(root)
        os.remove(tgz)
        print(f'  {name}: {"OK" if _looks_like(root, name) else "STRUCTURE UNEXPECTED"}')
    return root


def verify_datasets(required, root=None):
    """Print a per-dataset report and RAISE if anything required is missing, so the
    notebook fails here with an actionable message instead of deep inside training."""
    root = root or find_data_root(required)
    print(f'DATA_ROOT: {root}')
    if root is None:
        print('  searched:', [c for c in DATA_ROOT_CANDIDATES if c])
        raise FileNotFoundError(
            'MedIAnomaly data not found. Either:\n'
            f'  (a) download_datasets({required})   # needs internet + disk\n'
            '  (b) upload the extracted MedIAnomaly-Data folder as a Kaggle Dataset, or\n'
            '  (c) set MEDIANOMALY_DATA=/path/to/MedIAnomaly-Data')
    ok = True
    for name in required:
        d = os.path.join(root, name)
        if not _looks_like(root, name):
            print(f'  {name:<12} MISSING or wrong structure at {d}'); ok = False; continue
        if name == 'BraTS2021':
            n_tr = len(os.listdir(os.path.join(d, 'train')))
            n_no = len(os.listdir(os.path.join(d, 'test/normal')))
            n_tu = len(os.listdir(os.path.join(d, 'test/tumor')))
            n_an = len(os.listdir(os.path.join(d, 'test/annotation')))
            print(f'  {name:<12} train={n_tr}  test normal={n_no}  tumor={n_tu}  masks={n_an}')
            if n_tu != n_an:
                print(f'    WARNING: {n_tu} tumor images but {n_an} masks — pixel metrics need a mask per image')
                ok = False
        else:
            with open(os.path.join(d, 'data.json')) as f:
                dd = json.load(f)
            n_img = len(os.listdir(os.path.join(d, 'images')))
            tr, te0, te1 = len(dd['train']['0']), len(dd['test']['0']), len(dd['test']['1'])
            print(f'  {name:<12} train={tr}  test normal={te0}  abnormal={te1}  files={n_img}')
            if name == 'RSNA' and (tr, te0, te1) != (3851, 1000, 1000):
                print(f'    WARNING: expected 3851/1000/1000 (MedIAnomaly Table 2), got {tr}/{te0}/{te1}')
    if not ok:
        raise RuntimeError('dataset verification failed — see report above')
    print('All required datasets verified.')
    return root


DATA_ROOT = verify_datasets(REQUIRED_DATASETS)


# %% [CELL 2.1]  Loaders for the MedIAnomaly layout

def load_split_imagelevel(root, name, size=IMAGE_SIZE):
    """RSNA / VinCXR: returns (x_train, x_test, y_test) using THEIR split from data.json."""
    from PIL import Image
    d = os.path.join(root, name)
    with open(os.path.join(d, 'data.json')) as f:
        dd = json.load(f)

    def _load(names, tag):
        out = []
        for i, nm in enumerate(names):
            if i % 500 == 0:
                print(f'  {tag}: {i}/{len(names)}')
            im = Image.open(os.path.join(d, 'images', nm)).convert('L').resize((size, size),
                                                                               Image.BILINEAR)
            out.append(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)   # -> [-1, 1]
        return np.stack(out)[:, None] if out else np.zeros((0, 1, size, size), np.float32)

    tr  = dd['train']['0']
    te0, te1 = dd['test']['0'], dd['test']['1']
    if SAMPLE_MODE:                      # smoke test: tiny subsets, same code path
        tr, te0, te1 = tr[:30], te0[:10], te1[:5]
    x_train = _load(tr,  f'{name}-train')
    x_test  = np.concatenate([_load(te0, f'{name}-test-normal'), _load(te1, f'{name}-test-abn')])
    y_test  = np.array([0] * len(te0) + [1] * len(te1), dtype=np.int32)
    print(f'{name}: train {x_train.shape}  test {x_test.shape}  ({y_test.mean()*100:.1f}% abnormal)')
    return x_train, x_test, y_test


def load_split_brats(root, size=IMAGE_SIZE):
    """BraTS2021: returns (x_train, x_test, y_test, masks). The ONLY dataset here with
    pixel-level ground truth — masks are 0/255 PNGs named like the image with
    'flair'->'seg', matching MedIAnomaly's BraTSAD loader."""
    from PIL import Image
    d = os.path.join(root, 'BraTS2021')

    def _load(dirpath, names, tag, nearest=False):
        out = []
        for i, nm in enumerate(names):
            if i % 500 == 0:
                print(f'  {tag}: {i}/{len(names)}')
            im = Image.open(os.path.join(dirpath, nm)).convert('L').resize(
                (size, size), Image.NEAREST if nearest else Image.BILINEAR)
            out.append(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)   # -> [-1, 1]
        return np.stack(out)[:, None] if out else np.zeros((0, 1, size, size), np.float32)

    tr_names = sorted(os.listdir(os.path.join(d, 'train')))
    no_names = sorted(os.listdir(os.path.join(d, 'test/normal')))
    tu_names = sorted(os.listdir(os.path.join(d, 'test/tumor')))
    if SAMPLE_MODE:
        tr_names, no_names, tu_names = tr_names[:30], no_names[:10], tu_names[:5]
    mk_names = [e.replace('flair', 'seg') for e in tu_names]

    x_train = _load(os.path.join(d, 'train'), tr_names, 'brats-train')
    x_test  = np.concatenate([_load(os.path.join(d, 'test/normal'), no_names, 'brats-normal'),
                               _load(os.path.join(d, 'test/tumor'),  tu_names, 'brats-tumor')])
    y_test  = np.array([0] * len(no_names) + [1] * len(tu_names), dtype=np.int32)
    masks   = np.concatenate([np.zeros((len(no_names), 1, size, size), np.float32),
                               _load(os.path.join(d, 'test/annotation'), mk_names, 'brats-masks',
                                     nearest=True)])
    masks = (masks > 0.0).astype(np.float32)      # loader gives [-1,1]; >0 == was >127 == mask
    print(f'BraTS2021: train {x_train.shape}  test {x_test.shape}  '
          f'masks {masks.shape}  positive pixels {masks.mean()*100:.2f}%')
    return x_train, x_test, y_test, masks


# %% [markdown]
# ---
# ## **Cell 2.2** — Load RSNA into memory
# The whole dataset at 64px is ~30 MB, so we hold it as tensors and skip a `Dataset`
# class entirely. Loading ONCE here (rather than per experiment) is deliberate: every
# method below then sees byte-identical inputs, so a difference between two rows of the
# results table can only come from the method, never from a re-decoded image.

# %% [CELL 2.2]  Load RSNA

x_train_np, x_test_np, y_test_np = load_split_imagelevel(DATA_ROOT, DATASET)

X_TRAIN = torch.from_numpy(x_train_np)                 # (N,1,64,64) float32 in [-1,1]
X_TEST  = torch.from_numpy(x_test_np)
Y_TEST  = y_test_np

assert X_TRAIN.dtype == torch.float32 and X_TRAIN.shape[1] == 1
assert -1.01 <= float(X_TRAIN.min()) and float(X_TRAIN.max()) <= 1.01, \
    "inputs must be in [-1,1] — the ported SSIM/perceptual losses assume it"
print(f'X_TRAIN {tuple(X_TRAIN.shape)}  X_TEST {tuple(X_TEST.shape)}  '
      f'abnormal {Y_TEST.mean()*100:.1f}%  range [{X_TRAIN.min():.2f}, {X_TRAIN.max():.2f}]')


# %% [markdown]
# ---
# ## **Cell 3.0** — Losses and anomaly scores
# Ported from `MedIAnomaly/reconstruction/utils/losses.py` and `utils/util.py`.
#
# Every loss shares one interface, which is the key design idea of their codebase and
# the thing that makes experiment **A1** below possible at all:
#
# ```python
# criterion(x, net_out)                                     -> scalar training loss
# criterion(x, net_out, anomaly_score=True)                 -> (N,)      per-image score
# criterion(x, net_out, anomaly_score=True, keepdim=True)   -> (N,1,H,W) per-pixel map
# ```
#
# Because the training objective and the anomaly score are the *same* callable, they are
# normally locked together — train with L2, score with L2. A1 breaks that link on purpose
# and crosses them, which their paper never does.
#
# All losses assume inputs in **[-1, 1]** (`SSIMLoss` rescales to [0,1] internally,
# `PerceptualLoss._preprocess` de-normalises with mean=std=0.5).

# %% [CELL 3.0]  SSIM + the loss family

from typing import List, Optional, Tuple, Union
from collections import OrderedDict

# ---- SSIM, ported verbatim from MedIAnomaly/reconstruction/utils/util.py -------------
def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)
    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(f"Skipping Gaussian Smoothing at dim 2+{i} for {input.shape}")
    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    K1, K2 = K
    compensation = 1.0
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    win = win.to(X.device, dtype=X.dtype)

    mu1, mu2 = gaussian_filter(X, win), gaussian_filter(Y, win)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12   = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map   = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    return ssim_map


def ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None,
         K=(0.01, 0.03), nonnegative_ssim=False):
    """NOTE the unusual convention inherited from their code: `size_average=True` returns
    a per-IMAGE value (N,), `size_average=False` returns the full (N,C,h,w) map. Their
    SSIMLoss calls it with size_average=False and then reduces itself."""
    if not X.shape == Y.shape:
        raise ValueError(f"shape mismatch: {X.shape} vs {Y.shape}")
    for d in range(len(X.shape) - 1, 1, -1):
        X, Y = X.squeeze(dim=d), Y.squeeze(dim=d)
    if len(X.shape) not in (4, 5):
        raise ValueError(f"expected 4-d or 5-d, got {X.shape}")
    if win is not None:
        win_size = win.shape[-1]
    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
    ssim_map = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    return torch.mean(ssim_map, dim=[1, 2, 3]) if size_average else ssim_map


# ---- losses, ported verbatim from MedIAnomaly/reconstruction/utils/losses.py ----------
class AELoss(nn.Module):
    """Plain squared error — the 'AE' row of Table 6."""
    def __init__(self, grad_score=False):
        super().__init__()
        self.grad_score = grad_score

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        loss = (net_in - x_hat) ** 2
        if anomaly_score:
            if self.grad_score:
                grad = torch.abs(torch.autograd.grad(loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            return torch.mean(loss, dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        return loss.mean()


class SSIMLoss(nn.Module):
    """1 - SSIM. The Gaussian window shrinks the map by (win_size-1), hence the
    interpolate back to input size when a pixel map is requested."""
    def __init__(self, win_size=11):
        super().__init__()
        self.win_size = win_size

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        net_in_01 = ((net_in + 1) / 2.0).clamp(0., 1.)
        x_hat_01  = ((x_hat + 1) / 2.0).clamp(0., 1.)
        loss = 1. - ssim(net_in_01, x_hat_01, data_range=1., size_average=False, win_size=self.win_size)
        if anomaly_score:
            return torch.mean(F.interpolate(loss, size=net_in.shape[-2:], mode='bilinear'),
                              dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        return loss.mean()


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        loss = torch.abs(net_in - net_out['x_hat'])
        if anomaly_score:
            return torch.mean(loss, dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        return loss.mean()


class AEULoss(nn.Module):
    """AE-U: the net also predicts log_var, so the reconstruction error is divided by a
    learned per-pixel variance. That is why it wins on CXR — it learns to stop paying for
    the high-variance regions (ribs, edges) that a plain AE never reconstructs either.
    Returns a 3-tuple in TRAIN mode (loss, recon, log_var) — the driver unpacks it."""
    def __init__(self):
        super().__init__()

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, log_var = net_out['x_hat'], net_out['log_var']
        recon_loss = (net_in - x_hat) ** 2
        loss1 = torch.exp(-log_var) * recon_loss
        loss = loss1 + log_var
        if anomaly_score:
            # scored on loss1 only: the +log_var term is a normalising constant per pixel,
            # not evidence of anomaly
            return torch.mean(loss1, dim=[1], keepdim=True) if keepdim else torch.mean(loss1, dim=[1, 2, 3])
        return loss.mean(), recon_loss.mean().item(), log_var.mean().item()


class VAELoss(nn.Module):
    def __init__(self, kl_weight=0.005, grad=None):
        super().__init__()
        self.kl_weight = kl_weight
        self.grad = grad

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
        recon_loss = (net_in - x_hat) ** 2
        kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
        loss = recon_loss.mean() + self.kl_weight * kl_loss.mean()
        if anomaly_score:
            if self.grad in ('elbo', 'rec', 'kl', 'combi'):
                target = {'elbo': loss, 'rec': recon_loss.mean(), 'kl': kl_loss.mean(),
                          'combi': kl_loss.mean()}[self.grad]
                g = torch.abs(torch.autograd.grad(target, net_in)[0])
                out = recon_loss * g if self.grad == 'combi' else g
                return torch.mean(out, dim=[1], keepdim=True) if keepdim else torch.mean(out, dim=[1, 2, 3])
            return torch.mean(recon_loss, dim=[1], keepdim=True) if keepdim \
                else torch.mean(recon_loss, dim=[1, 2, 3])
        return loss, recon_loss.mean().item(), kl_loss.mean().item()


# %% [markdown]
# ### Cell 3.0b — Perceptual loss (AE-PL), the strongest baseline in Table 6
# `RelativePerceptualL1Loss` compares VGG19 `relu4_2` features instead of pixels, with
# each channel divided by its ImageNet-wide mean/std and the error taken relative to the
# feature magnitude.
#
# **Two deliberate deviations from their file**, both forced by "must run as one Kaggle
# notebook" and neither of which changes the objective:
#
# 1. They copy VGG19's weights into 16 hand-declared conv layers and tap `r42` by name.
#    We tap `torchvision.vgg19().features[:23]`, which *is* relu4_2 — same weights, same
#    padding (=1), same output. Verified in the assert at the bottom of the cell.
# 2. Their per-channel normalisation lives in a 106 KB `.pt` file we can't ship. Only the
#    512 `r42` means and vars are ever read (default `feature_weights={"r42": 1}`), so
#    those 1024 floats are embedded below as a compressed base64 blob. Byte-identical to
#    the file's values.
#
# If VGG19 weights can't be downloaded (Kaggle internet off), `HAS_VGG` goes False and
# the AE-PL rows are skipped with a message rather than silently substituting something
# else.

# %% [CELL 3.0b]  Perceptual loss

import base64, zlib

# r42 (relu4_2) channel means and variances from MedIAnomaly's
# utils/data/vgg19_ILSVRC2012_object_detection_mean_var.pt — first 512 floats are the
# means, next 512 the vars.
_R42_STATS_B64 = (
    "eNoNlXcgFoobhdta97ZLaQhRia6Vle+cbMnK3jOjiCg7fD7jQ6RhFxFRol2o0B7a/UhulNvet27KvQ0/f73/v+c5z1ExcoGn"
    "3lR8TsxFtXMSHkcV4qliJQ4ucIVUUiMOjdDFWlkhFJvaccApAxf3N+H2z0NIXp2PzvvRCPzujzT1cowcWQqpkBLoG3vhg+RN"
    "GL3Ixo7yLKybXYKh3qn4J8wTkUEnUdS2FbnGBWi03IshOtcgqa4DlfnJkHEPwuWLy5GUK8Yy13pc0DfAAeVsfJ56Cis787Hv"
    "7wRkWNZj8f+eCxq3VGBV71N4cihf7gzB18fdeC0swCubo/DXssMl1eNwzRAjQN0buvuOw2zTAnQ1ZWFgWxli3vpgj1QMbt2v"
    "QrGjL9bln8eSOesg01iCpI7tmN6xGYGnD2K57Vo4SR/EZ4O5qK2Pwz6NCGxDGkSPs/EwOwUhd/Ow8S8vfIluQeTCD4IhpTk4"
    "PGIT+rbKIly/XvDDJA3i1CrI0hlG86bg5sdjiM4ohZyiGfZKDYH3oRQ4K/pBaf42BDQ04tmVhSg8Gw67PSWo0qyH+gRNjFur"
    "i+AnL5Bx2wLhb+OhcTQD4bHNWBsUDYFhBladioN7eRrOXXeE+LMs+qRO4uQOExRtLMGeoVV4IXsG/RGpeNLxAGKPvej7zRft"
    "MqnQiQ2F8u6zaMh+g/uBYoT/W4N9D/JR67YbfQMpGHO0AkalRXD/txXwuy74UauP0kuuMHEfj0YPMc741mGKehOCGrZB0z0Q"
    "Sa3bEd9bi5AHQtjFHMUGOWeY3bqE2pjreDFJiIlflSFsDkfslUyUNojg7GiD/atTIZGZji978jG0uQCXpdbgt51OUPSyxIQR"
    "J+HU5INc3x3w105BRVsdnjU1wndkFmY4aMH0SQ6mrz6Oa2mVKFdww8shgAE0MWxkDOTjt+Pd6HuCb6eLkOhfiTavMkjWO0Ar"
    "rBJXby1H+fWleHLGFj8ObMARu5M4d7UcQWVpMDGTwdIyIc5Gu8NAKx1+3XG4eWAvJsTl4KjCPnidaMT5NiNkdLoh2LYIPzwS"
    "8Hz7XPQoz8ae8rcCo7xSvNcrhujwK0jbq6PfIwK3D1Uj+UohenebY0F2MCSivTBhuTS6+u8JJNeH43pXPV6p1cDp3CbkZHQJ"
    "ZlXqorR1PSa3WGKSWgnGVVzACG9XZDadRWldAZ78ewIt4n7UdGcg0c0JLaoPIfVPJoLjzTHdPgOnt+bAovgr7hRWI+JBCrxK"
    "TmPosgo43zDBuYDzWGTrgqd2uSjpiIXpVeJXawDem1dAKeshFjUp4HqCGHEyUZgnKcIxX3ucai7HurATkDl+FHLtCriUpInK"
    "3KvI0S5CdHQCSgZeCj6cuiNYIQpBiJYy3irG4InnLhSM1cW4PbXIcT6LueMj8CinBGPf5GP3pxZUZ6UhXGE7nrR9F/zoPgV7"
    "T2/k/BJjQFIJMns24tGOVth80YHkqEiohPkhLLIVS0zrsOzfUfiUmo+xQ28KtMNyEN6ehDt7bPEoLnXQE4G4bJyElItG+HX3"
    "Nkw0C9BzKwG7Y3Jx+40nOqe0Iup8PlR10yAcL4dl9mNQkKgB5cImVGoEI3/uFVwYfR+qLTk4e2opJH8XY5T/RBxqEsP6eK8g"
    "c1gyHI4JBznMgsOdGty2yoF2Yw1snAcZca+Fb0cplE4Ssx12wDZgJ56aNMDS1hZn8//CnGX3ceuxLroTffAmqw69oy9g9LBU"
    "FK/cCdMHd9EcYwXfC4sRvt4Dt9o3Y4Z6MrZWtsBH/6FgfVYYGi6lw3qYJkx2nIbrpkDEjUlBMfzxonYEXA4eQ9GGQR8Ub8Ih"
    "5ZOQ9t2OaoWdkFG6gXd+SdjokIS+wHNo+xkMb4EQLa1n4P8iBldki9HudRbdc5JR84cH/FmD6/JWg/nvQ9opD1iP8UBViTqe"
    "aejAV2I39D7shrFDJd6cm4ermy8LhNZrkHg7EX+Z5+P52CpBhHoa5kRFQNGhAhoDvyNwdApUOg0xcko2atwMIL3sAnaIr0NF"
    "0hYigyrMOueI/Kx8FEQ6INYnFad3q2BMoQgtS9bgW1k6OmRWoqd3C56Hn0L6wxJkm21C+qlr+D2jDooduzDm/ST4dBVDLzkL"
    "HQIRlI4XYXlKKa47zETmCkfs7EvCRcsWOK88Df/+rcgcfgMZrhdwrrsQn+rkMXPKkUHX5sPspDkMAjMh23ccC/++DO95g5w8"
    "rxG037GHRMN5fHuwBafuVCN2rwiG6mbwH7kd4q5prPlzK6zcBv3imAfh6Vasnf0CetXAl33vBQXPdiEoqBgK37JQ5LMI8pMW"
    "4li7EDdPHMAtxVL0J7gira0Q3l7Z+KNqOkZtzUSbVD0+eBnD4LQ7hLVjGF13FBrSHuhjNxaa1KJwvyx+Zk/GtJ5ItE2OhF1A"
    "EcYnnMQqfV0cspGB8x05dKolwz+0EWFlAXD+JxjxSzxQfskQljONMCrumuCo3jz0zFbAV0EWKuTLUbkzDvsH2Woqs+BBR1Oq"
    "zfRjlrYcI5MzmPwglDWjZGg2PIVxVKdotoh/rnalt4UlX+/N4BtnH9bU+jAy155PrQy5+qktHdLNmN/uR4mXmjQ5kc2gE2kU"
    "di5jYrsmN+7yZryzFg8ftqOMQi5j1oRSLS6O/73YwJwrs+j0QJMyn+15QleLLV/sOGJgBcVLtdnpYsWrOcl0mWo+eA05fHIa"
    "JZJ7oXMsk9MupLDHpZyT7dyZtyKeGpIe3Pg6njVPVKkx3IabN5qw96cxi+X8+d1Jnh+7RfSosOXO1y6Mt9Xj3IlCXjgfzZmj"
    "wlkXqsG4+iKaJIo59EkQzVurqBibzI91Xtz8YTjzDbzY7+lGHbMV7Jm0jiLDVXT030jvWUqM74qg/dIXgx4VMEEqgqvWjGa/"
    "7VhqLVXnPRt3xhcu5bwaVTbfiGZPkysXJpjyyBBpVnr48IiyN4/2OfLb32Ke71HkkW8CWnvZ0e6tI699Wswbn+Yw8WM2JX7Y"
    "c6LIjjOLI1jquJX/uPpxWLMGo2fGM9xYne+tFzHNdCrTpZxYJr+OBluEvJeQxEJ1MUvstlDvrwSGPI+jo4kpw/Zq0UPejms/"
    "OjEgRsQLz2z4PS2JK/TNaOMRx13fQ7m8Scw3ORt4PrSJ6cUzqBghw1tRmhx9cQDnR9hR/nMyD70SsfK6Fc/dU6OJvieTPTOY"
    "pGZGr6id/Nrrxv9dTWDyURH9PjhxlrkPq6+ZcuC7A+XtFvJxmzaFzd6sm6pEu0NraV7gQKMpTpywS4e1P1UZFp/H9J+WHKVi"
    "wm7rlexQ8OYTNzEdgu14e6IiQ5/68HJkEpXoSd/5K9kW7cotu+RpOmMFzZ97803ef4j7N55y3UK+N4vjU4tQLii35/KHmrS+"
    "okr/Nl22Gpmw0imTGbcjuV5Wl0/zZ/KxyIaT/lKivosRVwp2UskqmM+kbCjdUkXTl0ns5Qw+OyfFg23JnFQWRrcfUhzwXcLv"
    "p7sw/pkPm2Us+PN1KLfJzqClii5HrM5n6llHptjP4XVZHVZVyLF4twRr/n2DyGWGFP70ZsrMWFYZWFPJ7yt2NOszbZo7uV+N"
    "7yQM2X3Hj9YD4CiDJD7a4kKZy/7cl7mZM+w9qJBqxqyzm+lwxpajDpmyt8GAhyxSuD21lNX33OhgpU/94E28eiyJYqNFXL4j"
    "m64hc7gzJpGLNMx4p9mQvhNm8nb5Sh4xF/HjwGTmL1hFly/m/KXnyD+eKHHl6AC+mhfMYX5+7Hg2jnlq8tR8kEdVLTeeFrpx"
    "W8pzSD5+gTQ3AS07lWgp0uGoSjvarJrGX1EZ7JBNpJK/MZv3iaioL2S/Uwg3HHag9Me1TO8cwBALf26RleaH/Qb8ff0EThrt"
    "yuAQEZ0yB/+UR9ZbrWSDnifvD0ugpdE9eBaYULrzKjyuaXFzrBULthhyQ542U5VNuO1qEvtGL+RizWTuGunD0GxVrjxowWHD"
    "klhpnM4z6hEssg7j1keKzLygRBc/bXqcCKHamtWs27uF3+8ms1TbixP1ZnGWngnHX5xOAzMXbrZ4h1UlhqzrsuQZLWPK5Ym4"
    "Sd2DPVmBzOY6LtCz4R0jfQb/PY3nJgz2TMmK52/E8c8+bToXbeZsryQOlBnyztB1DPvsxA99Qs63XsWE3WbMXFbClpm6fPSH"
    "Il9+s6BOqAEvizV5QyqT41XHUOKqIoWSqRxw1GbIyc0s69Gj2Mma++qXcNvskbRa5zuYtwMDfwnY0CukpI0tcwJSWaksZLiM"
    "Cps+2XPu/iiGdLkxxtiQb8YJOaLYl3cfi6jyXzKfNXlyQa8V/Vs30jBRj90lUVwRt5BYoUJbVwnGChS4aagRx16z5IbHGxiV"
    "s4jiu114RnMePKrKi/FOrFo1gw0TrDipfzl/7s7n8F+zOK7Sn1nzdWmzbS2dg8xobhhAcb6Q6lpazFLzZf9EV86dH8jHjUr8"
    "5WvHnAOrB90UxOmvjXiz35JPfTWp+JsmOTaRs8q86GunRyXnQornJdPzH2uuyZXnCQtnJnQlsLpTn40TbFlW5sO3UbMp98iR"
    "s6vB1odJHKmxngqvQ3hfL5Jylbks9FHhhgFZuv/nzo1R0TzVo0knGxc+ig1im+5GrjAJ5L1f/8FLSo7xlkEMSbWhslMiq6vt"
    "WFGUxPYpkYy/e4CPBreyvdKacnMGO6jtTrWGTTTwXMDV379D891a7p+3ho4jPLm8fgr9Ximw9aIuD98MY2SYFb111bjuYjCv"
    "VQXxtxBpbh9kPK/RgpPkZbjCVZlhX6u5TSKcic+06NYnplyAJ8sz5tAr4HeuKycnZQhoMiyMsxbHMiRrPrumSlOOixkyIOS7"
    "7VkcnF7Od1Zm2zJ3Pl40jzrOxlxbMZS+h8fxa+44HpRYyIyIVQzs+INz9B35fwmht4A="
)


def _r42_mean_var():
    a = np.frombuffer(zlib.decompress(base64.b64decode(_R42_STATS_B64)), dtype=np.float32)
    assert a.size == 1024, a.size
    m = torch.from_numpy(a[:512].copy()).reshape(1, 512, 1, 1)
    v = torch.from_numpy(a[512:].copy()).reshape(1, 512, 1, 1)
    return m, v


class RelativePerceptualL1Loss(nn.Module):
    """MedIAnomaly's AE-PL objective: relative L1 on channel-normalised VGG19 relu4_2."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:23].eval().to(device)   # up to and incl. relu4_2
        for p in self.features.parameters():
            p.requires_grad = False
        m, v = _r42_mean_var()
        self.register_buffer('feat_mean', m)
        self.register_buffer('feat_var',  v)
        self.register_buffer('vgg_mean', torch.tensor(self.IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('vgg_std',  torch.tensor(self.IMAGENET_STD).reshape(1, 3, 1, 1))
        self.to(device)

    def _preprocess(self, x):
        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)
        x = x * 0.5 + 0.5                      # [-1,1] -> [0,1]
        return (x - self.vgg_mean) / self.vgg_std

    def _relative_l1(self, fx, fy):
        # relative to the magnitude of the REAL image's features, detached so the
        # denominator is a scale factor and not a second gradient path
        means = torch.abs(fx).mean(3).mean(2).mean(1).detach()
        return torch.abs(fx - fy) / means.reshape(-1, 1, 1, 1)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        y = net_out['x_hat']
        fx = self.features(self._preprocess(net_in))
        fy = self.features(self._preprocess(y))
        fx = (fx - self.feat_mean) / self.feat_var
        fy = (fy - self.feat_mean) / self.feat_var
        loss = self._relative_l1(fx, fy)
        if anomaly_score:
            if keepdim:
                loss = F.interpolate(loss, size=net_in.shape[-2:], mode='bilinear')
                return torch.mean(loss, dim=[1], keepdim=True)
            return torch.mean(loss, dim=[1, 2, 3])
        return loss.mean()


HAS_VGG = True
try:
    _pl_probe = RelativePerceptualL1Loss()
    with torch.no_grad():
        _z = torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        _f = _pl_probe.features(_pl_probe._preprocess(_z))
    assert _f.shape[1] == 512, _f.shape
    print(f'perceptual loss ready — relu4_2 feature map {tuple(_f.shape)} '
          f'(mean stats: {float(_pl_probe.feat_mean.mean()):.4f})')
    del _pl_probe, _z, _f
except Exception as e:
    HAS_VGG = False
    print(f'VGG19 unavailable ({type(e).__name__}: {e}) — AE-PL rows will be SKIPPED.\n'
          '  On Kaggle: enable "Internet" in the notebook settings, or attach a\n'
          '  torchvision-weights dataset.')


# %% [markdown]
# ---
# ## **Cell 3.1** — Method registry
# One table mapping a method name to (backbone, training loss, scoring loss, extras).
# Everything downstream reads this table, so adding a method is a one-line change and
# no experiment cell ever constructs a network by hand.
#
# **The AEU/VAE depth trap.** `AE.__init__` defaults `en_num_layers=1`, but `AEU` and
# `VAE` default both depths to `None` and pass them straight through to `BasicBlock`,
# which then does `range(None)` and dies with
# `TypeError: 'NoneType' object cannot be interpreted as an integer` — a message that
# points nowhere near the real cause. Their own `base_worker.set_network_loss` always
# passes the depths explicitly, so the defaults are simply dead. `build_net` below does
# the same and asserts, which is the only place in this notebook a network is built.

# %% [CELL 3.1]  Method registry and network factory

# ---- VAE, ported verbatim from MedIAnomaly/reconstruction/networks/vae.py -------------
class VAE(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048,
                 latent_size=16, en_num_layers=None, de_num_layers=None):
        super(VAE, self).__init__(input_size, in_planes, base_width, expansion, mid_num,
                                  latent_size, en_num_layers, de_num_layers)
        self.bottle_neck = VaeBottleNeck(4 * base_width * expansion, feature_size=self.fm,
                                         mid_num=mid_num, latent_size=latent_size)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)
        bottle_out = self.bottle_neck(en4)
        de4, mu, log_var = bottle_out['out'], bottle_out['mu'], bottle_out['log_var']
        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)
        return {'x_hat': x_hat, 'log_var': log_var, 'mu': mu,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}


_NET_CLASSES = {'ae': AE, 'aeu': AEU, 'vae': VAE}


def build_net(kind):
    """The ONLY place a backbone is instantiated. Depths are always passed explicitly —
    see the AEU/VAE None-default trap in the markdown above."""
    if kind == 'unet':
        # DAE does NOT use the AE backbone. base_worker.set_network_loss builds
        # UNet(in_channels, n_classes) for 'dae', i.e. the class defaults depth=5, wf=6.
        # The paper attributes DAE's segmentation win largely to this "customised UNet",
        # so substituting an AE here would not be a DAE.
        #
        # HEADS-UP for the write-up: Table 6 lists DAE as 2.79M params / 2.15 GFLOPs, but
        # those defaults give 31.0M params. The FLOPs side does check out (a depth-5 wf-6
        # UNet at 64px is ~1.07 G MACs ~= 2.15 GFLOPs, and thop reports MACs), so the
        # architecture is right and it is the params column that cannot be reconciled —
        # no (depth, wf) combination yields 2.79M. We follow the code, not the table, and
        # report our measured 31.0M. Worth stating explicitly rather than quietly
        # matching a number we cannot reproduce.
        return UNet(in_channels=1, n_classes=1,
                    depth=DAE_UNET_DEPTH, wf=DAE_UNET_WF).to(device)
    assert EN_DEPTH is not None and DE_DEPTH is not None, \
        'EN_DEPTH/DE_DEPTH must be ints — AEU and VAE forward None straight into range()'
    cls = _NET_CLASSES[kind]
    return cls(input_size=IMAGE_SIZE, in_planes=1, base_width=BASE_WIDTH, expansion=1,
               mid_num=HIDDEN_NUM, latent_size=LATENT_DIM,
               en_num_layers=EN_DEPTH, de_num_layers=DE_DEPTH).to(device)


_LOSS_CACHE = {}


def build_loss(name):
    """name -> criterion instance. Kept separate from build_net so experiment A1 can
    pair any training loss with any scoring loss.

    The perceptual criterion is cached: it wraps a 20M-parameter frozen VGG19, and A1
    asks for it once per grid cell. Rebuilding it each time re-allocated the whole
    feature extractor on the GPU — pure waste, and a plausible OOM on a 16 GB Kaggle
    accelerator. It holds no per-run state (frozen weights and two constant buffers),
    so a single shared instance is safe."""
    if name == 'perceptual':
        if 'perceptual' not in _LOSS_CACHE:
            _LOSS_CACHE['perceptual'] = RelativePerceptualL1Loss()
        return _LOSS_CACHE['perceptual']
    if name == 'l2':             return AELoss()
    if name == 'l2grad':         return AELoss(grad_score=True)
    if name == 'l1':             return L1Loss()
    if name == 'ssim':           return SSIMLoss()
    if name == 'aeu':            return AEULoss()
    if name == 'vae':            return VAELoss()
    if name == 'vaegrad-rec':    return VAELoss(grad='rec')
    if name == 'vaegrad-combi':  return VAELoss(grad='combi')
    raise KeyError(f'unknown loss {name!r}')


# Losses whose TRAIN-mode call returns (loss, extra1, extra2) instead of a bare scalar.
_TUPLE_LOSSES = {'aeu', 'vae'}
# Losses whose SCORE needs a gradient w.r.t. the input (so no torch.no_grad at eval).
_GRAD_LOSSES  = {'l2grad', 'vaegrad-rec', 'vaegrad-combi'}

# method name -> everything needed to run it. `train_loss`/`score_loss` are separate
# fields precisely so A1 can decouple them; for every baseline they are equal, which is
# the standard practice this benchmark (and the whole literature) assumes.
METHODS = {
    #                    net     train_loss    score_loss    extras
    'ae':         dict(net='ae',  train_loss='l2',         score_loss='l2'),
    'ae-l1':      dict(net='ae',  train_loss='l1',         score_loss='l1'),
    'ae-ssim':    dict(net='ae',  train_loss='ssim',       score_loss='ssim'),
    'ae-pl':      dict(net='ae',  train_loss='perceptual', score_loss='perceptual'),
    'aeu':        dict(net='aeu', train_loss='aeu',        score_loss='aeu'),
    'vae':        dict(net='vae', train_loss='vae',        score_loss='vae'),
    'dae':        dict(net='unet', train_loss='l2',        score_loss='l2',
                       denoise=True, noise_res=16, noise_std=0.2),
}

# MedIAnomaly Table 6, RSNA column (AUC ± sd, AP ± sd over 3 seeds). Our target.
TABLE6_RSNA = {
    'ae':      (67.5, 0.9, 66.7, 0.5),
    'ae-l1':   (68.1, 0.4, 67.9, 0.4),
    'ae-ssim': (80.9, 0.3, 78.6, 0.3),
    'ae-pl':   (87.5, 0.2, 84.8, 0.6),
    'vae':     (67.9, 0.8, 67.0, 0.8),
    'aeu':     (86.5, 0.9, 84.4, 1.1),
    'dae':     (86.1, 0.7, 83.3, 1.0),
    # score-swap rows: same backbone AND same training loss as ae / vae (Table 5)
    'ae-grad':        (73.7, 1.0, 71.0, 1.1),
    'vae-grad-rec':   (71.6, 0.8, 69.3, 0.9),
    'vae-grad-combi': (67.4, 0.3, 67.3, 0.4),
}

print(f'{len(METHODS)} methods registered: {", ".join(METHODS)}')


# %% [markdown]
# ---
# ## **Cell 3.2** — Training / evaluation driver
# A single function that runs any registry entry end to end and stores the result via
# `save_run`. Behaviour it enforces, each for a reason we hit the hard way earlier:
#
# * **`net.eval()` before scoring, always.** Scoring in train mode makes BatchNorm use
#   batch statistics *and* mutates its running statistics, which silently corrupts the
#   weights being saved. That single bug produced a 0.85-vs-0.95 discrepancy once.
# * **Per-run seeding of weights, batch order and DAE noise** from `seed`, so a "seed
#   spread" is a real measure of run-to-run variance and not of leftover global state.
# * **Skip / restore before training**: an existing local run is reused, otherwise wandb
#   is tried, otherwise it trains. `load_run` refuses a record made under a different
#   `config_fingerprint`, so a reused run can never be from different hyperparameters.
# * **Scores are always computed by `score_criterion`**, which may differ from the
#   training criterion — that is the whole mechanism of experiment A1.

# %% [CELL 3.2]  train_and_eval

from sklearn.metrics import roc_auc_score, average_precision_score


def add_noise(x, noise_res, noise_std):
    """DAE corruption, ported from MedIAnomaly dae_worker.add_noise (Kascenas et al.).
    Coarse noise (16x16) upsampled to full res and randomly rolled — NOT per-pixel
    Gaussian. That is the point: coarse noise forces the net to use context to repair a
    blob, which is what makes it transfer to blob-shaped pathology."""
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res),
                      std=noise_std).to(x.device)
    ns = F.interpolate(ns, size=x.shape[-1], mode='bilinear', align_corners=True)
    roll_x = random.choice(range(x.shape[-2]))
    roll_y = random.choice(range(x.shape[-1]))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
    ns = (ns - 0.5) * 2          # their `config.center` branch, always taken for CXR
    return x + ns, ns


@torch.no_grad()
def _forward_scores(net, criterion, x, batch_size=256):
    """Per-image anomaly scores for a whole tensor. net MUST already be in eval mode."""
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        out.append(criterion(xb, net(xb), anomaly_score=True).cpu())
    return torch.cat(out).numpy()


def _forward_scores_grad(net, criterion, x, batch_size=64):
    """Same, for gradient-based scores, which need graph + input grads."""
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device).requires_grad_(True)
        out.append(criterion(xb, net(xb), anomaly_score=True).detach().cpu())
    return torch.cat(out).numpy()


def evaluate_scores(scores, y):
    """Image-level metrics, matching AEWorker.evaluate."""
    return {
        'AUC': float(roc_auc_score(y, scores)),
        'AP':  float(average_precision_score(y, scores)),
        'normal_score':   float(np.mean(scores[y == 0])),
        'abnormal_score': float(np.mean(scores[y == 1])),
    }


def train_and_eval(method, seed=TRAIN_SEED, *, extra_params=None, epochs=None,
                   train_loss=None, score_loss=None, save_weights=True, verbose=True):
    """Run one (method, params, seed) and persist it. Returns the manifest dict."""
    spec = dict(METHODS[method])
    if train_loss is not None: spec['train_loss'] = train_loss
    if score_loss is not None: spec['score_loss'] = score_loss
    params = dict(extra_params or {})
    # A shortened run must never share an id with the full-length one, or a 2-epoch smoke
    # run would later be reused as if it were the real 250-epoch result. config_fingerprint
    # records the GLOBAL epoch count and cannot see this per-call override, so encode it here.
    if epochs is not None and epochs != EPOCHS:
        params['ep'] = epochs

    # --- skip / restore -------------------------------------------------------------
    rid = run_id(method, seed, **params)
    if run_exists(rid) or (USE_WANDB and fetch_run(rid)):
        try:
            man, _ = load_run(rid)
            if verbose:
                print(f"{rid}: reused  AUC={man['metrics']['AUC']:.4f}")
            return man
        except (RuntimeError, FileNotFoundError) as e:
            print(f'{rid}: stored record unusable, retraining\n    {e}')

    if spec['train_loss'] == 'perceptual' or spec['score_loss'] == 'perceptual':
        if not HAS_VGG:
            print(f'{rid}: SKIPPED — perceptual loss needs VGG19 weights (see Cell 3.0b)')
            return None

    # --- deterministic setup --------------------------------------------------------
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    net = build_net(spec['net'])
    train_crit = build_loss(spec['train_loss'])
    score_crit = train_crit if spec['score_loss'] == spec['train_loss'] else build_loss(spec['score_loss'])
    opt = Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_TRAIN), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=False, generator=g)

    n_epochs = epochs if epochs is not None else EPOCHS
    tuple_loss = spec['train_loss'] in _TUPLE_LOSSES
    n_params = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f"{rid}: {spec['net'].upper()} {n_params:,} params | train={spec['train_loss']} "
              f"score={spec['score_loss']} | {n_epochs} epochs")

    # --- train ----------------------------------------------------------------------
    epoch_loss, t0 = [], time.time()
    for ep in range(1, n_epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            net_in = xb
            if spec.get('denoise'):
                noisy, _ = add_noise(xb, spec['noise_res'], spec['noise_std'])
                out = net(noisy)          # corrupted IN ...
            else:
                out = net(xb)
            loss = train_crit(net_in, out)    # ... clean as TARGET
            if tuple_loss:
                loss = loss[0]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        epoch_loss.append(tot / n)
        if verbose and (ep == 1 or ep % max(1, n_epochs // 10) == 0 or ep == n_epochs):
            print(f'    ep {ep:>4}/{n_epochs}  loss {epoch_loss[-1]:.5f}  '
                  f'({time.time() - t0:.0f}s)')

    # --- evaluate -------------------------------------------------------------------
    net.eval()                       # never score in train mode (see markdown)
    if spec['score_loss'] in _GRAD_LOSSES:
        scores = _forward_scores_grad(net, score_crit, X_TEST)
    else:
        scores = _forward_scores(net, score_crit, X_TEST)
    metrics = evaluate_scores(scores, Y_TEST)
    metrics['train_loss_final'] = epoch_loss[-1]
    metrics['n_params'] = n_params
    metrics['minutes'] = (time.time() - t0) / 60
    if verbose:
        print(f"    -> AUC {metrics['AUC']:.4f}  AP {metrics['AP']:.4f}  "
              f"({metrics['minutes']:.1f} min)")

    return save_run(rid, method=method, seed=seed,
                    params={**params, 'train_loss': spec['train_loss'],
                            'score_loss': spec['score_loss']},
                    metrics=metrics, epoch_loss=epoch_loss,
                    arrays={'scores': scores, 'labels': Y_TEST},
                    weights={'net': net.state_dict()} if save_weights else None,
                    extra={'spec': {k: v for k, v in spec.items()}})


def rescore(base_method, score_loss, seed=TRAIN_SEED, *, name=None, verbose=True):
    """Score an ALREADY-TRAINED model with a different anomaly-score function. No
    training happens — the base run's stored weights are reused.

    This is not a shortcut, it is what the score-swap methods actually are: Table 5 gives
    AE-Grad the AE backbone and the AE objective, changing only the score, and likewise
    for both VAE-Grad variants. Training them from scratch would differ from `ae`/`vae`
    only by the random seed. Every off-diagonal cell of A1 is the same operation.

    Note on batching: a gradient score takes autograd.grad of `loss.mean()`, so scoring B
    images at once divides every score by exactly B relative to their batch_size=1 test
    loader. AUC and AP are rank statistics, so this is invisible to both — verified —
    but do not compare raw gradient-score magnitudes across batch sizes."""
    name = name or f'{base_method}-{score_loss}'
    rid  = run_id(name, seed)
    if run_exists(rid):
        try:
            man, _ = load_run(rid)
            if verbose:
                print(f"{rid}: reused  AUC={man['metrics']['AUC']:.4f}")
            return man
        except (RuntimeError, FileNotFoundError) as e:
            print(f'{rid}: stored record unusable, recomputing\n    {e}')

    if score_loss == 'perceptual' and not HAS_VGG:
        print(f'{rid}: SKIPPED — perceptual score needs VGG19 (see Cell 3.0b)')
        return None

    base = train_and_eval(base_method, seed=seed, verbose=verbose)   # trains only if needed
    if base is None:
        return None

    net = build_net(METHODS[base_method]['net'])
    load_run(run_id(base_method, seed), models={'net': net})         # also calls .eval()
    fn = _forward_scores_grad if score_loss in _GRAD_LOSSES else _forward_scores
    scores = fn(net, build_loss(score_loss), X_TEST)
    metrics = evaluate_scores(scores, Y_TEST)
    metrics['n_params'] = base['metrics'].get('n_params', float('nan'))
    del net
    if verbose:
        print(f"{rid}: rescored {base_method} with {score_loss} -> "
              f"AUC {metrics['AUC']:.4f}  AP {metrics['AP']:.4f}")
    # deliberately stores NO weights: this run owns no model, it re-scores another's.
    # `base_run` keeps that dependency explicit and auditable.
    return save_run(rid, method=name, seed=seed,
                    params={'train_loss': METHODS[base_method]['train_loss'],
                            'score_loss': score_loss},
                    metrics=metrics, arrays={'scores': scores},
                    extra={'base_run': run_id(base_method, seed), 'base_method': base_method})


# %% [markdown]
# ### Cell 3.2b — Smoke test
# Trains each backbone for 2 epochs and immediately reloads it, checking that the
# reloaded network reproduces the saved scores. If save/load drifts again — the
# train-mode-BatchNorm class of bug — it fails HERE, cheaply, rather than after a
# 250-epoch run.
#
# **The VAE is exempt from the exact check, and that is a property of the method, not a
# concession.** `VaeBottleNeck.forward` calls `reparameterize` unconditionally, with no
# `if self.training` guard — verbatim from MedIAnomaly — so the latent code is *sampled*
# at test time too and two scoring passes over identical weights give different numbers.
# That stochasticity is also why their VAE row carries a wider spread than the AE rows.
# For the VAE we therefore assert that the reloaded AUC agrees to within sampling noise.

# %% [CELL 3.2b]  Smoke test

import shutil

_STOCHASTIC_NETS = {'vae'}     # sample z at eval -> scores are not bit-reproducible


def smoke_test(epochs=2, methods=('ae', 'aeu', 'vae', 'dae')):
    ok = True
    for m in methods:
        rid = run_id(m, 0, ep=epochs)
        shutil.rmtree(_run_dir(rid), ignore_errors=True)      # always a fresh train
        man = train_and_eval(m, seed=0, epochs=epochs, verbose=False)
        if man is None:
            print(f'  {m:<6} skipped'); continue
        net = build_net(METHODS[m]['net'])
        _, arrays = load_run(rid, models={'net': net})        # load_run calls .eval()
        rescored = _forward_scores(net, build_loss(METHODS[m]['score_loss']), X_TEST)

        if METHODS[m]['net'] in _STOCHASTIC_NETS:
            d_auc = abs(roc_auc_score(Y_TEST, rescored) - man['metrics']['AUC'])
            good = d_auc < 0.02
            detail = f'reload |ΔAUC|={d_auc:.2e} (stochastic latent)'
        else:
            delta = float(np.abs(rescored - arrays['scores']).max())
            good = delta < 1e-5
            detail = f'reload max|Δscore|={delta:.2e}'
        ok &= good
        print(f'  {m:<6} AUC={man["metrics"]["AUC"]:.4f}  {detail}  '
              f'{"OK" if good else "MISMATCH"}')
    print('smoke test:', 'PASS' if ok else 'FAIL')
    return ok

# smoke_test()      # uncomment to run (fast; ~1 min on GPU)


# %% [markdown]
# ---
# # Experiment 1 — Baseline ladder (reproduction)
# ## **Cell 4.0** — Run the seven Table-6 methods
# Three seeds each, so the comparison table below carries a spread and a one-seed
# difference of 0.5 AUC can be recognised as noise rather than a finding.
#
# Order matters: cheap and diagnostic first (`ae`), so a broken data path shows up in
# minutes. Every run is skipped if already stored, so the cell is safe to re-execute and
# safe to interrupt — that is what makes it survivable on Kaggle's 9-hour limit.
#
# Expected wall clock on a P100: ~4 min per AE-family run × 3 seeds × 6 methods, plus
# DAE, which is ~70× the FLOPs (2.15 G vs 29.9 M) and dominates the total.

# %% [CELL 4.0]  Baseline ladder

BASELINE_METHODS = ['ae', 'ae-l1', 'ae-ssim', 'ae-pl', 'vae', 'aeu', 'dae']
SEEDS = [42, 43, 44] if not SAMPLE_MODE else [42]

# DAE costs 72x an AE run (2.15 GFLOP vs 29.9 MFLOP per step), so three seeds of it would
# outweigh the entire rest of the notebook. One seed still reproduces its Table 6 row and
# preserves the point that matters here -- AE-U and DAE are the only SOTA methods on this
# benchmark that do NOT use ImageNet weights -- we simply cannot quote a spread for it.
# Report it as n=1 and never compare it to another method on a sub-1-point difference.
EXPENSIVE = {'dae'}
SEEDS_FOR = {m: ([SEEDS[0]] if m in EXPENSIVE else SEEDS) for m in BASELINE_METHODS}

for m in BASELINE_METHODS:              # cheap methods first: a broken data path shows
    for s in SEEDS_FOR[m]:              # up in minutes, not after the expensive run
        train_and_eval(m, seed=s)

print('\nbaseline ladder complete')


# %% [markdown]
# ---
# ## **Cell 4.1** — Reproduction table
# Ours (mean ± sd over seeds) next to MedIAnomaly Table 6, with the gap in AUC points.
#
# How to read `ΔAUC`: within about ±1.5 points we have reproduced the row (their own sd
# is up to 1.1). A gap larger than that is a real difference and needs explaining before
# any experiment built on top of it means anything — most likely suspects, in order:
# a different train/test split, inputs not in [-1,1], or scoring in train mode.

# %% [CELL 4.1]  Reproduction table

def reproduction_table(methods=None):
    df = completed_runs()
    if df.empty:
        print('no runs stored yet'); return df
    if DATASET != 'RSNA':
        # TABLE6_RSNA is the RSNA column of the benchmark's Table 6. Comparing another
        # dataset's runs against it would print a table of meaningless deltas that could
        # easily end up in the report.
        print(f'reproduction table skipped: no reference column for {DATASET}')
        return df
    methods = methods or BASELINE_METHODS
    rows = []
    for m in methods:
        sub = df[df['method'] == m]
        if sub.empty:
            rows.append({'method': m, 'n_seeds': 0}); continue
        ref = TABLE6_RSNA.get(m)
        auc, ap = sub['AUC'] * 100, sub['AP'] * 100
        rows.append({
            'method':   m,
            'n_seeds':  len(sub),
            'AUC':      f'{auc.mean():.1f}±{auc.std(ddof=0):.1f}',
            'AP':       f'{ap.mean():.1f}±{ap.std(ddof=0):.1f}',
            'AUC_ref':  f'{ref[0]:.1f}±{ref[1]:.1f}' if ref else '—',
            'AP_ref':   f'{ref[2]:.1f}±{ref[3]:.1f}' if ref else '—',
            'dAUC':     f'{auc.mean() - ref[0]:+.1f}' if ref else '—',
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


REPRO = reproduction_table()


# %% [markdown]
# ---
# ## **Cell 4.2** — Three more Table-6 rows for free
# AE-Grad, VAE-Grad_rec and VAE-Grad_combi share both backbone and training objective
# with `ae` / `vae` (Table 5); only the anomaly score differs. So they need **no training
# at all** — `rescore` reuses the stored weights.
#
# They also make the point A1 is built on, using the benchmark's own numbers: AE-Grad
# reaches 73.7 against AE's 67.5 with *identical* training. +6.2 AUC from changing
# nothing but how the residual is read.

# %% [CELL 4.2]  Score-swap rows (no training)

SCORE_SWAPS = [('ae',  'l2grad',        'ae-grad'),
               ('vae', 'vaegrad-rec',   'vae-grad-rec'),
               ('vae', 'vaegrad-combi', 'vae-grad-combi')]

for _base, _score, _name in SCORE_SWAPS:
    for _s in SEEDS:
        rescore(_base, _score, _s, name=_name)

print()
REPRO_SWAPS = reproduction_table(BASELINE_METHODS + [n for _, _, n in SCORE_SWAPS])


# %% [markdown]
# ---
# # Experiment A1 (novel) — the training loss and the anomaly score are two decisions, not one
# ## **Cell 5.0** — the grid
# **State the prior work accurately first, because the benchmark already does part of
# this.** Table 5 shows AE-Grad, VAE-Grad_rec and VAE-Grad_combi all share a backbone and
# a training objective with plain AE / VAE and differ *only* in the anomaly score — and
# the effect is large (AE 67.5 → AE-Grad 73.7 with identical training). So MedIAnomaly
# does decouple training from scoring, and any claim that "they never separate the two"
# would be wrong.
#
# What they never do is cross the **reconstruction-metric** axis. Every metric variant is
# locked to its own objective: AE-SSIM trains on 1−SSIM and scores on 1−SSIM, AE-PL trains
# and scores on perceptual distance. So the headline **AE-SSIM 80.9 vs AE 67.5** carries a
# single label over two distinct mechanisms, and Table 6 cannot say which one earned the
# +13.4. The gradient rows show the question is worth asking; nobody has asked it here.
#
# So: train an AE with each of {L2, SSIM, perceptual}, then score every trained model with
# each of those three **plus the gradient score**, which costs no extra training. The
# diagonal reproduces Table 6, the `l2grad` column reproduces AE-Grad, and the rest is new.
#
# | If the gain comes from… | Signature in the grid |
# |---|---|
# | **the score** (SSIM/PL are better residual metrics on any reconstruction) | rows differ little, **columns** separate strongly |
# | **the objective** (training on SSIM/PL yields a genuinely different model) | **rows** separate strongly |
#
# ### The AE-U row is the most valuable one, and it is free
# AE-U is the largest unexplained jump in the whole table: **86.5 vs AE's 67.5, +19.0**,
# and unlike AE-PL it uses no ImageNet weights. It changes two things at once — it trains
# with an uncertainty-weighted loss, *and* it scores by dividing the residual by a learned
# per-pixel variance. Nobody has separated those.
#
# Adding `aeu` as a row costs **no extra training** (it is already in the ladder), and the
# `aeu` row scored with plain `l2` answers it outright:
#
# * **drops toward ~67** → the entire +19 is the *score*. Uncertainty training did not
#   build a better reconstructor; it built a better error metric. That is a strong,
#   compact claim, and it puts AE-U in the same category as AE-Grad.
# * **stays near ~86** → uncertainty training genuinely produced a different model, and
#   the variance-weighted score is incidental.
#
# The reverse cell (a plain AE scored with the AE-U rule) cannot exist: that score needs
# the `log_var` head only the AEU backbone has. So this is a 2-cell decomposition, not a
# full grid — worth stating plainly rather than implying symmetry we do not have.
#
# A column-dominated result is the more useful outcome: it would mean the expensive
# ImageNet-dependent part of AE-PL is only needed at *test* time, so a plain L2 AE scored
# perceptually recovers most of the 87.5 — and cheaply.
#
# 3 trainings cover all 12 cells; every off-diagonal cell is a `rescore` of stored weights.

# %% [CELL 5.0]  A1 — train-loss × score-loss grid

A1_TRAIN  = ['l2', 'ssim', 'perceptual', 'aeu']
A1_SCORE  = ['l2', 'ssim', 'perceptual', 'l2grad', 'aeu']
A1_SEEDS  = SEEDS


def a1_grid(seeds=None, train_losses=None, score_losses=None):
    seeds        = seeds or A1_SEEDS
    train_losses = train_losses or A1_TRAIN
    score_losses = score_losses or A1_SCORE
    if not HAS_VGG:
        train_losses = [l for l in train_losses if l != 'perceptual']
        score_losses = [l for l in score_losses if l != 'perceptual']
        print('VGG unavailable — running the grid without the perceptual axis')

    # the diagonal cells ARE the Table 6 baselines; reuse them rather than retrain
    base_of = {'l2': 'ae', 'ssim': 'ae-ssim', 'perceptual': 'ae-pl', 'aeu': 'aeu'}
    recs = []
    for tr in train_losses:
        for s_ in seeds:
            for sc in score_losses:
                # the AE-U score divides by a predicted per-pixel variance, so it needs
                # the log_var head that only the AEU backbone has -- that column exists
                # for the AE-U row alone, not for the whole grid
                if sc == 'aeu' and METHODS[base_of[tr]]['net'] != 'aeu':
                    continue
                if sc == tr:
                    man = train_and_eval(base_of[tr], seed=s_)
                else:
                    man = rescore(base_of[tr], sc, s_, name=f'a1-{tr}-{sc}')
                if man is None:
                    continue
                recs.append({'train_loss': tr, 'score_loss': sc, 'seed': s_,
                             'AUC': man['metrics']['AUC'], 'AP': man['metrics']['AP']})
    return pd.DataFrame(recs)


A1 = a1_grid()

if not A1.empty:
    piv = A1.pivot_table(index='train_loss', columns='score_loss', values='AUC',
                         aggfunc='mean') * 100
    print('\nA1 — image AUC (%), rows = training loss, columns = scoring function\n')
    print(piv.round(1).to_string())
    # Which axis actually carries the variance? Row spread = effect of the objective,
    # column spread = effect of the residual metric.
    print(f'\n  spread across training losses (row means) : '
          f'{piv.mean(axis=1).max() - piv.mean(axis=1).min():.1f} AUC pts')
    print(f'  spread across scoring losses  (col means) : '
          f'{piv.mean(axis=0).max() - piv.mean(axis=0).min():.1f} AUC pts')
    print(f'  best cell: train={piv.stack().idxmax()[0]} score={piv.stack().idxmax()[1]} '
          f'-> {piv.stack().max():.1f}   (Table 6 best img-rec: AE-PL 87.5)')


# %% [markdown]
# ---
# # Experiment A2 (novel) — what the denoising AE's noise scale actually controls
# ## **Cell 5.1** — noise_res × noise_std sweep
# DAE reaches 86.1 on RSNA with noise fixed at `noise_res=16, noise_std=0.2`
# (`dae_worker.py`) — two constants inherited from Kascenas et al.'s **brain MRI** work
# and never re-examined for chest radiographs, where lesions are far more diffuse and
# have no sharp boundary.
#
# `noise_res` sets the spatial scale of the corruption: it is sampled at
# `noise_res × noise_res` and bilinearly upsampled to 64×64, so *smaller* `noise_res`
# means *coarser*, blob-like noise. The hypothesis worth testing is that DAE works
# precisely when the corruption scale matches the lesion scale — which would predict a
# peak, not a plateau, and a peak whose location is dataset-specific rather than
# universal.
#
# Two readings, both publishable as findings:
# * **A peak away from res=16** — the inherited constant is mis-tuned for CXR, and
#   scale-matching is a real, transferable design rule.
# * **A flat curve** — DAE's gain is *not* about matching lesion scale, which contradicts
#   the intuitive story the method is usually explained with.
#
# This is the more expensive experiment (DAE runs at 2.15 GFLOPs), so it defaults to one
# seed and a coarse grid; widen `A2_SEEDS` once the shape is known.
#
# The grid deliberately includes the default cell `(16, 0.2)`, which retrains what the
# `dae` baseline already ran at the same seed under a different run id. It costs one run
# and buys a free consistency check: those two AUCs must agree, and if they don't, the
# training path is not deterministic and every other number in this notebook is suspect.

# %% [CELL 5.1]  A2 — denoising noise-scale sweep

# OFF BY DEFAULT, and the cost estimate below is now MEASURED rather than extrapolated.
# One DAE run took 82 min on a T4 (vs 1.6 min for an AE), so the grid as configured is
# 4 x 82 min = 5.5 h, and the original 4x3 grid would have been 16.4 h. An earlier comment
# here said "~45 h", which came from a FLOP-based extrapolation before DAE had ever been
# timed; it was wrong and is corrected here.
#
# It stays off for a reason that cost, not time, decides: DAE scores 83.87 on RSNA --
# 3.71 below AE-PL, 4.87 below the perceptual+uncertainty ensemble. For this sweep to
# change any conclusion in an IMAGE-LEVEL study, noise scale alone would have to be worth
# more than +4.87 AUC, and the largest lever measured anywhere in this project (the
# scoring function, 26.7 pts of spread) is not a corruption hyperparameter.
#
# The experiment is not cancelled, it is RELOCATED to the CV project, where DAE is the
# benchmark's best PIXEL-level method and where "must the corruption scale match the
# lesion scale?" actually pays off. At image level it answers a question this project
# cannot ask (no localisation claim is made or supported here).
RUN_A2  = False
A2_RES  = [4, 8, 16, 32]        # 4 = very coarse blobs … 32 = near per-pixel
A2_STD  = [0.2]                 # MedIAnomaly's default; widen only if res shows structure
A2_SEEDS = [42]


def a2_sweep(res_list=None, std_list=None, seeds=None):
    res_list, std_list = res_list or A2_RES, std_list or A2_STD
    seeds = seeds or A2_SEEDS
    recs = []
    for r in res_list:
        for sd in std_list:
            for s in seeds:
                METHODS['dae-sweep'] = dict(METHODS['dae'], noise_res=r, noise_std=sd)
                man = train_and_eval('dae-sweep', seed=s,
                                     extra_params={'res': r, 'std': sd})
                if man is None:
                    continue
                recs.append({'noise_res': r, 'noise_std': sd, 'seed': s,
                             'AUC': man['metrics']['AUC'], 'AP': man['metrics']['AP']})
    return pd.DataFrame(recs)


A2 = a2_sweep() if RUN_A2 else pd.DataFrame()
if not RUN_A2:
    print('A2 skipped (RUN_A2=False) — see the note above; it lives in the CV project.')

if not A2.empty:
    piv2 = A2.pivot_table(index='noise_res', columns='noise_std', values='AUC',
                          aggfunc='mean') * 100
    print('\nA2 — image AUC (%), rows = noise_res (smaller = coarser), '
          'columns = noise_std\n')
    print(piv2.round(1).to_string())
    best = piv2.stack().idxmax()
    print(f'\n  best: noise_res={best[0]} noise_std={best[1]} -> {piv2.stack().max():.1f}')
    if 16 in piv2.index and 0.2 in piv2.columns:
        print(f'  MedIAnomaly default (16, 0.2): {piv2.loc[16, 0.2]:.1f}  '
              f'(gain over default: {piv2.stack().max() - piv2.loc[16, 0.2]:+.1f} pts)')
        # determinism check described in the markdown: the (16,0.2) sweep cell and the
        # `dae` baseline are the same configuration at the same seed
        base = completed_runs().query("method == 'dae' and seed == @A2_SEEDS[0]")
        if not base.empty:
            d = abs(float(base['AUC'].iloc[0]) * 100 - piv2.loc[16, 0.2])
            print(f'  determinism check vs `dae` baseline: |Δ| = {d:.2f} pts '
                  f'{"OK" if d < 0.5 else "-- TRAINING IS NOT DETERMINISTIC, investigate"}')

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in piv2.columns:
        ax.plot(piv2.index, piv2[c], marker='o', label=f'std={c}')
    ax.axvline(16, ls='--', c='grey', lw=1)
    ax.text(16, ax.get_ylim()[0], ' MedIAnomaly default', fontsize=8, color='grey')
    ax.set_xscale('log', base=2); ax.set_xticks(A2_RES); ax.set_xticklabels(A2_RES)
    ax.set_xlabel('noise_res  (smaller = coarser corruption)')
    ax.set_ylabel('image AUC (%)'); ax.legend(); ax.set_title('A2 — DAE noise scale on RSNA')
    fig.tight_layout(); plt.show()


# %% [markdown]
# ---
# # Experiment A3 (novel) — do the scores see different things?
# ## **Cell 5.2** — score ensembling, no training at all
# A1 showed the scoring function carries most of each method's gain (column spread 26.7
# vs row spread 14.2). If SSIM, perceptual and AE-U's variance-weighted score each read a
# *different* part of the residual — local structure, semantics, calibrated uncertainty —
# then combining them should beat any one of them. If they are all reading the same
# signal through different lenses, combining changes nothing. Either answer is a result.
#
# This costs **zero training**: every run already stored its per-image `scores` array, so
# this cell only loads and combines them.
#
# **Rank averaging, not score averaging.** The four scores live on wildly different scales
# (L2 ~1e-2, perceptual ~0.3, AE-U's is negative) so a plain mean would just return the
# largest-scale member. Ranks are scale-free and monotone, and AUC/AP only depend on
# ordering, so nothing is lost by discarding the magnitudes.
#
# ### Pre-registering the combinations (this matters for the paper)
# With 4 members there are 15 subsets. Picking the best of 15 *on the test set* and
# reporting it as "our method" is test-set fitting and will not survive review. So two
# combinations are declared **a priori**, before looking:
#
# * `all4` — every diagonal score. The obvious, assumption-free choice.
# * `noIN` — AE + AE-SSIM + AE-U only. Pre-registered because it answers a question that
#   matters independently: MedIAnomaly notes that most of their SOTA relies on ImageNet
#   weights, with AE-U and DAE the exceptions. If `noIN` reaches AE-PL's 87.5 without
#   VGG19, that is a practically useful result for settings where ImageNet pre-training
#   is unavailable or unwanted.
#
# The full 15-subset scan is printed too, but explicitly labelled exploratory. Report the
# two pre-registered rows as findings; report anything else as a hypothesis for future work.

# %% [CELL 5.2]  A3 — score ensembling

from itertools import combinations

# members: (label, method name whose stored run holds the score)
# (label, method, params) — params are REQUIRED because a member's run id may carry them.
# The post-hoc head's id is run_id('ae-ssim-posthoc-u', seed, w='2'); resolving it with
# run_id(method, seed) alone silently returns nothing and every ensemble result comes back
# empty with no exception raised.
#
# A3_MEMBERS is the PRE-REGISTERED four. Do not add to it: the registered statistic
# (15 subsets, 150 splits, selection bias -0.20) is defined over exactly this set, and
# enlarging it would replace a pre-registered number with a post-hoc one. The extended
# set below is reported SEPARATELY and labelled exploratory.
A3_MEMBERS = [('l2', 'ae', {}), ('ssim', 'ae-ssim', {}),
              ('perceptual', 'ae-pl', {}), ('aeu', 'aeu', {})]
A3_MEMBERS_EXT = A3_MEMBERS + [('ssim+head', 'ae-ssim-posthoc-u', {'w': '2'})]
A3_PREREGISTERED = {
    'all4': ['l2', 'ssim', 'perceptual', 'aeu'],
    'noIN': ['l2', 'ssim', 'aeu'],          # no ImageNet weights anywhere in this one
}


def _ranks(x):
    """Ranks in [0,1]. argsort().argsort() is the rank; dividing makes members with
    different test-set sizes comparable and keeps the mean interpretable."""
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r / max(len(r) - 1, 1)


def load_member_scores(seed, members=None):
    """Per-image scores for each ensemble member at one seed. Returns {} if any member
    is missing, because a partial ensemble is not the ensemble we pre-registered."""
    out = {}
    for label, method, params in (members or A3_MEMBERS):
        rid = run_id(method, seed, **params)
        if not run_exists(rid) and not (USE_WANDB and fetch_run(rid)):
            print(f'  seed {seed}: member {label!r} ({rid}) missing — skipping this seed')
            return {}
        _, arrays = load_run(rid)
        out[label] = arrays['scores']
    return out


def a3_ensemble(seeds=None, members=None):
    seeds = seeds or SEEDS
    members_spec = members or A3_MEMBERS
    rows = []
    for s in seeds:
        member_scores = load_member_scores(s, members_spec)
        if not member_scores:
            continue
        ranked = {k: _ranks(v) for k, v in member_scores.items()}
        labels = [l for l, _, _ in members_spec]
        for r in range(1, len(labels) + 1):
            for combo in combinations(labels, r):
                m = evaluate_scores(np.mean([ranked[c] for c in combo], axis=0), Y_TEST)
                rows.append({'combo': '+'.join(combo), 'k': r, 'seed': s,
                             'AUC': m['AUC'], 'AP': m['AP'],
                             'preregistered': next((n for n, v in A3_PREREGISTERED.items()
                                                    if set(v) == set(combo)), '')})
    df = pd.DataFrame(rows)
    # persist only the pre-registered ones as runs; the scan is exploratory output
    for name, combo in A3_PREREGISTERED.items():
        for s in seeds:
            sub = df[(df.combo == '+'.join(combo)) & (df.seed == s)]
            if sub.empty:
                continue
            rid = run_id(f'ens-{name}', s)
            if run_exists(rid):
                continue
            member_scores = load_member_scores(s, members_spec)
            sc = np.mean([_ranks(member_scores[c]) for c in combo], axis=0)
            save_run(rid, method=f'ens-{name}', seed=s,
                     params={'members': '+'.join(combo)},
                     metrics=evaluate_scores(sc, Y_TEST), arrays={'scores': sc},
                     extra={'note': 'rank-average of stored scores; no training'})
    return df


A3 = a3_ensemble()

if not A3.empty:
    best_single = A3[A3.k == 1].groupby('combo').AUC.mean().max() * 100
    agg = (A3.groupby(['combo', 'k', 'preregistered']).AUC
             .agg(['mean', 'std']).reset_index().sort_values('mean', ascending=False))
    agg['mean'] *= 100; agg['std'] = agg['std'].fillna(0) * 100

    print('\nA3 — PRE-REGISTERED (report these)\n')
    pre = agg[agg.preregistered != '']
    for _, r in pre.iterrows():
        print(f"  {r['preregistered']:<6} {r['combo']:<32} "
              f"AUC {r['mean']:.2f}+-{r['std']:.2f}   vs best single {best_single:.2f} "
              f"({r['mean'] - best_single:+.2f})")
    print(f"\n  reference: AE-PL alone = {A3[A3.combo=='perceptual'].AUC.mean()*100:.2f}, "
          f"MedIAnomaly best img-rec = 87.5")

    print('\nA3 — full subset scan (EXPLORATORY — do not report as a result)\n')
    print(agg[['combo', 'k', 'mean', 'std']].to_string(index=False,
          float_format=lambda v: f'{v:.2f}'))


# %% [markdown]
# ---
# # Experiment A4 (novel) — is AE-U's advantage a training paradigm, or a wrapper?
# ## **Cell 5.3** — a variance head fitted post hoc onto a frozen AE
# A1's sharpest result: AE-U scored with plain L2 gives **65.4**, *below* the plain AE's
# 68.3. Uncertainty training does not build a better reconstructor — it builds a worse
# one — yet AE-U reaches 86.9. All of the gain is in dividing the residual by the learned
# per-pixel variance.
#
# If that is true, the variance should not need to be learned *jointly* at all. So:
#
# 1. take the **already-trained plain AE** and freeze it completely (`eval()`, no grads),
# 2. attach only a variance head — the same `BasicBlock` that AEU uses for its extra
#    output channel, so capacity is matched rather than quietly increased,
# 3. train **only that head**, with AE-U's own objective `exp(-log_var)*(x-x̂)² + log_var`,
#    on a reconstruction `x̂` that never changes,
# 4. score exactly as AE-U does, with `exp(-log_var)*(x-x̂)²`.
#
# | Outcome | Reading |
# |---|---|
# | reaches ~86 | uncertainty scoring is a **post-hoc wrapper**. Any trained AE can be upgraded without retraining it, and joint uncertainty training was never necessary. This is the version with a method contribution, not just an analysis. |
# | lands between 68 and 86 | joint training matters *partly* — the reconstruction and the variance co-adapt, and the decomposition is not clean. |
# | stays near 68 | the variance is only useful when the reconstruction was shaped alongside it; AE-U is genuinely a training paradigm. A negative result, and a clean one. |
#
# The AE stays in `eval()` throughout. That is not a detail: in train mode its BatchNorm
# would consume the running statistics of the model we are meant to be holding fixed, and
# the frozen baseline would silently drift while we measured it.

# %% [CELL 5.3]  A4 — post-hoc uncertainty head

class PostHocVarHead(nn.Module):
    """Predicts per-pixel log-variance from a frozen AE's last decoder feature map.

    Deliberately the SAME module AEU adds: AEU replaces `de_block4` with a BasicBlock
    emitting 2*in_planes channels and splits them into (x_hat, log_var). Here x_hat comes
    from the frozen AE and this block emits the log_var channel only, so A4 and AE-U differ
    in HOW the variance is trained, not in how much capacity is available to it.

    `width` is a CAPACITY CONTROL, and A4 is uninterpretable without it. At width=1 the
    head has 256 parameters -- exactly AEU's parameter increment (2,351,440 - 2,351,184),
    so the comparison is fair. But the post-hoc head reads decoder features that were
    optimised for reconstruction alone, never for variance, so a failure at width=1 has
    two possible causes: uncertainty genuinely needs joint training, or 256 parameters
    cannot extract it from features that do not encode it. Running a wide head separates
    them. If wide also fails, capacity is not the explanation."""
    def __init__(self, width=1, chans=None, linear=False):
        super().__init__()
        # NOTE the confound this parameterisation exists to break. BasicBlock(...,
        # last_layer=True) does `layers = layers[:-2]`, stripping the BatchNorm and the
        # ReLU -- so width=1 is a SINGLE ConvTranspose2d with no nonlinearity and no
        # normalisation, while width>=2 is Conv-BN-ReLU x2 + BasicBlock. The original
        # w=1 -> w=2 jump therefore varies depth, nonlinearity AND normalisation, not
        # width, and the flat plateau across w=2/4/8 shows width alone does nothing.
        #   chans=<int> : use exactly this many hidden channels (decouples capacity from
        #                 architecture -- a NONLINEAR head with ~AE-U's parameter budget)
        #   linear=True : widen with a 1x1 conv but keep the whole head LINEAR (no BN,
        #                 no ReLU) -- many parameters, still linearly decodable only
        if linear:
            w = chans if chans is not None else BASE_WIDTH * max(width, 1)
            self.block = nn.Sequential(
                nn.Conv2d(1 * BASE_WIDTH, w, 1, bias=False),          # linear widening
                BasicBlock(w, 1, DE_DEPTH, upsample=True, last_layer=True))
        elif width == 1 and chans is None:
            self.block = BasicBlock(1 * BASE_WIDTH, 1, DE_DEPTH, upsample=True, last_layer=True)
        else:
            w = chans if chans is not None else BASE_WIDTH * width
            self.block = nn.Sequential(
                nn.Conv2d(1 * BASE_WIDTH, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(True),
                nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(True),
                BasicBlock(w, 1, DE_DEPTH, upsample=True, last_layer=True))

    def forward(self, de1):
        return self.block(de1)


def _wtag(width):
    """Filesystem-safe tag for a width spec, which may be an int or a variant dict."""
    if isinstance(width, dict):
        return 'lin' + str(width.get('chans', '')) if width.get('linear') else \
               'c' + str(width.get('chans', width.get('width', '?')))
    return str(width)


def a4_posthoc_uncertainty(seed=TRAIN_SEED, width=1, base_method='ae', epochs=None,
                           verbose=True):
    """`base_method` names the FROZEN backbone the head is fitted onto. Default 'ae'
    reproduces the original experiment; 'ae-pl' and 'ae-ssim' test whether post-hoc
    uncertainty is a general wrapper or a property of the L2-trained reconstruction.
    All three share the AE architecture, so de_features[0] is 16ch at 32x32 either way."""
    # Encode a shortened run in the id, exactly as train_and_eval does. Without this a
    # 50-epoch debugging head and a 250-epoch real head share an id, and config_fingerprint
    # cannot see the difference because it records the GLOBAL epoch count.
    _p = {'w': _wtag(width)}
    if epochs is not None and epochs != EPOCHS:
        _p['ep'] = epochs
    rid = run_id(f'{base_method}-posthoc-u', seed, **_p)
    if run_exists(rid) or (USE_WANDB and fetch_run(rid)):
        try:
            man, _ = load_run(rid)
            if verbose:
                print(f"{rid}: reused  AUC={man['metrics']['AUC']:.4f}")
            return man
        except (RuntimeError, FileNotFoundError) as e:
            print(f'{rid}: stored record unusable, recomputing\n    {e}')

    base = train_and_eval(base_method, seed=seed, verbose=verbose)  # trains only if needed
    if base is None:
        return None
    ae = build_net(METHODS[base_method]['net'])
    load_run(run_id(base_method, seed), models={'net': ae})      # load_run calls .eval()
    for p in ae.parameters():
        p.requires_grad = False

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = (PostHocVarHead(**width) if isinstance(width, dict)
            else PostHocVarHead(width=width)).to(device)
    opt = Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_epochs = epochs if epochs is not None else EPOCHS
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_TRAIN), batch_size=BATCH_SIZE, shuffle=True,
                        generator=g)
    if verbose:
        print(f'{rid}: head {sum(p.numel() for p in head.parameters()):,} params on a '
              f'FROZEN {base_method} ({sum(p.numel() for p in ae.parameters()):,}) '
              f'| {n_epochs} epochs')

    epoch_loss, t0 = [], time.time()
    for ep in range(1, n_epochs + 1):
        head.train()
        ae.eval()                    # every epoch: nothing may put the AE back in train mode
        tot, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():                       # frozen reconstruction
                out = ae(xb)
                x_hat, de1 = out['x_hat'], out['de_features'][0]
            log_var = head(de1)
            recon = (xb - x_hat) ** 2                   # constant w.r.t. the head
            loss = (torch.exp(-log_var) * recon + log_var).mean()   # AEULoss, verbatim
            if not torch.isfinite(loss):
                print(f'    non-finite loss at epoch {ep} — aborting'); return None
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        epoch_loss.append(tot / n)
        if verbose and (ep == 1 or ep % max(1, n_epochs // 10) == 0 or ep == n_epochs):
            print(f'    ep {ep:>4}/{n_epochs}  loss {epoch_loss[-1]:.5f}  '
                  f'({time.time() - t0:.0f}s)')

    head.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_TEST), 256):
            xb = X_TEST[i:i + 256].to(device)
            out = ae(xb)
            lv = head(out['de_features'][0])
            # AEULoss anomaly_score: the loss1 term only, exp(-log_var)*(x-x_hat)^2
            s = torch.exp(-lv) * (xb - out['x_hat']) ** 2
            scores.append(torch.mean(s, dim=[1, 2, 3]).cpu())
    scores = torch.cat(scores).numpy()
    metrics = evaluate_scores(scores, Y_TEST)
    metrics['minutes'] = (time.time() - t0) / 60
    if verbose:
        print(f"    -> AUC {metrics['AUC']:.4f}  AP {metrics['AP']:.4f}")
    return save_run(rid, method=f'{base_method}-posthoc-u', seed=seed,
                    params={'w': width, 'base': base_method,
                            'train_loss': f'{base_method}-frozen+posthoc-var',
                            'score_loss': 'aeu'},
                    metrics=metrics, epoch_loss=epoch_loss,
                    arrays={'scores': scores}, weights={'head': head.state_dict()},
                    extra={'base_run': run_id(base_method, seed)})


# w=1 is capacity-matched to AEU (256 params, its exact increment); the rest map out how
# much capacity post-hoc extraction actually needs. The first run showed w=1 FAILS (66.94,
# below the plain AE) while w=8 reaches 81.81 -- so the interesting quantity is the
# smallest head that works, which is a direct measure of how much joint training helps by
# making the variance easy to read rather than by making it exist.
A4_WIDTHS = [1, 2, 4, 8, 16]
for _w in A4_WIDTHS:
    for _s in SEEDS:
        a4_posthoc_uncertainty(seed=_s, width=_w)

_a4 = completed_runs()
if not _a4.empty and 'ae-posthoc-u' in set(_a4.method):
    def _m(meth):
        sub = _a4[_a4.method == meth]
        return (sub.AUC.mean() * 100, sub.AUC.std(ddof=0) * 100, len(sub)) if not sub.empty \
            else (float('nan'), float('nan'), 0)
    ae_m, ae_s, _ = _m('ae')
    au_m, au_s, _ = _m('aeu')
    print('\nA4 — where does AE-U\'s advantage live?\n')
    print(f'  plain AE, L2 score                   {ae_m:6.2f}+-{ae_s:.2f}')
    # `w` is stored by _wtag as a STRING ('2', 'lin8', 'c32'), so comparing it against the
    # INTEGERS in A4_WIDTHS matched nothing and every width fell through the `continue` --
    # the per-width table printed as an empty block while the recovery line below still
    # printed, because that one never filtered. Compare on _wtag(_w), the stored form.
    if 'w' not in _a4:
        print('  (no `w` column stored — cannot separate widths)')
    for _w in A4_WIDTHS if 'w' in _a4 else []:
        sub = _a4[(_a4.method == 'ae-posthoc-u') & (_a4['w'].astype(str) == _wtag(_w))]
        if sub.empty:
            continue
        m, sd, n = sub.AUC.mean() * 100, sub.AUC.std(ddof=0) * 100, len(sub)
        tag = 'capacity-matched' if _w == 1 else 'capacity control '
        # Print n. A width reduced to 2 seeds by the non-finite guard used to render
        # identically to one with 3, so a silently-dropped run was invisible here.
        print(f'  frozen AE + var head (w={_w}, {tag}) {m:6.2f}+-{sd:.2f} n={n}  '
              f'({m - ae_m:+.2f} over the AE it was fitted onto)')
    print(f'  AE-U, jointly trained                {au_m:6.2f}+-{au_s:.2f}')
    gap = au_m - ae_m
    ph = _a4[_a4.method == 'ae-posthoc-u']
    if gap > 0 and not ph.empty:
        # Best WIDTH by its 3-seed mean, not best RUN. `ph.AUC.max()` took a max over ~24
        # rows (5 widths x 3 seeds, plus the M4 variants, which share this method string)
        # and divided it by a 3-seed mean baseline -- a max numerator against a mean
        # denominator, which inflated the headline. The width is still chosen on TEST AUC,
        # so the sentence has to say so rather than let the number imply otherwise.
        if 'w' in ph:
            per_w = ph.groupby(ph['w'].astype(str)).AUC.agg(['mean', 'size'])
            per_w = per_w[per_w['size'] >= 2]
        else:
            per_w = pd.DataFrame()
        if not per_w.empty:
            best_w = per_w['mean'].idxmax()
            best   = per_w.loc[best_w, 'mean'] * 100
            wtag   = f'w={best_w}, n={int(per_w.loc[best_w, "size"])}'
        else:
            best, wtag = ph.AUC.mean() * 100, 'all widths pooled'
        print(f'\n  best-WIDTH post-hoc head ({wtag}) recovers '
              f'{(best - ae_m) / gap * 100:.0f}% of AE-U\'s +{gap:.2f} advantage, '
              f'with the reconstruction never retrained.')
        print('  NOTE the width is selected on TEST AUC — report it as a test-set maximum,')
        print('       not as an unbiased estimate of what a post-hoc head recovers.')
    print('  (A1 cross-check: AE-U scored with plain L2 = 65.4, i.e. BELOW the plain AE —'
          '\n   uncertainty training does not improve the reconstruction itself.)')


# %% [markdown]
# ### Cell 5.2b — Making the ensemble result honest
# The pre-registered ensembles **lost**: `all4` 84.88 and `noIN` 81.91, both below AE-PL's
# 87.58. Equal-weight rank averaging has no defence against a weak member, and the L2
# score is 19 points worse than the others — combos containing it average 80.08, those
# without it 86.44.
#
# The scan's best cell, `perceptual+aeu` = **88.74**, beats AE-PL and MedIAnomaly's best
# (87.5). It is probably real: the spread is 0.13 and the next two best combos contain the
# same pair. But it was chosen by looking at 15 options **on the test set**, so as reported
# it is a selection artefact and a reviewer would reject it.
#
# This cell measures what that selection is worth honestly. Repeatedly: split the test set
# in half (stratified by label), pick the best combination on half A, and report its AUC on
# half B — which it has never seen. The gap between "best on A" and "that same combo on B"
# **is** the selection bias, quantified rather than assumed away.
#
# Compare three numbers:
# * **selected-then-held-out** — what the procedure actually delivers on unseen data
# * **AE-PL fixed** — the baseline, needing no selection at all
# * **oracle** — the best combo scored on the same half it was chosen on, i.e. the
#   optimistic number the exploratory scan reports
#
# If selected-then-held-out still beats AE-PL, the ensemble is a genuine finding and
# survives review. If it collapses to AE-PL, the 88.74 was selection noise. Note this is
# still a within-dataset check; the decisive test is whether the same combination wins on
# a *different* dataset, which is what the multi-dataset sweep would settle.

# %% [CELL 5.2b]  A3 — selection-aware validation

def a3_split_half(seeds=None, n_splits=50, rng_seed=0, members=None):
    seeds = seeds or SEEDS
    members_spec = members or A3_MEMBERS
    labels = [l for l, _, _ in members_spec]
    all_combos = [c for r in range(1, len(labels) + 1) for c in combinations(labels, r)]
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in seeds:
        member_scores = load_member_scores(s, members_spec)
        if not member_scores:
            continue
        ranked = {k: _ranks(v) for k, v in member_scores.items()}
        idx0 = np.where(Y_TEST == 0)[0]
        idx1 = np.where(Y_TEST == 1)[0]
        for _ in range(n_splits):
            # stratified halves: both sides keep the 50/50 class balance, so an AUC on
            # either half is comparable to an AUC on the full test set
            p0, p1 = rng.permutation(idx0), rng.permutation(idx1)
            A = np.concatenate([p0[:len(p0)//2], p1[:len(p1)//2]])
            B = np.concatenate([p0[len(p0)//2:], p1[len(p1)//2:]])
            def auc(combo, ix):
                sc = np.mean([ranked[c][ix] for c in combo], axis=0)
                return roc_auc_score(Y_TEST[ix], sc)
            scored = [(auc(c, A), c) for c in all_combos]
            best_a, best_combo = max(scored)
            rows.append({'seed': s,
                         'selected': '+'.join(best_combo),
                         'oracle_A': best_a,                      # optimistic
                         'heldout_B': auc(best_combo, B),         # honest
                         'ae_pl_B': auc(('perceptual',), B)})     # baseline, no selection
    return pd.DataFrame(rows)


A3V = a3_split_half()

if not A3V.empty:
    ho, orc, base = A3V.heldout_B.mean()*100, A3V.oracle_A.mean()*100, A3V.ae_pl_B.mean()*100
    wins = (A3V.heldout_B > A3V.ae_pl_B).mean()*100
    print('\nA3 — selection-aware validation ({} splits x {} seeds)\n'.format(
        len(A3V)//len(SEEDS), len(SEEDS)))
    print(f'  oracle (best combo, scored on the half it was picked on) {orc:6.2f}   <- optimistic')
    print(f'  held-out (that same combo on the UNSEEN half)            {ho:6.2f}   <- honest')
    print(f'  AE-PL alone on the same half (no selection)              {base:6.2f}   <- baseline')
    print(f'\n  selection bias                : {orc-ho:+.2f} AUC pts')
    print(f'  honest gain over AE-PL        : {ho-base:+.2f} AUC pts')
    print(f'  held-out beats AE-PL in       : {wins:.0f}% of splits')
    print(f'\n  most frequently selected combo: '
          f'{A3V.selected.value_counts().index[0]} '
          f'({A3V.selected.value_counts().iloc[0]/len(A3V)*100:.0f}% of splits)')
    verdict = ('SURVIVES selection -- report it' if ho - base > 0.3 and wins > 60 else
               'does NOT survive selection -- report the pre-registered negative result')
    print(f'\n  VERDICT: {verdict}')


# %% [markdown]
# ---
# # M0 (BLOCKER) — is the variance head learning uncertainty, or a static anatomical mask?
# ## **Cell 5.4** — degeneracy controls, no training
# A4 claims the head learns *per-image* uncertainty. There is a cheaper explanation that
# has to be ruled out before that claim can be made.
#
# The head reads `de_features[0]`, which sits downstream of a **16-dimensional latent** on
# globally-registered chest radiographs, and its training target is a residual that is
# **frozen** — constant with respect to the head's parameters. For a head that cannot see
# the residual and receives an almost-templated feature map, the optimal solution may
# simply be *the pixelwise mean training error*: a fixed anatomical mask that up-weights
# the lung fields and down-weights ribs, diaphragm and mediastinum. Since the score is
# `mean(exp(-log_var) * r^2)`, a constant `log_var` makes it exactly a fixed weighted mean
# of the squared residual — reproducible with **zero parameters and zero training**.
#
# A second degeneracy matters too: `exp(-log_var)` need not vary spatially at all. If
# `log_var` is roughly `static_map + per_image_scalar`, the ranking could come entirely
# from that scalar — an image-level contrast normalisation dressed as spatial uncertainty.
#
# Four scores from the SAME stored head, plus one that uses no head:
#
# | variant | `log_var` replaced by | tests |
# |---|---|---|
# | `full` | — (as trained) | the reported result |
# | `static` | its pixelwise mean over the training set | is it image-independent? |
# | `scalar` | its own spatial mean, per image | is it map-independent? |
# | `whiten` | **no head at all**: `r² / R̄` | can zero parameters do it? |
#
# `R̄` is the frozen AE's pixelwise mean squared residual over the training set.
#
# **Pre-registered reading.** If `static` ≈ `full`, the per-image component contributes
# nothing and the claim must be restated as a learned fixed error-normalisation mask. If
# `whiten` ≈ `full`, C2 is not a method at all — the finding becomes "AE-U's advantage is
# largely residual whitening by the training-set error profile", which is deflationary for
# C2 but *strengthens* C1. Either way we report it.

# %% [CELL 5.4]  M0 — degeneracy controls for the post-hoc variance head

M0_WIDTH = 2          # the smallest head that works; see the A4 capacity sweep
M0_BASE  = 'ae'


@torch.no_grad()
def _lv_and_resid(ae, head, X, batch=256):
    """Per-image log-variance maps and squared residuals for a tensor of images."""
    lvs, res = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        out = ae(xb)
        lvs.append(head(out['de_features'][0]).cpu())
        res.append(((xb - out['x_hat']) ** 2).cpu())
    return torch.cat(lvs), torch.cat(res)


def m0_degeneracy(seeds=None, width=M0_WIDTH, base_method=M0_BASE):
    seeds = seeds or SEEDS
    rows = []
    for s in seeds:
        rid = run_id(f'{base_method}-posthoc-u', s, w=_wtag(width))
        if not run_exists(rid) and not (USE_WANDB and fetch_run(rid)):
            print(f'  seed {s}: {rid} not available — run Cell 5.3 first'); continue
        ae = build_net(METHODS[base_method]['net'])
        load_run(run_id(base_method, s), models={'net': ae})
        head = (PostHocVarHead(**width) if isinstance(width, dict)
                else PostHocVarHead(width=width)).to(device)
        load_run(rid, models={'head': head})       # load_run calls .eval() on both

        lv_tr, res_tr = _lv_and_resid(ae, head, X_TRAIN)
        lv_te, res_te = _lv_and_resid(ae, head, X_TEST)

        LV_bar = lv_tr.mean(dim=0, keepdim=True)          # static map, from TRAIN only
        R_bar  = res_tr.mean(dim=0, keepdim=True)         # mean squared residual, TRAIN only

        def auc_of(w_map):
            return evaluate_scores(torch.mean(w_map * res_te, dim=[1, 2, 3]).numpy(), Y_TEST)

        full   = auc_of(torch.exp(-lv_te))
        static = auc_of(torch.exp(-LV_bar).expand_as(lv_te))
        scalar = auc_of(torch.exp(-lv_te.mean(dim=[1, 2, 3], keepdim=True)).expand_as(lv_te))
        whiten = auc_of(1.0 / (R_bar + EPS).expand_as(res_te))       # NO head, NO training

        # variance decomposition: lv_ij = LV_bar_j + d_ij, so
        #   Var_total = Var_j(LV_bar_j) + E_j[Var_i(d_ij)]
        v_static = LV_bar.flatten().var(unbiased=False).item()
        v_resid  = (lv_te - LV_bar).var(dim=0, unbiased=False).mean().item()
        frac_static = v_static / max(v_static + v_resid, 1e-12)

        # Is the per-image scalar NEW information, or just a restatement of the image's
        # own residual magnitude? If exp(-lv_bar) were a monotone function of mean(r^2),
        # the score would collapse to a monotone transform of the plain L2 score and the
        # AUC would fall back to the plain AE's -- so a HIGH correlation here would mean
        # the scalar is redundant, and a low one that it carries independent signal.
        lv_bar_img = lv_te.mean(dim=[1, 2, 3]).numpy()
        r_bar_img  = res_te.mean(dim=[1, 2, 3]).numpy()
        # Correlate log_var ITSELF with the residual — not its negation. An earlier
        # version correlated -log_var and then reported the result as if it were
        # log_var's, which flipped the sign of the mechanism story: the discount
        # interpretation needs log_var to RISE with the residual, i.e. a POSITIVE
        # correlation. Score the precision weight (-log_var) separately and label it
        # as such, since that is the quantity that enters the score.
        rho = float(np.corrcoef(np.argsort(np.argsort(lv_bar_img)),
                                np.argsort(np.argsort(r_bar_img)))[0, 1])
        prec_only = evaluate_scores(-lv_bar_img, Y_TEST)      # precision weight as a score
        # Class-conditional means. Without these the pooled correlation and the AUCs can
        # appear to disagree (a variable can correlate one way overall and separate the
        # classes the other way), and there is no way to tell which is happening.
        lv_norm = float(lv_bar_img[Y_TEST == 0].mean())
        lv_abn  = float(lv_bar_img[Y_TEST == 1].mean())

        rows.append({'seed': s, 'full': full['AUC'], 'static': static['AUC'],
                     'scalar': scalar['AUC'], 'whiten': whiten['AUC'],
                     'frac_static': frac_static, 'rho_logvar_resid': rho,
                     'precision_alone': prec_only['AUC'],
                     'logvar_normal': lv_norm, 'logvar_abnormal': lv_abn})
        del ae, head, lv_tr, res_tr, lv_te, res_te
    return pd.DataFrame(rows)


M0 = m0_degeneracy()

if not M0.empty:
    ref = completed_runs()
    ae_auc = ref[ref.method == M0_BASE].AUC.mean() * 100 if not ref.empty else float('nan')
    print(f'\nM0 — what is the variance head actually doing?  '
          f'(base={M0_BASE}, w={M0_WIDTH}, {len(M0)} seeds)\n')
    for k, label in [('full',   'full head, per-image log_var        '),
                     ('static', 'STATIC map (train-set pixelwise mean)'),
                     ('scalar', 'per-image SCALAR only (flat map)     '),
                     ('whiten', 'NO HEAD: r^2 / mean-train-residual   ')]:
        m, sd = M0[k].mean() * 100, M0[k].std(ddof=0) * 100
        print(f'  {label}  {m:6.2f}+-{sd:.2f}')
    print(f'  plain {M0_BASE} with its own score       {ae_auc:6.2f}')
    print(f'\n  variance of log_var explained by the static map: '
          f'{M0.frac_static.mean()*100:.1f}%')
    print(f'  precision weight (-log_var) used ALONE as a score : '
          f'{M0.precision_alone.mean()*100:6.2f}+-{M0.precision_alone.std(ddof=0)*100:.2f}')
    # Per-seed spread on BOTH mechanism quantities. These two numbers carry the primary
    # contribution ("residual relative to expected difficulty"), and were previously
    # printed as bare 3-seed means with no dispersion and no per-seed values -- so a
    # reader could not tell whether a 0.0997 gap on a variable centred near -5.1 was a
    # real effect or noise. Print sd and the individual seeds so it can be judged.
    _rho_v = M0.rho_logvar_resid.values
    print(f'  rank corr(log_var, that image\'s mean residual)     : '
          f'{_rho_v.mean():+.3f}+-{_rho_v.std(ddof=0):.3f}   '
          f'per-seed {", ".join(f"{v:+.3f}" for v in _rho_v)}')
    print(f'    -> {"REDUNDANT: log_var mostly restates residual magnitude" if abs(M0.rho_logvar_resid.mean()) > 0.8 else "log_var carries information the residual does not"}')
    _dlv_v = (M0.logvar_abnormal - M0.logvar_normal).values
    print(f'  mean log_var  normal {M0.logvar_normal.mean():+.4f} | '
          f'abnormal {M0.logvar_abnormal.mean():+.4f}  '
          f'(abnormal - normal = {_dlv_v.mean():+.4f}+-{_dlv_v.std(ddof=0):.4f})')
    print(f'    class contrast per-seed: {", ".join(f"{v:+.4f}" for v in _dlv_v)}   '
          f'sign consistent: {"YES" if (np.sign(_dlv_v) == np.sign(_dlv_v[0])).all() else "NO"}')
    # The weight ratio is what actually enters the score, so state it with its spread too.
    _wr = (np.exp(-M0.logvar_abnormal.values) / np.exp(-M0.logvar_normal.values) - 1) * 100
    print(f'    -> abnormal images weighted {_wr.mean():+.1f}%+-{_wr.std(ddof=0):.1f}% '
          f'relative to normal (precision weight exp(-log_var))')
    _rho = M0.rho_logvar_resid.mean()
    _dlv = M0.logvar_abnormal.mean() - M0.logvar_normal.mean()
    print(f'    -> rho is {"POSITIVE" if _rho > 0 else "NEGATIVE"}: globally-harder images get a '
          f'{"LARGER" if _rho > 0 else "SMALLER"} predicted log_var,')
    print(f'       so their residual is {"DISCOUNTED" if _rho > 0 else "AMPLIFIED"}. '
          f'The discount mechanism requires a POSITIVE rho.')
    print(f'    -> class-conditional: abnormal images get a '
          f'{"LARGER" if _dlv > 0 else "SMALLER"} log_var than normal ones '
          f'({_dlv:+.4f}),')
    print(f'       i.e. the weighting {"works AGAINST" if _dlv > 0 else "works FOR"} '
          f'detection on its own. The pooled rho and this contrast can disagree;')
    print('       when they do, THIS is the one that bears on anomaly separation.')
    d_static = (M0.full - M0.static).mean() * 100
    d_whiten = (M0.full - M0.whiten).mean() * 100
    print(f'\n  full - static : {d_static:+.2f} pts  '
          f'{"-> per-image component contributes little" if abs(d_static) < 1.5 else "-> per-image component matters"}')
    print(f'  full - whiten : {d_whiten:+.2f} pts  '
          f'{"-> ZERO-PARAMETER whitening reproduces it; C2 is not a method" if abs(d_whiten) < 1.5 else "-> the head does more than whitening"}')


# %% [markdown]
# ---
# # M2 — does post-hoc uncertainty transfer to other frozen backbones?
# ## **Cell 5.5** — the experiment that decides the paper's shape
# C2 currently rests on ONE backbone. "You can retrofit uncertainty onto *a* plain
# autoencoder" is an observation; "onto any frozen reconstruction model" is a recipe.
#
# AE-PL and AE-SSIM share the AE architecture, so `de_features[0]` is 16 channels at
# 32×32 in all three cases and the same head fits unchanged. Both backbones are already
# trained and stored, so this only fits heads.
#
# **Pre-register the reading before running.** If the head lifts AE-PL above 88.74 — the
# best number this project has measured — then C2 is a transferable method, leads the
# paper, and takes the conditional title. If it lifts the weaker backbones but not AE-PL,
# uncertainty and perceptual scoring are redundant rather than complementary, which is
# itself informative given that A3 found them complementary *across* models.

# %% [CELL 5.5]  M2 — generality of the post-hoc head

M2_BASES = ['ae', 'ae-ssim', 'ae-pl']
M2_WIDTH = 2

for _b in M2_BASES:
    for _s in SEEDS:
        a4_posthoc_uncertainty(seed=_s, width=M2_WIDTH, base_method=_b)

_m2 = completed_runs()
if not _m2.empty:
    print('\nM2 — post-hoc variance head on three frozen backbones\n')
    print(f"  {'backbone':<10}{'frozen model':>16}{'+ post-hoc head':>18}{'delta':>9}")
    for b in M2_BASES:
        base_rows = _m2[_m2.method == b]
        # MUST filter on width too: every head variant shares the method name
        # f'{b}-posthoc-u', so matching on the name alone silently averages w=1, w=8,
        # w=16 and the M4 variants into this row.
        want = {run_id(f'{b}-posthoc-u', s_, w=_wtag(M2_WIDTH)) for s_ in SEEDS}
        ph_rows = _m2[_m2.run_id.isin(want)]
        if base_rows.empty or ph_rows.empty:
            continue
        bm = base_rows.AUC.mean() * 100
        pm, ps = ph_rows.AUC.mean() * 100, ph_rows.AUC.std(ddof=0) * 100
        print(f'  {b:<10}{bm:>16.2f}{pm:>13.2f}+-{ps:<4.2f}{pm-bm:>+9.2f}')
    print(f"\n  references: AE-U (jointly trained) 86.86 | "
          f"best measured configuration (perceptual+uncertainty ensemble) 88.74")


# %% [markdown]
# ---
# # M4 — is it capacity, or is it nonlinearity?
# ## **Cell 5.6** — breaking the confound in the A4 width sweep
# The A4 sweep reported a "sharp capacity threshold" between 256 and 14,528 parameters.
# It is not one. `BasicBlock(..., last_layer=True)` executes `layers = layers[:-2]`,
# removing the BatchNorm and the ReLU, so the 256-parameter head is a **single
# `ConvTranspose2d` with no nonlinearity and no normalisation**, while every wider head is
# `Conv-BN-ReLU ×2` followed by that block. The w=1 → w=2 comparison therefore varies
# depth, nonlinearity and normalisation together — and the flat plateau across w=2/4/8
# (11.6× parameters, 0.24 spread) shows width alone changes nothing.
#
# Two heads separate the factors:
#
# | variant | parameters | nonlinear? | tests |
# |---|---|---|---|
# | `tiny-nonlinear` | ~AE-U's budget | yes | does a *tiny* nonlinear head work? |
# | `wide-linear` | large | no | do *many* linear parameters work? |
#
# **Pre-registered reading.** If `tiny-nonlinear` reaches ~80 and `wide-linear` stays
# ~67, capacity is irrelevant and the real claim is that **the variance is not linearly
# decodable from a frozen decoder's features** — joint training's function is to make it
# linearly readable. That is a sharper, more mechanistic finding than the capacity story
# it replaces.

# %% [CELL 5.6]  M4 — disentangling capacity from nonlinearity

M4_VARIANTS = [
    ('linear-256   (as A4 w=1)',   1),
    ('tiny-nonlinear',             {'chans': 4}),      # ~AE-U's parameter budget, nonlinear
    ('wide-linear',                {'chans': 256, 'linear': True}),
    ('nonlinear-32 (as A4 w=2)',   2),
]

for _label, _w in M4_VARIANTS:
    for _s in SEEDS:
        a4_posthoc_uncertainty(seed=_s, width=_w, base_method='ae')

_m4 = completed_runs()
if not _m4.empty:
    print('\nM4 — capacity vs nonlinearity in the post-hoc head\n')
    print(f"  {'variant':<28}{'params':>10}{'nonlin':>8}{'AUC':>16}")
    for label, w in M4_VARIANTS:
        h = (PostHocVarHead(**w) if isinstance(w, dict) else PostHocVarHead(width=w))
        n = sum(p.numel() for p in h.parameters())
        nl = any(isinstance(m, nn.ReLU) for m in h.modules())
        rid = run_id('ae-posthoc-u', SEEDS[0], w=_wtag(w))
        sub = _m4[_m4.run_id.isin([run_id('ae-posthoc-u', s, w=_wtag(w)) for s in SEEDS])]
        if sub.empty:
            print(f'  {label:<28}{n:>10,}{"yes" if nl else "no":>8}{"(not run)":>16}'); continue
        print(f'  {label:<28}{n:>10,}{"yes" if nl else "no":>8}'
              f'{sub.AUC.mean()*100:>11.2f}+-{sub.AUC.std(ddof=0)*100:<4.2f}')
    print('\n  If tiny-nonlinear works and wide-linear does not, the threshold is')
    print('  NONLINEARITY, not capacity, and the A4 headline must be restated.')


# %% [markdown]
# ---
# # M3 — inferential statistics on the headline gaps
# ## **Cell 5.7** — DeLong's paired test
# Every comparison in this project is currently an arithmetic subtraction, and the
# protocol note says differences below ~1.5 AUC points are not interpretable. That bar
# came from an **unpaired** standard error (Hanley–McNeil, 0.76 points at AUC 0.887 with
# 1000/1000 images), and it is the wrong instrument here.
#
# Our comparisons are between two scores evaluated on the **same 2000 images**. Those
# scores are strongly correlated — they come from the same data and often the same
# weights — and a paired test exploits that correlation, giving a substantially tighter
# interval on the *difference* than the unpaired bound suggests. DeLong's method is the
# standard estimator: it forms the structural components of each AUC, takes their
# covariance, and tests the difference directly.
#
# This is what lets us say honestly whether +1.16 (the ensemble over AE-PL) and +1.19
# (the post-hoc head over retrained AE-SSIM) are real. Under the unpaired bar both are
# "not interpretable"; under the correct test they may or may not be. Either answer is
# reportable — what is not defensible is quoting the gaps with no test at all.

# %% [CELL 5.7]  M3 — DeLong paired comparison of AUCs

def _midrank(x):
    """Ranks with ties averaged — the tie handling is what makes DeLong exact."""
    J = np.argsort(x, kind='mergesort')
    z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and z[j] == z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_variance(scores, labels):
    """Fast DeLong (Sun & Xu, 2014). `scores` is (k, n) for k predictors on the SAME n
    samples. Returns (aucs, covariance matrix) — the covariance is what a paired test
    needs and what an unpaired standard error throws away."""
    labels = np.asarray(labels)
    order = np.argsort(-labels, kind='mergesort')          # positives first
    s = np.asarray(scores, dtype=float)[:, order]
    m = int((labels == 1).sum())                            # positives
    n = len(labels) - m                                     # negatives
    k = s.shape[0]

    tx = np.array([_midrank(s[r, :m]) for r in range(k)])
    ty = np.array([_midrank(s[r, m:]) for r in range(k)])
    tz = np.array([_midrank(s[r, :])  for r in range(k)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n                              # structural components
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(score_a, score_b, labels, alpha=0.05):
    """Paired comparison of two AUCs on identical samples. Returns a dict with the
    difference, its standard error, CI and two-sided p-value."""
    from scipy import stats
    aucs, cov = delong_variance(np.vstack([score_a, score_b]), labels)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {'auc_a': aucs[0], 'auc_b': aucs[1], 'diff': diff, 'se': se,
            'ci_lo': diff - half, 'ci_hi': diff + half,
            'p': float(2 * stats.norm.sf(abs(z)))}


def ensemble_scores(seed, members=(('ae-pl', {}), ('aeu', {}))):
    """Rank-average of the given members' stored score vectors, rebuilt on demand.
    The winning perceptual+uncertainty combination is EXPLORATORY (Cell 5.2) and is
    therefore not persisted as a run, unlike the two pre-registered ensembles. Comparing
    against `ens-all4` instead would silently test the combination that LOST."""
    vecs = []
    for m, prm in members:
        v = _scores_for(run_id(m, seed, **prm))
        if v is None:
            return None
        vecs.append(_ranks(v))
    return np.mean(vecs, axis=0)


def _scores_for(rid, quiet=False):
    """Per-image score vector for a stored run, fetched from wandb if not local.

    Returns None if unavailable -- and SAYS SO. This used to fail silently, which is how
    a single missing run turned a headline into a 2-seed mean printed with the same
    formatting as a 3-seed one: m3_delong turns None into `continue`, so the row simply
    lost a seed with no trace in the output."""
    if not run_exists(rid) and not (USE_WANDB and fetch_run(rid)):
        if not quiet:
            print(f'  !! MISSING RUN {rid} — this seed is being DROPPED from the statistic')
        return None
    try:
        _, arrays = load_run(rid)
    except (RuntimeError, FileNotFoundError) as e:
        if not quiet:
            print(f'  !! UNUSABLE RUN {rid} ({type(e).__name__}) — seed DROPPED: {e}')
        return None
    return arrays.get('scores')


# Headline comparisons, each stated as (label, run-id builder A, run-id builder B).
# Every one of these is currently reported as an arithmetic difference somewhere.
M3_COMPARISONS = [
    ('ensemble(perc+unc) vs AE-PL',
     lambda s: ensemble_scores(s), lambda s: run_id('ae-pl', s)),
    ('ensemble(perc+unc) vs AE-U',
     lambda s: ensemble_scores(s), lambda s: run_id('aeu', s)),
    ('post-hoc head (w=2) vs AE-SSIM',
     lambda s: run_id('ae-posthoc-u', s, w='2'), lambda s: run_id('ae-ssim', s)),
    ('post-hoc head (w=2) vs plain AE',
     lambda s: run_id('ae-posthoc-u', s, w='2'), lambda s: run_id('ae', s)),
    ('AE-U vs post-hoc head (w=2)',
     lambda s: run_id('aeu', s), lambda s: run_id('ae-posthoc-u', s, w='2')),
    ('AE rescored with SSIM vs AE',
     lambda s: run_id('a1-l2-ssim', s), lambda s: run_id('ae', s)),
    # C3's headline (86.99 vs 86.86, +0.13) sat UNTESTED behind a contribution heading
    # that called it "an ImageNet-free route to AE-U's accuracy". One line, and the score
    # vectors were already stored. An untested +0.13 with sds 0.57/0.44 supports
    # "indistinguishable from", not "reaches" -- this decides which verb is allowed.
    ('frozen-AE-SSIM+head vs AE-U',
     lambda s: run_id('ae-ssim-posthoc-u', s, w='2'), lambda s: run_id('aeu', s)),
]


def m3_delong(comparisons=None, seeds=None):
    comparisons = comparisons or M3_COMPARISONS
    seeds = seeds or SEEDS
    rows = []
    for label, ra, rb in comparisons:
        for s in seeds:
            a, b = ra(s), rb(s)
            sa = _scores_for(a) if isinstance(a, str) else a
            sb = _scores_for(b) if isinstance(b, str) else b
            if sa is None or sb is None:
                continue
            r = delong_test(sa, sb, Y_TEST)
            # keep the score vectors (or their ids) so m3_report can compute the exact
            # multi-seed contrast rather than pooling as if the seeds were independent
            rows.append({'comparison': label, 'seed': s, 'run_a': a, 'run_b': b, **r})
    return pd.DataFrame(rows)


M3 = m3_delong()

def _pooled_delong_se(g):
    """Exact standard error for the MEAN of k paired AUC differences measured on ONE test
    set. Stacks all 2k score vectors, takes a single DeLong covariance, and applies the
    contrast c = (1..1, -1..-1)/k. Falls back to the naive pooling only if the underlying
    score vectors are unavailable, and says so."""
    a_ids = g['run_a'].tolist() if 'run_a' in g else None
    b_ids = g['run_b'].tolist() if 'run_b' in g else None
    if not a_ids or not b_ids:
        print('    !! score vectors not recorded for this comparison; falling back to the'
              ' independence-assuming pool, which UNDERSTATES the SE by ~sqrt(k)')
        return float(np.sqrt((g['se'].values ** 2).mean() / len(g)))
    A = [_scores_for(r, quiet=True) if isinstance(r, str) else r for r in a_ids]
    B = [_scores_for(r, quiet=True) if isinstance(r, str) else r for r in b_ids]
    keep = [(x, y) for x, y in zip(A, B) if x is not None and y is not None]
    k = len(keep)
    if k == 0:
        return float('nan')
    _, cov = delong_variance(np.vstack([x for x, _ in keep] + [y for _, y in keep]), Y_TEST)
    c = np.r_[np.ones(k), -np.ones(k)] / k
    return float(np.sqrt(max(c @ cov @ c, 0.0)))


def m3_report(M3, alpha=0.05):
    """Two DIFFERENT tests, reported side by side, because they answer different questions.

    The previous version averaged the three per-seed DeLong endpoints and called the result
    a 95% CI. It is not one: averaging endpoints yields the precision of a TYPICAL SINGLE
    SEED, which does not shrink as seeds are added, so it is not an interval for the
    quantity actually being reported (the seed mean). It also took p = max across seeds,
    an intersection-union test of a different null from the one the interval addressed --
    and the two were then quoted in whichever direction supported the conclusion.

    What is reported instead:
      DeLong (pooled) -- mean difference, SE = sqrt(mean(se_i^2)/k). Treats the 2000 TEST
        IMAGES as the sampling unit. Valid CONDITIONAL ON THESE TRAINED MODELS; it carries
        no seed-variance component, so it cannot speak to retraining.
      seed-level t   -- paired t on the k per-seed differences, k-1 df. Treats the SEED as
        the sampling unit, so it does address retraining. With k=3 the 95% multiplier is
        4.30 against DeLong's 1.96, i.e. >2x wider before any variance is counted. This is
        the test that bears on the project's own ~1.5-point rule (07 T5), which is a
        SEED-COUNT argument and is NOT retired by a paired image-level test.
      Holm           -- family-wise correction across the comparisons reported here.
    A claim is only safe if BOTH tests agree. Where they disagree, say which is which."""
    from scipy import stats
    rows = []
    for label, g in M3.groupby('comparison', sort=False):
        d_i = g['diff'].values
        k   = len(d_i)
        d   = float(d_i.mean())
        # WRONG, and it was here: se_pool = sqrt(mean(se_i^2)/k). That divisor assumes the
        # k per-seed differences are independent draws. They are not -- every seed is
        # evaluated on the SAME test images, so the image-sampling variance that DeLong
        # measures is very nearly common to all k. Dividing by k understated the standard
        # error by a factor of ~sqrt(k) on all three datasets and turned two null results
        # into significant ones.
        #
        # The exact statistic uses one covariance over all 2k predictors and the contrast
        # c = (1..1, -1..-1)/k, which keeps every cross-seed covariance term. Computed in
        # `m3_pooled_se` below from the stored score vectors.
        se_pool = _pooled_delong_se(g) if 'se' in g else float('nan')
        z   = d / se_pool if se_pool > 0 else 0.0
        p_dl = float(2 * stats.norm.sf(abs(z)))
        half = stats.norm.ppf(1 - alpha / 2) * se_pool
        # Seed-level: needs >=2 seeds and non-zero spread to be defined at all.
        if k >= 2 and np.std(d_i, ddof=1) > 0:
            t_stat, p_seed = stats.ttest_1samp(d_i, 0.0)
            t_half = stats.t.ppf(1 - alpha / 2, k - 1) * np.std(d_i, ddof=1) / np.sqrt(k)
        else:
            p_seed, t_half = float('nan'), float('nan')
        rows.append({'comparison': label, 'n_seeds': k, 'diff': d * 100,
                     'dl_lo': (d - half) * 100, 'dl_hi': (d + half) * 100, 'p_delong': p_dl,
                     'seed_lo': (d - t_half) * 100, 'seed_hi': (d + t_half) * 100,
                     'p_seed': p_seed})
    R = pd.DataFrame(rows)
    # Holm-Bonferroni over the DeLong p-values of THIS family. State the family size --
    # an uncorrected p from a family of 6+ is not a defensible number on its own.
    order = R.p_delong.values.argsort()
    n_c   = len(R)
    holm  = np.empty(n_c)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (n_c - i) * R.p_delong.values[idx])
        holm[idx] = min(running, 1.0)
    R['p_holm'] = holm
    return R


if not M3.empty:
    M3R = m3_report(M3)
    print('\nM3 — paired comparisons on the same test images\n')
    print(f"  {'comparison':<32}{'n':>3}{'diff':>8}"
          f"{'DeLong 95%':>19}{'p_holm':>9}{'seed-level 95%':>21}{'p_seed':>9}")
    for _, r in M3R.iterrows():
        mark = '***' if r.p_holm < 0.001 else '**' if r.p_holm < 0.01 else \
               '*' if r.p_holm < 0.05 else 'n.s.'
        seed_ci = (f'[{r.seed_lo:>+7.2f},{r.seed_hi:>+7.2f}]'
                   if np.isfinite(r.seed_lo) else f'{"n/a":>18}')
        print(f'  {r.comparison:<32}{int(r.n_seeds):>3}{r["diff"]:>+8.2f}'
              f'[{r.dl_lo:>+7.2f},{r.dl_hi:>+7.2f}]{r.p_holm:>9.2g}'
              f'{seed_ci:>21}{r.p_seed:>9.2g}  {mark}')
    print(f'\n  DeLong: 2000 test images are the sampling unit. Pooled SE = '
          f'sqrt(mean(se_i^2)/k).')
    print( '          Valid CONDITIONAL ON THESE TRAINED MODELS — no seed variance.')
    print(f'  seed  : the SEED is the sampling unit, k-1 df. This is the one that speaks')
    print( '          to retraining, and the one the ~1.5-point rule (07 T5) is about.')
    print(f'  p_holm: Holm-corrected across the {len(M3R)} comparisons in THIS family.')
    _both = M3R[(M3R.p_holm < 0.05) & (M3R.p_seed < 0.05)]
    _dlonly = M3R[(M3R.p_holm < 0.05) & ~(M3R.p_seed < 0.05)]
    print(f'\n  SURVIVES BOTH TESTS ({len(_both)}/{len(M3R)}): '
          f'{", ".join(_both.comparison) if len(_both) else "none"}')
    if len(_dlonly):
        print(f'  DeLong ONLY — do NOT call these robust to retraining:')
        for c in _dlonly.comparison:
            print(f'      {c}')
    M3R.to_csv(f'{OUTPUT_DIR}/m3_tests_{RUN_VERSION}.csv', index=False)


# %% [markdown]
# ---
# # Item 4 — is the ImageNet-free head a drop-in substitute for AE-U?
# ## **Cell 5.8** — the C3 substitution control
# C3 claims frozen-AE-SSIM + post-hoc head (86.99) is equivalent to jointly-trained AE-U
# (86.86). An equivalence claim should survive **substitution**: if the two are truly
# interchangeable, swapping one for the other inside a fixed ensemble should move nothing.
#
# **This does NOT test C1**, and it is worth being explicit about that because an earlier
# draft claimed it did. Both scores here are applied to *compatible* backbones — the
# perceptual score reads its own perceptually-trained model, the uncertainty score reads a
# pixel-trained one. No score is ever applied across the space boundary, so no C1
# prediction is at risk. It is a C3 control, nothing more.
#
# **Pre-registered:** |ens(perceptual, ssim+head) − ens(perceptual, aeu)| < 1.5 AUC points
# → the head is a drop-in substitute for AE-U *as a component*, which upgrades C3 from
# "matches AE-U standalone" to "matches AE-U in use". > 3.0 → C3 is limited to standalone
# use and must say so.
#
# **The pre-registered 4-member ensemble result is NOT overwritten.** Enlarging
# `A3_MEMBERS` from 4 to 5 takes the subset scan from 15 to 31 and would replace a
# pre-registered statistic (150 splits, selection bias −0.20) with a post-hoc one. The
# extended scan is computed separately and reported beside the original, labelled
# exploratory.

# %% [CELL 5.8]  Item 4 — C3 substitution control

def c3_substitution_control(seeds=None):
    seeds = seeds or SEEDS
    rows = []
    for s in seeds:
        a = ensemble_scores(s, members=(('ae-pl', {}), ('aeu', {})))
        b = ensemble_scores(s, members=(('ae-pl', {}),
                                        ('ae-ssim-posthoc-u', {'w': '2'})))
        if a is None or b is None:
            print(f'  seed {s}: a member is missing — run Cell 5.5 first'); continue
        rows.append({'seed': s,
                     'perc+aeu': evaluate_scores(a, Y_TEST)['AUC'],
                     'perc+ssimhead': evaluate_scores(b, Y_TEST)['AUC'],
                     'delong_p': delong_test(b, a, Y_TEST)['p']})
    return pd.DataFrame(rows)


C3SUB = c3_substitution_control()

if not C3SUB.empty:
    m_a, m_b = C3SUB['perc+aeu'].mean() * 100, C3SUB['perc+ssimhead'].mean() * 100
    d = m_b - m_a
    print('\nItem 4 — is the post-hoc head a drop-in substitute for AE-U in the ensemble?\n')
    print(f'  perceptual + AE-U (jointly trained)       {m_a:6.2f}')
    print(f'  perceptual + frozen AE-SSIM + head        {m_b:6.2f}')
    print(f'  difference                                {d:+6.2f}   '
          f'(paired DeLong p = {C3SUB.delong_p.max():.3g}, least significant seed)')
    verdict = ('SUBSTITUTABLE — C3 upgrades to "matches AE-U as a component"' if abs(d) < 1.5
               else 'NOT substitutable — C3 is limited to standalone use'
               if abs(d) > 3.0 else 'INCONCLUSIVE (between the 1.5 and 3.0 bands)')
    print(f'\n  VERDICT: {verdict}')

    # exploratory 31-subset scan, reported BESIDE the pre-registered 15-subset result
    print('\n  --- exploratory: the 5-member subset scan (NOT the pre-registered result) ---')
    A3X = a3_ensemble(members=A3_MEMBERS_EXT)
    if not A3X.empty:
        top = (A3X.groupby('combo').AUC.mean().sort_values(ascending=False) * 100).head(5)
        print(top.to_string())
        print(f'\n  pre-registered 4-member result stands unchanged; the above is'
              f' {len(A3X.combo.unique())} subsets and is exploratory.')


# %% [markdown]
# ### Cell 5.8b — selection-aware validation of the EXTENDED member set
# The 5-member scan produced the best number in the project — but it is the top of **31
# subsets chosen on the test set**, which is precisely the position the 4-member result
# was in before Cell 5.2b validated it. It is not reportable until it survives the same
# protocol.
#
# Same machinery, larger space: pick the best combination on half the test set, score it
# on the unseen half, 50 splits x 3 seeds. Two things decide it — whether the held-out
# gain over the best single score survives, and whether the SAME combination keeps
# winning. A shuffling winner over 31 options is noise wearing a large number.
#
# The pre-registered 4-member result is untouched and reported beside this one.

# %% [CELL 5.8b]  Extended (5-member) selection-aware validation

A3V_EXT = a3_split_half(members=A3_MEMBERS_EXT)

if not A3V_EXT.empty:
    ho  = A3V_EXT.heldout_B.mean() * 100
    orc = A3V_EXT.oracle_A.mean() * 100
    base = A3V_EXT.ae_pl_B.mean() * 100
    wins = (A3V_EXT.heldout_B > A3V_EXT.ae_pl_B).mean() * 100
    top  = A3V_EXT.selected.value_counts()
    print(f'\nA3 EXTENDED (5 members, 31 subsets) — selection-aware, '
          f'{len(A3V_EXT)//len(SEEDS)} splits x {len(SEEDS)} seeds\n')
    print(f'  oracle   (best combo on the half it was picked on)  {orc:6.2f}')
    print(f'  held-out (that combo on the UNSEEN half)            {ho:6.2f}')
    print(f'  AE-PL alone on the same half                        {base:6.2f}')
    print(f'\n  selection bias          : {orc - ho:+.2f} pts')
    print(f'  honest gain over AE-PL  : {ho - base:+.2f} pts')
    print(f'  beats AE-PL in          : {wins:.0f}% of splits')
    print(f'  most selected combo     : {top.index[0]}  ({top.iloc[0]/len(A3V_EXT)*100:.0f}%)')
    if 'A3V' in globals() and not A3V.empty:
        ho4 = A3V.heldout_B.mean() * 100
        print(f'\n  PRE-REGISTERED 4-member held-out: {ho4:6.2f}   '
              f'(extended - preregistered = {ho - ho4:+.2f})')
    stable = top.iloc[0] / len(A3V_EXT) > 0.6
    print(f'\n  VERDICT: {"survives — a stable winner over 31 options" if stable and ho - base > 0.3 else "does NOT survive — the winner shuffles; report the 4-member result only"}')


# %% [markdown]
# ### Cell 5.8c — does ADDING the head member to the winning pair help?
# The 31-subset scan's best combination (perceptual + uncertainty + ssim-head, 89.46,
# held-out 89.54) sits above the pre-registered pair (perceptual + uncertainty, 88.74,
# held-out 88.83). Whether that +0.71 is real has not been tested.
#
# **Do not test it as "winner of 31 subsets vs winner of 15 subsets".** Both sides of
# that comparison were chosen by looking at data, and comparing two selected quantities
# invites the selection bias straight back in.
#
# Test the single fixed question instead: *take the pre-registered winning pair and ADD
# one member — does the score improve?* That is one pre-specifiable comparison with no
# selection in it, and it happens to be the same two score vectors.
#
# Expect a TIGHT interval: the two ensembles share both of the pair's members, so they
# are strongly paired — the same structural reason the ensemble-vs-AE-PL comparison had
# a half-width of 0.595 while the post-hoc-vs-AE-SSIM one had 1.34.

# %% [CELL 5.8c]  Is the 3-member ensemble better than the pre-registered pair?

PAIR    = (('ae-pl', {}), ('aeu', {}))
PLUS_HEAD = PAIR + (('ae-ssim-posthoc-u', {'w': '2'}),)


def ensemble_add_member_test(seeds=None):
    seeds = seeds or SEEDS
    rows = []
    for s in seeds:
        a = ensemble_scores(s, members=PLUS_HEAD)
        b = ensemble_scores(s, members=PAIR)
        if a is None or b is None:
            print(f'  seed {s}: a member is missing — run Cells 5.5 and 5.2 first'); continue
        r = delong_test(a, b, Y_TEST)
        rows.append({'seed': s, 'pair': r['auc_b'], 'pair_plus_head': r['auc_a'],
                     'diff': r['diff'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
                     'p': r['p']})
    return pd.DataFrame(rows)


ADDTEST = ensemble_add_member_test()

if not ADDTEST.empty:
    b, a = ADDTEST.pair.mean() * 100, ADDTEST.pair_plus_head.mean() * 100
    d, lo, hi = (ADDTEST['diff'].mean() * 100,
                 ADDTEST.ci_lo.mean() * 100, ADDTEST.ci_hi.mean() * 100)
    pmax = ADDTEST['p'].max()
    print('\nDoes adding the frozen-AE-SSIM+head member to the pre-registered pair help?\n')
    print(f'  perceptual + uncertainty                  {b:6.2f}   (pre-registered winner)')
    print(f'  perceptual + uncertainty + ssim-head      {a:6.2f}')
    print(f'  difference   {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  '
          f'p = {pmax:.3g} (least significant seed)')
    sig = pmax < 0.05 and lo > 0
    print(f'\n  VERDICT: {"the third member ADDS significantly — report the 3-member set" if sig else "no significant gain — report the pre-registered PAIR as the headline, and the 3-member set as a consistent but untested extension"}')
    print('  NOTE both sides share two members, so this interval is tight by construction;')
    print('  that is a property of the comparison, not extra evidence.')


# %% [markdown]
# ---
# # Item 5 — what do these models actually reconstruct?
# ## **Cell 5.9** — the qualitative figure
# Two of the project's most striking numbers currently rest on an *unverified* narrative:
#
# * the A1 cell where a perceptually-trained model scored in pixel space gives **44.9** —
#   below chance;
# * the post-hoc head **destroying** that same model, −30.56 to 57.01.
#
# The proposed explanation for both is one mechanism: perceptual loss does not constrain
# absolute pixel intensity, so the reconstruction is free to drift there, and any
# pixel-space reading of it becomes meaningless. That is a hypothesis. This figure is what
# turns it into evidence — or refutes it.
#
# **What to look for.** If the AE-PL reconstruction is a plausible radiograph at the wrong
# brightness or contrast, the mechanism is confirmed and the finding becomes more
# interesting, not less. If it is structurally wrong, the explanation is different and the
# mechanism sentence must be rewritten.
#
# Zero GPU training — it reloads stored weights, so it can run on CPU in parallel with a
# GPU job in another session.

# %% [CELL 5.9]  Item 5 — input / reconstruction / error map

FIG_METHODS = ['ae', 'ae-ssim', 'ae-pl', 'aeu']
FIG_N       = 4        # test images shown


def qualitative_panel(seed=None, methods=None, n=FIG_N, save=True):
    seed    = seed if seed is not None else SEEDS[0]
    methods = methods or FIG_METHODS
    # a fixed, reproducible pick: the n most-abnormal and n most-normal by label order
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(Y_TEST == 1)[0], n // 2, replace=False),
                          rng.choice(np.where(Y_TEST == 0)[0], n - n // 2, replace=False)])
    x = X_TEST[idx]

    avail = []
    for m in methods:
        rid = run_id(m, seed)
        if run_exists(rid) or (USE_WANDB and fetch_run(rid)):
            avail.append(m)
        else:
            print(f'  {m}: not available, skipping')
    if not avail:
        print('  nothing to plot'); return None

    rows = 1 + 2 * len(avail)          # input, then (recon, error) per method
    fig, ax = plt.subplots(rows, len(idx), figsize=(2.0 * len(idx), 2.0 * rows))
    ax = np.atleast_2d(ax)
    for j in range(len(idx)):
        ax[0, j].imshow(x[j, 0], cmap='gray', vmin=-1, vmax=1)
        ax[0, j].set_title(f'{"ABNORMAL" if Y_TEST[idx[j]] else "normal"}', fontsize=8)
    ax[0, 0].set_ylabel('input', fontsize=9)

    for i, m in enumerate(avail):
        net = build_net(METHODS[m]['net'])
        load_run(run_id(m, seed), models={'net': net})
        with torch.no_grad():
            out = net(x.to(device))
            xh  = out['x_hat'].cpu()
        err = ((x - xh) ** 2)[:, 0]
        for j in range(len(idx)):
            ax[1 + 2 * i, j].imshow(xh[j, 0], cmap='gray', vmin=-1, vmax=1)
            # a SHARED error scale across methods, so the panels are comparable; a
            # per-panel scale would hide exactly the intensity drift we are looking for
            ax[2 + 2 * i, j].imshow(err[j], cmap='inferno', vmin=0, vmax=float(err.max()))
        ax[1 + 2 * i, 0].set_ylabel(f'{m}\nrecon', fontsize=8)
        ax[2 + 2 * i, 0].set_ylabel(f'{m}\nsq. error', fontsize=8)
        # the quantity the mechanism is about: does this model preserve absolute intensity?
        print(f'  {m:<9} recon mean {xh.mean():+.3f} (input {x.mean():+.3f})   '
              f'std {xh.std():.3f} (input {x.std():.3f})   '
              f'mean sq err {err.mean():.4f}')
        del net

    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'{DATASET} — reconstruction and pixel error by training objective '
                 f'(seed {seed})', fontsize=11)
    fig.tight_layout()
    if save:
        out_png = f'{OUTPUT_DIR}/qualitative_{DATASET}_s{seed}.png'
        fig.savefig(out_png, dpi=140, bbox_inches='tight')
        print(f'\n  saved -> {out_png}')
        # The ONLY output of this notebook that is not already inside a per-run wandb
        # artifact. Everything else -- weights, per-image scores, metrics -- is uploaded
        # by save_run and versioned; every results table is DERIVED from those scores and
        # recomputes in seconds. A figure is not derivable, so log it explicitly rather
        # than letting it die with the Kaggle session's disk.
        if USE_WANDB and wandb.run is not None:
            try:
                wandb.log({f'qualitative/{DATASET}_s{seed}': wandb.Image(out_png)})
                print(f'  logged to wandb as qualitative/{DATASET}_s{seed}')
            except Exception as e:
                print(f'  wandb image log failed: {type(e).__name__}: {e}')
    plt.show()
    return fig


_ = qualitative_panel()
print('\n  READ THE INTENSITY STATISTICS ABOVE, not just the pictures:')
print('  if ae-pl\'s reconstruction mean/std depart from the input while ae\'s do not,')
print('  that IS the mechanism behind the 44.9 cell and the -30.56, made measurable.')


# %% [markdown]
# ---
# ## **Cell 5.10** — Why one LAG head run never completes
# `ae-ssim-posthoc-u_ds-LAG_w-2_s43` aborts on the non-finite-loss guard, so LAG stores 99
# runs rather than 100 and one comparison is reported at n=2. Re-running does not help:
# the seed fixes the initialisation and the batch order, so the run takes the identical
# trajectory every time and fails at the identical point. It is deterministic, not flaky.
#
# This cell finds out *where* and *why*, without changing anything. The heteroscedastic
# objective is
#
# $$\mathcal{L} = \mathrm{mean}\left[\exp(-\log\sigma^2)\,(x-\hat{x})^2 + \log\sigma^2\right]$$
#
# which is unbounded below in $\log\sigma^2$: as the head drives the log-variance down on
# a pixel whose residual is near zero, $\exp(-\log\sigma^2)$ grows without limit and one
# bad batch overflows. That is a known fragility of the objective, not a bug in the port.
#
# **Read the output before deciding anything.** If the run diverges in the first few epochs
# it is an initialisation problem; if it survives 200 epochs and then dies it is a late
# instability, and the two have different fixes.
#
# ### On "fixing" it
# Any mitigation — gradient clipping, clamping $\log\sigma^2$, a lower learning rate —
# produces a **different training protocol**. Storing one mitigated run beside two
# unmitigated ones would give an n=3 mean over three models that were not trained the same
# way, which is worse than an honest n=2. So the mitigation below is written to run **all
# three seeds** under the same variant and to store them under a **different run id**, where
# they cannot be confused with the originals. Leave `RUN_MITIGATION = False` unless you
# intend to report the variant as a variant.

# %% [CELL 5.10]  Diagnose the failing LAG head run

DIAG_BASE   = 'ae-ssim'      # the frozen backbone the head sits on
DIAG_SEED   = 43             # the seed that fails
DIAG_WIDTH  = 2
RUN_MITIGATION = False       # see the markdown above before setting this True


def diagnose_posthoc(base_method=DIAG_BASE, seed=DIAG_SEED, width=DIAG_WIDTH,
                     max_epochs=None, clip=None, lv_clamp=None, lr=None, store_as=None):
    """Re-run one post-hoc head with per-batch instrumentation.

    Returns a dict describing the first non-finite step, or the completed run. Nothing is
    saved unless `store_as` is given, so the default call cannot alter the record."""
    n_epochs = max_epochs if max_epochs is not None else EPOCHS
    base_rid = run_id(base_method, seed)
    if not (run_exists(base_rid) or (USE_WANDB and fetch_run(base_rid))):
        print(f'  backbone {base_rid} not available'); return None
    ae = build_net(METHODS[base_method]['net'])
    load_run(base_rid, models={'net': ae})
    for prm in ae.parameters():
        prm.requires_grad = False
    ae.eval()

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    head = PostHocVarHead(width=width).to(device)
    opt = Adam(head.parameters(), lr=lr if lr is not None else LR,
               weight_decay=WEIGHT_DECAY)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_TRAIN), batch_size=BATCH_SIZE, shuffle=True,
                        generator=g)

    print(f'  diagnosing {base_method} w={width} seed={seed} on {DATASET}'
          f'{"" if clip is None else f", grad-clip {clip}"}'
          f'{"" if lv_clamp is None else f", log_var clamped to {lv_clamp}"}'
          f'{"" if lr is None else f", lr {lr}"}')
    print(f"  {'ep':>5}{'loss':>12}{'log_var min':>13}{'log_var max':>13}"
          f"{'exp(-lv) max':>14}{'grad norm':>11}")
    epoch_loss = []
    for ep in range(1, n_epochs + 1):
        head.train(); ae.eval()
        tot, n = 0.0, 0
        lv_lo, lv_hi, gmax = float('inf'), -float('inf'), 0.0
        for bi, (xb,) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                out = ae(xb); x_hat, de1 = out['x_hat'], out['de_features'][0]
            log_var = head(de1)
            if lv_clamp is not None:
                log_var = log_var.clamp(min=lv_clamp[0], max=lv_clamp[1])
            recon = (xb - x_hat) ** 2
            loss = (torch.exp(-log_var) * recon + log_var).mean()
            lv_lo = min(lv_lo, float(log_var.min())); lv_hi = max(lv_hi, float(log_var.max()))
            if not torch.isfinite(loss):
                # THE POINT OF THIS CELL: report the state at the moment it breaks.
                print(f'\n  *** NON-FINITE LOSS at epoch {ep}, batch {bi} ***')
                print(f'      loss                     {float(loss)}')
                print(f'      log_var  min / max       {float(log_var.min()):.4f} / '
                      f'{float(log_var.max()):.4f}')
                print(f'      exp(-log_var) max        {float(torch.exp(-log_var).max()):.4g}')
                print(f'      residual max             {float(recon.max()):.4g}')
                print(f'      non-finite in log_var?   {not bool(torch.isfinite(log_var).all())}')
                print(f'      non-finite in recon?     {not bool(torch.isfinite(recon).all())}')
                print(f'\n      DIAGNOSIS: {"the head output itself went non-finite" if not bool(torch.isfinite(log_var).all()) else "log_var drifted low enough that exp(-log_var) overflowed"}')
                print(f'      epochs survived: {ep - 1} of {n_epochs}')
                return {'ok': False, 'epoch': ep, 'batch': bi,
                        'lv_min': float(log_var.min()), 'lv_max': float(log_var.max()),
                        'epoch_loss': epoch_loss}
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(head.parameters(),
                                                clip if clip is not None else float('inf'))
            gmax = max(gmax, float(gn))
            opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        epoch_loss.append(tot / n)
        if ep == 1 or ep % max(1, n_epochs // 12) == 0 or ep == n_epochs:
            print(f'  {ep:>5}{epoch_loss[-1]:>12.5f}{lv_lo:>13.3f}{lv_hi:>13.3f}'
                  f'{np.exp(-lv_lo):>14.4g}{gmax:>11.2f}')
    print(f'\n  COMPLETED {n_epochs} epochs, final loss {epoch_loss[-1]:.5f}')
    return {'ok': True, 'epoch_loss': epoch_loss, 'head': head, 'ae': ae}


_diag = diagnose_posthoc()

if _diag is not None and not _diag['ok']:
    print(f"""
  WHAT TO DO WITH THIS
  --------------------
  The run is deterministic, so re-running the notebook will reproduce this exactly and
  LAG will keep storing 99 runs. There are two defensible options and one indefensible
  one.

  (1) REPORT n=2, which is what the paper currently does. The comparison using this run
      is marked n=2 wherever it appears, and this cell's output is the explanation.
      Costs nothing, hides nothing.

  (2) RE-RUN ALL THREE SEEDS under a documented variant (gradient clipping, or clamping
      log_var) and report that variant as a variant, under its own run id. Set
      RUN_MITIGATION = True below. This gives a clean n=3 for a protocol that differs
      from the rest of the paper, so it belongs in the appendix, not in Table 1.

  (3) Re-run ONLY seed {DIAG_SEED} with a mitigation and store it beside the two
      unmitigated seeds. DO NOT DO THIS. The resulting n=3 mean would average three
      models that were not trained the same way, which is worse than an honest n=2.
""")

if RUN_MITIGATION:
    print('\n  MITIGATED VARIANT — all three seeds, gradient clipping at 1.0,'
          ' stored under a separate run id\n')
    for _s in SEEDS:
        _r = diagnose_posthoc(seed=_s, clip=1.0)
        print(f"    seed {_s}: {'completed' if _r and _r['ok'] else 'still diverged'}")
    print('\n  If all three completed, report this as an appendix variant. It is NOT'
          '\n  interchangeable with the main-protocol runs.')


# %% [markdown]
# ---
# ## **Cell 6.0** — Everything stored so far
# One table of every run under the current `RUN_VERSION`. Because `load_run` refuses a
# record whose `config_fingerprint` differs from the current one, every row here was
# produced under the same image size, latent size, epoch count, learning rate and pixel
# range — the table can be read as a like-for-like comparison without further checking.

# %% [CELL 6.0]  Run inventory

ALL = completed_runs()
if ALL.empty:
    print('nothing stored yet')
else:
    cols = [c for c in ['run_id', 'method', 'seed', 'train_loss', 'score_loss',
                        'res', 'std', 'AUC', 'AP', 'n_params', 'minutes'] if c in ALL]
    print(f'{len(ALL)} runs in {CKPT_DIR}\n')
    print(ALL[cols].sort_values('AUC', ascending=False).to_string(index=False))
    ALL.to_csv(f'{OUTPUT_DIR}/all_runs_{RUN_VERSION}.csv', index=False)
    print(f'\nwritten -> {OUTPUT_DIR}/all_runs_{RUN_VERSION}.csv')


# %% [markdown]
# ---
# ## **Cell 6.1** — AUROC and AP side by side
# The project reports AUROC everywhere and AP for four runs only. AP is already computed
# and stored by `evaluate_scores` for **every** run, so the full table costs nothing —
# it is pure retrieval, no retraining, no rescoring.
#
# Why it matters: the test set is constructed at 50% prevalence, so AP tracks AUROC
# closely *here* — and that is precisely why it cannot be omitted. A reader needs both to
# reason about the realistic-prevalence case, where AP would be far lower (07 T3).
# MedIAnomaly reports both; a paper arguing the field measures carelessly cannot itself
# report a single rank metric on a constructed prevalence.

# %% [CELL 6.1]  AUROC + AP for every stored run

_AP = completed_runs()
if _AP.empty or 'AP' not in _AP:
    print('no runs with AP stored yet')
else:
    # Group by method AND width. Grouping on `method` alone pools all five head widths
    # plus the M4 variants into one `ae-posthoc-u` row (n=21), which is not a quantity --
    # it averages a 66.94 linear head with an 82.05 nonlinear one. Same defect as the A4
    # table had; keep the key identical to how the runs are actually distinguished.
    _AP = _AP.copy()
    _AP['group'] = _AP.method + (('_w' + _AP['w'].astype(str)) if 'w' in _AP else '')
    _AP['group'] = _AP['group'].fillna(_AP.method)
    g = (_AP.groupby('group')[['AUC', 'AP']]
         .agg(['mean', 'std', 'size']).sort_values(('AUC', 'mean'), ascending=False))
    print(f'\nAUROC and AP by method — {DATASET}, {RUN_VERSION}\n')
    print(f"  {'method':<24}{'AUROC':>16}{'AP':>16}{'n':>4}")
    for meth, r in g.iterrows():
        n = int(r[('AUC', 'size')])
        sd_a = 0.0 if n < 2 or not np.isfinite(r[('AUC', 'std')]) else r[('AUC', 'std')]
        sd_p = 0.0 if n < 2 or not np.isfinite(r[('AP', 'std')]) else r[('AP', 'std')]
        print(f'  {meth:<24}{r[("AUC","mean")]*100:>9.2f}+-{sd_a*100:<5.2f}'
              f'{r[("AP","mean")]*100:>9.2f}+-{sd_p*100:<5.2f}{n:>4}')
    _flat = g.copy()
    _flat.columns = ['_'.join(c) for c in _flat.columns]
    _flat.to_csv(f'{OUTPUT_DIR}/auc_ap_{RUN_VERSION}.csv')
    print(f'\n  written -> {OUTPUT_DIR}/auc_ap_{RUN_VERSION}.csv')
    print('  NOTE: 50% prevalence is a benchmark construction (07 T3). AP at clinical')
    print('        prevalence would be substantially lower; state this beside the table.')


# %% [markdown]
# ---
# ## **Cell 6.2** — the selection card: a REAL held-out protocol
# Every "validated" number in this project so far was selected on the same test set it is
# reported on. The split-half protocol splits the test set in half — the code does that
# correctly, but both halves are still test data, so it measures *selection stability*,
# not generalisation. There is no held-out set anywhere (07 T2).
#
# This cell creates one, for **zero GPU cost**, by using the dataset axis:
#
# 1. On **RSNA**, every choice that was made by looking at test AUC — the head width, the
#    ensemble membership — is written to a `selection_card.json`.
# 2. On **VinCXR / LAG**, the card is *read* and those choices are applied **frozen**. No
#    scan, no maximum, no re-selection. Whatever the card says is what gets evaluated.
#
# The resulting VinCXR and LAG numbers are genuinely held out: nothing about them
# influenced the configuration. This answers "you selected on test", "you have no
# validation set" and "does it generalise" with one protocol.
#
# **Carrying the card across sessions:** each dataset runs in its own Kaggle session, so
# download the RSNA bundle (Cell 7.0), and either upload `selection_card.json` as a Kaggle
# dataset input or paste its contents into `SELECTION_CARD_FALLBACK` below.

# %% [CELL 6.2]  Selection card — write on RSNA, apply frozen elsewhere

CARD_PATH = f'{OUTPUT_DIR}/selection_card.json'

# THE EASY PATH — no file transfer between Kaggle sessions.
# Each Kaggle session runs one DATASET and cannot see another session's files, so the
# RSNA choices have to reach the VinCXR and LAG sessions somehow. Rather than downloading
# and re-uploading a file, just leave this dict as it is: it already states the choices
# RSNA makes. On RSNA the cell OVERWRITES it from the actual runs and writes the file, so
# if the two ever disagree you will see it in the printed output.
SELECTION_CARD_FALLBACK = {
    'selected_on': 'RSNA',
    'run_version': 'dl-v1',
    'posthoc_head_width': '2',
    'ensemble_members': [['ae-pl', {}], ['aeu', {}]],
    'ensemble_members_extended': [['ae-pl', {}], ['aeu', {}],
                                  ['ae-ssim-posthoc-u', {'w': '2'}]],
    'note': 'Chosen on RSNA. Applied unchanged elsewhere, which is what makes those '
            'datasets held out.',
}


def _find_card():
    """A real card file wins over the pasted literal, so that uploading one actually
    takes effect; the literal is the no-transfer convenience path."""
    cands = [CARD_PATH] + sorted(_glob.glob('/kaggle/input/**/selection_card.json',
                                            recursive=True))
    for c in cands:
        if os.path.exists(c):
            with open(c) as f:
                return json.load(f), c
    if SELECTION_CARD_FALLBACK:
        return dict(SELECTION_CARD_FALLBACK), '<SELECTION_CARD_FALLBACK literal>'
    return None, None


def write_selection_card():
    """RSNA only. Records every test-selected choice so other datasets inherit it."""
    runs = completed_runs()
    ph = runs[runs.method == 'ae-posthoc-u'] if not runs.empty else pd.DataFrame()
    width = '2'
    if not ph.empty and 'w' in ph:
        # SMALLEST width within one sd of the best, NOT argmax. Argmax picks w=4 (82.05)
        # over w=2 (82.00) on a 0.05 difference against sds of ~0.6 -- i.e. it chases
        # noise, and it contradicts the paper's own stated rule ("the smallest head that
        # works"). The smallest-within-noise rule is pre-committable, is what the write-up
        # already claims, and is far easier to defend than a test-set maximum.
        per_w = ph.groupby(ph['w'].astype(str)).AUC.agg(['mean', 'std', 'size'])
        per_w = per_w[per_w['size'] >= 2]
        num = per_w[per_w.index.str.fullmatch(r'\d+')]      # drop the M4 variants
        if not num.empty:
            best = num['mean'].max()
            tol  = float(num.loc[num['mean'].idxmax(), 'std'] or 0.0)
            ok   = num[num['mean'] >= best - tol]
            width = str(min(int(w) for w in ok.index))
    card = {
        'selected_on': DATASET,
        'run_version': RUN_VERSION,
        'posthoc_head_width': width,
        'ensemble_members': [['ae-pl', {}], ['aeu', {}]],
        'ensemble_members_extended': [['ae-pl', {}], ['aeu', {}],
                                      ['ae-ssim-posthoc-u', {'w': width}]],
        'note': ('Every entry here was chosen by inspecting TEST AUC on the selecting '
                 'dataset. Applying them unchanged to another dataset is what makes that '
                 "dataset's numbers held out."),
    }
    with open(CARD_PATH, 'w') as f:
        json.dump(card, f, indent=2)
    return card


def apply_selection_card():
    """Non-RSNA. Evaluates the card's frozen choices — no scan, no maximum."""
    card, src = _find_card()
    if card is None:
        print('  no selection_card.json found. Run this cell on RSNA first, download the')
        print('  bundle from Cell 7.0, and upload the card as a Kaggle dataset input')
        print('  (or paste it into SELECTION_CARD_FALLBACK).')
        return None
    if card['selected_on'] == DATASET:
        print(f'  card was selected on {DATASET} — these numbers are NOT held out here.')
    print(f'  card loaded from {src}, selected on {card["selected_on"]}\n')
    rows = []
    for s in SEEDS:
        ens = ensemble_scores(s, members=[(m, p) for m, p in card['ensemble_members']])
        ext = ensemble_scores(s, members=[(m, p) for m, p in
                                          card['ensemble_members_extended']])
        head = _scores_for(run_id('ae-posthoc-u', s, w=card['posthoc_head_width']))
        base = _scores_for(run_id('ae-pl', s))
        r = {'seed': s}
        for name, v in [('ensemble_pair', ens), ('ensemble_ext', ext),
                        ('posthoc_head', head), ('ae_pl_baseline', base)]:
            r[name] = evaluate_scores(v, Y_TEST)['AUC'] * 100 if v is not None else np.nan
        rows.append(r)
    H = pd.DataFrame(rows)
    print(f'  HELD-OUT on {DATASET} — configuration frozen from {card["selected_on"]}\n')
    for c in ['ae_pl_baseline', 'posthoc_head', 'ensemble_pair', 'ensemble_ext']:
        v = H[c].dropna()
        if len(v):
            print(f'    {c:<18}{v.mean():6.2f}+-{v.std(ddof=0):.2f}  n={len(v)}')
    if H.ensemble_pair.notna().any() and H.ae_pl_baseline.notna().any():
        from scipy import stats as _st
        d = (H.ensemble_pair - H.ae_pl_baseline).dropna()
        print(f'\n    ensemble_pair - AE-PL = {d.mean():+.2f}  '
              f'per-seed {", ".join(f"{v:+.2f}" for v in d)}')
        print('    ^ THIS is a held-out gain: no part of it was tuned on this dataset.')
        # Two tests, same reasoning as M3: the seed-level one answers "does this survive
        # retraining", the image-level one answers "is it real for these models". With
        # n=3 the seed interval is very wide, so a null there means UNDERPOWERED, not
        # absent -- say which, and report the sign count, which is the honest summary of
        # a 3-seed direction.
        if len(d) >= 2 and d.std(ddof=1) > 0:
            _t, _p = _st.ttest_1samp(d.values, 0.0)
            _hw = _st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f'    seed-level 95% CI [{d.mean()-_hw:+.2f}, {d.mean()+_hw:+.2f}]  '
                  f'p={_p:.3f}  ({"significant" if _p < 0.05 else "NOT significant — n=3"})')
        print(f'    sign: {(d > 0).sum()}/{len(d)} seeds positive')
        # Paired DeLong on the same images, per seed. Much tighter than the seed test,
        # and valid conditional on these trained models.
        _dl = []
        for _s in SEEDS:
            _e = ensemble_scores(_s, members=[(m, pr) for m, pr in card['ensemble_members']])
            _b = _scores_for(run_id('ae-pl', _s), quiet=True)
            if _e is not None and _b is not None:
                _dl.append(delong_test(_e, _b, Y_TEST))
        if _dl:
            _dd = np.mean([r['diff'] for r in _dl]) * 100
            _se = float(np.sqrt(np.mean([r['se'] ** 2 for r in _dl]) / len(_dl)))
            _z  = _dd / (_se * 100) if _se > 0 else 0.0
            print(f'    DeLong pooled  {_dd:+.2f}  '
                  f'[{_dd - 1.96*_se*100:+.2f}, {_dd + 1.96*_se*100:+.2f}]  '
                  f'p={2*_st.norm.sf(abs(_z)):.2g}   (conditional on these models)')
    H.to_csv(f'{OUTPUT_DIR}/heldout_{DATASET}_{RUN_VERSION}.csv', index=False)
    print(f'\n  written -> {OUTPUT_DIR}/heldout_{DATASET}_{RUN_VERSION}.csv')
    return H


if DATASET == 'RSNA':
    _card = write_selection_card()
    print(f'  selection card WRITTEN -> {CARD_PATH}')
    print(f'  head width {_card["posthoc_head_width"]}, '
          f'{len(_card["ensemble_members"])}-member pair, '
          f'{len(_card["ensemble_members_extended"])}-member extended')
    _lit = SELECTION_CARD_FALLBACK or {}
    _agree = (_lit.get('posthoc_head_width') == _card['posthoc_head_width'])
    print(f'\n  The VinCXR and LAG sessions read these settings from the'
          f' SELECTION_CARD_FALLBACK\n  literal in Cell 6.2 — nothing has to be'
          f' downloaded or uploaded between sessions.')
    print(f'  literal says width {_lit.get("posthoc_head_width")!r}, '
          f'this run computed {_card["posthoc_head_width"]!r} -> '
          f'{"AGREE, nothing to do" if _agree else "DISAGREE: edit the literal in Cell 6.2 to match"}')
    HELDOUT = None
else:
    HELDOUT = apply_selection_card()


# %% [markdown]
# ---
# ## **Cell 7.0** — Bundle everything into one downloadable zip
# Collects **every** artifact this notebook produced — model weights, per-image score
# vectors, per-run manifests, all CSV tables, all figures, and a written summary — into a
# single zip under `/kaggle/working` for direct download. No wandb involved.
#
# `INCLUDE_WEIGHTS` controls the size. Weights are ~9 MB per run, so a full 100-run
# session is roughly 1 GB; the per-image **scores** are a few KB each and are what every
# table in the paper is actually derived from. Set it to `False` for a small bundle that
# still reproduces every number, `True` when you want the trained models themselves.

# %% [CELL 7.0]  Collect and zip all results

INCLUDE_WEIGHTS = True        # False -> scores + manifests + tables only (~50x smaller)

import zipfile, datetime, shutil


def _human(nbytes):
    for u in ['B', 'KB', 'MB', 'GB']:
        if nbytes < 1024 or u == 'GB':
            return f'{nbytes:.1f} {u}'
        nbytes /= 1024


def write_summary(path):
    """A plain-text record of what this session produced, so the zip is readable on its
    own without re-running anything."""
    runs = completed_runs()
    L = []
    A = L.append
    A('=' * 78)
    A(f'DL PROJECT RESULTS BUNDLE')
    A(f'dataset      : {DATASET}')
    A(f'run version  : {RUN_VERSION}')
    A(f'generated    : {datetime.datetime.now().isoformat(timespec="seconds")}')
    A(f'seeds        : {SEEDS}')
    A(f'epochs       : {EPOCHS}   sample mode: {SAMPLE_MODE}')
    A(f'runs stored  : {len(runs)}')
    A('=' * 78)
    if not runs.empty:
        A('')
        A('ALL RUNS (sorted by AUROC)')
        A('-' * 78)
        cols = [c for c in ['run_id', 'method', 'seed', 'AUC', 'AP', 'n_params',
                            'minutes'] if c in runs]
        A(runs[cols].sort_values('AUC', ascending=False).to_string(index=False))
        A('')
        A('BY METHOD (mean +- population sd over seeds)')
        A('-' * 78)
        gg = runs.groupby('method')[['AUC', 'AP']].agg(['mean', 'std', 'size'])
        for meth, r in gg.sort_values(('AUC', 'mean'), ascending=False).iterrows():
            n = int(r[('AUC', 'size')])
            A(f'  {meth:<26}AUROC {r[("AUC","mean")]*100:6.2f}   '
              f'AP {r[("AP","mean")]*100:6.2f}   n={n}')
    A('')
    A('CAVEATS THAT TRAVEL WITH THESE NUMBERS')
    A('-' * 78)
    A('  * sd values are POPULATION sd (ddof=0); on n=3 this understates the sample sd')
    A('    by a factor of 1.22.')
    A('  * DeLong p-values condition on THESE trained models and carry no seed variance.')
    A('    The seed-level column in m3_tests_*.csv is the one that speaks to retraining.')
    A('  * the post-hoc head WIDTH was selected on test AUC; the reported recovery % is')
    A('    therefore a test-set maximum.')
    A('  * the test set is 50% prevalence by construction, so AP is optimistic relative')
    A('    to clinical prevalence.')
    A(f'  * held-out numbers exist only where a selection card from another dataset was')
    A(f'    applied (heldout_*.csv). Everything else was selected on the data it reports.')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return path


BUNDLE_DIR = f'/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
_stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
BUNDLE = f'{BUNDLE_DIR}/dl_results_{DATASET}_{RUN_VERSION}_{_stamp}.zip'

write_summary(f'{OUTPUT_DIR}/SUMMARY_{DATASET}.txt')

_skip_ext = () if INCLUDE_WEIGHTS else ('.pt', '.pth', '.ckpt')
_counts, _bytes = {}, 0
with zipfile.ZipFile(BUNDLE, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, _dirs, files in os.walk(OUTPUT_DIR):
        for fn in sorted(files):
            if fn.endswith(_skip_ext):
                continue
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, os.path.dirname(OUTPUT_DIR)))
            ext = os.path.splitext(fn)[1] or '(none)'
            _counts[ext] = _counts.get(ext, 0) + 1
            _bytes += os.path.getsize(full)

print(f'\nBUNDLE WRITTEN\n  {BUNDLE}')
print(f'  zipped size   : {_human(os.path.getsize(BUNDLE))}   '
      f'(raw {_human(_bytes)})')
_wmsg = 'INCLUDED' if INCLUDE_WEIGHTS else 'EXCLUDED (set INCLUDE_WEIGHTS=True to keep)'
print(f'  weights       : {_wmsg}')
print('\n  contents by type:')
for ext, n in sorted(_counts.items(), key=lambda kv: -kv[1]):
    print(f'    {ext:<10}{n:>5}')
_size = os.path.getsize(BUNDLE)
_LIMIT = 900 * 1024 ** 2       # Kaggle's browser download is unreliable well below 1 GB

# Kaggle serves /kaggle/working through the Output pane, but clicking Download on a large
# single file often does nothing at all -- no error, no progress. Splitting into volumes
# under the limit is what actually makes it downloadable, so do it automatically.
PARTS = [BUNDLE]
if _size > _LIMIT:
    print(f'\n  {_human(_size)} is too large for a reliable browser download — splitting.')
    PARTS = []
    with open(BUNDLE, 'rb') as src:
        i = 0
        while True:
            chunk = src.read(_LIMIT)
            if not chunk:
                break
            part = f'{BUNDLE}.part{i:02d}'
            with open(part, 'wb') as out:
                out.write(chunk)
            PARTS.append(part)
            i += 1
    os.remove(BUNDLE)          # keep only the parts, or the Output pane shows both
    print(f'  {len(PARTS)} parts written. Rejoin them locally with:')
    print(f'    cat {os.path.basename(BUNDLE)}.part* > {os.path.basename(BUNDLE)}')

print('\n  DOWNLOAD — try these in order:')
print('   1. Click each file below (works even when the Output pane button does not).')
try:
    from IPython.display import FileLink, display
    for p_ in PARTS:
        display(FileLink(os.path.relpath(p_, '/kaggle/working')
                         if p_.startswith('/kaggle/working') else p_))
except Exception as _e:
    print(f'   (FileLink unavailable: {type(_e).__name__})')
print('   2. If nothing downloads: Save Version -> "Save & Run All (Commit)". When it')
print('      finishes, open the finished version and use its Output tab. Committed')
print('      output downloads reliably; an interactive session\'s often does not.')
print('   3. Still stuck? Set INCLUDE_WEIGHTS=False and re-run this cell — that drops')
print('      the bundle to a few MB and every table still recomputes from the scores.')
if not INCLUDE_WEIGHTS:
    print('\n  NOTE weights excluded; every table still recomputes from the stored scores.')
