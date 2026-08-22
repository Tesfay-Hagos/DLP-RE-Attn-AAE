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

# %% [markdown]
# # C1.3c The AE backbone family
# Cell 1.3b gave us the UNet, which is the DAE's network and nothing else's. Every other
# method in this study rides the same convolutional autoencoder: four stride-2 encoder
# blocks, a fully-connected bottleneck, four transposed-conv decoder blocks. AE, AE-U and
# VAE differ ONLY in the bottleneck and the final layer, which is why they are subclasses
# here rather than three separate networks.
#
# All of it is ported verbatim from MedIAnomaly (`reconstruction/networks/`). Verbatim is
# the point: this project's job is to reproduce their pixel-level numbers before it
# changes anything, and a "cleaned up" backbone makes any gap uninterpretable.
#
# `MemBottleNeck` (MemAE) is NOT here -- it lives in Cell 1.3d with the memory module it
# depends on. The DL sibling notebook carried `MemBottleNeck` while omitting `MemModule`,
# leaving a latent NameError in the class body; MedIAnomaly does define `MemModule`, in
# `networks/base_units/memory_module.py`, and Cell 1.3d ports both together.

# %% [CELL 1.3c]  AE / AE-U / VAE backbone, ported from MedIAnomaly reconstruction/networks/

# ---- ported verbatim from networks/base_units/conv_layers.py ------------------
def down_conv(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)


def up_conv(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


# ---- ported verbatim from networks/base_units/blocks.py -----------------------
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, num_layers, downsample=False, upsample=False,
                 last_layer=False):
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

        # Deeper block. NOTE the asymmetry, which is theirs, not a typo of ours: the
        # upsample branch PREPENDS its extra layers (they must run at the input width,
        # before the transposed conv changes it), the other branches APPEND.
        if upsample:
            for _ in range(1, num_layers):
                layers = [conv3x3(inplanes, inplanes), nn.BatchNorm2d(inplanes),
                          nn.ReLU(inplace=True)] + layers
        else:
            for _ in range(1, num_layers):
                layers = layers + [conv3x3(planes, planes), nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True)]

        if last_layer:
            layers = layers[:-2]      # drop BN + ReLU so the output is unbounded

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    """Residual variant of BasicBlock. Unused by the seven registered methods -- kept
    because MedIAnomaly's `--expansion`/ResBlock path is the obvious architecture knob to
    turn in the extension, and porting it now is cheaper than porting it later."""
    def __init__(self, inplanes, planes, num_layers, downsample=False, upsample=False,
                 last_layer=False):
        super(ResBlock, self).__init__()
        assert not (downsample and upsample)
        self.last_layer = last_layer
        self.relu = nn.ReLU(inplace=True)

        layers = []
        if downsample:
            layers.append(down_conv(inplanes, planes))
            self.skip = nn.Sequential(down_conv(inplanes, planes), nn.BatchNorm2d(planes))
        elif upsample:
            layers.append(up_conv(inplanes, planes))
            self.skip = nn.Sequential(up_conv(inplanes, planes), nn.BatchNorm2d(planes))
        else:
            layers.append(conv3x3(inplanes, planes))
            self.skip = nn.Identity()
        layers.append(nn.BatchNorm2d(planes))

        if upsample:
            for _ in range(1, num_layers):
                layers = [conv3x3(inplanes, inplanes), nn.BatchNorm2d(inplanes),
                          nn.ReLU(inplace=True)] + layers
        else:
            for _ in range(1, num_layers):
                layers = layers + [nn.ReLU(inplace=True), conv3x3(planes, planes),
                                   nn.BatchNorm2d(planes)]

        if last_layer:
            layers = layers[:-1]      # drop the BN only

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x) + self.skip(x)
        return out if self.last_layer else self.relu(out)


class BottleNeck(nn.Module):
    """Dense bottleneck: flatten -> mid_num -> latent_size -> mid_num -> unflatten.

    This is where the AE's capacity limit actually lives, and it is why IMAGE_SIZE and
    LATENT_DIM are not independent knobs: `feature_size` is input_size//16, so the first
    Linear is (4*base_width * (size/16)^2) x mid_num. Doubling the input resolution
    QUADRUPLES that matrix. At 64px it is 64*16 -> 1024; at 128px, 64*64 -> 1024."""
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(BottleNeck, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.linear_enc = nn.Sequential(
            nn.Linear(in_planes * feature_size * feature_size, mid_num),
            nn.BatchNorm1d(mid_num), nn.ReLU(True),
            nn.Linear(mid_num, latent_size))
        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, mid_num),
            nn.BatchNorm1d(mid_num), nn.ReLU(True),
            nn.Linear(mid_num, in_planes * feature_size * feature_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)
        out = self.linear_dec(z)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)
        return {'out': out, 'z': z}


class SpatialBottleNeck(nn.Module):
    """1x1-conv bottleneck that keeps the spatial grid instead of flattening it. Their
    `--spatial` flag. Retains localisation the dense bottleneck destroys, so it is a
    natural architecture arm for a PIXEL-level study -- unused by the reproduction."""
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(SpatialBottleNeck, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.linear_enc = nn.Sequential(
            nn.Conv2d(in_planes, mid_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_num), nn.ReLU(True),
            nn.Conv2d(mid_num, latent_size, kernel_size=1, stride=1, padding=0, bias=False))
        self.linear_dec = nn.Sequential(
            nn.Conv2d(latent_size, mid_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_num), nn.ReLU(True),
            nn.Conv2d(mid_num, in_planes, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        z = self.linear_enc(x)
        return {'out': self.linear_dec(z), 'z': z}


class VaeBottleNeck(BottleNeck):
    """Same shape, but the encoder emits 2*latent_size and splits it into (mu, log_var)."""
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(VaeBottleNeck, self).__init__(in_planes, feature_size, mid_num, latent_size)
        self.linear_enc = nn.Sequential(
            nn.Linear(in_planes * feature_size * feature_size, mid_num),
            nn.BatchNorm1d(mid_num), nn.ReLU(True),
            nn.Linear(mid_num, 2 * latent_size))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return torch.randn_like(std) * std + mu

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, log_var = self.linear_enc(x).chunk(2, dim=1)
        z_hat = self.reparameterize(mu, log_var)
        out = self.linear_dec(z_hat)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)
        return {'out': out, 'mu': mu, 'log_var': log_var}


# ---- ported verbatim from networks/ae.py -------------------------------------
class AE(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048,
                 latent_size=16, en_num_layers=1, de_num_layers=1, spatial=False):
        super(AE, self).__init__()
        bottleneck = SpatialBottleNeck if spatial else BottleNeck
        self.fm = input_size // 16          # four stride-2 blocks, 2^4 = 16

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)
        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers, downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers, downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers, downsample=True)

        self.bottle_neck = bottleneck(4 * base_width * expansion, feature_size=self.fm,
                                      mid_num=mid_num, latent_size=latent_size)

        self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers, upsample=True)
        self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers, upsample=True)
        self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers, upsample=True)
        self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, de_num_layers, upsample=True, last_layer=True)

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
        return {'x_hat': x_hat, 'z': z,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}


# ---- ported verbatim from networks/aeu.py ------------------------------------
class AEU(AE):
    """AE-U: the last decoder block emits 2*in_planes and splits into (x_hat, log_var),
    a LEARNED PER-PIXEL variance. For a pixel-level study this matters more than it does
    image-level -- the anomaly map is the squared error DIVIDED by that variance, so AE-U
    is the one method here whose map is reweighted spatially by the model itself."""
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048,
                 latent_size=16, en_num_layers=None, de_num_layers=None):
        super(AEU, self).__init__(input_size, in_planes, base_width, expansion, mid_num,
                                  latent_size, en_num_layers, de_num_layers)
        self.de_block4 = BasicBlock(1 * base_width * expansion, 2 * in_planes, de_num_layers,
                                    upsample=True, last_layer=True)

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


# ---- ported verbatim from networks/vae.py ------------------------------------
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


# ---- shape contract ----------------------------------------------------------
# AEU and VAE declare `en_num_layers=None, de_num_layers=None` and hand those straight to
# `range(1, num_layers)`. None never reaches range() only because every call site passes
# ints. That is a trap one careless default away from a TypeError deep in a training run,
# so it is asserted at construction time instead of hoped for.
def _selftest_backbones(size=64):
    """Every registered backbone builds, runs, and returns its input's shape."""
    x = torch.randn(2, 1, size, size)
    for name, cls in (('AE', AE), ('AEU', AEU), ('VAE', VAE)):
        net = cls(input_size=size, in_planes=1, base_width=16, expansion=1, mid_num=1024,
                  latent_size=16, en_num_layers=1, de_num_layers=1).eval()
        out = net(x)
        assert out['x_hat'].shape == x.shape, f'{name}: {out["x_hat"].shape} != {x.shape}'
        if name == 'AEU':
            assert out['log_var'].shape == x.shape, f'AEU log_var {out["log_var"].shape}'
        if name == 'VAE':
            assert out['mu'].shape == (2, 16), f'VAE mu {out["mu"].shape}'
        print(f'  {name:<4} OK  x_hat {tuple(out["x_hat"].shape)}  '
              f'{sum(p.numel() for p in net.parameters()):,} params')
    u = UNet(in_channels=1, n_classes=1, depth=4, wf=5).eval()
    assert u(x)['x_hat'].shape == x.shape, u(x)['x_hat'].shape
    print(f'  UNet OK  x_hat {tuple(u(x)["x_hat"].shape)}  '
          f'{sum(p.numel() for p in u.parameters()):,} params')


print('Backbone self-test:')
_selftest_backbones(size=64)


# %% [markdown]
# # C1.3d MemAE and Constrained-AE
# The last two backbones. Both subclass `AE` and change exactly one thing, which is why
# they belong here rather than in files of their own.
#
# * **MemAE** replaces the dense bottleneck with a memory-addressed one: the latent code
#   is re-expressed as a convex combination of 25 learned prototypes. The premise is that
#   a code assembled only from prototypes of healthy anatomy cannot reconstruct pathology.
#
#   **Their shrinkage is inert at their own settings, and the self-test measures it.**
#   MemAE's sparsity comes from `hard_shrink_relu`, which zeroes attention weights below
#   `shrink_thres=0.0025`. But MedIAnomaly sets `mem_size=25`, so a uniform attention
#   weight is 1/25 = 0.04 — sixteen times the threshold. The shrinkage therefore never
#   fires, and the memory degenerates to a dense 25-way soft combination. The original
#   MemAE paper uses a memory two orders of magnitude larger (mem_dim ~2000), where 1/N
#   sits near the threshold and the shrinkage does bite. We reproduce their setting, not
#   the paper's, and `_selftest_backbones_d` prints the occupancy so this stays visible
#   rather than becoming folklore. `mem_size` is an obvious knob for the extension.
# * **Constrained-AE** keeps the bottleneck and adds a term to the objective: re-encode
#   the reconstruction and require its latent code to match the original's. Note the
#   `istrain` flag on `forward` — the second encoder pass runs only in training, so the
#   driver must pass it. At test time `z_rec` is `None` and the score is the plain
#   squared residual.
#
# Ported verbatim from `networks/mem_ae.py`, `networks/base_units/memory_module.py` and
# `networks/constrained_ae.py`.

# %% [CELL 1.3d]  MemAE + Constrained-AE, ported from MedIAnomaly

# ---- ported verbatim from networks/base_units/memory_module.py ----------------
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    """ReLU-based hard shrinkage; only meaningful for positive values."""
    return (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))   # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)      # (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)          # MxC -> CxM
        output = F.linear(att_weight, mem_trans)       # (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(self.mem_dim, self.fea_dim is not None)


class MemModule(nn.Module):
    """NxCxHxW -> (NxHxW)xC -> address memory -> NxCxHxW. Our bottleneck is dense (l==2),
    so the permute branches are dead here -- kept because they are theirs and because the
    spatial bottleneck arm would use them."""
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)
        if l == 2:
            x = input
        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f'wrong feature map size {s}')
        x = x.contiguous().view(-1, s[1]) if l != 2 else x.contiguous()

        y_and = self.memory(x)
        y, att = y_and['output'], y_and['att']

        if l == 2:
            pass
        elif l == 3:
            y = y.view(s[0], s[2], s[1]).permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim).permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1]).permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim).permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1]).permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim).permute(0, 4, 1, 2, 3)
        return {'output': y, 'att': att}


# ---- ported verbatim from networks/base_units/blocks.py ----------------------
class MemBottleNeck(BottleNeck):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16, mem_size=25,
                 shrink_thres=0.0025):
        super(MemBottleNeck, self).__init__(in_planes, feature_size, mid_num, latent_size)
        self.memory_module = MemModule(mem_dim=mem_size, fea_dim=latent_size,
                                       shrink_thres=shrink_thres)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)
        mem_out = self.memory_module(z)
        z_hat, att = mem_out['output'], mem_out['att']
        out = self.linear_dec(z_hat)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)
        return {'out': out, 'att': att, 'z': z, 'z_hat': z_hat}


# ---- ported verbatim from networks/mem_ae.py --------------------------------
class MemAE(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048,
                 latent_size=16, en_num_layers=None, de_num_layers=None, mem_size=25,
                 shrink_thres=0.0025):
        super(MemAE, self).__init__(input_size, in_planes, base_width, expansion, mid_num,
                                    latent_size, en_num_layers, de_num_layers)
        self.bottle_neck = MemBottleNeck(4 * base_width * expansion, feature_size=self.fm,
                                         mid_num=mid_num, latent_size=latent_size,
                                         mem_size=mem_size, shrink_thres=shrink_thres)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)
        bottle_out = self.bottle_neck(en4)
        z, z_hat, att, de4 = (bottle_out['z'], bottle_out['z_hat'],
                              bottle_out['att'], bottle_out['out'])
        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)
        return {'x_hat': x_hat, 'z': z, 'z_hat': z_hat, 'att': att,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}


# ---- ported verbatim from networks/constrained_ae.py ------------------------
class ConstrainedAE(AE):
    """`forward(x, istrain=True)` runs the encoder a SECOND time, on the reconstruction,
    to produce `z_rec`. The driver must pass istrain during training or the objective
    silently loses its constraint term (z_rec is None and ConstrainedAELoss raises)."""
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048,
                 latent_size=16, en_num_layers=None, de_num_layers=None):
        super(ConstrainedAE, self).__init__(input_size, in_planes, base_width, expansion,
                                            mid_num, latent_size, en_num_layers, de_num_layers)

    def forward(self, x, istrain=False):
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

        if istrain:
            en1_rec = self.en_block1(x_hat)
            en2_rec = self.en_block2(en1_rec)
            en3_rec = self.en_block3(en2_rec)
            en4_rec = self.en_block4(en3_rec)
            z_rec = self.bottle_neck(en4_rec)['z']
        else:
            z_rec = None

        return {'x_hat': x_hat, 'z': z, 'z_rec': z_rec,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}


def _selftest_backbones_d(size=64):
    x = torch.randn(2, 1, size, size)
    m = MemAE(input_size=size, in_planes=1, base_width=16, expansion=1, mid_num=1024,
              latent_size=16, en_num_layers=1, de_num_layers=1).eval()
    o = m(x)
    assert o['x_hat'].shape == x.shape, o['x_hat'].shape
    assert o['att'].shape == (2, 25), f"att {o['att'].shape} -- expected (N, mem_size)"
    # attention must remain a distribution over prototypes
    row_sums = o['att'].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), row_sums
    occupancy = (o['att'] > 0).float().mean().item()
    print(f"  MemAE OK  x_hat {tuple(o['x_hat'].shape)}  att {tuple(o['att'].shape)}  "
          f"{sum(p.numel() for p in m.parameters()):,} params")
    print(f"    memory occupancy {occupancy*100:.0f}% of {o['att'].shape[1]} slots "
          f"(uniform weight {1/o['att'].shape[1]:.3f} vs shrink_thres 0.0025 -> "
          f"{'shrinkage INERT' if 1/o['att'].shape[1] > 0.0025 else 'shrinkage active'})")

    c = ConstrainedAE(input_size=size, in_planes=1, base_width=16, expansion=1, mid_num=1024,
                      latent_size=16, en_num_layers=1, de_num_layers=1).eval()
    o_test  = c(x, istrain=False)
    o_train = c(x, istrain=True)
    assert o_test['z_rec'] is None, 'z_rec must be None at test time'
    assert o_train['z_rec'].shape == o_train['z'].shape, 'z_rec/z shape mismatch'
    print(f"  ConstrainedAE OK  z {tuple(o_train['z'].shape)}  "
          f"z_rec {tuple(o_train['z_rec'].shape)} (train) / None (test)  "
          f"{sum(p.numel() for p in c.parameters()):,} params")


print('Backbone self-test (1.3d):')
_selftest_backbones_d(size=64)


# %% [CELL 1.4]  Configuration — MedIAnomaly reference hyperparameters
# All values below are taken from the benchmark's own defaults
# (MedIAnomaly/reconstruction/options.py + data_utils.get_transform), NOT tuned by us.
# Rationale: our first goal is to REPRODUCE their numbers so ours are comparable to
# Table 6. Hyperparameter search (manual ablation / Optuna) comes later, on top of a
# verified baseline -- tuning before reproducing makes any gap uninterpretable.
SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))

RUN_VERSION    = 'cv-v7.0'
SKIP_COMPLETED = True
WANDB_PROJECT  = 'MedIAnomaly-CV'
WANDB_GROUP    = f'{RUN_VERSION}'
# wandb run ids are not free-form: a '.' is not reliably accepted, and RUN_VERSION now
# carries one. Artifact names were already sanitised with the same '.'->'p' mapping in
# save_run/fetch_run, so this keeps the id consistent with them instead of inventing a
# second convention. CKPT_DIR keeps the dot -- a filesystem is happy with it.
WANDB_ID       = RUN_VERSION.lower().replace('.', 'p')

OUTPUT_DIR = ('/kaggle/working/results_cv' if os.path.isdir('/kaggle/working')
              else 'results_cv') + ('' if not SAMPLE_MODE else '_sample')
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
EPOCHS        = 250 if not SAMPLE_MODE else 2      # options.py epochs['rsna'/'brats']
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

print(f"SAMPLE_MODE : {SAMPLE_MODE}")
print(f"RUN_VERSION : {RUN_VERSION}  (SKIP_COMPLETED={SKIP_COMPLETED})")
print(f"image {IMAGE_SIZE}px | latent {LATENT_DIM} | epochs {EPOCHS} | bs {BATCH_SIZE} | lr {LR}")
print(f"OUTPUT_DIR  : {OUTPUT_DIR}")

# Name of the Kaggle Secret holding your wandb API key.
# Kaggle: Notebook -> Add-ons -> Secrets -> add a secret with THIS label.
# Locally: set WANDB_API_KEY, or just run `wandb login` once.
WANDB_SECRET_NAME = os.environ.get('WANDB_SECRET_NAME', 'REATTN_KEY')

USE_WANDB = False

# ── what a "Run All" actually EXECUTES ─────────────────────────────────────────
# Every cell up to 3.3 defines and self-tests; none of them touch the study. These two
# gate the cells that do (4.2 and 5.3). They default to True because running the notebook
# should run the notebook -- an earlier version defined everything, printed
# "Driver ready", and computed nothing, which reads exactly like a silent failure.
# Set False to load the definitions for inspection without starting hours of training.
RUN_REPRODUCTION = True     # Cell 4.2: the 5-method grid vs MedIAnomaly Table 7
RUN_EXPERIMENTS  = True     # Cell 5.3: E2 (free) then E1 (one DAE per noise_res)


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
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'pixel_range': list(PIXEL_RANGE),
        'sample_mode': SAMPLE_MODE, 'run_version': RUN_VERSION,
    }


def _slug(v):
    """Compact, filesystem- and wandb-safe rendering of a parameter value."""
    if isinstance(v, float):
        return f'{v:g}'.replace('.', 'p').replace('-', 'm')
    return str(v).replace('.', 'p').replace('/', '-').replace(' ', '')


def run_id(method, seed, **params):
    """Deterministic id: method[_k-v...]_sN. Same inputs -> same id, always."""
    parts = [method] + [f'{k}-{_slug(v)}' for k, v in sorted(params.items())] + [f's{seed}']
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
    diff = {k: (old.get(k), cur[k]) for k in cur if old.get(k) != cur[k]}
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


def stored_runs_by_method():
    """Every stored run, grouped by the method recorded in its OWN manifest.

    Deliberately NOT rebuilt from run_id(): reconstructing ids at the call site
    duplicates the id logic in train_and_eval, and the two drifting apart makes a lookup
    report 'not run yet' for runs that exist on disk -- a silent wrong answer rather than
    an error. The manifest records method, seed and params, so it is the one truth."""
    out = {}
    for mp in sorted(_glob.glob(os.path.join(CKPT_DIR, '*', 'manifest.json'))):
        with open(mp) as f:
            man = json.load(f)
        out.setdefault(man['method'], []).append(man)
    return out


def find_run(method, seed=None, **params_match):
    """Stored runs of `method` whose manifest params match every given key.

    The one supported way to locate an existing run. Callers that rebuilt ids by hand had
    to re-derive which params train_and_eval folds in and under what condition (`sz` only
    when the size differs from the global, `ep` only when epochs are overridden, ...);
    getting that subtly wrong silently returns nothing."""
    hits = []
    for man in stored_runs_by_method().get(method, []):
        if seed is not None and man['seed'] != seed:
            continue
        if all(man['params'].get(k) == v for k, v in params_match.items()):
            hits.append(man)
    return hits


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
               name=WANDB_ID,
               config=dict(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM,
                           hidden_num=HIDDEN_NUM, base_width=BASE_WIDTH,
                           epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                           weight_decay=WEIGHT_DECAY, pixel_range=PIXEL_RANGE,
                           run_version=RUN_VERSION),
               tags=['medianomaly', 'cv', RUN_VERSION],
               resume='allow', id=WANDB_ID,
               settings=wandb.Settings(init_timeout=120))
    print(f'WandB ready  project={WANDB_PROJECT}  version={RUN_VERSION}')
except Exception as _e:
    # Any failure here (no internet, no key, user declines login, etc.)
    # falls back to USE_WANDB=False so the rest of the notebook still runs.
    USE_WANDB = False
    print(f'WandB unavailable ({_e}) — continuing without.')



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

# Datasets this project needs. BraTS2021 is the only one MedIAnomaly computes pixel
# metrics on (ae_worker.py:18), so for a segmentation study it is the only requirement.
#
# This lived at the tail of the wandb cell until a run that skipped wandb setup died here
# with a bare NameError. Dataset requirements are not a logging concern; they belong in
# the dataset cell, so that Cell 2.0 is runnable on its own.
REQUIRED_DATASETS = ['BraTS2021']


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


def _has_payload(d, name):
    """True if directory `d` IS the dataset payload for `name` (not its parent)."""
    if name == 'BraTS2021':
        return all(os.path.isdir(os.path.join(d, p))
                   for p in ['train', 'test/normal', 'test/tumor', 'test/annotation'])
    return os.path.isdir(os.path.join(d, 'images')) and os.path.isfile(os.path.join(d, 'data.json'))


def dataset_dir(root, name):
    """The directory holding `name`'s payload under `root`, or None.

    Two layouts are accepted, because Kaggle produces both depending on what was zipped:

      <root>/<name>/train/...   the archive contained the dataset FOLDER  (our zip does)
      <root>/train/...          the archive contained the dataset CONTENTS

    Every caller must resolve through this rather than assuming os.path.join(root, name).
    A flat mount used to fail discovery entirely and report 'data not found', which sends
    you looking for a missing upload instead of a naming difference."""
    nested = os.path.join(root, name)
    if _has_payload(nested, name):
        return nested
    if _has_payload(root, name):
        return root
    return None


def _looks_like(root, name):
    """True if `name` is reachable from `root` in either accepted layout."""
    return dataset_dir(root, name) is not None


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
        d = dataset_dir(root, name)
        if d is None:
            d = os.path.join(root, name)
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
    d = dataset_dir(root, name) or os.path.join(root, name)
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
    d = dataset_dir(root, 'BraTS2021') or os.path.join(root, 'BraTS2021')

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
    # Masks are loaded RAW (0..255) and never pass through the image normalisation.
    #
    # This is not a style choice. An earlier version pushed masks through `_load`, which
    # maps 0..255 -> [-1,1], while building the normal images' all-zero masks with
    # `np.zeros` -- i.e. normalised 0.0, which on that scale is raw 127.5, not raw 0. The
    # threshold in use was `> 0.0`, so the normals came out correctly empty and the two
    # errors cancelled. Correcting the threshold to MedIAnomaly's `> 0` (normalised
    # `> -1.0`) removed the compensation and every one of the 828 normal images became
    # 100% lesion -- which inflates prevalence, destroys PixAP, and does so uniformly
    # enough across methods that the ranking could still look plausible.
    #
    # Keeping masks in their native 0/255 space removes the whole class of error: `np.zeros`
    # now means raw 0, which IS no-lesion, and the threshold is `> 0`, literally theirs
    # (dataload.py:152).
    def _load_mask(dirpath, names, tag):
        out = []
        for i, nm in enumerate(names):
            if i % 500 == 0:
                print(f'  {tag}: {i}/{len(names)}')
            im = Image.open(os.path.join(dirpath, nm)).convert('L').resize(
                (size, size), Image.NEAREST)
            out.append(np.asarray(im, dtype=np.float32))          # RAW 0..255
        return np.stack(out)[:, None] if out else np.zeros((0, 1, size, size), np.float32)

    masks_raw = np.concatenate([
        np.zeros((len(no_names), 1, size, size), np.float32),     # raw 0 == no lesion
        _load_mask(os.path.join(d, 'test/annotation'), mk_names, 'brats-masks')])

    # NEAREST resampling of a 0/255 mask must stay 0/255. If it ever does not, `> 0` and a
    # mid-grey threshold would disagree and Dice would quietly change meaning.
    stray = np.unique(masks_raw[(masks_raw > 0.5) & (masks_raw < 254.5)])
    assert stray.size == 0, (
        f"annotation masks are not binary 0/255 -- found intermediate values {stray[:8]}. "
        "NEAREST resampling should preserve 0/255; investigate before trusting Dice.")
    masks = (masks_raw > 0).astype(np.float32)          # BraTSAD dataload.py:152
    print(f'BraTS2021: train {x_train.shape}  test {x_test.shape}  '
          f'masks {masks.shape}  positive pixels {masks.mean()*100:.2f}%')
    return x_train, x_test, y_test, masks


# %% [markdown]
# ---
# ## **Cell 2.2** — Materialise BraTS2021, at every resolution a method needs
# BraTS2021 is the only dataset MedIAnomaly computes pixel metrics on. That is not our
# choice: `ae_worker.py` line 18 reads
# `self.pixel_metric = True if self.opt.dataset == "brats" else False`. Six of their seven
# datasets are image-level only, so BraTS2021 is the entire published pixel-level
# evidence base and therefore the entire reproduction target.
#
# **Why this is a cache and not four globals.** Their `train_eval.sh` trains every method
# at 64px — except DAE, which it runs at `--input-size 128 -bs 16`. Reproducing DAE
# therefore needs the data at 128px, and it must be *loaded* at 128px, not upsampled from
# 64px, which would carry no extra information. `get_data(size)` loads and caches per
# resolution; the module-level globals are just the `IMAGE_SIZE` view, kept so that the
# common path reads the same as before.
#
# `MASKS_TEST` is aligned index-for-index with `X_TEST`, normals carrying an all-zero
# mask, exactly as `BraTSAD` does. Their pixel metrics flatten the whole test set, so the
# normals' zero-masks supply most of the negative pixels; evaluating on abnormal images
# alone would report a different, easier number.

# %% [CELL 2.2]  BraTS2021 at any resolution, loaded once and cached

_DATA_CACHE = {}


def get_data(size):
    """(X_TRAIN, X_TEST, Y_TEST, MASKS_TEST) for BraTS2021 at `size`, loaded once.

    Tensors are float32 in [-1,1]; masks are float32 binary and index-aligned with X_TEST.
    Cached per resolution because a 128px reload costs minutes and DAE needs it."""
    if size in _DATA_CACHE:
        return _DATA_CACHE[size]

    x_train, x_test, y_test, masks = load_split_brats(DATA_ROOT, size=size)
    x_train = torch.from_numpy(x_train).float()
    x_test  = torch.from_numpy(x_test).float()
    masks   = torch.from_numpy(masks).float()

    # --- contract checks ---------------------------------------------------------
    # Each prevents a specific silent failure. None of these would crash training; they
    # would produce plausible-looking Dice numbers that mean nothing.
    assert x_train.ndim == 4 and x_train.shape[1] == 1, x_train.shape
    assert x_train.shape[-1] == size, (x_train.shape, size)
    assert x_test.shape[1:] == x_train.shape[1:], (x_test.shape, x_train.shape)
    assert masks.shape == x_test.shape, (masks.shape, x_test.shape)
    assert len(y_test) == len(x_test), (len(y_test), len(x_test))
    assert set(np.unique(masks.numpy())) <= {0.0, 1.0}, 'masks must be binary'
    assert y_test.min() == 0 and y_test.max() == 1, 'test needs both classes'

    # Normal images carry an all-zero mask; abnormal ones must not. A failure here means
    # the 'flair'->'seg' name mapping has paired masks to the wrong images, and every
    # pixel metric downstream would be scoring a permutation of the ground truth.
    is_norm = torch.from_numpy(y_test == 0)
    is_abn  = torch.from_numpy(y_test == 1)
    n_sum = masks[is_norm].sum().item()
    a_sum = masks[is_abn].sum().item()
    assert n_sum == 0.0, f'normal images carry {n_sum:.0f} lesion pixels'
    assert a_sum > 0.0, 'abnormal images carry no lesion pixels -- mask mapping is wrong'

    # [-1,1] is not negotiable: SSIM rescales by ((x+1)/2) internally and the DAE/CeAE
    # corruptions are centred for that range. A [0,1] tensor would train without error.
    assert x_train.min() >= -1.001 and x_train.max() <= 1.001, (x_train.min(), x_train.max())

    print(f'  {size}px: train {tuple(x_train.shape)}  test {tuple(x_test.shape)}  '
          f'{int(is_norm.sum())} normal / {int(is_abn.sum())} abnormal  '
          f'lesion pixels {masks.mean().item()*100:.3f}% of test '
          f'({masks[is_abn].mean().item()*100:.2f}% within abnormal)')

    _DATA_CACHE[size] = (x_train, x_test, y_test, masks)
    return _DATA_CACHE[size]


print(f'Loading BraTS2021 at {IMAGE_SIZE}px:')
X_TRAIN, X_TEST, Y_TEST, MASKS_TEST = get_data(IMAGE_SIZE)

# The prevalence is the number to hold in mind when reading PixAP: a random scorer earns
# exactly this, so a PixAP is only interpretable against it.
PIXEL_PREVALENCE = float(MASKS_TEST.mean())
print(f'random-scorer PixAP baseline = prevalence = {PIXEL_PREVALENCE:.4f}')


# %% [markdown]
# ---
# ## **Cell 3.0** — SSIM and the loss family
# Every criterion here answers to the same three-argument contract, ported from
# MedIAnomaly `reconstruction/utils/losses.py`:
#
# | call | returns | used for |
# |---|---|---|
# | `crit(x, out)` | scalar | the training objective |
# | `crit(x, out, anomaly_score=True)` | `(N,)` | image-level score |
# | `crit(x, out, anomaly_score=True, keepdim=True)` | `(N,1,H,W)` | **the anomaly map** |
#
# The third column is the one this project lives on, and it is worth being explicit about
# what it does NOT guarantee.
#
# **The map and the image score are not the same quantity reduced two ways.** For every
# criterion except SSIM they agree: `keepdim=True` averages over channels only, and taking
# the spatial mean afterwards reproduces `keepdim=False` exactly. SSIM breaks that. Its
# Gaussian window is valid-padded, so the raw map is `(H-win+1)` on a side — at 64px with
# the default 11-wide window that is 54x54, **29% of the image area discarded at the
# border**. `keepdim=False` averages that shrunken map. `keepdim=True` bilinearly
# interpolates it back to HxW first. The two therefore integrate over different regions
# and are genuinely different numbers.
#
# We flag it rather than fix it, for two reasons. It is what the benchmark's own code
# does, so reproducing their pixel numbers requires reproducing it. And it is a
# quantified, one-line-of-code confound sitting inside the score the benchmark
# recommends — which makes it the natural first thing for the extension to measure rather
# than something to quietly paper over. `_selftest_losses` below asserts the
# agreement for the four criteria that have it and asserts the DISAGREEMENT for SSIM, so
# neither property can change without the notebook failing loudly.

# %% [CELL 3.0]  SSIM + the loss family, ported from MedIAnomaly reconstruction/utils/

from typing import List, Optional, Tuple, Union
from collections import OrderedDict

# ---- SSIM, ported verbatim from reconstruction/utils/util.py ------------------
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


# ---- losses, ported verbatim from reconstruction/utils/losses.py --------------
class AELoss(nn.Module):
    """Plain squared error. `keepdim=True` gives the per-pixel squared residual, which is
    the canonical reconstruction anomaly map."""
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
    """1 - SSIM. The valid-padded Gaussian window shrinks the map by (win_size - 1) on
    each side; `keepdim=True` interpolates it back to the input size. See the markdown
    above for why that makes the map and the image score different quantities."""
    def __init__(self, win_size=11):
        super().__init__()
        self.win_size = win_size

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        net_in_01 = ((net_in + 1) / 2.0).clamp(0., 1.)
        x_hat_01  = ((x_hat + 1) / 2.0).clamp(0., 1.)
        loss = 1. - ssim(net_in_01, x_hat_01, data_range=1., size_average=False,
                         win_size=self.win_size)
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
    """AE-U. The net predicts a per-pixel log-variance and the residual is divided by it.
    That division IS a spatial reweighting of the anomaly map -- the one method here whose
    localisation is shaped by the model rather than by the residual alone.

    Scored on `loss1` only: the `+log_var` term is a per-pixel normalising constant, not
    evidence of anomaly. Returns a 3-tuple in TRAIN mode; the driver unpacks it."""
    def __init__(self):
        super().__init__()

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, log_var = net_out['x_hat'], net_out['log_var']
        recon_loss = (net_in - x_hat) ** 2
        loss1 = torch.exp(-log_var) * recon_loss
        loss = loss1 + log_var
        if anomaly_score:
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


# ---- the map/score contract, asserted ----------------------------------------
# How much of the image the valid-padded SSIM window throws away, at this resolution.
SSIM_WIN = 11
SSIM_VALID_FRAC = ((IMAGE_SIZE - SSIM_WIN + 1) / IMAGE_SIZE) ** 2


def _selftest_losses(size=None, tol=1e-5):
    """Prove the map/score relationship for every criterion -- agreement where it should
    hold, and a measured disagreement for SSIM where it should not."""
    size = size or IMAGE_SIZE
    torch.manual_seed(0)
    x = torch.randn(4, 1, size, size).clamp(-1, 1)
    out = {'x_hat': torch.randn(4, 1, size, size).clamp(-1, 1),
           'log_var': torch.randn(4, 1, size, size) * 0.1,
           'mu': torch.randn(4, 16)}
    out['log_var_dense'] = torch.randn(4, 16) * 0.1

    checks = [('l2', AELoss(), out), ('l1', L1Loss(), out),
              ('aeu', AEULoss(), out),
              ('vae', VAELoss(), {**out, 'log_var': out['log_var_dense']})]
    for name, crit, o in checks:
        smap  = crit(x, o, anomaly_score=True, keepdim=True)
        score = crit(x, o, anomaly_score=True)
        assert smap.shape == (4, 1, size, size), f'{name}: map {smap.shape}'
        assert score.shape == (4,), f'{name}: score {score.shape}'
        d = (smap.mean(dim=[1, 2, 3]) - score).abs().max().item()
        assert d < tol, f'{name}: map mean and image score disagree by {d:.2e}'
        print(f'  {name:<5} map {tuple(smap.shape)}  mean(map) == score  (|d| {d:.1e})')

    crit = SSIMLoss(win_size=SSIM_WIN)
    smap  = crit(x, out, anomaly_score=True, keepdim=True)
    score = crit(x, out, anomaly_score=True)
    raw   = 1. - ssim(((x + 1) / 2).clamp(0, 1), ((out['x_hat'] + 1) / 2).clamp(0, 1),
                      data_range=1., size_average=False, win_size=SSIM_WIN)
    assert smap.shape == (4, 1, size, size), f'ssim: map {smap.shape}'
    assert raw.shape[-1] == size - SSIM_WIN + 1, f'ssim: raw {raw.shape}'
    d = (smap.mean(dim=[1, 2, 3]) - score).abs().max().item()
    assert d > tol, ('ssim: map mean and image score AGREE -- the border-discard '
                     'asymmetry documented above has disappeared; re-read the cell')
    print(f'  ssim  map {tuple(smap.shape)}  raw {tuple(raw.shape)}  '
          f'valid area {SSIM_VALID_FRAC*100:.0f}%  mean(map) != score (|d| {d:.1e})')


print(f'Loss self-test at {IMAGE_SIZE}px:')
_selftest_losses()
print(f'\nSSIM window {SSIM_WIN}px keeps {SSIM_VALID_FRAC*100:.1f}% of the '
      f'{IMAGE_SIZE}px image; {100*(1-SSIM_VALID_FRAC):.1f}% is border-discarded.')




# %% [markdown]
# ---
# ## **Cell 3.0b** — The remaining criteria
# Three more, completing the set MedIAnomaly's `train_eval.sh` actually runs.
#
# * **`MemAELoss`** — squared residual plus an entropy penalty on the memory attention,
#   weighted 2e-4. The penalty is what keeps the addressing sparse; the anomaly map is the
#   plain residual, so the memory affects localisation only through the reconstruction.
# * **`ConstrainedAELoss`** — squared residual plus the latent-consistency term. Returns a
#   3-tuple in train mode and needs `z_rec`, so the driver must call the network with
#   `istrain=True`.
# * **`RelativePerceptualL1Loss`** (AE-PL) — relative L1 on channel-normalised VGG19
#   `relu4_2` features. Ported from the DL sibling notebook, where it was verified against
#   MedIAnomaly's own layer-by-layer construction.
#
# **AE-PL's map is not native pixel resolution.** `relu4_2` at 64px input is 8x8, so
# `keepdim=True` bilinearly upsamples an 8x8 map to 64x64. Every "lesion" it can draw is
# therefore at least 8px across and blob-shaped by construction. That is not a bug in the
# port — it is what the method is — but it means AE-PL's BestDice is capped by its own
# output resolution, and the report must say so rather than reading it as a like-for-like
# localiser.

# %% [CELL 3.0b]  MemAE / Constrained-AE criteria, then the perceptual loss

class MemAELoss(nn.Module):
    """Ported verbatim from reconstruction/utils/losses.py."""
    def __init__(self):
        super(MemAELoss, self).__init__()
        self.entropy_loss_weight = 0.0002
        self.eps = 1e-12

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, att = net_out['x_hat'], net_out['att']
        recon_loss = (net_in - x_hat) ** 2
        entro_loss = self.entropy_loss(att)
        loss = recon_loss.mean() + self.entropy_loss_weight * entro_loss
        if anomaly_score:
            return torch.mean(recon_loss, dim=[1], keepdim=True) if keepdim \
                else torch.mean(recon_loss, dim=[1, 2, 3])
        return loss.mean(), recon_loss.mean().item(), entro_loss.item()

    def entropy_loss(self, x):
        x = self.feature_map_permute(x)
        b = x * torch.log(x + self.eps)
        return (-1. * b.sum(dim=1)).mean()

    def feature_map_permute(self, input):
        s = input.data.shape
        l = len(s)
        if l == 2:
            x = input
        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f'wrong feature map size {s}')
        return x.contiguous().view(-1, s[1])


class ConstrainedAELoss(nn.Module):
    """Ported verbatim. Needs net_out['z_rec'], i.e. forward(x, istrain=True)."""
    def __init__(self):
        super(ConstrainedAELoss, self).__init__()

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, z = net_out['x_hat'], net_out['z']
        loss_x = (net_in - x_hat) ** 2
        if anomaly_score:
            return torch.mean(loss_x, dim=[1], keepdim=True) if keepdim \
                else torch.mean(loss_x, dim=[1, 2, 3])
        z_rec = net_out['z_rec']
        if z_rec is None:
            raise RuntimeError(
                'ConstrainedAELoss got z_rec=None -- the network was called without '
                'istrain=True, so the latent-consistency term is missing. See Cell 1.3d.')
        loss_z = (z - z_rec) ** 2
        loss = loss_x.mean() + loss_z.mean()
        return loss.mean(), loss_x.mean().item(), loss_z.mean().item()


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
# The registry is the full set of reconstruction methods MedIAnomaly's `train_eval.sh`
# runs, restricted to those that can produce a pixel map. One line per method; nothing
# downstream ever constructs a network by hand.
#
# **What their script runs:**
# ```
# methods="ae ae-l1 ae-ssim ae-perceptual ae-spatial vae constrained-ae memae ceae
#          ganomaly aeu ae-grad vae-rec vae-combi"
# ... plus: train.py -m dae --input-size 128 -bs 16
# ```
#
# **GANomaly is excluded, and this is not a shortcut.** Its anomaly score is
# `mean((z - z_hat)**2, dim=1)` — a distance between two *latent vectors*, with no spatial
# extent to map back onto the image. Their own `ganomaly_worker.evaluate` computes AUC and
# AP and no pixel metrics at all, on BraTS as on everything else. There is no GANomaly row
# in the pixel-level table to reproduce, so including it would mean inventing a scoring
# rule and attributing it to them.
#
# **DAE trains at 128px, batch 16.** Their shell script gives it its own loop with
# `--input-size 128 -bs 16` while everything else takes the 64px defaults. A DAE trained at
# 64px is a different experiment from the one they report, so `input_size` and
# `batch_size` are per-method overrides rather than globals.
#
# **Two methods corrupt their input.** DAE adds coarse rolled noise; CeAE erases a square
# patch. Both reconstruct toward the *clean* image, so `corrupt` is a training-time
# property and never touches evaluation.

# %% [CELL 3.1]  Method registry and network / loss factories

# DAE's UNet. base_worker.set_network_loss builds UNet(in_channels, n_classes), taking the
# class defaults depth=5, wf=6.
#
# UNRESOLVED, and the write-up must say so. MedIAnomaly Table 6 lists DAE at 2.79M params
# / 2.15 GFLOPs. These defaults give 31.042M -- and no (depth, wf) pair in depth 3..6 x
# wf 2..7 yields 2.79M (the nearest are ~1.9M), so the table's params column cannot be
# reconciled with the shipped code by any setting.
#
# An earlier draft of this comment argued the FLOPs column reconciles (~1.07 G MACs at
# 64px ~= 2.15 GFLOPs) and took that as evidence the architecture was right. That argument
# is void: train_eval.sh runs DAE at --input-size 128, where the same network is ~4x those
# MACs and matches nothing. The agreement at 64px was a coincidence of the wrong
# resolution.
#
# So we follow the code because it is the only executable specification, not because the
# numbers reconcile -- they do not. If DAE lands short of Table 7's 75.5, capacity is a
# live suspect alongside the corruption, and neither can be ruled out from the repository.
DAE_UNET_DEPTH = 5
DAE_UNET_WF    = 6

# CeAE's corruption, from dataload.py: RandomErasing(p=1., scale=(0.024, 0.024),
# ratio=(1., 1.), value=-1) -- one square of 2.4% of the image area, filled with the
# minimum intensity. sqrt(0.024) * 64 ~= 10px at 64px input.
CEAE_ERASE_SCALE = 0.024
CEAE_ERASE_VALUE = -1.0

_NET_CLASSES = {'ae': AE, 'aeu': AEU, 'vae': VAE, 'memae': MemAE,
                'constrained-ae': ConstrainedAE}


def build_net(kind, input_size=None):
    """The ONLY place a backbone is instantiated."""
    input_size = input_size or IMAGE_SIZE
    if kind == 'unet':
        return UNet(in_channels=1, n_classes=1,
                    depth=DAE_UNET_DEPTH, wf=DAE_UNET_WF).to(device)
    # AEU, VAE, MemAE and ConstrainedAE all default en/de_num_layers to None and forward
    # them straight into range(), so a None here is a TypeError minutes into a run.
    assert isinstance(EN_DEPTH, int) and isinstance(DE_DEPTH, int), \
        f'EN_DEPTH/DE_DEPTH must be ints, got {EN_DEPTH!r}/{DE_DEPTH!r}'
    kw = dict(input_size=input_size, in_planes=1, base_width=BASE_WIDTH, expansion=1,
              mid_num=HIDDEN_NUM, latent_size=LATENT_DIM,
              en_num_layers=EN_DEPTH, de_num_layers=DE_DEPTH)
    # ae-spatial is the same AE class with the 1x1-conv bottleneck (their --spatial),
    # so it is a flag here rather than an entry in _NET_CLASSES.
    if kind == 'ae-spatial':
        return AE(**kw, spatial=True).to(device)
    return _NET_CLASSES[kind](**kw).to(device)


_LOSS_CACHE = {}


def build_loss(name):
    """name -> criterion. Separate from build_net so any training loss can be paired with
    any scoring loss -- that separation is what makes `rescore` possible."""
    if name == 'perceptual':
        # wraps a frozen 20M-param VGG19; one shared instance, it holds no run state
        if 'perceptual' not in _LOSS_CACHE:
            _LOSS_CACHE['perceptual'] = RelativePerceptualL1Loss()
        return _LOSS_CACHE['perceptual']
    if name == 'l2':             return AELoss()
    if name == 'l2grad':         return AELoss(grad_score=True)
    if name == 'l1':             return L1Loss()
    if name == 'ssim':           return SSIMLoss(win_size=SSIM_WIN)
    if name == 'aeu':            return AEULoss()
    if name == 'vae':            return VAELoss()
    if name == 'vaegrad-elbo':   return VAELoss(grad='elbo')
    if name == 'vaegrad-kl':     return VAELoss(grad='kl')
    if name == 'vaegrad-rec':    return VAELoss(grad='rec')
    if name == 'vaegrad-combi':  return VAELoss(grad='combi')
    if name == 'memae':          return MemAELoss()
    if name == 'constrained':    return ConstrainedAELoss()
    raise KeyError(f'unknown loss {name!r}')


# Losses whose TRAIN call returns (loss, extra1, extra2) instead of a bare scalar.
_TUPLE_LOSSES = {'aeu', 'vae', 'memae', 'constrained'}
# Losses whose SCORE differentiates w.r.t. the input (so no torch.no_grad at eval).
_GRAD_LOSSES  = {'l2grad', 'vaegrad-elbo', 'vaegrad-kl', 'vaegrad-rec', 'vaegrad-combi'}

METHODS = {
    # their name          net               train_loss        score_loss       extras
    'ae':             dict(net='ae',             train_loss='l2',         score_loss='l2'),
    'ae-l1':          dict(net='ae',             train_loss='l1',         score_loss='l1'),
    'ae-ssim':        dict(net='ae',             train_loss='ssim',       score_loss='ssim'),
    'ae-perceptual':  dict(net='ae',             train_loss='perceptual', score_loss='perceptual'),
    'ae-spatial':     dict(net='ae-spatial',     train_loss='l2',         score_loss='l2'),
    'vae':            dict(net='vae',            train_loss='vae',        score_loss='vae'),
    'constrained-ae': dict(net='constrained-ae', train_loss='constrained', score_loss='constrained',
                           istrain=True),
    'memae':          dict(net='memae',          train_loss='memae',      score_loss='memae'),
    'ceae':           dict(net='ae',             train_loss='l2',         score_loss='l2',
                           corrupt='erase'),
    'aeu':            dict(net='aeu',            train_loss='aeu',        score_loss='aeu'),
    'ae-grad':        dict(net='ae',             train_loss='l2',         score_loss='l2grad'),
    'vae-rec':        dict(net='vae',            train_loss='vae',        score_loss='vaegrad-rec'),
    'vae-combi':      dict(net='vae',            train_loss='vae',        score_loss='vaegrad-combi'),
    # foreground_mask=True is REQUIRED for BraTS -- dae_worker.py:50-53 applies it for
    # datasets 'brain' and 'brats', and this project is BraTS-only. See add_noise.
    'dae':            dict(net='unet',           train_loss='l2',         score_loss='l2',
                           corrupt='noise', noise_res=16, noise_std=0.2,
                           foreground_mask=True, input_size=128, batch_size=16),
}

# Excluded from the pixel study, with the reason, so the omission is auditable rather
# than silent. See the markdown above.
EXCLUDED_METHODS = {
    'ganomaly': 'latent-space score mean((z-z_hat)^2, dim=1) has no spatial extent, so '
                'ganomaly_worker.evaluate computes only AUC/AP and no pixel metrics on '
                'any dataset; it also evaluates in net.train() mode by design '
                '(ganomaly_worker.py:88-91), which makes it doubly non-comparable here',
}

# ---------------------------------------------------------------------------------
# MedIAnomaly Table 7 -- BraTS2021 pixel-level, (AP_pix, Dice_ceiling), mean +- sd over
# their repeats. These are the published numbers; we reproduce a SUBSET of them to
# validate this harness, not all of them, because re-deriving a table that already exists
# buys nothing. See ACTIVE_METHODS below for which, and why.
#
# `ae-spatial` is absent from Table 7 -- their shell script trains it, but the paper does
# not report a pixel-level row for it. No target, so no reproduction claim is possible.
TARGET_BRATS_PIXEL = {
    'dae':            ((75.5, 0.7), (71.1, 0.6)),
    'ae-perceptual':  ((44.8, 0.4), (45.2, 0.2)),
    'vae':            ((44.0, 5.3), (47.1, 3.4)),
    'vae-rec':        ((35.6, 1.2), (42.1, 0.6)),
    'ae':             ((33.2, 2.2), (39.2, 1.7)),
    'constrained-ae': ((30.9, 3.9), (37.5, 3.1)),
    'ae-grad':        ((29.9, 1.0), (36.6, 0.5)),
    'ceae':           ((28.6, 3.2), (35.8, 2.4)),
    'ae-l1':          ((26.5, 3.7), (34.4, 3.1)),
    'ae-ssim':        ((25.7, 0.4), (35.7, 0.4)),
    'vae-combi':      ((24.3, 1.2), (32.9, 1.5)),
    'memae':          ((22.8, 5.6), (30.8, 4.8)),
    'aeu':            ((22.2, 4.0), (35.9, 4.4)),
    'ae-spatial':     (None, None),      # not reported pixel-level
}

# ---------------------------------------------------------------------------------
# THE ACTIVE SUBSET
#
# Table 7 has one real result and a cluster. DAE scores 75.5 AP_pix; the next best is
# 44.8; the bottom nine sit between 22.2 and 33.2 with standard deviations of 3-6, which
# makes most of their ordering noise. Training all fourteen would spend most of the
# compute separating methods the benchmark itself cannot separate.
#
# So we run five, chosen to SPAN the table rather than to sample it:
#
#   dae            75.5  the only method that segments; the ceiling
#   ae-perceptual  44.8  best non-DAE by AP_pix
#   vae            44.0  best non-DAE by Dice (47.1)
#   ae             33.2  the baseline every other method is a variant of
#   aeu            22.2  the floor -- and the interesting one, see below
#
# **Why AE-U earns a slot despite being last.** It is the WORST pixel-level method here
# (22.2 AP_pix) and one of the BEST image-level methods in the same benchmark (86.5 AUROC
# on RSNA, Table 6). A method can rank near the top on detection and dead last on
# localisation. That inversion is the most interesting thing in this table and it is
# free to observe -- it needs no extra experiment, only that AE-U is in the set.
#
# Adding a method back is one line: put its name in ACTIVE_METHODS. The full registry is
# retained precisely so the reduction is a choice, not a limitation.
ACTIVE_METHODS = ['dae', 'ae-perceptual', 'vae', 'ae', 'aeu']

# Relative training cost, used only to order the grid so that a session that dies early
# has banked the cheap runs. These are HINTS, not measurements: 1.0 is a plain 64px AE.
#   ae-perceptual  two extra VGG19 forwards per step (net_in and x_hat) plus the backward
#                  through them; VGG19 is ~20M params against the AE's 2.35M
#   constrained-ae a second encoder pass over the reconstruction
#   dae            4x the pixels (128px), 31M params, and 4x the steps (batch 16)
METHOD_COST = {'ae-perceptual': 12.0, 'constrained-ae': 1.5, 'dae': 40.0}


def method_cost(m):
    return METHOD_COST.get(m, 1.0) * len(seeds_for(m))

# Seeds per method. DAE has the smallest variance in Table 7 (+-0.7 AP_pix, +-0.6 Dice)
# and by far the largest cost: a 31M-parameter UNet over 4,211 training slices at 128px
# for 250 epochs, roughly 4x the per-step cost of the 64px AEs on 4x the pixels. One seed
# for it and three for the noisy methods spends the compute where the uncertainty is.
# BraTS2021 (their Table 2): 4,211 train / 828 test normal / 1,948 test abnormal.
#
# The cost of that choice, stated so the report does not overclaim: DAE's row will have
# NO measured spread. Its "gap in sd units" against Table 7 is a single point estimate
# against their +-0.7, so a gap of 1-2 sd there is not evidence of anything. If DAE ends
# up carrying a claim, give it a second seed -- SEEDS_BY_METHOD = {'dae': [42, 43]} -- and
# accept the extra run. Note also that Cell 1.3 sets torch.backends.cudnn.benchmark=True,
# which MedIAnomaly does not; kernel autotuning makes repeated runs differ in the last
# float digits. That is run-to-run noise, not systematic bias, but it means an exactly
# identical rerun is not guaranteed on GPU.
SEEDS_DEFAULT = [42, 43, 44]
SEEDS_BY_METHOD = {'dae': [42]}


def seeds_for(method):
    return SEEDS_BY_METHOD.get(method, SEEDS_DEFAULT)


assert set(ACTIVE_METHODS) <= set(METHODS), set(ACTIVE_METHODS) - set(METHODS)

print(f'{len(METHODS)} methods registered, {len(ACTIVE_METHODS)} active:\n')
print(f"  {'method':<15} {'target AP_pix':>13} {'target Dice':>12}  seeds  notes")
for m, spec in METHODS.items():
    on = m in ACTIVE_METHODS
    ap, dc = TARGET_BRATS_PIXEL.get(m, (None, None))
    aps = f'{ap[0]:.1f}+-{ap[1]:.1f}' if ap else '--'
    dcs = f'{dc[0]:.1f}+-{dc[1]:.1f}' if dc else '--'
    tag = []
    if spec.get('corrupt'):  tag.append(f"corrupt={spec['corrupt']}")
    if spec.get('istrain'):  tag.append('istrain')
    sz, bs = spec.get('input_size', IMAGE_SIZE), spec.get('batch_size', BATCH_SIZE)
    if sz != IMAGE_SIZE or bs != BATCH_SIZE: tag.append(f'{sz}px/bs{bs}')
    n_seeds = len(seeds_for(m)) if on else 0
    print(f"  {'*' if on else ' '} {m:<13} {aps:>13} {dcs:>12}  "
          f"{n_seeds if on else '-':>5}  {' '.join(tag)}")
print('\n  * = active. Add a method by putting its name in ACTIVE_METHODS.')
for m, why in EXCLUDED_METHODS.items():
    print(f'  x {m:<13} EXCLUDED -- {why}')
print(f'\n  total runs: {sum(len(seeds_for(m)) for m in ACTIVE_METHODS)}')


# %% [markdown]
# ---
# ## **Cell 3.2** — Pixel-level evaluation
# This is the cell the whole project exists for. It reproduces MedIAnomaly's
# `AEWorker.evaluate` pixel branch exactly, with one algorithmic substitution that is
# proved equivalent rather than assumed.
#
# **What their code does, and what we keep.** Four properties of their protocol are
# reproduced deliberately, because changing any of them would make our numbers
# incomparable to theirs:
#
# 1. **The image score is the mean of the map**, not an independently computed score:
#    `test_scores = torch.mean(test_score_maps, dim=[1,2,3])`. For SSIM this means the
#    image score is taken from the *interpolated* map, so it differs from what
#    `keepdim=False` returns — the asymmetry Cell 3.0 measured at 28.8% of image area.
# 2. **Pixel metrics flatten the whole test set**, normals included. The normals'
#    all-zero masks supply most of the negative pixels; dropping them would inflate
#    PixAP substantially.
# 3. **`BestDice` is an oracle.** The threshold maximising Dice is chosen on the test
#    masks themselves, over 200 candidates drawn from the test predictions. No deployment
#    can pick that threshold. It is a ceiling, and the report must call it one.
# 4. **200 thresholds, sampled from the sorted predictions**, not evenly spaced in value —
#    so they are dense where the predictions are dense.
#
# **The substitution.** Their `compute_best_dice` re-binarises the entire prediction array
# once per threshold across an 8-process pool: 200 full passes over ~8M pixels. The same
# numbers fall out of a single sort. Sort the pixels by score, accumulate the lesion
# labels from the top down, and every threshold's true-positive count is a lookup. That is
# one O(N log N) pass instead of 200 O(N) passes, with no multiprocessing to misbehave
# inside a notebook. `_selftest_best_dice` asserts the two agree exactly on random data,
# so the speedup cannot silently become a different metric.

# %% [CELL 3.2]  Pixel-level metrics — MedIAnomaly's protocol, one algorithm swapped

def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """Sorensen-Dice, ported verbatim from MedIAnomaly reconstruction/utils/util.py.
    Both arrays must already be binary."""
    preds, targets = np.array(preds), np.array(targets)
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError('Predictions must be binary')
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')
    return 2 * np.sum(preds[targets == 1]) / (np.sum(preds) + np.sum(targets))


def _best_dice_thresholds(flat_preds: np.ndarray, n_thresh: int = 200) -> np.ndarray:
    """Their threshold grid: n_thresh order statistics of the predictions themselves.

    Reproduced quirk and all -- `np.arange(0, num, step)` with `step = num // n_thresh`
    yields n_thresh+1 candidates, not n_thresh. Harmless (one extra threshold), but it is
    theirs, and matching it means our grid is theirs."""
    num = flat_preds.size
    step = num // n_thresh
    if step < 1:
        raise ValueError(f'{num} pixels is fewer than n_thresh={n_thresh}')
    indices = np.arange(0, num, step)
    return np.sort(flat_preds)[indices]


def compute_best_dice_reference(preds, targets, n_thresh=200):
    """Their algorithm, single-process. Correct, slow, and only used to prove the fast
    path below. Do not call it on a full test set."""
    preds, targets = np.asarray(preds).reshape(-1), np.asarray(targets).reshape(-1)
    thresholds = _best_dice_thresholds(preds, n_thresh)
    scores = np.array([compute_dice(np.where(preds > t, 1, 0), targets) for t in thresholds])
    return float(scores.max()), float(thresholds[scores.argmax()])


def compute_best_dice(preds, targets, n_thresh=200):
    """One sort, then every threshold is a lookup. Exactly equivalent to the reference.

    For a threshold t: P = #{pred > t}, TP = #{pred > t and target == 1}, T = #{target==1},
    and Dice = 2*TP / (P + T). Sorting ascending once makes P a searchsorted lookup and TP
    a suffix sum of the reordered targets."""
    preds   = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    assert preds.shape == targets.shape, (preds.shape, targets.shape)

    order = np.argsort(preds, kind='stable')
    sorted_preds   = preds[order]
    sorted_targets = targets[order].astype(np.int64)

    # suffix[k] = number of lesion pixels among the k-th and all higher-scoring pixels.
    # Written with an explicit temporary rather than an in-place reversal: assigning a
    # reversed view of an array onto itself is an overlapping copy, and numpy's guarantees
    # there are not worth relying on for a metric.
    suffix = np.zeros(sorted_targets.size + 1, dtype=np.int64)
    suffix[:-1] = np.cumsum(sorted_targets[::-1])[::-1]

    T = int(sorted_targets.sum())
    thresholds = _best_dice_thresholds(sorted_preds, n_thresh)   # already sorted
    k  = np.searchsorted(sorted_preds, thresholds, side='right') # #{pred <= t}
    P  = sorted_preds.size - k                                   # #{pred >  t}
    TP = suffix[k]
    denom = P + T
    dice = np.where(denom > 0, 2.0 * TP / np.maximum(denom, 1), 0.0)
    return float(dice.max()), float(thresholds[dice.argmax()])


def _selftest_best_dice(trials=5, n=4000, n_thresh=50, seed=0):
    """The fast path must equal the reference exactly, including the argmax tie-break."""
    rng = np.random.default_rng(seed)
    for t in range(trials):
        # deliberately lumpy: ties and repeated values are where a sort-based
        # reformulation would diverge from a threshold-scan if it were wrong
        preds = np.round(rng.random(n) * 20) / 20
        targets = (rng.random(n) < 0.08).astype(np.float32)
        d_fast, th_fast = compute_best_dice(preds, targets, n_thresh)
        d_ref,  th_ref  = compute_best_dice_reference(preds, targets, n_thresh)
        assert abs(d_fast - d_ref) < 1e-12, f'trial {t}: dice {d_fast} != {d_ref}'
        assert th_fast == th_ref, f'trial {t}: threshold {th_fast} != {th_ref}'
    print(f'  compute_best_dice == reference on {trials} random trials '
          f'(n={n}, ties present)')


def evaluate_maps(score_maps, y, masks, n_thresh=200):
    """Every metric MedIAnomaly reports for a pixel-capable dataset, from the maps alone.

    score_maps : (N,1,H,W) anomaly maps
    y          : (N,)      image labels, 0 normal / 1 abnormal
    masks      : (N,1,H,W) binary ground truth, all-zero for normal images
    """
    maps = score_maps.detach().cpu().numpy() if torch.is_tensor(score_maps) else np.asarray(score_maps)
    gt   = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    y    = np.asarray(y)
    assert maps.shape == gt.shape, (maps.shape, gt.shape)
    assert maps.shape[0] == y.shape[0], (maps.shape, y.shape)

    # image-level: THEIR definition -- the spatial mean of the map (see markdown)
    scores = maps.mean(axis=(1, 2, 3))
    res = {
        'AUC': float(roc_auc_score(y, scores)),
        'AP':  float(average_precision_score(y, scores)),
        'normal_score':   float(scores[y == 0].mean()),
        'abnormal_score': float(scores[y == 1].mean()),
    }

    flat_p, flat_g = maps.reshape(-1), gt.reshape(-1)
    res['PixAUC'] = float(roc_auc_score(flat_g, flat_p))
    res['PixAP']  = float(average_precision_score(flat_g, flat_p))
    best_dice, best_thresh = compute_best_dice(flat_p, flat_g, n_thresh)
    res['BestDice']   = best_dice
    res['BestThresh'] = best_thresh
    res['pixel_prevalence'] = float(flat_g.mean())
    return res


def _selftest_evaluate_maps(size=16, n_norm=6, n_abn=6, seed=0):
    """A map that IS the ground truth must score perfectly; pure noise must not."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_norm + [1] * n_abn)
    masks = np.zeros((n_norm + n_abn, 1, size, size), np.float32)
    for i in range(n_norm, n_norm + n_abn):
        r0, c0 = rng.integers(0, size - 5, 2)
        masks[i, 0, r0:r0 + 5, c0:c0 + 5] = 1.0

    perfect = evaluate_maps(torch.from_numpy(masks), y, torch.from_numpy(masks), n_thresh=20)
    assert perfect['BestDice'] > 0.999, perfect['BestDice']
    assert perfect['PixAP']   > 0.999, perfect['PixAP']
    assert perfect['AUC']     > 0.999, perfect['AUC']

    noise = rng.random((n_norm + n_abn, 1, size, size)).astype(np.float32)
    chance = evaluate_maps(torch.from_numpy(noise), y, torch.from_numpy(masks), n_thresh=20)
    prev = masks.mean()
    assert abs(chance['PixAUC'] - 0.5) < 0.15, chance['PixAUC']
    assert chance['PixAP'] < prev * 3, (chance['PixAP'], prev)
    print(f"  oracle map  -> BestDice {perfect['BestDice']:.3f}  PixAP {perfect['PixAP']:.3f}  "
          f"AUC {perfect['AUC']:.3f}")
    print(f"  noise  map  -> BestDice {chance['BestDice']:.3f}  PixAP {chance['PixAP']:.3f}  "
          f"PixAUC {chance['PixAUC']:.3f}  (prevalence {prev:.3f})")


print('Pixel-metric self-test:')
_selftest_best_dice()
_selftest_evaluate_maps()


# %% [markdown]
# ---
# ## **Cell 3.3** — Training / evaluation driver
# One function runs any registry entry end to end and persists it. Everything the fourteen
# methods differ by is a field on the spec, so there is exactly one training loop.
#
# What it enforces, each preventing a specific failure:
#
# * **`net.eval()` before scoring, always.** Scoring in train mode makes BatchNorm use
#   batch statistics *and* mutate its running statistics, corrupting the weights being
#   saved. This exact bug cost the DL sibling notebook a 0.85-vs-0.95 discrepancy once.
# * **Per-run seeding** of weights, batch order and corruption, so a seed spread measures
#   run-to-run variance rather than leftover global state.
# * **Resolution travels with the method.** DAE trains at 128px; the data for its
#   resolution comes from `get_data`, and `input_size` enters the run id, so a 128px DAE
#   can never be confused with a 64px one.
# * **`load_run` refuses a record** whose stored `config_fingerprint` differs from the
#   current one, so a run from different hyperparameters is never silently reused.
# * **The stored array is the MAP, not the score.** Every image-level number is
#   recoverable from a map by taking the spatial mean, as MedIAnomaly does. Storing scores
#   instead would make re-analysis require a retrain.
#
# **On batched scoring.** MedIAnomaly evaluates with `test batch_size=1`; we batch. For
# every criterion this is identical, with one exception: gradient scores differentiate
# `loss.mean()`, whose gradient w.r.t. each input carries a `1/B` factor. That is a single
# positive constant across the whole test set, and AUC, AP, PixAUC, PixAP and BestDice are
# all rank-based and invariant to it. Only `BestThresh` shifts, which is why `BestThresh`
# must never be compared across batch sizes.

# %% [CELL 3.3]  train_and_eval — one registry entry, end to end

def add_noise(x, noise_res, noise_std, foreground_mask=True):
    """DAE corruption, ported from MedIAnomaly dae_worker.add_noise (Kascenas et al.).

    Coarse noise (16x16) upsampled to full resolution and randomly rolled -- NOT per-pixel
    Gaussian. That is the entire point: coarse noise forces the network to use context to
    repair a blob, which is what makes the learned correction transfer to blob-shaped
    pathology. Per-pixel noise would train a denoiser that generalises to nothing.

    **The foreground mask is not optional for MRI.** dae_worker.py:50-53 applies
    `ns *= (x > x.min())` for datasets 'brain' and 'brats' -- and applies it BEFORE the
    centering, which is easy to misread. The consequence is not "less noise in the
    background": masked pixels get ns=0, then `(0 - 0.5) * 2` sets them to exactly -1.
    So the background is shifted by a DETERMINISTIC -1 while only the foreground carries
    the stochastic corruption.

    Omitting it is not a rounding error. BraTS flair slices are roughly half background,
    so an unmasked DAE spends much of its reconstruction budget learning to denoise air --
    a task the reference model never sees -- and its test-time background residuals are
    correspondingly miscalibrated. Since evaluate_maps flattens the ENTIRE test set,
    background included, that lands straight on PixAP and BestDice, in the one method
    whose whole mechanism is the corruption.

    `x.min()` is taken over the WHOLE BATCH, as theirs is, not per image. With data
    normalised to [-1,1] and black background, it is -1.0 for any batch containing one
    background pixel, so the batch-wide reduction is stable in practice."""
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res),
                      std=noise_std).to(x.device)
    ns = F.interpolate(ns, size=x.shape[-1], mode='bilinear', align_corners=True)
    roll_x = random.choice(range(x.shape[-2]))
    roll_y = random.choice(range(x.shape[-1]))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
    if foreground_mask:                       # dae_worker.py:50-53, for 'brain'/'brats'
        ns = ns * (x > x.min())
    ns = (ns - 0.5) * 2          # their `config.center` branch, always taken here
    return x + ns, ns


def _selftest_add_noise(size=32):
    """Background must receive exactly -1; foreground must not be constant."""
    x = torch.full((2, 1, size, size), -1.0)
    x[:, :, 8:24, 8:24] = 0.5                      # a 'brain' in a black field
    bg = x <= x.min()
    _, ns = add_noise(x, noise_res=8, noise_std=0.2, foreground_mask=True)
    assert torch.allclose(ns[bg], torch.full_like(ns[bg], -1.0), atol=1e-6), \
        f'masked background got {ns[bg].unique()[:5]}, expected exactly -1'
    assert ns[~bg].std() > 0, 'foreground received no stochastic noise'
    _, ns_um = add_noise(x, noise_res=8, noise_std=0.2, foreground_mask=False)
    assert ns_um[bg].std() > 0, 'unmasked path should put noise in the background'
    print(f'  add_noise: masked background == -1 exactly; foreground sd '
          f'{ns[~bg].std():.3f}; unmasked background sd {ns_um[bg].std():.3f}')


print('DAE corruption self-test:')
_selftest_add_noise()


_ERASER = None


def erase_patch(x):
    """CeAE corruption, matching dataload.py's
    RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1).

    Theirs is applied per-sample inside the Dataset; torchvision's RandomErasing samples
    one rectangle for the whole batch when handed a 4-D tensor, so we loop. The difference
    matters: a batch-wide erase would put the hole in the same place for all 64 images in a
    step, which is a materially easier task."""
    global _ERASER
    if _ERASER is None:
        from torchvision.transforms import RandomErasing
        _ERASER = RandomErasing(p=1.0, scale=(CEAE_ERASE_SCALE, CEAE_ERASE_SCALE),
                                ratio=(1.0, 1.0), value=CEAE_ERASE_VALUE)
    return torch.stack([_ERASER(img) for img in x])


def corrupt_input(x, spec):
    """Training-time input corruption. Returns x unchanged for the methods that use none.
    Never called at evaluation: every method is scored on the clean image."""
    mode = spec.get('corrupt')
    if mode is None:
        return x
    if mode == 'noise':
        return add_noise(x, spec['noise_res'], spec['noise_std'],
                         foreground_mask=spec.get('foreground_mask', True))[0]
    if mode == 'erase':
        return erase_patch(x)
    raise KeyError(f'unknown corruption {mode!r}')


def _net_forward(net, x, spec, training):
    """ConstrainedAE needs istrain=True during training to produce z_rec; every other
    network takes a single argument. Centralised so the loop has no per-method branch."""
    if spec.get('istrain') and training:
        return net(x, istrain=True)
    return net(x)


@torch.no_grad()
def _forward_maps(net, criterion, x, batch_size=64):
    """Anomaly MAPS for a whole tensor. net MUST already be in eval mode."""
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        out.append(criterion(xb, net(xb), anomaly_score=True, keepdim=True).cpu())
    return torch.cat(out)


def _forward_maps_grad(net, criterion, x, batch_size=32):
    """Same, for gradient-based scores, which need the graph and input grads."""
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device).requires_grad_(True)
        out.append(criterion(xb, net(xb), anomaly_score=True, keepdim=True).detach().cpu())
    return torch.cat(out)


def compute_maps(net, criterion, x, score_loss):
    """Dispatch to the gradient or no-grad path. net must already be in eval mode."""
    fn = _forward_maps_grad if score_loss in _GRAD_LOSSES else _forward_maps
    return fn(net, criterion, x)


def train_and_eval(method, seed=TRAIN_SEED, *, extra_params=None, epochs=None,
                   train_loss=None, score_loss=None, spec_override=None,
                   save_weights=True, save_maps=True, verbose=True):
    """Run one (method, seed) and persist it. Returns the manifest dict.

    `spec_override` changes registry fields for this run only (e.g. {'noise_res': 32}).
    Every overridden key is folded into the run id, so a swept variant can never collide
    with the registry default -- without that, a noise_res=32 DAE would overwrite the
    noise_res=16 one and the sweep would silently compare a run against itself."""
    if method in EXCLUDED_METHODS:
        raise ValueError(f'{method} is excluded from the pixel study: '
                         f'{EXCLUDED_METHODS[method]}')
    spec = dict(METHODS[method])
    if train_loss is not None: spec['train_loss'] = train_loss
    if score_loss is not None: spec['score_loss'] = score_loss
    if spec_override:
        unknown = set(spec_override) - set(spec)
        assert not unknown, f'spec_override introduces unknown keys {unknown} for {method}'
        spec.update(spec_override)

    size = spec.get('input_size', IMAGE_SIZE)
    bs   = spec.get('batch_size', BATCH_SIZE)

    params = dict(extra_params or {})
    # Anything that changes the numbers but is NOT in config_fingerprint (which only sees
    # globals) must enter the run id here, or two different experiments share a record.
    for k, v in (spec_override or {}).items():
        if k not in ('input_size', 'batch_size'):    # those are covered by 'sz'/bs below
            params[k] = v
    if size != IMAGE_SIZE:
        params['sz'] = size
    if bs != METHODS[method].get('batch_size', BATCH_SIZE):
        params['bs'] = bs
    if epochs is not None and epochs != EPOCHS:
        params['ep'] = epochs

    # --- skip / restore ------------------------------------------------------------
    rid = run_id(method, seed, **params)
    if run_exists(rid) or (USE_WANDB and fetch_run(rid)):
        try:
            man, _ = load_run(rid)
            if verbose:
                print(f"{rid}: reused  PixAP={man['metrics']['PixAP']:.4f} "
                      f"BestDice={man['metrics']['BestDice']:.4f}")
            return man
        except (RuntimeError, FileNotFoundError) as e:
            print(f'{rid}: stored record unusable, retraining\n    {e}')

    if 'perceptual' in (spec['train_loss'], spec['score_loss']) and not HAS_VGG:
        print(f'{rid}: SKIPPED -- perceptual loss needs VGG19 weights (see Cell 3.0b)')
        return None

    # --- data at this method's resolution ------------------------------------------
    x_train, x_test, y_test, masks = get_data(size)

    # --- deterministic setup -------------------------------------------------------
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    net = build_net(spec['net'], input_size=size)
    train_crit = build_loss(spec['train_loss'])
    score_crit = (train_crit if spec['score_loss'] == spec['train_loss']
                  else build_loss(spec['score_loss']))
    opt = Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=EPS)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_train), batch_size=bs, shuffle=True,
                        drop_last=False, generator=g)

    n_epochs = epochs if epochs is not None else EPOCHS
    tuple_loss = spec['train_loss'] in _TUPLE_LOSSES
    n_params = sum(p.numel() for p in net.parameters())
    if verbose:
        extra = f" corrupt={spec['corrupt']}" if spec.get('corrupt') else ''
        print(f"{rid}: {spec['net']} {n_params:,} params | train={spec['train_loss']} "
              f"score={spec['score_loss']} | {size}px bs{bs} {n_epochs} epochs{extra}")

    # --- train ---------------------------------------------------------------------
    epoch_loss, t0 = [], time.time()
    for ep in range(1, n_epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            net_in = corrupt_input(xb, spec)        # corrupted IN ...
            out = _net_forward(net, net_in, spec, training=True)
            loss = train_crit(xb, out)              # ... clean as TARGET
            if tuple_loss:
                loss = loss[0]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        epoch_loss.append(tot / n)
        if verbose and (ep == 1 or ep % max(1, n_epochs // 10) == 0 or ep == n_epochs):
            print(f'    ep {ep:>4}/{n_epochs}  loss {epoch_loss[-1]:.5f}  '
                  f'({time.time() - t0:.0f}s)')

    # --- evaluate ------------------------------------------------------------------
    net.eval()                       # never score in train mode (see markdown)
    maps = compute_maps(net, score_crit, x_test, spec['score_loss'])
    assert maps.shape == x_test.shape, f'map shape {maps.shape} != input {x_test.shape}'

    metrics = evaluate_maps(maps, y_test, masks)
    metrics['train_loss_final'] = epoch_loss[-1]
    metrics['n_params'] = n_params
    metrics['input_size'] = size
    metrics['minutes'] = (time.time() - t0) / 60
    if verbose:
        print(f"    -> PixAP {metrics['PixAP']:.4f}  BestDice {metrics['BestDice']:.4f}  "
              f"PixAUC {metrics['PixAUC']:.4f} | AUC {metrics['AUC']:.4f}  "
              f"AP {metrics['AP']:.4f}  ({metrics['minutes']:.1f} min)")

    arrays = {'labels': y_test}
    if save_maps:
        # float32, not float16. BestDice's thresholds are order statistics of these very
        # values; half precision collapses ties the metric can see.
        arrays['maps'] = maps.numpy().astype(np.float32)
    return save_run(rid, method=method, seed=seed,
                    params={**params, 'train_loss': spec['train_loss'],
                            'score_loss': spec['score_loss']},
                    metrics=metrics, epoch_loss=epoch_loss, arrays=arrays,
                    weights={'net': net.state_dict()} if save_weights else None,
                    extra={'spec': {k: v for k, v in spec.items()}})


def rescore(base_method, score_loss, seed=TRAIN_SEED, *, name=None, verbose=True):
    """Re-score ALREADY-TRAINED weights with a different anomaly-map rule.

    Identical weights, a different map, so any change in the pixel metrics is
    attributable to the scoring rule and to nothing else. This is the one mechanism that
    separates the training objective from the scoring rule without confounding them."""
    spec = dict(METHODS[base_method])
    size = spec.get('input_size', IMAGE_SIZE)
    hits = find_run(base_method, seed)
    if not hits:
        raise FileNotFoundError(
            f'no stored {base_method!r} run at seed {seed} -- '
            f'run train_and_eval({base_method!r}, seed={seed}) first')
    if len(hits) > 1:
        raise RuntimeError(
            f'{len(hits)} stored {base_method!r} runs at seed {seed}: '
            f"{[h['run_id'] for h in hits]}. Pass the one you mean via name=, or clear "
            'the others -- silently picking one would make this result unattributable.')
    base_rid = hits[0]['run_id']

    rid = name or run_id(f'{base_method}-rescored', seed, score=score_loss)
    if run_exists(rid):
        m, _ = load_run(rid)
        if verbose:
            print(f"{rid}: reused  PixAP={m['metrics']['PixAP']:.4f}")
        return m

    net = build_net(spec['net'], input_size=size)
    man, _ = load_run(base_rid, models={'net': net})   # load_run leaves net in eval mode
    _, x_test, y_test, masks = get_data(size)

    crit = build_loss(score_loss)
    t0 = time.time()
    maps = compute_maps(net, crit, x_test, score_loss)
    metrics = evaluate_maps(maps, y_test, masks)
    metrics['minutes'] = (time.time() - t0) / 60
    metrics['n_params'] = man['metrics'].get('n_params')
    metrics['input_size'] = size
    if verbose:
        print(f"{rid}: weights from {base_rid}, scored with {score_loss!r}\n"
              f"    -> PixAP {metrics['PixAP']:.4f}  BestDice {metrics['BestDice']:.4f}  "
              f"PixAUC {metrics['PixAUC']:.4f}")
    return save_run(rid, method=f'{base_method}-rescored', seed=seed,
                    params={'base': base_method,
                            'train_loss': spec['train_loss'], 'score_loss': score_loss},
                    metrics=metrics,
                    arrays={'maps': maps.numpy().astype(np.float32), 'labels': y_test},
                    weights=None,          # the weights belong to the base run
                    extra={'base_run': base_rid})


print('Driver ready: train_and_eval(method, seed) / rescore(base_method, score_loss, seed)')

# %% [markdown]
# ---
# ## **Cell 3.4** — Preflight
# Everything above this point defines things and self-tests them; nothing above touches
# the study. This cell answers the one question that is otherwise easy to get wrong after
# a Run All: **is this notebook actually ready, and what has already been done?**
#
# It exists because the self-tests in Cells 3.0-3.3 pass whether or not the data loaded.
# A failed Cell 2.0 followed by a wall of green self-test output reads like success.

# %% [CELL 3.4]  Preflight — state, and what will run

def preflight():
    ok = True
    print('DATA')
    have_data = 'X_TRAIN' in globals() and X_TRAIN is not None
    if have_data:
        print(f'  root      {DATA_ROOT}')
        print(f'  train     {tuple(X_TRAIN.shape)}')
        print(f'  test      {tuple(X_TEST.shape)}  '
              f'{int((Y_TEST == 0).sum())} normal / {int((Y_TEST == 1).sum())} abnormal')
        print(f'  lesions   {PIXEL_PREVALENCE*100:.3f}% of test pixels '
              f'(= the PixAP a random scorer earns)')
    else:
        ok = False
        print('  NOT LOADED — Cell 2.0/2.2 did not complete. The self-tests above pass')
        print('  without data, so green output there does not mean the notebook is ready.')

    print('\nCOMPUTE')
    print(f'  device    {device}' + (f'  ({torch.cuda.get_device_name(0)})'
                                     if device.type == 'cuda' else ''))
    if device.type != 'cuda' and not SAMPLE_MODE:
        ok = False
        print('  !! NO GPU. This grid is not feasible on CPU. For scale: one epoch of')
        print('     ae-perceptual measured ~640s on CPU, so its 250 epochs alone are ~44')
        print('     HOURS, and dae is several times heavier again. On Kaggle, enable an')
        print('     accelerator (Settings -> Accelerator -> GPU) and re-run. Set')
        print('     SAMPLE_MODE=1 if you only want to smoke-test the code path.')

    print('\nPLAN')
    total = sum(len(seeds_for(m)) for m in ACTIVE_METHODS)
    _ord = sorted(ACTIVE_METHODS, key=lambda m: (method_cost(m), m))
    print(f'  {len(ACTIVE_METHODS)} active methods, {total} runs, cheapest first:')
    print('    ' + ', '.join(f'{m}x{len(seeds_for(m))} (cost {method_cost(m):g})'
                             for m in _ord))
    print(f'  epochs {EPOCHS} | {IMAGE_SIZE}px (DAE {METHODS["dae"].get("input_size")}px) '
          f'| SAMPLE_MODE={SAMPLE_MODE}')
    if SAMPLE_MODE:
        print('  !! SAMPLE_MODE is ON — tiny subsets and 2 epochs. Results are')
        print('     smoke-test artefacts, NOT comparable to Table 7.')

    print('\nALREADY DONE')
    stored = stored_runs_by_method()
    if not stored:
        print('  nothing stored yet — a full run starts from scratch')
    for m in sorted(stored):
        done = len(stored[m])
        want = len(seeds_for(m)) if m in ACTIVE_METHODS else 0
        print(f'  {m:<18} {done} stored' + (f' / {want} planned' if want else ''))

    print('\nWILL EXECUTE')
    print(f'  Cell 4.2  reproduction   RUN_REPRODUCTION={RUN_REPRODUCTION}')
    print(f'  Cell 5.3  experiments    RUN_EXPERIMENTS={RUN_EXPERIMENTS}')
    if not (RUN_REPRODUCTION or RUN_EXPERIMENTS):
        print('  -> both are False, so Run All will define everything and compute nothing.')
    print('\n' + ('READY' if ok else 'NOT READY — fix DATA above before running'))
    return ok


preflight()


# %% [markdown]
# ---
# ## **Cell 4.0** — The grid
# Every active method at every one of its seeds. `train_and_eval` skips anything already
# stored, so this cell is safe to re-run after a Kaggle session dies: it picks up where
# the last one stopped rather than retraining from scratch. That is the whole reason the
# run store exists.
#
# Order matters for a session-limited machine. Cheap methods run first, so a session that
# dies early still leaves you a usable partial table; DAE — one 31M-parameter UNet at
# 128px, the single most expensive run here — goes last, when everything else is banked.

# %% [CELL 4.0]  Run the grid

def run_grid(methods=None, verbose=True):
    """Train every active method at every seed. Returns the list of manifests.

    Re-entrant: stored runs are reused, so re-running after a crash resumes."""
    methods = methods or ACTIVE_METHODS
    # Cheapest first, so an interrupted session still leaves a usable partial table.
    #
    # An earlier version sorted on (input_size, is_unet) and called that "cheapest first".
    # It is not: every 64px method ties, so the order fell back to registry order and
    # ae-perceptual -- the most expensive 64px method by an order of magnitude -- ran
    # FIRST. A session that died early therefore banked nothing. Sort on actual cost.
    order = sorted(methods, key=lambda m: (method_cost(m), m))
    plan = [(m, s) for m in order for s in seeds_for(m)]
    print(f'{len(plan)} runs planned: ' +
          ', '.join(f'{m}x{len(seeds_for(m))}' for m in order) + '\n')

    out, t0 = [], time.time()
    for k, (m, sd) in enumerate(plan, 1):
        print(f'[{k}/{len(plan)}] {m} seed={sd}')
        try:
            man = train_and_eval(m, seed=sd, verbose=verbose)
            if man is not None:
                out.append(man)
        except Exception as e:
            # One method failing must not cost the whole grid. Report and continue.
            print(f'  !! {m} seed={sd} FAILED -- {type(e).__name__}: {e}')
        print(f'  elapsed {(time.time() - t0)/60:.1f} min\n')
    print(f'grid complete: {len(out)}/{len(plan)} runs in {(time.time() - t0)/60:.1f} min')
    return out


# %% [markdown]
# ---
# ## **Cell 4.1** — The table
# Our numbers next to MedIAnomaly's Table 7, with the gap in units of THEIR reported
# standard deviation. That last column is the one to read: a gap of 0.4 sd is a
# reproduction, a gap of 6 sd is a bug in our harness or a difference in protocol we have
# not found yet. Comparing raw point estimates without it invites reading a 2-point
# difference as meaningful when their own repeats span 5.
#
# `⌈Dice⌉` is a CEILING, not an achievable score: Cell 3.2 selects its threshold on the
# test masks. It is reported because they report it, and it must be named as an oracle
# wherever it appears in the write-up.

# %% [CELL 4.1]  Results against MedIAnomaly Table 7

def results_table(methods=None):
    """Aggregate stored runs into (mean, sd) per method and compare to their Table 7."""
    methods = methods or ACTIVE_METHODS
    stored = stored_runs_by_method()
    rows = []
    for m in methods:
        vals = {'PixAP': [], 'BestDice': [], 'PixAUC': [], 'AUC': [], 'AP': []}
        n = 0
        for man in stored.get(m, []):
            mt = man['metrics']
            if not all(k in mt for k in vals):
                continue
            for k in vals:
                vals[k].append(mt[k] * 100)
            n += 1
        if n == 0:
            rows.append({'method': m, 'n': 0}); continue

        tgt_ap, tgt_dc = TARGET_BRATS_PIXEL.get(m, (None, None))
        row = {'method': m, 'n': n}
        for k in vals:
            row[k] = float(np.mean(vals[k]))
            row[k + '_sd'] = float(np.std(vals[k], ddof=1)) if n > 1 else float('nan')
        for k, tgt in (('PixAP', tgt_ap), ('BestDice', tgt_dc)):
            if tgt is None:
                row[k + '_target'] = float('nan'); row[k + '_gap_sd'] = float('nan')
            else:
                row[k + '_target'] = tgt[0]
                # gap in units of THEIR sd; guard a zero sd rather than dividing by it
                row[k + '_gap_sd'] = (row[k] - tgt[0]) / tgt[1] if tgt[1] > 0 else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def print_results_table(methods=None):
    df = results_table(methods)
    print(f"{'method':<15} {'n':>2}  {'AP_pix':>14}  {'target':>10}  {'gap':>7}   "
          f"{'Dice_ceil':>14}  {'target':>10}  {'gap':>7}")
    print('-' * 92)
    for _, r in df.iterrows():
        if r.get('n', 0) == 0:
            print(f"{r['method']:<15} {'0':>2}  (not run yet)"); continue
        def fmt(k):
            sd = r[k + '_sd']
            return f"{r[k]:.1f}+-{sd:.1f}" if np.isfinite(sd) else f"{r[k]:.1f}"
        def tgt(k):
            t = r[k + '_target']
            return f'{t:.1f}' if np.isfinite(t) else '--'
        def gap(k):
            g = r[k + '_gap_sd']
            return f'{g:+.1f}sd' if np.isfinite(g) else '--'
        print(f"{r['method']:<15} {int(r['n']):>2}  {fmt('PixAP'):>14}  {tgt('PixAP'):>10}  "
              f"{gap('PixAP'):>7}   {fmt('BestDice'):>14}  {tgt('BestDice'):>10}  "
              f"{gap('BestDice'):>7}")
    print('\n  gap = (ours - theirs) / their sd.  |gap| < 2 is a reproduction;')
    print('  a large gap means a protocol difference or a bug, not a finding.')
    print('  Dice_ceil is an ORACLE threshold chosen on the test masks (Cell 3.2).')
    return df


# %% [markdown]
# ---
# ## **Cell 4.2** — Run the reproduction
# The cells above only DEFINE things. This one executes. `RUN_REPRODUCTION` (Cell 1.4)
# gates it so the notebook can be loaded for inspection without launching hours of
# training, but it defaults to True: running the cells should run the study.
#
# Re-running is cheap. `train_and_eval` reuses any stored run, so a second pass over a
# finished grid prints the table in seconds.

# %% [CELL 4.2]  EXECUTE — the reproduction grid and its table

if RUN_REPRODUCTION:
    _grid = run_grid()
    print()
    _df_repro = print_results_table()
else:
    print('RUN_REPRODUCTION is False -- nothing executed.')
    print('  Set it True in Cell 1.4, or call run_grid() then print_results_table().')


# %% [markdown]
# ---
# # Part 5 — The two experiments
# Reproduction establishes the harness. These two ask something the benchmark does not.
# Both descend from the same finding in the DL sibling project: **a spatial-scale
# hyperparameter that nobody ablates can silently control the sign and size of a reported
# effect.** There, it was the Gaussian bandwidth inside an SSIM score. Here it appears
# twice, and neither instance has been ablated in print.
#
# ### E1 — Is DAE's win an architecture, or a scale match?
# DAE beats second place by +30.7 AP_pix. MedIAnomaly attributes this substantially to its
# "customised UNet". But DAE's corruption is
# `add_noise(x, noise_res=16, noise_std=0.2)`: Gaussian noise sampled on a 16x16 grid,
# bilinearly upsampled, randomly rolled. The network is therefore trained to repair blobs
# of ONE relative size — 1/16 of the image width — and `noise_res` is never ablated in the
# benchmark.
#
# **H1: DAE's advantage is a match between its corruption scale and the lesion size
# distribution, not primarily an architectural property.**
#
# Prediction if H1 holds: pixel performance is strongly non-monotone in `noise_res`, and
# the optimum **shifts with lesion size** when the test set is stratified. Prediction if
# H1 fails: performance is broadly flat, the architecture story stands, and DAE is robust
# to a knob nobody tuned — which would itself be worth reporting.
#
# The stratified version is what makes this a manipulation rather than a correlation: we
# are not observing that big lesions are easier, we are asking whether the corruption
# scale that wins *moves* when the target scale moves.
#
# ### E2 — Why is the best detector the worst localiser?
# AE-U is near the top image-level in this benchmark (86.5 AUROC on RSNA, Table 6) and
# **last** pixel-level on BraTS (22.2 AP_pix, Table 7). Its map is not a residual; it is a
# residual divided by a learned per-pixel variance:
# `loss1 = exp(-log_var) * (x - x_hat)**2`.
#
# **H2: the learned variance suppresses exactly the regions where lesions are.** The
# network learns to be uncertain where the image is hard to reconstruct, and lesions are
# hard, so the denominator discounts the evidence.
#
# This costs no training at all. It is the DL project's own method — hold the weights
# fixed, change only the scoring rule — applied to pixels: score the trained AE-U with
# plain `l2` and see whether *removing* its uncertainty weighting improves localisation.
# If it does, AE-U's uncertainty term is trading localisation for detection, and the
# benchmark's two tables are measuring a real trade-off rather than a quirk.

# %% [CELL 5.0]  Lesion-size strata — shared by both experiments

def lesion_strata(masks, y, n_strata=3):
    """Split the ABNORMAL test images into equal-count bins by lesion area.

    Returns (index_arrays, labels). Normal images are excluded from the strata because
    they have no lesion size; they are re-added to every stratum's evaluation so that the
    negative-pixel pool stays comparable across strata. Without that, a stratum with fewer
    images would be scored against a smaller background and its AP would not be comparable
    to its neighbours'."""
    y = np.asarray(y)
    m = masks.numpy() if torch.is_tensor(masks) else np.asarray(masks)
    area = m.reshape(len(m), -1).sum(axis=1)          # lesion pixels per image
    abn = np.where(y == 1)[0]
    order = abn[np.argsort(area[abn])]
    chunks = np.array_split(order, n_strata)
    labels = [f'{lo}-{hi} px' for lo, hi in
              ((int(area[c].min()), int(area[c].max())) for c in chunks)]
    return chunks, labels


def evaluate_by_stratum(maps, y, masks, n_strata=3, n_thresh=200):
    """evaluate_maps on each lesion-size stratum, normals included in every one."""
    y = np.asarray(y)
    normals = np.where(y == 0)[0]
    chunks, labels = lesion_strata(masks, y, n_strata)
    rows = []
    for idx, lab in zip(chunks, labels):
        sel = np.concatenate([normals, idx])
        r = evaluate_maps(maps[sel], y[sel], masks[sel], n_thresh=n_thresh)
        r['stratum'] = lab
        r['n_abnormal'] = len(idx)
        rows.append(r)
    return pd.DataFrame(rows)


# %% [markdown]
# ## **Cell 5.1** — E1: the DAE noise-scale sweep
# `noise_res` is swept; everything else is held fixed. Blob width is `input_size /
# noise_res` pixels, i.e. `1/noise_res` of the image — a RELATIVE scale, which is what
# makes the sweep resolution-independent and lets us run it cheaply.
#
# **The scales, measured on the real data (not assumed).** BraTS2021 lesion areas at 64px
# fall in terciles of 12-112, 113-226 and 227-616 px^2 -- equivalent square widths of
# roughly 3.5-10.6, 15 and 24.8 px. DAE's default corruption is noise_res=16 at 128px
# input, a blob 8px wide, which is 4px in these 64px-equivalent units.
#
# So their default blob sits at the SMALL end of the lesion distribution -- narrower than
# the median lesion in every tercile. That sharpens H1 into a directional prediction:
# LOWERING noise_res (bigger blobs) should help, and should help MOST on the large-lesion
# tercile. If instead performance is flat, or peaks at their default, H1 is wrong and DAE's
# win is not a scale match. The sweep values below bracket the lesion range: at 64px,
# noise_res 4/8/16/32 gives blobs of 16/8/4/2 px against lesion widths of ~3.5-25 px.
#
# **Deliberate protocol deviation, stated up front.** The sweep runs at
# `SWEEP_INPUT_SIZE` (64px), not the 128px their `train_eval.sh` uses for DAE. That makes
# each run about four times cheaper. It is legitimate here and not in the reproduction:
# every cell of the sweep is identical except `noise_res`, so the comparison is internal.
# The absolute numbers from this cell must NOT be compared to Table 7's 75.5 — only to
# each other. `sweep_baseline()` re-runs their `noise_res=16` at 64px so the sweep has its
# own in-protocol reference point.

# %% [CELL 5.1]  E1 — DAE noise-scale sweep

SWEEP_INPUT_SIZE = 64
SWEEP_NOISE_RES  = [4, 8, 16, 32]     # blob = 1/4, 1/8, 1/16, 1/32 of image width
SWEEP_SEED       = 42


def run_noise_sweep(noise_values=None, seed=SWEEP_SEED, size=SWEEP_INPUT_SIZE,
                    epochs=None, verbose=False):
    """Train DAE once per noise_res. Re-entrant: stored runs are reused."""
    noise_values = noise_values or SWEEP_NOISE_RES
    print(f'DAE noise sweep at {size}px, seed {seed}: noise_res {noise_values}')
    print(f'  blob width = {size}/noise_res px = ' +
          ', '.join(f'{size//r}px' for r in noise_values) + '\n')
    out = []
    for r in noise_values:
        man = train_and_eval('dae', seed=seed, epochs=epochs, verbose=verbose,
                             spec_override={'noise_res': r, 'input_size': size,
                                            'batch_size': 16})
        if man is not None:
            man['_noise_res'] = r
            out.append(man)
            print(f"  noise_res {r:>3} (blob {size//r:>3}px)  "
                  f"PixAP {man['metrics']['PixAP']*100:5.1f}  "
                  f"Dice {man['metrics']['BestDice']*100:5.1f}  "
                  f"AUC {man['metrics']['AUC']*100:5.1f}")
    return out


def noise_sweep_table(noise_values=None, seed=SWEEP_SEED, size=SWEEP_INPUT_SIZE):
    """Sweep results, overall and per lesion-size stratum. The stratified columns are the
    test of H1: if the winning noise_res is the same in every stratum, H1 is wrong."""
    noise_values = noise_values or SWEEP_NOISE_RES
    _, x_test, y_test, masks = get_data(size)
    rows = []
    for r in noise_values:
        hits = find_run('dae', seed, noise_res=r)
        if not hits:
            print(f'  noise_res {r}: not run yet'); continue
        man, arrays = load_run(hits[0]['run_id'])
        maps = torch.from_numpy(arrays['maps'])
        overall = evaluate_maps(maps, y_test, masks)
        strata = evaluate_by_stratum(maps, y_test, masks)
        row = {'noise_res': r, 'blob_px': size // r,
               'PixAP': overall['PixAP'] * 100, 'BestDice': overall['BestDice'] * 100,
               'AUC': overall['AUC'] * 100}
        for _, sr in strata.iterrows():
            row[f"PixAP[{sr['stratum']}]"] = sr['PixAP'] * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        print(df.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
        strat_cols = [c for c in df.columns if c.startswith('PixAP[')]
        if strat_cols:
            best = {c: int(df.loc[df[c].idxmax(), 'noise_res']) for c in strat_cols}
            print(f'\n  best noise_res per lesion-size stratum: {best}')
            print('  H1 predicts these DIFFER (corruption scale tracks lesion scale).')
            print('  If they are all equal, H1 is not supported by this sweep.')
    return df


# %% [markdown]
# ## **Cell 5.2** — E2: does AE-U's uncertainty destroy its own localisation?
# No training. `rescore` loads the trained AE-U weights and scores them with plain `l2`,
# so the ONLY difference between the two rows is the presence of the `exp(-log_var)`
# factor. Any change in the pixel metrics is attributable to that factor and to nothing
# else — the same weights produced both maps.
#
# Read the `AUC` column alongside `PixAP`. H2's interesting form is not "the variance term
# is bad", it is that the variance term helps DETECTION while hurting LOCALISATION — one
# knob, two tables, opposite signs.

# %% [CELL 5.2]  E2 — AE-U variance decomposition

def aeu_variance_decomposition(seed=TRAIN_SEED):
    """AE-U scored with its own uncertainty-weighted rule vs the plain residual."""
    base = train_and_eval('aeu', seed=seed, verbose=False)
    if base is None:
        print('aeu has not been trained yet'); return None
    plain = rescore('aeu', 'l2', seed=seed, verbose=False)

    rows = []
    for tag, man, rule in (('AE-U (theirs)', base, 'exp(-log_var) * (x-x_hat)^2'),
                           ('AE-U, no 1/var', plain, '(x-x_hat)^2')):
        m = man['metrics']
        rows.append({'scoring rule': tag, 'formula': rule,
                     'PixAP': m['PixAP'] * 100, 'BestDice': m['BestDice'] * 100,
                     'PixAUC': m['PixAUC'] * 100, 'AUC': m['AUC'] * 100,
                     'AP': m['AP'] * 100})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

    d_pix = df.loc[1, 'PixAP'] - df.loc[0, 'PixAP']
    d_img = df.loc[1, 'AUC'] - df.loc[0, 'AUC']
    print(f'\n  removing the variance weighting: PixAP {d_pix:+.1f}, image AUC {d_img:+.1f}')
    if d_pix > 0 and d_img < 0:
        print('  -> H2 SUPPORTED: the uncertainty term buys detection and costs '
              'localisation.\n     Same weights, one scoring factor, opposite signs on '
              'the two tables.')
    elif d_pix > 0:
        print('  -> the variance term hurts localisation, but detection did not improve '
              'with it either;\n     H2 as stated (a trade-off) is not what is happening.')
    else:
        print('  -> H2 NOT supported: the variance weighting does not explain AE-U\'s '
              'poor localisation.\n     Look elsewhere -- the reconstruction itself, or '
              'the bottleneck.')
    return df



# %% [markdown]
# ---
# ## **Cell 5.3** — Run the experiments
# E2 is nearly free: it reuses the trained AE-U and only re-scores it. E1 trains one DAE
# per `noise_res`, so it is the expensive half — it runs second, and only after the
# reproduction has had a chance to show whether our DAE is trustworthy in the first place.

# %% [CELL 5.3]  EXECUTE — E2 then E1

if RUN_EXPERIMENTS:
    print('=' * 78)
    print('E2 — does AE-U\'s uncertainty weighting destroy its own localisation?')
    print('=' * 78)
    _df_e2 = aeu_variance_decomposition()

    print('\n' + '=' * 78)
    print('E1 — is DAE\'s win an architecture, or a corruption-scale match?')
    print('=' * 78)
    _sweep = run_noise_sweep()
    print()
    _df_e1 = noise_sweep_table()
else:
    print('RUN_EXPERIMENTS is False -- nothing executed.')
    print('  Set it True in Cell 1.4, or call aeu_variance_decomposition(), '
          'run_noise_sweep(), noise_sweep_table().')