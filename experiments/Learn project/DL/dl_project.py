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

RUN_VERSION    = 'dl-v1'
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
REQUIRED_DATASETS = ['RSNA']


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

x_train_np, x_test_np, y_test_np = load_split_imagelevel(DATA_ROOT, 'RSNA')

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
A3_MEMBERS = [('l2', 'ae'), ('ssim', 'ae-ssim'), ('perceptual', 'ae-pl'), ('aeu', 'aeu')]
A3_PREREGISTERED = {
    'all4': ['l2', 'ssim', 'perceptual', 'aeu'],
    'noIN': ['l2', 'ssim', 'aeu'],          # no ImageNet weights anywhere in this one
}


def _ranks(x):
    """Ranks in [0,1]. argsort().argsort() is the rank; dividing makes members with
    different test-set sizes comparable and keeps the mean interpretable."""
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r / max(len(r) - 1, 1)


def load_member_scores(seed):
    """Per-image scores for each ensemble member at one seed. Returns {} if any member
    is missing, because a partial ensemble is not the ensemble we pre-registered."""
    out = {}
    for label, method in A3_MEMBERS:
        rid = run_id(method, seed)
        if not run_exists(rid) and not (USE_WANDB and fetch_run(rid)):
            print(f'  seed {seed}: member {label!r} ({rid}) missing — skipping this seed')
            return {}
        _, arrays = load_run(rid)
        out[label] = arrays['scores']
    return out


def a3_ensemble(seeds=None):
    seeds = seeds or SEEDS
    rows = []
    for s in seeds:
        members = load_member_scores(s)
        if not members:
            continue
        ranked = {k: _ranks(v) for k, v in members.items()}
        labels = [l for l, _ in A3_MEMBERS]
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
            members = load_member_scores(s)
            sc = np.mean([_ranks(members[c]) for c in combo], axis=0)
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
    rid = run_id(f'{base_method}-posthoc-u', seed, w=_wtag(width))
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
    for _w in A4_WIDTHS:
        sub = _a4[(_a4.method == 'ae-posthoc-u') & (_a4.get('w', pd.Series(dtype=float)) == _w)] \
              if 'w' in _a4 else _a4[_a4.method == 'ae-posthoc-u']
        if sub.empty:
            continue
        m, sd = sub.AUC.mean() * 100, sub.AUC.std(ddof=0) * 100
        tag = 'capacity-matched' if _w == 1 else 'capacity control '
        print(f'  frozen AE + var head (w={_w}, {tag}) {m:6.2f}+-{sd:.2f}   '
              f'({m - ae_m:+.2f} over the AE it was fitted onto)')
    print(f'  AE-U, jointly trained                {au_m:6.2f}+-{au_s:.2f}')
    gap = au_m - ae_m
    ph = _a4[_a4.method == 'ae-posthoc-u']
    if gap > 0 and not ph.empty:
        best = ph.AUC.max() * 100
        print(f'\n  best post-hoc head recovers {(best - ae_m) / gap * 100:.0f}% of AE-U\'s '
              f'+{gap:.2f} advantage, with the reconstruction never retrained.')
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

def a3_split_half(seeds=None, n_splits=50, rng_seed=SPLIT_SEED if 'SPLIT_SEED' in dir() else 0):
    seeds = seeds or SEEDS
    labels = [l for l, _ in A3_MEMBERS]
    all_combos = [c for r in range(1, len(labels) + 1) for c in combinations(labels, r)]
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in seeds:
        members = load_member_scores(s)
        if not members:
            continue
        ranked = {k: _ranks(v) for k, v in members.items()}
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

        rows.append({'seed': s, 'full': full['AUC'], 'static': static['AUC'],
                     'scalar': scalar['AUC'], 'whiten': whiten['AUC'],
                     'frac_static': frac_static})
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
        ph_rows   = _m2[_m2.method == f'{b}-posthoc-u']
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


def _scores_for(rid):
    """Per-image score vector for a stored run, fetched from wandb if not local."""
    if not run_exists(rid) and not (USE_WANDB and fetch_run(rid)):
        return None
    try:
        _, arrays = load_run(rid)
    except (RuntimeError, FileNotFoundError):
        return None
    return arrays.get('scores')


# Headline comparisons, each stated as (label, run-id builder A, run-id builder B).
# Every one of these is currently reported as an arithmetic difference somewhere.
M3_COMPARISONS = [
    ('ensemble(perc+unc) vs AE-PL',
     lambda s: run_id('ens-all4', s), lambda s: run_id('ae-pl', s)),
    ('post-hoc head (w=2) vs AE-SSIM',
     lambda s: run_id('ae-posthoc-u', s, w='2'), lambda s: run_id('ae-ssim', s)),
    ('post-hoc head (w=2) vs plain AE',
     lambda s: run_id('ae-posthoc-u', s, w='2'), lambda s: run_id('ae', s)),
    ('AE-U vs post-hoc head (w=2)',
     lambda s: run_id('aeu', s), lambda s: run_id('ae-posthoc-u', s, w='2')),
    ('AE rescored with SSIM vs AE',
     lambda s: run_id('a1-l2-ssim', s), lambda s: run_id('ae', s)),
]


def m3_delong(comparisons=None, seeds=None):
    comparisons = comparisons or M3_COMPARISONS
    seeds = seeds or SEEDS
    rows = []
    for label, ra, rb in comparisons:
        for s in seeds:
            sa, sb = _scores_for(ra(s)), _scores_for(rb(s))
            if sa is None or sb is None:
                continue
            r = delong_test(sa, sb, Y_TEST)
            rows.append({'comparison': label, 'seed': s, **r})
    return pd.DataFrame(rows)


M3 = m3_delong()

if not M3.empty:
    print('\nM3 — DeLong PAIRED tests on the same 2000 test images\n')
    print(f"  {'comparison':<34}{'diff (pts)':>13}{'95% CI':>20}{'p':>10}{'':>4}")
    for label, g in M3.groupby('comparison', sort=False):
        d = g['diff'].mean() * 100
        lo, hi = g.ci_lo.mean() * 100, g.ci_hi.mean() * 100
        p = g['p'].max()                       # most conservative across seeds
        mark = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'  {label:<34}{d:>+13.2f}[{lo:>+7.2f},{hi:>+7.2f}]{p:>10.2g}  {mark}')
    print('\n  p is the LEAST significant across seeds (conservative).')
    # Compare the paired half-widths against the UNPAIRED Hanley-McNeil bound rather
    # than asserting which is tighter -- the whole point is to measure it, and the
    # paired advantage depends on how correlated the two scores actually are.
    hw = ((M3.ci_hi - M3['diff']).mean()) * 100
    print(f'  mean paired 95% half-width : {hw:.2f} pts')
    print(f'  unpaired Hanley-McNeil bound: 1.48 pts (AUC 0.887, n=1000/1000)')
    print(f'  -> paired interval is {"TIGHTER" if hw < 1.48 else "WIDER"}; '
          f'apply the ~1.5-point rule only to UNPAIRED comparisons.')


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
