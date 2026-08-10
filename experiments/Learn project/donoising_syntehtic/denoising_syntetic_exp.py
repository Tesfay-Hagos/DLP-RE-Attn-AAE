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
    pkg          : the name you 'import' in code (e.g. 'sklearn', 'pydicom')
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
    'seaborn':    {'install_name': None,            'min_version': None},
    'pydicom':    {'install_name': None,            'min_version': '2.3.0'}
}

for pkg, spec in REQUIRED_PACKAGES.items():
    check_import(pkg, install_name=spec['install_name'], min_version=spec['min_version'])



# %% [markdown]
# # C1.3 Import the required packages for the project
# %% [CELL 1.3] import the required packages for the project
import os, time, json, random, warnings
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
import pydicom

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

PAL = {
    'C1': '#4878CF',   # blue      — CNN-AE baseline
    'C2': '#F5A623',   # orange    — VAE baseline
    'C3': '#7B68EE',   # slate     — CNN-AAE ablation
    'C4': '#E84C3D',   # red       — RE-Attn-AAE (novel)
    'C5': '#95A5A6',   # grey      — ResNet frozen (failure case)
    'C6': '#2ECC71',   # green     — ResNet partial fine-tune
    'C7': '#1A5276',   # dark blue — ResNet mostly fine-tuned
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == 'cuda':
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True





# %% [markdown]
# # C1.4 Define the configuration for the project  this includes
# %% [markdown]
# Define the configuration for the project  this includes 
# the project constant hyperparameter and constant values used throughout the project.

# All hyperparameters are defined in one place so comparisons across conditions are fair.
# Every condition uses **identical** `EPOCHS`, `WARMUP_EPOCHS`, `LR`, `BATCH_SIZE`, and `LATENT_DIM`.
#
# | Parameter | Value | Notes | 
# |-----------|-------|-------| 
# | `IMAGE_SIZE` | 128 | Downsampled from 1024 × 1024 DICOM |
# | `LATENT_DIM` | 128 | Shared across all encoders |
# | `LR` | 1e-4 | Adam with cosine annealing |
# | `EPOCHS` | 80 | Main training phase per condition |
# | `WARMUP_EPOCHS` | 20 | Reconstruction-only warm-start for C3/C4/C5 — 20 epochs critical to prevent discriminator collapse |
# | `LAMBDA_ADV` | 0.3 | Weight of adversarial generator loss |
# | `BATCH_SIZE` | 32 | Per-GPU mini-batch size |
#
# Set environment variable `SAMPLE_MODE=1` to run a minimal smoke-test
# (2 epochs, small data) without touching the real dataset.



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


# %% [CELL 1.4]  define the configuration for the project  this includes
SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))

# ── Version + skip control (mirrors bone_fracture_kaggle.py) ─────────
# Bump RUN_VERSION to force a full re-run (old checkpoints are ignored).
# Set SKIP_COMPLETED=False to retrain within the same version.
RUN_VERSION    = 'v3.1'
SKIP_COMPLETED = True
WANDB_PROJECT  = 'RE-Attn-AAE-RSNA'
WANDB_GROUP    = f'ablation-{RUN_VERSION}'   # groups all 7 conditions under one experiment

BASE       = '/kaggle/input/competitions/rsna-pneumonia-detection-challenge'
TRAIN_DIR  = f'{BASE}/stage_2_train_images'
OUTPUT_DIR = '/kaggle/working/results_rsna_resnet' if not SAMPLE_MODE else 'results_rsna_resnet_sample'
CKPT_DIR   = f'{OUTPUT_DIR}/ckpt_{RUN_VERSION}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

IMAGE_SIZE    = 128
ORIG_SIZE     = 1024
FLAT_DIM      = IMAGE_SIZE * IMAGE_SIZE
LATENT_DIM    = 128
LR            = 1e-4
BETA1         = 0.5
EPOCHS        = 80  if not SAMPLE_MODE else 2
WARMUP_EPOCHS = 20  if not SAMPLE_MODE else 1
LAMBDA_ADV    = 0.3
LAMBDA_REC2   = 0.5   # C3.5/C4: weight of the second (attention-gated) reconstruction loss
LAMBDA_AE     = 0.3   # C4 (rejected): weight of the attention-expansion regularizer mean(1-att)
LAMBDA_SEG    = 1.0   # C3.5: weight of the BCE mask-supervision loss against synthetic anomalies
BATCH_SIZE    = 32  if not SAMPLE_MODE else 4
EPS           = 1e-8
# MedIAnomaly RSNA protocol (Sec 3.1.1 / Table 2): 3851 normal TRAIN,
# TEST = 1000 normal + 1000 abnormal. Matching it is what makes our numbers
# directly comparable to their Table 6 (AE-PL 87.5 / AE-U 86.5 / GANomaly 80.0).
# The old 2000+2000 split is NOT comparable.
TEST_NORMAL   = 1000 if not SAMPLE_MODE else 10
TEST_OPACITY  = 1000 if not SAMPLE_MODE else 5
TRAIN_NORMAL  = 3851 if not SAMPLE_MODE else 30   # cap train set to theirs

SPLIT_SEED    = 42                                        # NEVER change — fixes train/test split
TRAIN_SEED    = int(os.environ.get('TRAIN_SEED', '42'))    # vary this: 42, 1337, 2024
DOWNSAMPLE    = os.environ.get('DOWNSAMPLE', 'stride')      # 'max' | 'avg' | 'stride'
SEED          = SPLIT_SEED   # keeps bootstrap_auc/compute_metrics (which default to seed=SEED) tied to the split, not the sweep

print(f"SAMPLE_MODE    : {SAMPLE_MODE}")
print(f"RUN_VERSION    : {RUN_VERSION}  (SKIP_COMPLETED={SKIP_COMPLETED})")
print(f"EPOCHS/WARMUP  : {EPOCHS} / {WARMUP_EPOCHS}")
print(f"OUTPUT_DIR     : {OUTPUT_DIR}")
print(f"CKPT_DIR       : {CKPT_DIR}")


USE_WANDB = False


# %% [markdown]
# # C1.5 Checkpoint save/load helpers

# %% [CELL 1.5]  Checkpoint save/load helpers
# ── Checkpoint helpers ────────────────────────────────────────────────
# These functions expect two globals to already exist before they're called:
#   all_results  = {}   # dict: condition key (e.g. 'C4') -> metrics dict
#   loss_history = {}   # dict: condition key -> list of per-epoch losses
# Define both in the training-loop cell before calling save_ckpt/load_ckpt,
# otherwise you'll get a NameError the first time they run.
all_results={}
loss_history={}
def ckpt_path(cond):
    # Where this condition's "done" marker (metrics) lives on disk.
    return f'{CKPT_DIR}/{cond}_done.json'
def restore_from_wandb(cond, run_version, entity=None):
    """Download a past run_version's checkpoint artifact for `cond` back into
    CKPT_DIR, so load_ckpt(cond)/load_weights(cond, ...) work normally
    afterward — even in a brand-new session with an empty /kaggle/working.
    Needs only wandb.login() to have succeeded; does NOT need wandb.init()."""
    api = wandb.Api()
    entity = entity or api.default_entity
    group = f'ablation-{run_version}'   # matches _art_name in save_ckpt exactly — no sanitizing here
    art_name = f'{entity}/{WANDB_PROJECT}/{group}-{cond.lower()}-ckpt:latest'
    art = api.artifact(art_name)
    # Remove any pre-existing files for this condition FIRST. art.download() overwrites only
    # the files the artifact contains, so downloading on top of a different run's leftovers
    # can silently produce a checkpoint whose .npy arrays and .pth weights came from
    # DIFFERENT models — which then yields plausible-looking but wrong analysis numbers.
    import glob as _glob
    for _stale in _glob.glob(f'{CKPT_DIR}/{cond}_*'):
        os.remove(_stale)
    art.download(root=CKPT_DIR)
    print(f'  [{cond}] restored from wandb ({run_version}) -> {CKPT_DIR}')

def ensure_local(cond, run_version=None):
    """Fetch cond's checkpoint from wandb into CKPT_DIR if it's not already
    on local disk, so is_done()/load_ckpt() see it. Call this before is_done()."""
    if os.path.exists(ckpt_path(cond)):
        return   # already local, nothing to do
    try:
        restore_from_wandb(cond, run_version or RUN_VERSION)
    except Exception as e:
        print(f'  [{cond}] no local or wandb checkpoint found ({e}) — will train fresh.')

def is_done(cond):
    """Return True if this condition is already completed for RUN_VERSION."""
    # Lets you re-run the notebook and skip conditions already trained under this RUN_VERSION.
    return SKIP_COMPLETED and os.path.exists(ckpt_path(cond))

def save_ckpt(cond, result_keys, scores, disc_scores, epoch_loss,
              attn_maps=None, wandb_group=None, **model_states):
    """
    Persist a completed condition to disk and log to wandb immediately.
    result_keys: list of keys to pull from all_results, e.g. ['C4','C4_disc','C4_fuse']
    model_states: keyword args of name→state_dict, e.g. enc1=enc1.state_dict()
    wandb_group: override WANDB_GROUP for THIS save only (default: real ablation
                 group). Pass e.g. 'test-runs' for throwaway/sanity-check saves —
                 the artifact gets that group in its name, metadata, and tags, so
                 later you can search "test-runs" in the wandb Artifacts tab and
                 bulk-delete everything that matched, without touching real results.
    """
    # Snapshot just this condition's slice of the global results/loss dicts to JSON.
    info = {
        'all_results':   {k: all_results[k] for k in result_keys if k in all_results},
        'loss_history':  [float(v) for v in epoch_loss],
    }
    with open(ckpt_path(cond), 'w') as f:
        json.dump(info, f, indent=2)
    np.save(f'{CKPT_DIR}/{cond}_scores.npy', scores)
    if disc_scores is not None:
        np.save(f'{CKPT_DIR}/{cond}_disc.npy', disc_scores)
    if attn_maps is not None:
        np.save(f'{CKPT_DIR}/{cond}_attn.npy', attn_maps)
    for name, state in model_states.items():
        torch.save(state, f'{CKPT_DIR}/{cond}_{name}.pth')
    # Everything below is optional: if wandb never logged in (USE_WANDB=False)
    # or wandb.init() was never called (wandb.run is None), checkpointing still
    # works locally — you just don't get the online dashboard/artifact copies.
    if USE_WANDB and wandb.run is not None:
        # ── 1. Log per-condition metrics (grouped by condition prefix) ──
        log = {'condition': cond}
        for k in result_keys:
            if k in all_results:
                r = all_results[k]
                tag = k.lower().replace(cond.lower()+'_','').replace(cond.lower(),'ssim')
                for m in ['auc_roc','auc_pr','f1']:
                    if m in r: log[f'{cond}/{tag}_{m}'] = r[m]
        wandb.log(log)
        # ── 2. Log per-epoch loss curve ──
        for ep, val in enumerate(epoch_loss):
            wandb.log({f'loss/{cond}': val, f'step_{cond}': ep})
        # ── 3. Upload ALL checkpoint files as versioned artifact ──────────
        # Artifact name: {group}-{cond}-ckpt  e.g. ablation-v2-c1-ckpt
        # wandb auto-versions each upload (:v0, :v1, …); :latest always points here.
        # Cell 3b restores by downloading :latest → CKPT_DIR on session reset.
        _group = wandb_group or WANDB_GROUP
        _art_name = f'{_group}-{cond.lower()}-ckpt'
        try:
            art = wandb.Artifact(
                _art_name,
                type='checkpoint',
                metadata={'cond': cond, 'version': RUN_VERSION, 'group': _group},
            )
            art.add_file(ckpt_path(cond))                       # {COND}_done.json
            art.add_file(f'{CKPT_DIR}/{cond}_scores.npy')       # SSIM anomaly scores
            disc_p = f'{CKPT_DIR}/{cond}_disc.npy'
            attn_p = f'{CKPT_DIR}/{cond}_attn.npy'
            if os.path.exists(disc_p): art.add_file(disc_p)     # discriminator scores
            if os.path.exists(attn_p): art.add_file(attn_p)     # attention maps
            for name in model_states:
                wp = f'{CKPT_DIR}/{cond}_{name}.pth'
                if os.path.exists(wp): art.add_file(wp)         # model weights
            art = wandb.log_artifact(art)
            art.wait()                 # block until the artifact is fully registered server-side
            art.tags = [_group.replace('.', '_')]   # wandb tags reject '.'; only settable on a waited-on artifact
            art.save()                 # push the tag change back to the server
            print(f'  [{cond}] artifact logged → wandb:{_art_name}:latest')
        except Exception as _art_e:
            print(f'  [{cond}] wandb artifact upload failed: {_art_e}')
    print(f'  [{cond}] checkpoint saved to {CKPT_DIR}/')

def load_ckpt(cond):
    """Load saved condition results back into all_results and loss_history."""
    with open(ckpt_path(cond)) as f:
        info = json.load(f)
    all_results.update(info['all_results'])
    loss_history[cond] = info['loss_history']
    scores     = np.load(f'{CKPT_DIR}/{cond}_scores.npy')
    disc_p     = f'{CKPT_DIR}/{cond}_disc.npy'
    attn_p     = f'{CKPT_DIR}/{cond}_attn.npy'
    disc_sc    = np.load(disc_p)    if os.path.exists(disc_p) else None
    attn_maps  = np.load(attn_p)   if os.path.exists(attn_p) else None
    # Integrity check: save_ckpt writes every file for a condition in one call, so their
    # mtimes should be seconds apart. A large spread means the .npy arrays and .pth weights
    # came from different runs — exactly the failure that makes saved metrics disagree with
    # fresh inference from the same checkpoint.
    import glob as _glob
    _files = _glob.glob(f'{CKPT_DIR}/{cond}_*')
    if len(_files) > 1:
        _mt = {f: os.path.getmtime(f) for f in _files}
        _spread = max(_mt.values()) - min(_mt.values())
        if _spread > 300:   # >5 min apart == not one save
            print(f'  [{cond}] WARNING: checkpoint files span {_spread/60:.1f} min — arrays and '
                  f'weights may be from DIFFERENT runs. Verify with fresh inference before '
                  f'trusting saved metrics.')
            for f in sorted(_mt, key=_mt.get):
                print(f'      {time.strftime("%H:%M:%S", time.localtime(_mt[f]))}  {os.path.basename(f)}')
    print(f'  [{cond}] loaded from checkpoint (version {RUN_VERSION}).')
    return scores, disc_sc, attn_maps

def load_weights(cond, **models):
    """Load saved weights into model objects. Pass name=model_instance."""
    for name, model in models.items():
        p = f'{CKPT_DIR}/{cond}_{name}.pth'
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
            # Put the module in eval mode immediately. load_weights is only ever called on the
            # reload path, where the very next thing is inference — but nn.Module defaults to
            # TRAIN mode, so BatchNorm would use batch statistics AND mutate its running stats
            # on every forward. Measured effect: ~0.003 output difference per forward, which is
            # what made reloaded-checkpoint metrics disagree with fresh inference from the same
            # weights. Training cells call .train() explicitly in their loops, so this is safe.
            model.eval()
        else:
            # Previously this only printed — so a missing file left that sub-module at RANDOM
            # INIT while everything downstream carried on and produced meaningless-but-plausible
            # metrics. Fail loudly instead.
            raise FileNotFoundError(
                f"[{cond}] weight file missing: {p}\n"
                f"  Refusing to continue with '{name}' left at random initialisation. "
                f"Delete the condition's checkpoint and retrain, or restore a complete artifact.")



# %% [markdown]
# # C1.6 Wandb setup and login

# %% [CELL 1.6]  Wandb setup and login
# USE_WANDB is the flag every later cell should check before calling wandb.*
# — that's what makes wandb fully optional (see save_ckpt above).
try:
    import wandb
    # Two login paths: Kaggle reads the API key from its Secrets vault;
    # anywhere else falls back to the normal interactive/browser login.
    if os.path.exists('/kaggle/working'):
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret('REATTN_KEY'), relogin=True)
    else:
        wandb.login()
    USE_WANDB = True
    # id=f'ablation-{RUN_VERSION}' + resume='allow' means re-running this cell
    # (e.g. after a Kaggle session reset) reattaches to the SAME wandb run
    # instead of creating a new one, so metrics keep appending to one history.
    wandb.init(project=WANDB_PROJECT,
               group=WANDB_GROUP,
               name=f'ablation-C1-C7-{RUN_VERSION}',
               config=dict(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM,
                           lambda_adv=LAMBDA_ADV, warmup_epochs=WARMUP_EPOCHS,
                           epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                           run_version=RUN_VERSION),
               tags=['ablation', 'RE-attention', 'AAE', 'CXR', RUN_VERSION],
               resume='allow', id=f'ablation-{RUN_VERSION}',
               settings=wandb.Settings(init_timeout=120))
    print(f'WandB ready  project={WANDB_PROJECT}  version={RUN_VERSION}')
except Exception as _e:
    # Any failure here (no internet, no key, user declines login, etc.)
    # falls back to USE_WANDB=False so the rest of the notebook still runs.
    USE_WANDB = False
    print(f'WandB unavailable ({_e}) — continuing without.')



# %% [markdown]
# ---
# ## **Cell 2.0** — Data preparation
#
# Loads DICOM chest X-rays from the RSNA Pneumonia Detection dataset, applies
# **CLAHE contrast enhancement**, and bilinearly downsamples to `IMAGE_SIZE × IMAGE_SIZE`.
#
# **Train / test split strategy:**
# - Training set: normal scans only (no anomalies seen during training).
# - Test set: 2 000 normal + 2 000 lung-opacity images (50 / 50 balance).
# - Bounding-box annotations are loaded for all opacity images that have them
#   — used later for **pixel-level localisation AUROC**.
#
# In `SAMPLE_MODE` random arrays substitute for real images so the full pipeline
# can be validated in seconds without the dataset.



# %% [CELL 2.0]  Data preparation

def _clahe_uint8(img_f32):
    import cv2
    img_u8 = (img_f32 * 255).clip(0, 255).astype(np.uint8)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_u8).astype(np.float32) / 255.0

def load_dcm_resized(patient_id, train_dir, size):
    dcm = pydicom.dcmread(f'{train_dir}/{patient_id}.dcm')
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = _clahe_uint8(img)
    t   = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    t   = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()

def load_images(patient_ids, train_dir, size, tag):
    imgs, n = [], len(patient_ids)
    for i, pid in enumerate(patient_ids):
        if i % 500 == 0:
            print(f"  {tag}: {i}/{n}")
        imgs.append(load_dcm_resized(pid, train_dir, size))
    arr = np.stack(imgs)[:, None, :, :]
    print(f"  {tag} done → {arr.shape}")
    return arr

if SAMPLE_MODE:
    # Seed here so the smoke-test data is IDENTICAL across runs. Without this the fake test
    # set is regenerated differently every run, so a checkpoint saved in one run and reloaded
    # in the next is evaluated on different images — which looks exactly like a corrupted
    # checkpoint and makes local reload verification meaningless.
    np.random.seed(SPLIT_SEED)
    x_train_norm = np.random.rand(30, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    x_test_norm  = np.random.rand(TEST_NORMAL,  1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    x_test_opa   = np.random.rand(TEST_OPACITY, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    raw_boxes    = {i: [(100, 200, 300, 200)] for i in range(TEST_OPACITY)}
    print(f"SAMPLE_MODE — train:{x_train_norm.shape}  "
          f"test_norm:{x_test_norm.shape}  test_opa:{x_test_opa.shape}")
else:
    labels = pd.read_csv(f'{BASE}/stage_2_train_labels.csv')
    detail = pd.read_csv(f'{BASE}/stage_2_detailed_class_info.csv')
    patient_class = (detail.drop_duplicates('patientId')
                           .set_index('patientId')['class'])
    normal_ids  = patient_class[patient_class == 'Normal'].index.tolist()
    opacity_ids = patient_class[patient_class == 'Lung Opacity'].index.tolist()
    np.random.shuffle(normal_ids); np.random.shuffle(opacity_ids)
    test_nml_ids  = normal_ids[:TEST_NORMAL]
    # Cap the training set at TRAIN_NORMAL (3851) to match MedIAnomaly's RSNA
    # protocol. Without this cap we would train on ~7851 normals against their
    # 3851 and the comparison to their Table 6 would be unfair in our favour.
    train_nml_ids = normal_ids[TEST_NORMAL:TEST_NORMAL + TRAIN_NORMAL]
    test_opa_ids  = opacity_ids[:TEST_OPACITY]
    print(f"Train normal  : {len(train_nml_ids)}")
    print(f"Test  normal  : {len(test_nml_ids)}")
    print(f"Test  opacity : {len(test_opa_ids)}")
    print(f"\nLoading images ...")
    t0 = time.time()
    x_train_norm = load_images(train_nml_ids, TRAIN_DIR, IMAGE_SIZE, 'Train-normal')
    x_test_norm  = load_images(test_nml_ids,  TRAIN_DIR, IMAGE_SIZE, 'Test-normal')
    x_test_opa   = load_images(test_opa_ids,  TRAIN_DIR, IMAGE_SIZE, 'Test-opacity')
    print(f"All images loaded in {time.time()-t0:.0f}s")
    box_df    = labels[labels['Target'] == 1][['patientId','x','y','width','height']]
    raw_boxes = {}
    for i, pid in enumerate(test_opa_ids):
        rows = box_df[box_df['patientId'] == pid]
        if len(rows):
            raw_boxes[i] = list(zip(rows['x'], rows['y'], rows['width'], rows['height']))

x_test      = np.concatenate([x_test_norm, x_test_opa], axis=0)
binary_test = np.array([0]*len(x_test_norm) + [1]*len(x_test_opa), dtype=np.int32)
test_boxes  = {k + len(x_test_norm): v for k, v in raw_boxes.items()}

print(f"\nTrain (normal only) : {x_train_norm.shape}")
print(f"Test                : {x_test.shape}  ({binary_test.mean()*100:.1f}% anomaly)")
print(f"Opacity with boxes  : {len(test_boxes)}")

# %% [markdown]
# ---
# ## **Cell 2.1** — update wandb config with dataset sizes
# Dataset size wasn't known yet at wandb.init() time (Cell 1, before data loading) --
# enrich the *same* run's config now instead of opening a second run for it.
# %% [CELL 2.1]  Wandb config update with dataset sizes
if USE_WANDB and wandb.run is not None:
    wandb.config.update({
        'dataset':      'RSNA Pneumonia Detection',
        'train_normal': int(x_train_norm.shape[0]),
        'test_normal':  int((binary_test == 0).sum()),
        'test_opacity': int((binary_test == 1).sum()),
    }, allow_val_change=True)



# %% [markdown]
# ---
# ## **Cell 2.2** — DataLoader factory
#
# A thin wrapper around `TensorDataset` + `DataLoader`.
# `pin_memory=True` on GPU environments speeds up CPU→GPU transfers.
# Each condition creates its own loader from this function to ensure independent shuffling.

# %% [CELL 2.2]  DataLoader helper

def make_loader(x_np, batch_size, shuffle=True, drop_last=True):
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last,
                      pin_memory=(device.type == 'cuda'),
                      num_workers=2)


# %% [markdown]
# ---
# ## **Cell 3.0** — Evaluation utilities

# %% [CELL 3.0]  Evaluation utilities

def bootstrap_auc(scores, binary_labels, n_boot=1000, seed=SEED):
    """Bootstrap resample AUC-ROC to get mean/std/95% CI (replaces hand-typed stability labels)."""
    if len(np.unique(binary_labels)) < 2:
        return {'auc_mean': np.nan, 'auc_std': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    rng = np.random.default_rng(seed)
    n = len(scores)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(binary_labels[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(binary_labels[idx], scores[idx])
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {'auc_mean': float(aucs.mean()), 'auc_std': float(aucs.std()),
            'ci_lo': float(lo), 'ci_hi': float(hi)}


def bootstrap_paired_diff(scores_a, scores_b, binary_labels, n_boot=1000, seed=SEED):
    """Paired bootstrap on AUC-ROC(a) - AUC-ROC(b), resampling both scores with the same
    indices each draw. Returns the diff CI and a two-sided bootstrap p-value for diff == 0."""
    if len(np.unique(binary_labels)) < 2:
        return {'diff_mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'p_value': np.nan}
    rng = np.random.default_rng(seed)
    n = len(binary_labels)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(binary_labels[idx])) < 2:
            diffs[i] = np.nan
            continue
        auc_a = roc_auc_score(binary_labels[idx], scores_a[idx])
        auc_b = roc_auc_score(binary_labels[idx], scores_b[idx])
        diffs[i] = auc_a - auc_b
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {'diff_mean': float(diffs.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'p_value': float(min(p_value, 1.0))}


def disc_stability_label(auc_stats):
    """Computed replacement for the old hardcoded disc_stable dict: classifies the
    discriminator-only score using its bootstrap CI instead of an asserted label."""
    ci_lo, ci_hi, auc_std = auc_stats['ci_lo'], auc_stats['ci_hi'], auc_stats['auc_std']
    if np.isnan(ci_lo):
        return '-'
    if ci_lo <= 0.5 <= ci_hi:
        return f'COLLAPSED ({ci_lo:.2f}-{ci_hi:.2f})'
    elif auc_std > 0.03:
        return f'unstable (σ={auc_std:.3f})'
    else:
        return f'stable (σ={auc_std:.3f})'


def compute_metrics(scores, binary_labels, n_boot=1000, seed=SEED):
    if len(np.unique(binary_labels)) < 2:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'f1': np.nan,
                'auc_std': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    auc_roc          = roc_auc_score(binary_labels, scores)
    auc_pr           = average_precision_score(binary_labels, scores)
    fpr, tpr, thresh = roc_curve(binary_labels, scores)
    best = np.argmax(tpr - fpr)
    pred = (scores >= thresh[best]).astype(int)
    boot = bootstrap_auc(scores, binary_labels, n_boot=n_boot, seed=seed)
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'f1': f1_score(binary_labels, pred, zero_division=0),
            'auc_std': boot['auc_std'], 'ci_lo': boot['ci_lo'], 'ci_hi': boot['ci_hi']}

def boxes_to_mask(boxes, size=IMAGE_SIZE, orig=ORIG_SIZE):
    scale = size / orig
    mask  = np.zeros((size, size), dtype=np.float32)
    for (x, y, w, h) in boxes:
        x1, y1 = int(x*scale), int(y*scale)
        x2 = min(size, int((x+w)*scale))
        y2 = min(size, int((y+h)*scale))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask

def pixel_auroc(maps_np, boxes_dict, binary_arr):
    gt_all, pred_all = [], []
    for idx in range(len(binary_arr)):
        if binary_arr[idx] == 0 or idx not in boxes_dict:
            continue
        gt_all.append(boxes_to_mask(boxes_dict[idx]).flatten())
        pred_all.append(maps_np[idx].flatten())
    if not gt_all:
        return np.nan
    gt, pred = np.concatenate(gt_all), np.concatenate(pred_all)
    return roc_auc_score(gt, pred) if len(np.unique(gt)) > 1 else np.nan

def pixel_auroc_inlung(maps_np, boxes_dict, binary_arr, lung_mask, thr=0.5):
    """Same as pixel_auroc, but restricted to lung-field pixels only. pixel_auroc
    pools ALL pixels (including image borders/background), so any lung-shaped map
    scores well above chance for free — RSNA boxes are always inside the lung
    field, "pneumonia occurs in lungs, not corners" isn't a real localization
    finding. Restricting evaluation to in-lung pixels asks the real question:
    within the lung, does the map find the lesion specifically? A uniform
    in-lung map (e.g. a fixed lung-shaped prior) should collapse to ~0.5 here."""
    lung = (lung_mask.squeeze().cpu().numpy() > thr).flatten() if torch.is_tensor(lung_mask) \
           else (lung_mask.squeeze() > thr).flatten()
    gt_all, pred_all = [], []
    for idx in range(len(binary_arr)):
        if binary_arr[idx] == 0 or idx not in boxes_dict:
            continue
        gt_all.append(boxes_to_mask(boxes_dict[idx]).flatten()[lung])
        pred_all.append(maps_np[idx].flatten()[lung])
    if not gt_all:
        return np.nan
    gt, pred = np.concatenate(gt_all), np.concatenate(pred_all)
    return roc_auc_score(gt, pred) if len(np.unique(gt)) > 1 else np.nan

def _pixel_pairs(maps_np, boxes_dict, binary_arr, lung_mask=None, thr=0.5):
    """Shared GT/prediction flattening for every pixel-level metric.
    lung_mask=None -> pooled (all pixels); otherwise restricted to the lung field."""
    sel = None
    if lung_mask is not None:
        m = lung_mask.squeeze().cpu().numpy() if torch.is_tensor(lung_mask) else lung_mask.squeeze()
        sel = (m > thr).flatten()
    gt_all, pred_all = [], []
    for idx in range(len(binary_arr)):
        if binary_arr[idx] == 0 or idx not in boxes_dict:
            continue
        g = boxes_to_mask(boxes_dict[idx]).flatten()
        p = maps_np[idx].flatten()
        gt_all.append(g if sel is None else g[sel])
        pred_all.append(p if sel is None else p[sel])
    if not gt_all:
        return None, None
    return np.concatenate(gt_all), np.concatenate(pred_all)


def pixel_ap(maps_np, boxes_dict, binary_arr, lung_mask=None):
    """AP_pix — MedIAnomaly's pixel metric. They explicitly REJECT pixel-level AUC
    (Sec 3.1.4): "we do not employ the pixel-level AUC since it is insensitive to
    false positives, which is particularly relevant in medical images where the
    majority of pixels are negative." Report this alongside our in-lung AUROC."""
    gt, pred = _pixel_pairs(maps_np, boxes_dict, binary_arr, lung_mask)
    if gt is None or len(np.unique(gt)) < 2:
        return np.nan
    return float(average_precision_score(gt, pred))


def best_dice(maps_np, boxes_dict, binary_arr, lung_mask=None, n_thr=50):
    """[Dice] — best achievable Dice over thresholds, computed dataset-wise, exactly
    as MedIAnomaly define it (avoids having to pick an operating point)."""
    gt, pred = _pixel_pairs(maps_np, boxes_dict, binary_arr, lung_mask)
    if gt is None or len(np.unique(gt)) < 2:
        return np.nan
    lo, hi = float(pred.min()), float(pred.max())
    best = 0.0
    for t in np.linspace(lo, hi, n_thr):
        b = (pred >= t)
        inter = float((b & (gt > 0.5)).sum())
        denom = float(b.sum() + (gt > 0.5).sum())
        if denom > 0:
            best = max(best, 2.0 * inter / denom)
    return best


def make_lung_prior(size=IMAGE_SIZE, device='cpu'):
    """Two overlapping ellipses approximating the lung field. Used BOTH to restrict
    pixel metrics (in-lung) and as the fixed non-learned CONTROL that any real
    localiser must beat. On our data it scores ~0.754 pooled AUROC but exactly
    0.500 in-lung — that gap is the anatomy confound."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
    left  = (((xx - 0.32) / 0.22) ** 2 + ((yy - 0.52) / 0.38) ** 2) <= 1.0
    right = (((xx - 0.68) / 0.22) ** 2 + ((yy - 0.52) / 0.38) ** 2) <= 1.0
    return (left | right).float().to(device).view(1, 1, size, size)


def pixel_report(maps_np, tag=''):
    """One call -> every pixel metric, pooled AND in-lung, next to the fixed-prior
    control. This is the table the protocol contribution rests on."""
    lp = LUNG_PRIOR if 'LUNG_PRIOR' in globals() else make_lung_prior(IMAGE_SIZE, device)
    const = np.repeat(lp.squeeze().cpu().numpy()[None], len(maps_np), axis=0)
    rows = [('learned', maps_np), ('fixed lung prior', const)]
    print(f"  {tag}{'':<18}{'AUROC':>9}{'AP_pix':>9}{'[Dice]':>9}   |{'AUROC':>9}{'AP_pix':>9}{'[Dice]':>9}")
    print(f"  {'':<18}{'---- pooled ----':>27}   |{'--- in-lung ---':>27}")
    out = {}
    for name, mp in rows:
        r = (pixel_auroc(mp, test_boxes, binary_test),
             pixel_ap(mp, test_boxes, binary_test),
             best_dice(mp, test_boxes, binary_test),
             pixel_auroc_inlung(mp, test_boxes, binary_test, lp),
             pixel_ap(mp, test_boxes, binary_test, lp),
             best_dice(mp, test_boxes, binary_test, lp))
        out[name] = r
        print(f"  {name:<18}{r[0]:>9.4f}{r[1]:>9.4f}{r[2]:>9.4f}   |{r[3]:>9.4f}{r[4]:>9.4f}{r[5]:>9.4f}")
    return out


def show(obj, title=None):
    """Render DataFrames/objects as real tables in a notebook, plain text in a
    terminal. Keeps the Kaggle notebook output readable instead of monospace dumps."""
    if title:
        print(f"\n=== {title} ===")
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


mse_fn = nn.MSELoss()

def ssim_anomaly_map(x, x_hat, window=11):
    """Per-pixel (1-SSIM) → (B, H*W). Higher = more anomalous."""
    pad  = window // 2
    mu_x = F.avg_pool2d(x,     window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(x_hat, window, stride=1, padding=pad)
    s_x  = F.avg_pool2d(x**2,     window, stride=1, padding=pad) - mu_x**2
    s_y  = F.avg_pool2d(x_hat**2, window, stride=1, padding=pad) - mu_y**2
    s_xy = F.avg_pool2d(x*x_hat,  window, stride=1, padding=pad) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_x*mu_y + c1)*(2*s_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1)*(s_x + s_y + c2))
    return (1.0 - ssim.clamp(-1, 1)).view(x.size(0), -1)

def ssim_loss_fn(x, x_hat):
    return ssim_anomaly_map(x, x_hat).mean()

def anomaly_score(x, x_hat):
    """99th-pct SSIM score — primary metric, consistent across all conditions."""
    return torch.quantile(ssim_anomaly_map(x, x_hat), 0.99, dim=1)

def normalise_scores(s):
    """Min-max normalise to [0,1] so fusion weights both scores equally."""
    s_min, s_max = s.min(), s.max()
    return (s - s_min) / (s_max - s_min + 1e-8)

def vae_elbo_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = 0.7 * mse_fn(x_hat, x) + 0.3 * ssim_loss_fn(x_hat, x)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl

all_results  = {}
loss_history = {}
print("Utilities defined.")

# %% [markdown]
# ---
# ## **Cell 4.0** — Training signals: coarse noise (DAE) vs synthetic lesions
# The two arms of the factorial differ ONLY in what supervises the network.
# * `add_coarse_noise` — MedIAnomaly's DAE recipe: 16x16 Gaussian noise, bilinearly
#   upsampled, randomly translated. UNSTRUCTURED, so there is no memorisable proxy.
#   Their DAE is the best pixel-level method in the benchmark (~20% Dice over 2nd).
# * `make_synthetic_lesion` — a lesion-shaped, mask-supervised anomaly (our v2
#   generator). This IS memorisable, which is what we think drives proxy overfitting.

# %% [CELL 4.0]  Training signals + losses

LUNG_PRIOR = make_lung_prior(IMAGE_SIZE, device)

def add_coarse_noise(x, noise_res=16, noise_std=0.2):
    """DAE noise, ported from MedIAnomaly utils/dae_worker.py.
    NOTE: their version ends with `ns = (ns - 0.5) * 2`, which shifts zero-mean
    noise to mean -1. That only makes sense for their [-1,1] normalisation (ours is
    [0,1]) and would darken every image, so it is deliberately NOT reproduced here."""
    n, c, h, w = x.shape
    ns = torch.normal(mean=torch.zeros(n, c, noise_res, noise_res, device=x.device),
                       std=noise_std)
    ns = F.interpolate(ns, size=(h, w), mode='bilinear', align_corners=True)
    ns = torch.roll(ns, shifts=[random.randrange(h), random.randrange(w)], dims=[-2, -1])
    return (x + ns).clamp(0, 1), ns


def make_synthetic_lesion(xb, lung_prior, q_range=(0.80, 0.95), delta=(0.08, 0.30)):
    """Lesion-shaped synthetic anomaly restricted to the lung field, with randomised
    size / intensity and locally smoothed texture (our v2 generator: ~5% coverage)."""
    n, c, h, w = xb.shape
    blob = F.interpolate(torch.rand(n, 1, 8, 8, device=xb.device), size=(h, w),
                          mode='bilinear', align_corners=False)
    flat = blob.view(n, -1)
    q    = torch.empty(n, device=xb.device).uniform_(*q_range)
    k    = (q * (flat.shape[1] - 1)).long()
    thr  = flat.sort(dim=1).values.gather(1, k.view(n, 1)).view(n, 1, 1, 1)
    mask = F.avg_pool2d((blob > thr).float(), 9, stride=1, padding=4) * lung_prior
    d    = torch.empty(n, 1, 1, 1, device=xb.device).uniform_(*delta)
    base = F.avg_pool2d(xb, 5, stride=1, padding=2)
    return (xb * (1 - mask) + (base + d).clamp(0, 1) * mask), mask


def focal_bce_logits(logits, target, gamma=2.0, alpha=0.25):
    """Stable focal loss on PRE-sigmoid logits (clamped post-sigmoid focal has zero
    gradient past |logit|~20, which previously stalled training)."""
    p  = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.where(target == 1, p, 1 - p)
    at = torch.where(target == 1, torch.as_tensor(alpha, device=logits.device),
                                   torch.as_tensor(1 - alpha, device=logits.device))
    return (at * (1 - pt).pow(gamma) * ce).mean()


def dice_loss(pred, target, eps=1.0):
    num = 2.0 * (pred * target).sum(dim=[1, 2, 3]) + eps
    den = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) + eps
    return (1.0 - num / den).mean()


class PerceptualLoss(nn.Module):
    """AE-PL: reconstruction error in VGG19 FEATURE space instead of pixel space.
    Best image-level method on RSNA in MedIAnomaly (87.5 AUC). Motivation matches
    our own diagnosis: pixel-space error is dominated by anatomical edges (ribs,
    diaphragm), which is exactly what polluted our error maps."""
    def __init__(self, layer=16):
        super().__init__()
        self.vgg = tv_models.vgg19(weights='IMAGENET1K_V1').features[:layer].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):                       # grayscale [0,1] -> ImageNet-normalised 3ch
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std

    def forward(self, x, x_hat, keepdim=False):
        f, fh = self.vgg(self._prep(x)), self.vgg(self._prep(x_hat))
        if keepdim:                            # per-pixel map, upsampled to input size
            m = ((f - fh) ** 2).mean(dim=1, keepdim=True)
            return F.interpolate(m, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return F.mse_loss(fh, f)


# %% [markdown]
# ---
# ## **Cell 4.1** — THE FACTORIAL: training signal x capacity
# The claim under test: **capacity is harmful specifically when the supervision
# signal is a memorisable synthetic proxy**, not in general.
# MedIAnomaly reports that bigger reconstruction networks do not help, and separately
# that SSL/synthetic methods struggle when synthetic != real. Neither is measured as
# an interaction. This grid measures it.
#
# Prediction: along `denoise`, real performance is flat or rising with capacity;
# along `synthmask`, it falls while SYNTHETIC performance rises to ~1.0.
# The divergence between those two curves is the finding.

# %% [CELL 4.1]  Factorial sweep — {denoise, synthmask} x capacity

CAPACITY_LADDER = [(3, 3), (4, 3), (4, 4), (4, 5)] if not SAMPLE_MODE else [(3, 3), (4, 3)]
ARMS            = ['denoise', 'synthmask']
SWEEP_SEEDS     = [42, 1337] if not SAMPLE_MODE else [42]

factorial_results = []

for arm in ARMS:
    for (depth, wf) in CAPACITY_LADDER:
        for seed in SWEEP_SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            net = UNet(in_channels=1, n_classes=1, depth=depth, wf=wf).to(device)
            n_par = sum(p.numel() for p in net.parameters())
            cond  = f'FACT_{arm}_d{depth}w{wf}_s{seed}'
            print(f"\n--- {cond}  ({n_par:,} params) ---")

            ensure_local(cond)
            if is_done(cond):
                scores, amaps, _ = load_ckpt(cond)
                load_weights(cond, net=net)
            else:
                opt   = Adam(net.parameters(), lr=LR, betas=(BETA1, 0.999))
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
                loader, ep_loss = make_loader(x_train_norm, BATCH_SIZE), []
                for epoch in range(EPOCHS):
                    net.train(); losses = []
                    for (xb,) in loader:
                        xb = xb.to(device); n = xb.size(0)
                        flip = torch.rand(n, device=device) > 0.5
                        xb[flip] = xb[flip].flip(dims=[3])
                        opt.zero_grad()
                        if arm == 'denoise':
                            x_in, _ = add_coarse_noise(xb)
                            x_hat   = torch.sigmoid(net(x_in)['x_hat'])
                            loss    = mse_fn(x_hat, xb)          # restore CLEAN from noisy
                        else:
                            apply  = (torch.rand(n, device=device) < 0.5).view(-1, 1, 1, 1).float()
                            x_syn, m_full = make_synthetic_lesion(xb, LUNG_PRIOR)
                            x_in   = xb * (1 - apply) + x_syn * apply
                            m      = (m_full * apply > 0.5).float()
                            logits = net(x_in)['x_hat']
                            loss   = focal_bce_logits(logits, m) + dice_loss(torch.sigmoid(logits), m)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"{cond}: non-finite loss at epoch {epoch+1}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step(); losses.append(loss.item())
                    sched.step(); ep_loss.append(float(np.mean(losses)))
                    if (epoch + 1) % 20 == 0 or epoch == 0:
                        print(f"  ep {epoch+1:02d}/{EPOCHS}  loss={ep_loss[-1]:.5f}")

                net.eval(); scores, amaps = [], []
                with torch.no_grad():
                    for i in range(0, len(x_test), BATCH_SIZE):
                        xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device); n = xb.size(0)
                        out = net(xb)['x_hat']
                        # anomaly map: reconstruction error (denoise) or segmentation (synthmask)
                        amap = ((xb - torch.sigmoid(out)) ** 2) if arm == 'denoise' else torch.sigmoid(out)
                        scores.append(torch.quantile(amap.view(n, -1), 0.99, dim=1).cpu().numpy())
                        amaps.append(amap.squeeze(1).cpu().numpy())
                scores, amaps = np.concatenate(scores), np.concatenate(amaps)
                all_results[cond] = {**compute_metrics(scores, binary_test), 'arm': arm,
                                      'depth': depth, 'wf': wf, 'params': n_par, 'seed': seed}
                save_ckpt(cond, [cond], scores, amaps, ep_loss, net=net.state_dict())

            m_img = compute_metrics(scores, binary_test)
            rec = {'arm': arm, 'depth': depth, 'wf': wf, 'params': n_par, 'seed': seed,
                   'image_auc': m_img['auc_roc'],
                   'pix_auroc_inlung': pixel_auroc_inlung(amaps, test_boxes, binary_test, LUNG_PRIOR),
                   'pix_ap_inlung':    pixel_ap(amaps, test_boxes, binary_test, LUNG_PRIOR),
                   'dice_inlung':      best_dice(amaps, test_boxes, binary_test, LUNG_PRIOR)}

            # SYNTHETIC-task performance: how well does it solve its own proxy?
            # For 'denoise' there is no lesion proxy, so this is only defined for synthmask.
            if arm == 'synthmask':
                torch.manual_seed(0)
                pool = x_test[np.where(binary_test == 0)[0]]
                fl   = np.zeros(len(pool), np.float32); fl[: len(pool)//2] = 1.0
                ss, sl = [], []
                with torch.no_grad():
                    for i in range(0, len(pool), BATCH_SIZE):
                        xb = torch.tensor(pool[i:i+BATCH_SIZE]).to(device); n = xb.size(0)
                        cf = torch.tensor(fl[i:i+BATCH_SIZE], device=device).view(-1,1,1,1)
                        xs, _ = make_synthetic_lesion(xb, LUNG_PRIOR)
                        xi = xb*(1-cf) + xs*cf
                        sg = torch.sigmoid(net(xi)['x_hat'])
                        ss.append(torch.quantile(sg.view(n,-1), 0.99, dim=1).cpu().numpy())
                        sl.append(cf.view(-1).cpu().numpy())
                rec['synthetic_auc'] = float(roc_auc_score(np.concatenate(sl).astype(int),
                                                            np.concatenate(ss)))
            else:
                rec['synthetic_auc'] = float('nan')

            print(f"  image AUC={rec['image_auc']:.4f}  in-lung AUROC={rec['pix_auroc_inlung']:.4f}  "
                  f"AP={rec['pix_ap_inlung']:.4f}  Dice={rec['dice_inlung']:.4f}  "
                  f"synthAUC={rec['synthetic_auc']:.4f}")
            factorial_results.append(rec)

# %% [CELL 4.2]  Factorial summary — the interaction figure
df_f = pd.DataFrame(factorial_results)
agg = df_f.groupby(['arm', 'params'])[['image_auc', 'pix_auroc_inlung', 'pix_ap_inlung',
                                        'dice_inlung', 'synthetic_auc']].mean().round(4)
show(agg, 'FACTORIAL: mean over seeds  (capacity x training signal)')
show(df_f.round(4), 'per-run results')
df_f.to_csv(f'{OUTPUT_DIR}/factorial_results.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for arm, mk in [('denoise', 'o-'), ('synthmask', 's--')]:
    d = df_f[df_f.arm == arm].groupby('params').mean(numeric_only=True).reset_index()
    axes[0].plot(d['params'], d['image_auc'], mk, label=f'{arm} (real)')
    axes[1].plot(d['params'], d['pix_auroc_inlung'], mk, label=f'{arm} (real)')
    if arm == 'synthmask':
        axes[0].plot(d['params'], d['synthetic_auc'], '^:', color='grey', label='synthmask (SYNTHETIC)')
for ax, ttl, base in [(axes[0], 'Image-level AUC vs capacity', None),
                       (axes[1], 'In-lung pixel-AUROC vs capacity', 0.5)]:
    ax.set_xscale('log'); ax.set_xlabel('segmentation/reconstruction net params')
    ax.set_title(ttl); ax.legend(fontsize=8)
    if base: ax.axhline(base, color='k', ls=':', lw=1, label='chance')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/factorial_interaction.png', dpi=150); plt.show()
print(f"Saved {OUTPUT_DIR}/factorial_interaction.png  and factorial_results.csv")
