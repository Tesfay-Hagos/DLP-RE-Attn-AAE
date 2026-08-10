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


# %% [CELL 1.4]  Configuration — MedIAnomaly reference hyperparameters
# All values below are taken from the benchmark's own defaults
# (MedIAnomaly/reconstruction/options.py + data_utils.get_transform), NOT tuned by us.
# Rationale: our first goal is to REPRODUCE their numbers so ours are comparable to
# Table 6. Hyperparameter search (manual ablation / Optuna) comes later, on top of a
# verified baseline -- tuning before reproducing makes any gap uninterpretable.
SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))

RUN_VERSION    = 'cv-v1'
SKIP_COMPLETED = True
WANDB_PROJECT  = 'MedIAnomaly-CV'
WANDB_GROUP    = f'{RUN_VERSION}'

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
        print(f'  [{rid}] not on wandb ({type(e).__name__}) — will train fresh')
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
               tags=['medianomaly', 'cv', RUN_VERSION],
               resume='allow', id=f'{RUN_VERSION}',
               settings=wandb.Settings(init_timeout=120))
    print(f'WandB ready  project={WANDB_PROJECT}  version={RUN_VERSION}')
except Exception as _e:
    # Any failure here (no internet, no key, user declines login, etc.)
    # falls back to USE_WANDB=False so the rest of the notebook still runs.
    USE_WANDB = False
    print(f'WandB unavailable ({_e}) — continuing without.')

# Datasets this project needs: BraTS2021 carries the pixel ground truth (positive anchor); RSNA is the CXR contrast.
# Add 'VinCXR' once you want the second CXR dataset.
REQUIRED_DATASETS = ['BraTS2021', 'RSNA']


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
DATA_ROOT_CANDIDATES = [
    os.environ.get('MEDIANOMALY_DATA', ''),
    os.path.expanduser('~/MedIAnomaly-Data'),
    '/kaggle/working/MedIAnomaly-Data',
] + sorted(_glob.glob('/kaggle/input/*/MedIAnomaly-Data')) + sorted(_glob.glob('/kaggle/input/*'))


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
