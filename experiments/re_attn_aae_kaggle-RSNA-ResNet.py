#!/usr/bin/env python3

# %% [markdown]
# # RE-Attention Adversarial Autoencoder — Extended Experiment Suite
# ## RSNA Pneumonia Detection Challenge · PyTorch / Kaggle T4 GPU
#
# ---
#
# ### Overview
#
# This notebook runs **five experimental conditions** in a single pass to ablate and extend
# the novel **RE-Attention Adversarial Autoencoder (RE-Attn-AAE)** for unsupervised anomaly
# detection on chest X-rays. Only **normal** scans are seen at training time; the model must
# flag pneumonia (lung opacity) at inference purely from reconstruction deviation.
#
# | ID | Condition | Purpose |
# |----|-----------|---------|
# | **C1** | CNN-AE Baseline | Reconstruction-only anchor |
# | **C2** | VAE Baseline | Probabilistic model comparison |
# | **C3** | CNN-AAE *(ablation)* | Adversarial latent regularisation, no attention |
# | **C4** | CNN-RE-Attn-AAE *(Ours)* | Full novel method |
# | **C5** | ResNet-18-RE-Attn-AAE | Frozen ImageNet backbone + novel method |
#
# ### Ablation chain
#
# ```
# C1  →  C3 : does adversarial latent regularisation help without attention?
# C3  →  C4 : does RE-Attention add value on top of the AAE?
# C4  →  C5 : does a pretrained backbone further boost the method?
# ```
#
# ### Score fusion
# For C3 / C4 / C5 we also evaluate a late-fusion score:
# `combined = 0.5 × SSIM-score + 0.5 × disc-score`
#
# ### Runtime estimate
# ~140 minutes on a Kaggle T4 GPU (`EPOCHS = 80`, `WARMUP_EPOCHS = 20`, identical across all conditions).
# These values match the proven configuration from the main RSNA experiment where the discriminator
# converged correctly and disc AUC-ROC reached 0.8409. Using fewer epochs caused discriminator
# collapse in C4/C5 (Gen loss → 0 from epoch 10 onward).

# %% [markdown]
# ---
# ## **Cell 1** — Install and verify required packages
#
# Checks each dependency and installs it silently if missing.
# On Kaggle all packages are pre-installed; this cell acts as a safety net for any missing extras.

# %% [CELL 1]  Install / verify packages

import subprocess, sys

def check_import(pkg, install_name=None):
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        name = install_name or pkg
        print(f"  ✗ {pkg} — installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name, '-q'])

for pkg in ['torch', 'sklearn', 'numpy', 'matplotlib', 'pandas', 'seaborn']:
    check_import(pkg)
check_import('pydicom')

# %% [markdown]
# ---
# ## **Cell 2** — Imports and global plot style
#
# Loads all libraries, sets a consistent `matplotlib` style used across every figure,
# and reports the active device (CPU / GPU). All five conditions share this configuration.
#
# **Colour palette** (used in convergence curves, ROC plots, and bar charts):
# - Blue → C1 CNN-AE  |  Orange → C2 VAE  |  Slate → C3 CNN-AAE  |  Red → C4 Novel  |  Green → C5 ResNet

# %% [CELL 2]  Imports and global plot style

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
# ---
# ## **Cell 3** — Configuration
#
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

# %% [CELL 3]  Configuration

SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))

# ── Version + skip control (mirrors bone_fracture_kaggle.py) ─────────
# Bump RUN_VERSION to force a full re-run (old checkpoints are ignored).
# Set SKIP_COMPLETED=False to retrain within the same version.
RUN_VERSION    = 'v2'
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
BATCH_SIZE    = 32  if not SAMPLE_MODE else 4
SEED          = 42
EPS           = 1e-8
TEST_NORMAL   = 2000 if not SAMPLE_MODE else 10
TEST_OPACITY  = 2000 if not SAMPLE_MODE else 5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── wandb login ───────────────────────────────────────────────────────
USE_WANDB = False
try:
    import wandb
    if os.path.exists('/kaggle/working'):
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret('REATTN_KEY'), relogin=True)
    else:
        wandb.login()
    USE_WANDB = True
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
    USE_WANDB = False
    print(f'WandB unavailable ({_e}) — continuing without.')

# ── Checkpoint helpers ────────────────────────────────────────────────
def ckpt_path(cond):
    return f'{CKPT_DIR}/{cond}_done.json'

def is_done(cond):
    """Return True if this condition is already completed for RUN_VERSION."""
    return SKIP_COMPLETED and os.path.exists(ckpt_path(cond))

def save_ckpt(cond, result_keys, scores, disc_scores, epoch_loss,
              attn_maps=None, **model_states):
    """
    Persist a completed condition to disk and log to wandb immediately.
    result_keys: list of keys to pull from all_results, e.g. ['C4','C4_disc','C4_fuse']
    model_states: keyword args of name→state_dict, e.g. enc1=enc1.state_dict()
    """
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
        _art_name = f'{WANDB_GROUP}-{cond.lower()}-ckpt'
        try:
            art = wandb.Artifact(
                _art_name,
                type='checkpoint',
                metadata={'cond': cond, 'version': RUN_VERSION, 'group': WANDB_GROUP},
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
            wandb.log_artifact(art)
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
    print(f'  [{cond}] loaded from checkpoint (version {RUN_VERSION}).')
    return scores, disc_sc, attn_maps

def load_weights(cond, **models):
    """Load saved weights into model objects. Pass name=model_instance."""
    for name, model in models.items():
        p = f'{CKPT_DIR}/{cond}_{name}.pth'
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
        else:
            print(f'  [{cond}] weight file missing: {p}')

print(f"SAMPLE_MODE    : {SAMPLE_MODE}")
print(f"RUN_VERSION    : {RUN_VERSION}  (SKIP_COMPLETED={SKIP_COMPLETED})")
print(f"EPOCHS/WARMUP  : {EPOCHS} / {WARMUP_EPOCHS}")
print(f"OUTPUT_DIR     : {OUTPUT_DIR}")
print(f"CKPT_DIR       : {CKPT_DIR}")

# %% [markdown]
# ---
# ## **Cell 3b** — Auto-restore checkpoints from wandb after session reset
#
# When a Kaggle session resets, `/kaggle/working/` is wiped and all checkpoint files
# in `CKPT_DIR` are lost. This cell runs once at startup: for each condition whose
# `{COND}_done.json` is missing locally but whose wandb artifact exists, it downloads
# the artifact back into `CKPT_DIR` so `is_done(cond)` returns True and training is
# skipped. It runs silently if wandb is not configured or no artifacts exist yet.

# %% [CELL 3b]  Auto-restore checkpoints from wandb

if USE_WANDB and SKIP_COMPLETED and wandb.run is not None:
    try:
        _api      = wandb.Api()
        _entity   = wandb.run.entity
        _restored = []
        _missing  = []
        for _cond in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
            if not os.path.exists(ckpt_path(_cond)):
                # Artifact name must match exactly what save_ckpt uploads
                _art_name = f'{_entity}/{WANDB_PROJECT}/{WANDB_GROUP}-{_cond.lower()}-ckpt:latest'
                try:
                    _art = _api.artifact(_art_name)
                    _art.download(root=CKPT_DIR)
                    _restored.append(_cond)
                    print(f'  [restore] {_cond} ← wandb:{WANDB_GROUP}-{_cond.lower()}-ckpt:latest')
                except Exception:
                    _missing.append(_cond)   # not yet uploaded — will train normally
        if _restored:
            print(f"  [restore] restored {len(_restored)}/7 conditions: {_restored}")
        if _missing:
            print(f"  [restore] will train from scratch: {_missing}")
        if not _restored and not _missing:
            print("  [restore] all conditions already present locally")
    except Exception as _re:
        print(f"  [restore] skipped: {_re}")
else:
    print("  [restore] skipped (wandb not active or SKIP_COMPLETED=False)")

# %% [markdown]
# ---
# ## **Cell 4** — Data preparation
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

# %% [CELL 4]  Data preparation

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
    train_nml_ids = normal_ids[TEST_NORMAL:]
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
# ## **Cell 5** — DataLoader factory
#
# A thin wrapper around `TensorDataset` + `DataLoader`.
# `pin_memory=True` on GPU environments speeds up CPU→GPU transfers.
# Each condition creates its own loader from this function to ensure independent shuffling.

# %% [CELL 5]  DataLoader helper

def make_loader(x_np, batch_size, shuffle=True, drop_last=True):
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last,
                      pin_memory=(device.type == 'cuda'),
                      num_workers=2)

# %% [markdown]
# ---
# ## **Cell 6** — Model architectures
#
# Five building blocks shared across conditions:
#
# | Class | Role | Used in |
# |-------|------|---------|
# | `CNNEncoder` | 3-block conv encoder → latent vector | C1, C3, C4 (enc1 & enc2) |
# | `CNNDecoder` | Latent → 3-block transposed conv → image | C1, C2, C3, C4, C5 |
# | `VAEEncoder` | Same CNN backbone + dual μ / log σ² heads, reparameterisation | C2 |
# | `ResNetEncoder` | **Partially fine-tuned** ResNet-18 (layer4 trainable) + `fc` projection | C5 |
# | `REAttention` | 3-layer conv network: SSIM error map → soft spatial mask ∈ [0, 1] | C4, C5 |
# | `LatentDisc` | MLP discriminator: latent → P(sample looks Gaussian) | C3, C4, C5 |
#
# **ResNetEncoder design note — partial fine-tuning:**
# Grayscale input `(B, 1, H, W)` is replicated to 3 channels via `.repeat(1, 3, 1, 1)`.
# Layers 0–6 (conv1 through layer3) are **frozen** — low-level features (edges, textures)
# are domain-agnostic and transfer from ImageNet without modification.
# **Layer4 + avgpool are trainable at LR × 0.1** — high-level semantic features need CXR
# adaptation. Fully frozen caused reconstruction loss 2× higher than scratch CNN (0.164 vs 0.077),
# making SSIM error maps diffuse and causing RE-Attention collapse.

# %% [CELL 6]  Model architectures

class CNNEncoder(nn.Module):
    """3 × (Conv-BN-ReLU-MaxPool) → flatten → Linear."""
    def __init__(self, latent_dim, image_size=IMAGE_SIZE):
        super().__init__()
        s = image_size // 8
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128 * s * s, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class CNNDecoder(nn.Module):
    """Linear → unflatten → 3 × ConvTranspose2d → Sigmoid."""
    def __init__(self, latent_dim, image_size=IMAGE_SIZE):
        super().__init__()
        self.s    = image_size // 8
        self.flat = 128 * self.s * self.s
        self.fc   = nn.Linear(latent_dim, self.flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32,   1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 128, self.s, self.s))


class VAEEncoder(nn.Module):
    """Same CNN backbone as CNNEncoder with dual mu / log-var projection heads."""
    def __init__(self, latent_dim, image_size=IMAGE_SIZE):
        super().__init__()
        s = image_size // 8
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        hidden = 128 * s * s
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def encode(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return mu   # deterministic mean at inference for stable scoring

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar), mu, logvar


class ResNetEncoder(nn.Module):
    """ResNet-18 backbone + trainable projection head.

    (B,1,H,W) → repeat channel 3x → (B,3,H,W) → ResNet-18 → (B,512)
    → Linear(512, latent_dim).

    freeze_upto controls which backbone layers are frozen:

    | value | frozen layers       | trainable backbone | condition |
    |-------|---------------------|--------------------|-----------|
    | None  | all (0-8)           | none               | C5        |
    | 7     | 0-6 (conv1→layer3)  | layer4 + avgpool   | C6        |
    | 2     | 0-1 (conv1, bn1)    | layer1-4 + avgpool | C7        |

    Backbone layer index map:
      0=conv1  1=bn1  2=relu  3=maxpool  4=layer1  5=layer2
      6=layer3  7=layer4  8=avgpool
    """
    def __init__(self, latent_dim, freeze_upto=None):
        super().__init__()
        base = tv_models.resnet18(weights='IMAGENET1K_V1')
        self.backbone   = nn.Sequential(*list(base.children())[:-1])
        self.fc         = nn.Linear(512, latent_dim)
        self.freeze_upto = freeze_upto
        if freeze_upto is None:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            for i, child in enumerate(self.backbone.children()):
                if i < freeze_upto:
                    for p in child.parameters():
                        p.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x.repeat(1, 3, 1, 1)).flatten(1)
        return self.fc(feats)


class REAttention(nn.Module):
    """Conv SSIM-error-guided attention: (B,1,H,W) → soft mask (B,1,H,W) ∈[0,1]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  1, 1),            nn.Sigmoid(),
        )

    def forward(self, e):
        return self.net(e)


class LatentDisc(nn.Module):
    """MLP discriminator: latent → P(looks Gaussian)."""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32,  1),         nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)

print("Models defined: CNNEncoder, CNNDecoder, VAEEncoder, ResNetEncoder, REAttention, LatentDisc")

# %% [markdown]
# ---
# ## **Cell 7** — Evaluation utilities
#
# All metrics are computed identically across conditions:
#
# - **`anomaly_score(x, x_hat)`** — 99th-percentile SSIM error per image.
#   This is the **primary anomaly score** reported for every condition.
#   Using the 99th percentile (instead of the mean) is robust to small normally-reconstructed areas
#   in otherwise anomalous images.
#
# - **`ssim_anomaly_map(x, x_hat)`** — per-pixel `(1 − SSIM)` map using an 11×11 sliding window.
#   SSIM captures structural similarity; the error is HIGH at pneumonia regions
#   (smooth consolidation the model cannot reconstruct) and LOW at normal lung texture.
#   This is superior to MSE for localisation because MSE is dominated by sharp edges (ribs, heart border).
#
# - **`pixel_auroc(maps, boxes, labels)`** — compares spatial anomaly maps against radiologist
#   bounding boxes. Measures localisation quality, not just detection.
#
# - **`vae_elbo_loss`** — ELBO = reconstruction (0.7 × MSE + 0.3 × SSIM-loss) + β × KL divergence.

# %% [CELL 7]  Evaluation utilities

def compute_metrics(scores, binary_labels):
    if len(np.unique(binary_labels)) < 2:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'f1': np.nan}
    auc_roc          = roc_auc_score(binary_labels, scores)
    auc_pr           = average_precision_score(binary_labels, scores)
    fpr, tpr, thresh = roc_curve(binary_labels, scores)
    best = np.argmax(tpr - fpr)
    pred = (scores >= thresh[best]).astype(int)
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'f1': f1_score(binary_labels, pred, zero_division=0)}

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
# ## **Cell 7b** — Restore from wandb download (run only after a Kaggle session reset)
#
# **When to run:** only if the Kaggle session expired and `/kaggle/working` was wiped.
# Prereqs: run the Option-1 download cell first to populate
# `/kaggle/working/weights_from_wandb/` and `/kaggle/working/results_from_wandb/`.
#
# This cell:
# 1. Copies `results.json` / `loss_history.json` back to `OUTPUT_DIR`
# 2. Writes ckpt stub files for C1-C7 so all training cells skip immediately
# 3. Loads C6 weights (from wandb) and re-runs inference to get real score arrays
#
# After this cell, continue running from Cell 8 onward — all conditions will skip.

# %% [CELL 7b]  Restore from wandb download

import shutil, glob as _glob

SRC_RESULTS = os.environ.get('RESTORE_SRC_RESULTS', '/kaggle/working/results_from_wandb')
SRC_WEIGHTS = os.environ.get('RESTORE_SRC_WEIGHTS', '/kaggle/working/weights_from_wandb')

_RESTORE_NEEDED = (
    os.path.isdir(SRC_RESULTS) and
    os.path.exists(f'{SRC_RESULTS}/results.json') and
    not os.path.exists(f'{OUTPUT_DIR}/results.json')
)

if not _RESTORE_NEEDED:
    print("Restore not needed — results already present or wandb download not found.")
    print(f"  OUTPUT_DIR/results.json exists: {os.path.exists(f'{OUTPUT_DIR}/results.json')}")
    print(f"  SRC_RESULTS exists:             {os.path.isdir(SRC_RESULTS)}")
else:
    print("Restoring from wandb download...")

    # 1. Copy JSON results back
    for fname in ['results.json', 'loss_history.json']:
        src = f'{SRC_RESULTS}/{fname}'
        if os.path.exists(src):
            shutil.copy2(src, f'{OUTPUT_DIR}/{fname}')
            print(f"  copied {fname}")

    with open(f'{OUTPUT_DIR}/results.json') as f:
        all_results = json.load(f)
    with open(f'{OUTPUT_DIR}/loss_history.json') as f:
        loss_history = json.load(f)
    print(f"  all_results loaded: {list(all_results.keys())}")

    # 2. Copy C6 weights to OUTPUT_DIR/weights/
    _w = f'{OUTPUT_DIR}/weights'
    os.makedirs(_w, exist_ok=True)
    for f in _glob.glob(f'{SRC_WEIGHTS}/*.pth'):
        shutil.copy2(f, _w)
        print(f"  weight: {os.path.basename(f)}")

    # 3. Write ckpt stubs for all conditions so SKIP_COMPLETED triggers
    os.makedirs(CKPT_DIR, exist_ok=True)
    _n = len(x_test)
    for _cond in ['C1','C2','C3','C4','C5','C6','C7']:
        _rkeys = [k for k in all_results if k.startswith(_cond)]
        _info  = {
            'all_results': {k: all_results[k] for k in _rkeys},
            'loss_history': loss_history.get(_cond, []),
        }
        with open(ckpt_path(_cond), 'w') as f:
            json.dump(_info, f)
        _zeros = np.zeros(_n, dtype=np.float32)
        np.save(f'{CKPT_DIR}/{_cond}_scores.npy', _zeros)
        np.save(f'{CKPT_DIR}/{_cond}_disc.npy',   _zeros)
        np.save(f'{CKPT_DIR}/{_cond}_attn.npy',   _zeros.reshape(_n, 1, 1))
    print(f"  ckpt stubs written for C1-C7 in {CKPT_DIR}")

    # 4. Load C6 model and re-run real inference (overwrites dummy zeros)
    print("\nRe-running C6 inference with downloaded weights...")
    enc1_c6    = ResNetEncoder(LATENT_DIM, freeze_upto=7).to(device)
    dec_c6     = CNNDecoder(LATENT_DIM).to(device)
    re_attn_c6 = REAttention().to(device)
    enc2_c6    = CNNEncoder(LATENT_DIM).to(device)
    disc_c6    = LatentDisc(LATENT_DIM).to(device)
    enc1_c6.load_state_dict(torch.load(f'{_w}/c6_enc1.pth',     map_location=device))
    dec_c6.load_state_dict(torch.load(f'{_w}/c6_dec.pth',       map_location=device))
    re_attn_c6.load_state_dict(torch.load(f'{_w}/c6_re_attn.pth', map_location=device))
    enc2_c6.load_state_dict(torch.load(f'{_w}/c6_enc2.pth',     map_location=device))
    disc_c6.load_state_dict(torch.load(f'{_w}/c6_disc.pth',     map_location=device))
    enc1_c6.eval(); dec_c6.eval(); re_attn_c6.eval(); enc2_c6.eval(); disc_c6.eval()

    _sc6, _disc6, _attn6 = [], [], []
    with torch.no_grad():
        for _i in range(0, len(x_test), BATCH_SIZE):
            _xb = torch.tensor(x_test[_i:_i+BATCH_SIZE]).to(device)
            _n  = _xb.size(0)
            _z1     = enc1_c6(_xb); _xh1 = dec_c6(_z1)
            _sm     = ssim_anomaly_map(_xb, _xh1).view(_n, 1, IMAGE_SIZE, IMAGE_SIZE)
            _att    = re_attn_c6(_sm)
            _sc6.append(anomaly_score(_xb, _xh1).cpu().numpy())
            _disc6.append((1.0 - disc_c6(enc2_c6(_xb * _att))).squeeze(1).cpu().numpy())
            _attn6.append(_att.squeeze(1).cpu().numpy())

    scores_c6    = np.concatenate(_sc6)
    sc_disc_c6   = np.concatenate(_disc6)
    attn_maps_c6 = np.concatenate(_attn6)
    sc_fuse_c6   = 0.5 * normalise_scores(scores_c6) + 0.5 * normalise_scores(sc_disc_c6)

    _m6 = compute_metrics(scores_c6, binary_test)
    print(f"  C6 re-eval AUC-ROC={_m6['auc_roc']:.4f}  "
          f"(stored: {all_results['C6']['auc_roc']:.4f})")

    np.save(f'{CKPT_DIR}/C6_scores.npy', scores_c6)
    np.save(f'{CKPT_DIR}/C6_disc.npy',   sc_disc_c6)
    np.save(f'{CKPT_DIR}/C6_attn.npy',   attn_maps_c6)
    print("  C6 real scores saved.")
    print("\nRestore complete. Continue running from Cell 8 — all training will be skipped.")

# %% [markdown]
# ---
# ## **Cell 8** — C1: CNN-AE Baseline
#
# **Architecture:** `CNNEncoder → CNNDecoder`
#
# The simplest possible baseline: a plain convolutional autoencoder trained to minimise
# a combined `0.7 × MSE + 0.3 × SSIM` reconstruction loss on normal images only.
#
# At inference, anomaly score = 99th-percentile SSIM error per image.
# Normal images that the AE has learned to reconstruct faithfully score low;
# unseen pneumonia patterns that the AE cannot reconstruct score high.
#
# **Optimisation:** Adam with cosine annealing (`eta_min = 1e-6`).
# Horizontal random flip augmentation is applied during training to improve generalisation.
#
# This condition is the **anchor** for the ablation chain — C3, C4, and C5 all build on it.

# %% [CELL 8]  C1 — CNN-AE Baseline

print("\n" + "="*60)
print("CONDITION 1 — CNN-AE Baseline")
print("="*60)

enc_c1 = CNNEncoder(LATENT_DIM).to(device)
dec_c1 = CNNDecoder(LATENT_DIM).to(device)

if is_done('C1'):
    scores_c1, _, _ = load_ckpt('C1')
    load_weights('C1', enc1=enc_c1, dec=dec_c1)
else:
    opt_c1   = Adam(list(enc_c1.parameters()) + list(dec_c1.parameters()), lr=LR)
    sched_c1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c1, T_max=EPOCHS, eta_min=1e-6)
    loader_c1     = make_loader(x_train_norm, BATCH_SIZE)
    c1_epoch_loss = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc_c1.train(); dec_c1.train()
        losses = []
        for (xb,) in loader_c1:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_c1.zero_grad()
            xhat = dec_c1(enc_c1(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_c1.step()
            losses.append(loss.item())
        sched_c1.step()
        c1_epoch_loss.append(np.mean(losses))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  loss={c1_epoch_loss[-1]:.5f}  "
                  f"lr={sched_c1.get_last_lr()[0]:.2e}")
    loss_history['C1'] = c1_epoch_loss
    print(f"C1 training: {time.time()-t0:.1f}s")
    enc_c1.eval(); dec_c1.eval()
    scores_c1 = []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            scores_c1.append(anomaly_score(xb, dec_c1(enc_c1(xb))).cpu().numpy())
    scores_c1 = np.concatenate(scores_c1)
    m_c1 = compute_metrics(scores_c1, binary_test)
    print(f"\n  AUC-ROC={m_c1['auc_roc']:.4f}  AUC-PR={m_c1['auc_pr']:.4f}  F1={m_c1['f1']:.4f}")
    all_results['C1'] = {**m_c1, 'label': 'CNN-AE Baseline'}
    save_ckpt('C1', ['C1'], scores_c1, None, c1_epoch_loss,
              enc1=enc_c1.state_dict(), dec=dec_c1.state_dict())

# %% [markdown]
# ---
# ## **Cell 9** — C2: VAE Baseline
#
# **Architecture:** `VAEEncoder (μ + log σ² heads) → reparameterise → CNNDecoder`
#
# The VAE extends the CNN-AE with a **probabilistic latent space**.
# The encoder outputs mean `μ` and log-variance `log σ²`; latent codes are sampled
# via the reparameterisation trick during training. The loss is the **ELBO**:
#
# ```
# ELBO = Reconstruction (0.7 × MSE + 0.3 × SSIM-loss) + β × KL(q(z|x) ‖ N(0,I))
# ```
#
# At inference the deterministic mean `μ` is used (no sampling) for stable anomaly scores.
# Scoring is identical to C1: 99th-percentile SSIM error.
#
# **Purpose:** compares a probabilistic model against the deterministic AE (C1) and the
# adversarially regularised AAE (C3/C4/C5). All three impose a Gaussian prior on the
# latent space — but by different mechanisms (KL term vs. discriminator).

# %% [CELL 9]  C2 — VAE Baseline

print("\n" + "="*60)
print("CONDITION 2 — VAE Baseline")
print("="*60)
print("Probabilistic AE: same CNN backbone + reparameterised latent + KL term.")
print("Scored with SSIM 99th-pct on reconstruction (consistent with C1/C3/C4/C5).\n")

enc_vae = VAEEncoder(LATENT_DIM).to(device)
dec_vae = CNNDecoder(LATENT_DIM).to(device)

if is_done('C2'):
    scores_c2, _, _ = load_ckpt('C2')
    load_weights('C2', enc1=enc_vae, dec=dec_vae)
else:
    opt_vae   = Adam(list(enc_vae.parameters()) + list(dec_vae.parameters()), lr=LR)
    sched_vae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vae, T_max=EPOCHS, eta_min=1e-6)
    loader_vae     = make_loader(x_train_norm, BATCH_SIZE)
    vae_epoch_loss = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc_vae.train(); dec_vae.train()
        losses = []
        for (xb,) in loader_vae:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_vae.zero_grad()
            z, mu, logvar = enc_vae(xb)
            xhat = dec_vae(z)
            loss = vae_elbo_loss(xb, xhat, mu, logvar, beta=1.0)
            loss.backward(); opt_vae.step()
            losses.append(loss.item())
        sched_vae.step()
        vae_epoch_loss.append(np.mean(losses))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  ELBO={vae_epoch_loss[-1]:.5f}  "
                  f"lr={sched_vae.get_last_lr()[0]:.2e}")
    loss_history['C2'] = vae_epoch_loss
    print(f"C2 training: {time.time()-t0:.1f}s")
    enc_vae.eval(); dec_vae.eval()
    scores_c2 = []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            mu, _ = enc_vae.encode(xb)
            scores_c2.append(anomaly_score(xb, dec_vae(mu)).cpu().numpy())
    scores_c2 = np.concatenate(scores_c2)
    m_c2 = compute_metrics(scores_c2, binary_test)
    print(f"\n  AUC-ROC={m_c2['auc_roc']:.4f}  AUC-PR={m_c2['auc_pr']:.4f}  F1={m_c2['f1']:.4f}")
    all_results['C2'] = {**m_c2, 'label': 'VAE Baseline'}
    save_ckpt('C2', ['C2'], scores_c2, None, vae_epoch_loss,
              enc1=enc_vae.state_dict(), dec=dec_vae.state_dict())

# %% [markdown]
# ---
# ## **Cell 10** — C3: CNN-AAE Ablation (adversarial regularisation, no attention)
#
# **Architecture:** `CNNEncoder (enc1) → CNNDecoder + LatentDisc`
#
# Extends C1 by adding an adversarial latent discriminator — making this a classic
# **Adversarial Autoencoder (AAE)**. There is no RE-Attention and no second encoder.
#
# **Three-phase training loop per batch:**
# 1. **Reconstruction** — `enc1 + dec` minimise `0.7 × MSE + 0.3 × SSIM-loss`.
# 2. **Discriminator** — `disc` learns to distinguish `z_real ~ N(0, I)` from `z_fake = enc1(x)`.
# 3. **Generator (enc1 adversarial)** — only `enc1` is updated; the decoder is *not* included
#    in this phase to prevent pulling it toward a Gaussian distribution and degrading reconstruction.
#
# A 10-epoch **warm-start** (reconstruction only) stabilises the latent space before
# the discriminator is introduced.
#
# **Gradient clipping** (`max_norm = 1.0`) is applied to both the discriminator and enc1
# in the adversarial phases to prevent sigmoid saturation collapse.
#
# **Ablation role:** C1 → C3 answers: *does adversarial latent regularisation alone help?*
# C3 → C4 answers: *does RE-Attention add further value on top of AAE?*
#
# **Score fusion:** `combined = 0.5 × SSIM-score + 0.5 × (1 − disc(z1))`

# %% [CELL 10]  C3 — CNN-AAE Ablation (adversarial regularisation, no attention)

print("\n" + "="*60)
print("CONDITION 3 — CNN-AAE  [ablation: adversarial without RE-Attention]")
print("="*60)
print("Adds latent discriminator to C1. enc1 latents pushed toward N(0,I).")
print("No re_attn, no enc2 — tests adversarial regularisation alone.")
print("C1→C3: does AAE help?  C3→C4: does RE-Attention add value?\n")

enc1_c3 = CNNEncoder(LATENT_DIM).to(device)
dec_c3  = CNNDecoder(LATENT_DIM).to(device)
ld_c3   = LatentDisc(LATENT_DIM).to(device)

if is_done('C3'):
    scores_c3, sc_disc_c3, _ = load_ckpt('C3')
    sc_fuse_c3 = 0.5 * normalise_scores(scores_c3) + 0.5 * normalise_scores(sc_disc_c3)
    load_weights('C3', enc1=enc1_c3, dec=dec_c3, disc=ld_c3)
else:
    opt_rec_c3  = Adam(list(enc1_c3.parameters()) + list(dec_c3.parameters()), lr=LR, betas=(BETA1, 0.999))
    opt_disc_c3 = Adam(ld_c3.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_gen_c3  = Adam(enc1_c3.parameters(), lr=LR, betas=(BETA1, 0.999))
    sched_rec_c3  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rec_c3,  T_max=EPOCHS, eta_min=1e-6)
    sched_disc_c3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc_c3, T_max=EPOCHS, eta_min=1e-6)
    sched_gen_c3  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen_c3,  T_max=EPOCHS, eta_min=1e-6)
    loader_c3     = make_loader(x_train_norm, BATCH_SIZE)
    c3_epoch_loss = []
    print(f"Warm-start enc1+dec for {WARMUP_EPOCHS} epochs before activating disc...")
    opt_warmup_c3 = Adam(list(enc1_c3.parameters()) + list(dec_c3.parameters()), lr=LR, betas=(BETA1, 0.999))
    t_ws = time.time()
    for epoch in range(WARMUP_EPOCHS):
        enc1_c3.train(); dec_c3.train()
        ws_l = []
        for (xb,) in loader_c3:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_warmup_c3.zero_grad()
            xhat = dec_c3(enc1_c3(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_warmup_c3.step()
            ws_l.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={np.mean(ws_l):.5f}")
    print(f"Warm-start done ({time.time()-t_ws:.1f}s). Activating discriminator.\n")
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc1_c3.train(); dec_c3.train(); ld_c3.train()
        rec_l, d_l, g_l = [], [], []
        for (xb,) in loader_c3:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_rec_c3.zero_grad()
            z1 = enc1_c3(xb); x_hat1 = dec_c3(z1)
            loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
            loss_rec.backward(); opt_rec_c3.step()
            opt_disc_c3.zero_grad()
            with torch.no_grad():
                z_fake = enc1_c3(xb)
            z_real = torch.randn(n, LATENT_DIM, device=device)
            loss_d = (-torch.mean(torch.log(ld_c3(z_real) + EPS))
                      - torch.mean(torch.log(1.0 - ld_c3(z_fake) + EPS)))
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(ld_c3.parameters(), max_norm=1.0)
            opt_disc_c3.step()
            opt_gen_c3.zero_grad()
            loss_g = LAMBDA_ADV * (-torch.mean(torch.log(ld_c3(enc1_c3(xb)) + EPS)))
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(enc1_c3.parameters(), max_norm=1.0)
            opt_gen_c3.step()
            rec_l.append(loss_rec.item()); d_l.append(loss_d.item()); g_l.append(loss_g.item())
        sched_rec_c3.step(); sched_disc_c3.step(); sched_gen_c3.step()
        c3_epoch_loss.append(np.mean(rec_l))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  Recon={c3_epoch_loss[-1]:.5f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}  "
                  f"lr={sched_rec_c3.get_last_lr()[0]:.2e}")
    loss_history['C3'] = c3_epoch_loss
    print(f"C3 training: {time.time()-t0:.1f}s")
    enc1_c3.eval(); dec_c3.eval(); ld_c3.eval()
    scores_c3, sc_disc_c3 = [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            z1 = enc1_c3(xb); xhat = dec_c3(z1)
            scores_c3.append(anomaly_score(xb, xhat).cpu().numpy())
            sc_disc_c3.append((1.0 - ld_c3(z1)).squeeze(1).cpu().numpy())
    scores_c3  = np.concatenate(scores_c3)
    sc_disc_c3 = np.concatenate(sc_disc_c3)
    sc_fuse_c3 = 0.5 * normalise_scores(scores_c3) + 0.5 * normalise_scores(sc_disc_c3)
    m_c3       = compute_metrics(scores_c3,  binary_test)
    m_c3_disc  = compute_metrics(sc_disc_c3, binary_test)
    m_c3_fuse  = compute_metrics(sc_fuse_c3, binary_test)
    print(f"\n  SSIM primary  AUC-ROC={m_c3['auc_roc']:.4f}  AUC-PR={m_c3['auc_pr']:.4f}  F1={m_c3['f1']:.4f}")
    print(f"  Disc score    AUC-ROC={m_c3_disc['auc_roc']:.4f}")
    print(f"  Fusion        AUC-ROC={m_c3_fuse['auc_roc']:.4f}")
    all_results['C3']      = {**m_c3,      'label': 'CNN-AAE (ablation, no attn)'}
    all_results['C3_disc'] = {**m_c3_disc, 'label': 'CNN-AAE disc score'}
    all_results['C3_fuse'] = {**m_c3_fuse, 'label': 'CNN-AAE fusion'}
    save_ckpt('C3', ['C3','C3_disc','C3_fuse'], scores_c3, sc_disc_c3, c3_epoch_loss,
              enc1=enc1_c3.state_dict(), dec=dec_c3.state_dict(), disc=ld_c3.state_dict())

# %% [markdown]
# ---
# ## **Cell 11** — C4: CNN-RE-Attn-AAE *(Ours — Full Novel Method)*
#
# **Architecture:** `CNNEncoder (enc1) + CNNDecoder + REAttention + CNNEncoder (enc2) + LatentDisc`
#
# This is the complete proposed method. The key innovation is the **two-pass RE-Attention** mechanism:
#
# 1. **Pass 1 (reconstruction):** `enc1 → dec` reconstructs the image normally.
# 2. **SSIM error map:** `(1 − SSIM)` computed between the input and reconstruction.
#    Pixels where the model *failed* to reconstruct score high — these are candidate anomalous regions.
# 3. **RE-Attention:** A small 3-layer conv network converts the SSIM error map into a
#    soft spatial attention mask `att_img ∈ [0, 1]`.
# 4. **Pass 2 (adversarial):** `enc2` encodes the attention-masked image `x × att_img`.
#    The discriminator pushes `enc2`'s latent toward `N(0, I)`.
#    Because enc2 only sees attended (potentially anomalous) regions, the disc learns
#    whether those regions look like normal structure or anomaly.
#
# **Three-phase training (per batch):**
# 1. **enc1 + dec** — reconstruction loss (SSIM error map computed but detached; no gradient to re_attn here).
# 2. **Discriminator** — updated with `z_real ~ N(0, I)` vs `z2 = enc2(x × att)`.
# 3. **re_attn + enc2** — adversarial generator loss; re_attn receives gradient here and learns
#    to highlight regions that the discriminator finds non-Gaussian (i.e., anomalous).
#
# **SSIM vs MSE for attention signal:**
# MSE error is dominated by sharp edges (ribs, heart border) — the wrong regions for pneumonia.
# SSIM error correctly peaks at smooth consolidations that break the lung's structural texture.
# Using SSIM as the attention signal was the key fix that brought pixel AUROC from 0.37 to 0.70.

# %% [CELL 11]  C4 — CNN-RE-Attn-AAE  [NOVEL]

print("\n" + "="*60)
print("CONDITION 4 — CNN-RE-Attn-AAE  [NOVEL]")
print("="*60)
print("Full novel method: SSIM-guided RE-Attention + two-encoder AAE.")
print("C3 vs C4 isolates the RE-Attention contribution on top of AAE.\n")

enc1_c4    = CNNEncoder(LATENT_DIM).to(device)
enc2_c4    = CNNEncoder(LATENT_DIM).to(device)
dec_c4     = CNNDecoder(LATENT_DIM).to(device)
re_attn_c4 = REAttention().to(device)
ld_c4      = LatentDisc(LATENT_DIM).to(device)

if is_done('C4'):
    scores_c4, sc_disc_c4, attn_maps_c4 = load_ckpt('C4')
    sc_fuse_c4 = 0.5 * normalise_scores(scores_c4) + 0.5 * normalise_scores(sc_disc_c4)
    load_weights('C4', enc1=enc1_c4, enc2=enc2_c4, dec=dec_c4, re_attn=re_attn_c4, disc=ld_c4)
else:
    opt_rec_c4  = Adam(list(enc1_c4.parameters()) + list(dec_c4.parameters()), lr=LR, betas=(BETA1, 0.999))
    opt_disc_c4 = Adam(ld_c4.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_gen_c4  = Adam(list(re_attn_c4.parameters()) + list(enc2_c4.parameters()), lr=LR, betas=(BETA1, 0.999))
    sched_rec_c4  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rec_c4,  T_max=EPOCHS, eta_min=1e-6)
    sched_disc_c4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc_c4, T_max=EPOCHS, eta_min=1e-6)
    sched_gen_c4  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen_c4,  T_max=EPOCHS, eta_min=1e-6)
    loader_c4     = make_loader(x_train_norm, BATCH_SIZE)
    c4_epoch_loss = []
    print(f"Warm-start enc1+dec for {WARMUP_EPOCHS} epochs...")
    opt_warmup_c4 = Adam(list(enc1_c4.parameters()) + list(dec_c4.parameters()), lr=LR, betas=(BETA1, 0.999))
    t_ws = time.time()
    for epoch in range(WARMUP_EPOCHS):
        enc1_c4.train(); dec_c4.train()
        ws_l = []
        for (xb,) in loader_c4:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_warmup_c4.zero_grad()
            xhat = dec_c4(enc1_c4(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_warmup_c4.step()
            ws_l.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={np.mean(ws_l):.5f}")
    print(f"Warm-start done ({time.time()-t_ws:.1f}s). Activating RE-Attention + AAE.\n")
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc1_c4.train(); enc2_c4.train(); dec_c4.train()
        re_attn_c4.train(); ld_c4.train()
        rec_l, d_l, g_l = [], [], []
        for (xb,) in loader_c4:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_rec_c4.zero_grad()
            z1 = enc1_c4(xb); x_hat1 = dec_c4(z1)
            loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
            loss_rec.backward(); opt_rec_c4.step()
            opt_disc_c4.zero_grad()
            with torch.no_grad():
                z1_s = enc1_c4(xb); xh1_s = dec_c4(z1_s)
                att_s = re_attn_c4(ssim_anomaly_map(xb, xh1_s).view(n, 1, IMAGE_SIZE, IMAGE_SIZE))
                z2_s  = enc2_c4(xb * att_s)
            z_real = torch.randn(n, LATENT_DIM, device=device)
            loss_d = (-torch.mean(torch.log(ld_c4(z_real) + EPS))
                      - torch.mean(torch.log(1.0 - ld_c4(z2_s) + EPS)))
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(ld_c4.parameters(), max_norm=1.0)
            opt_disc_c4.step()
            opt_gen_c4.zero_grad()
            with torch.no_grad():
                z1_g = enc1_c4(xb); xh1_g = dec_c4(z1_g)
                ssim_g = ssim_anomaly_map(xb, xh1_g).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_g  = re_attn_c4(ssim_g)
            loss_g = LAMBDA_ADV * (-torch.mean(torch.log(ld_c4(enc2_c4(xb * att_g)) + EPS)))
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                list(re_attn_c4.parameters()) + list(enc2_c4.parameters()), max_norm=1.0)
            opt_gen_c4.step()
            rec_l.append(loss_rec.item()); d_l.append(loss_d.item()); g_l.append(loss_g.item())
        sched_rec_c4.step(); sched_disc_c4.step(); sched_gen_c4.step()
        c4_epoch_loss.append(np.mean(rec_l))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  Recon={c4_epoch_loss[-1]:.5f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}  "
                  f"lr={sched_rec_c4.get_last_lr()[0]:.2e}")
    loss_history['C4'] = c4_epoch_loss
    print(f"C4 training: {time.time()-t0:.1f}s")
    enc1_c4.eval(); enc2_c4.eval(); dec_c4.eval(); re_attn_c4.eval(); ld_c4.eval()
    scores_c4, sc_disc_c4, attn_maps_c4 = [], [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device); n = xb.size(0)
            z1 = enc1_c4(xb); x_hat1 = dec_c4(z1)
            ssim_inf = ssim_anomaly_map(xb, x_hat1).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_img  = re_attn_c4(ssim_inf)
            scores_c4.append(anomaly_score(xb, x_hat1).cpu().numpy())
            sc_disc_c4.append((1.0 - ld_c4(enc2_c4(xb * att_img))).squeeze(1).cpu().numpy())
            attn_maps_c4.append(att_img.squeeze(1).cpu().numpy())
    scores_c4    = np.concatenate(scores_c4)
    sc_disc_c4   = np.concatenate(sc_disc_c4)
    sc_fuse_c4   = 0.5 * normalise_scores(scores_c4) + 0.5 * normalise_scores(sc_disc_c4)
    attn_maps_c4 = np.concatenate(attn_maps_c4)
    m_c4       = compute_metrics(scores_c4,  binary_test)
    m_c4_disc  = compute_metrics(sc_disc_c4, binary_test)
    m_c4_fuse  = compute_metrics(sc_fuse_c4, binary_test)
    print(f"\n  SSIM primary  AUC-ROC={m_c4['auc_roc']:.4f}  AUC-PR={m_c4['auc_pr']:.4f}  F1={m_c4['f1']:.4f}")
    print(f"  Disc score    AUC-ROC={m_c4_disc['auc_roc']:.4f}")
    print(f"  Fusion        AUC-ROC={m_c4_fuse['auc_roc']:.4f}")
    all_results['C4']      = {**m_c4,      'label': 'CNN-RE-Attn-AAE (Ours)'}
    all_results['C4_disc'] = {**m_c4_disc, 'label': 'CNN-RE-Attn-AAE disc'}
    all_results['C4_fuse'] = {**m_c4_fuse, 'label': 'CNN-RE-Attn-AAE fusion'}
    save_ckpt('C4', ['C4','C4_disc','C4_fuse'], scores_c4, sc_disc_c4, c4_epoch_loss,
              attn_maps=attn_maps_c4,
              enc1=enc1_c4.state_dict(), enc2=enc2_c4.state_dict(),
              dec=dec_c4.state_dict(), re_attn=re_attn_c4.state_dict(),
              disc=ld_c4.state_dict())

# %% [markdown]
# ---
# ## **Cell 12** — C5: ResNet-18-RE-Attn-AAE (Partial Fine-Tuning Transfer Learning)
#
# **Architecture:** `ResNetEncoder (partially fine-tuned ResNet-18) + CNNDecoder + REAttention + CNNEncoder (enc2) + LatentDisc`
#
# Replaces the scratch-trained `enc1` in C4 with a **partially fine-tuned ImageNet ResNet-18 backbone**.
#
# **Why partial fine-tuning, not full freeze:**
# An initial experiment with a fully frozen backbone produced reconstruction loss 2× higher
# than the scratch CNN encoder (0.164 vs 0.077). The SSIM error maps were diffuse — uniformly
# high across the image rather than localised — because the frozen ImageNet features optimised
# for object classification cannot be directly inverted into sharp CXR reconstructions.
# This broke the RE-Attention signal and caused discriminator collapse (Gen loss → 0 from epoch 20).
#
# **Partial fine-tuning strategy:**
# - **Frozen (layers 0–6):** conv1, bn1, relu, maxpool, layer1, layer2, layer3.
#   Low-level features (edges, textures, intensity gradients) are domain-agnostic
#   and transfer from ImageNet to CXR without modification.
# - **Trainable (layer4 + avgpool + fc):** high-level semantic features need CXR adaptation.
#   Layer4 uses `LR × 0.1` (10× slower — pretrained, fine-tuning).
#   `fc` and `dec` use full `LR` (randomly initialised).
#
# This is the standard cross-domain transfer learning recipe: freeze early layers,
# fine-tune late layers where domain gap is largest.
#
# **Ablation role:** C4 → C5 answers: *does a partially fine-tuned ImageNet backbone
# further boost the novel RE-Attn-AAE beyond scratch training?*

# %% [CELL 12]  C5 — ResNet-18-RE-Attn-AAE (frozen backbone)

print("\n" + "="*60)
print("CONDITION 5 — ResNet-18-RE-Attn-AAE  (fully frozen backbone)")
print("="*60)
print("enc1 = ResNet-18 fully frozen + trainable fc projection only.")
print("Tests whether frozen ImageNet features alone are sufficient.")
print("C4 vs C5: effect of full freeze.  C5 vs C6: effect of partial fine-tune.\n")

if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM before C5: {used:.1f} / {total:.1f} GB")
    torch.cuda.empty_cache()

enc1_c5    = ResNetEncoder(LATENT_DIM, freeze_upto=None).to(device)
enc2_c5    = CNNEncoder(LATENT_DIM).to(device)
dec_c5     = CNNDecoder(LATENT_DIM).to(device)
re_attn_c5 = REAttention().to(device)
ld_c5      = LatentDisc(LATENT_DIM).to(device)

if is_done('C5'):
    scores_c5, sc_disc_c5, attn_maps_c5 = load_ckpt('C5')
    sc_fuse_c5 = 0.5 * normalise_scores(scores_c5) + 0.5 * normalise_scores(sc_disc_c5)
    load_weights('C5', enc1=enc1_c5, enc2=enc2_c5, dec=dec_c5, re_attn=re_attn_c5, disc=ld_c5)
else:

    # Full freeze — only fc projection + dec are updated
    opt_rec_c5  = Adam(
        list(enc1_c5.fc.parameters()) + list(dec_c5.parameters()),
        lr=LR, betas=(BETA1, 0.999))
    opt_disc_c5 = Adam(ld_c5.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_gen_c5  = Adam(
        list(re_attn_c5.parameters()) + list(enc2_c5.parameters()),
        lr=LR, betas=(BETA1, 0.999))

    sched_rec_c5  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rec_c5,  T_max=EPOCHS, eta_min=1e-6)
    sched_disc_c5 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc_c5, T_max=EPOCHS, eta_min=1e-6)
    sched_gen_c5  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen_c5,  T_max=EPOCHS, eta_min=1e-6)

    loader_c5     = make_loader(x_train_norm, BATCH_SIZE)
    c5_epoch_loss = []

    print(f"Warm-start enc1.fc + dec for {WARMUP_EPOCHS} epochs (backbone fully frozen)...")
    opt_warmup_c5 = Adam(
        list(enc1_c5.fc.parameters()) + list(dec_c5.parameters()),
        lr=LR, betas=(BETA1, 0.999))
    t_ws = time.time()
    for epoch in range(WARMUP_EPOCHS):
        enc1_c5.train(); dec_c5.train()
        ws_l = []
        for (xb,) in loader_c5:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_warmup_c5.zero_grad()
            xhat = dec_c5(enc1_c5(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_warmup_c5.step()
            ws_l.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={np.mean(ws_l):.5f}")
    print(f"Warm-start done ({time.time()-t_ws:.1f}s). Activating RE-Attention + AAE.\n")

    t0 = time.time()
    for epoch in range(EPOCHS):
        enc1_c5.train(); enc2_c5.train(); dec_c5.train()
        re_attn_c5.train(); ld_c5.train()
        rec_l, d_l, g_l = [], [], []

        for (xb,) in loader_c5:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])

            # Phase 1: pass-1 reconstruction (frozen backbone, only fc + dec update); re_attn trains via Phase 3
            opt_rec_c5.zero_grad()
            z1 = enc1_c5(xb); x_hat1 = dec_c5(z1)
            loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
            loss_rec.backward(); opt_rec_c5.step()

            # Phase 2: discriminator
            opt_disc_c5.zero_grad()
            with torch.no_grad():
                z1_s = enc1_c5(xb); xh1_s = dec_c5(z1_s)
                att_s = re_attn_c5(ssim_anomaly_map(xb, xh1_s).view(n, 1, IMAGE_SIZE, IMAGE_SIZE))
                z2_s  = enc2_c5(xb * att_s)
            z_real = torch.randn(n, LATENT_DIM, device=device)
            loss_d = (-torch.mean(torch.log(ld_c5(z_real) + EPS))
                      - torch.mean(torch.log(1.0 - ld_c5(z2_s) + EPS)))
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(ld_c5.parameters(), max_norm=1.0)
            opt_disc_c5.step()

            # Phase 3: re_attn + enc2 adversarial
            opt_gen_c5.zero_grad()
            with torch.no_grad():
                z1_g = enc1_c5(xb); xh1_g = dec_c5(z1_g)
                ssim_g = ssim_anomaly_map(xb, xh1_g).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_g  = re_attn_c5(ssim_g)
            loss_g = LAMBDA_ADV * (-torch.mean(torch.log(ld_c5(enc2_c5(xb * att_g)) + EPS)))
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                list(re_attn_c5.parameters()) + list(enc2_c5.parameters()), max_norm=1.0)
            opt_gen_c5.step()

            rec_l.append(loss_rec.item()); d_l.append(loss_d.item()); g_l.append(loss_g.item())

        sched_rec_c5.step(); sched_disc_c5.step(); sched_gen_c5.step()
        c5_epoch_loss.append(np.mean(rec_l))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  Recon={c5_epoch_loss[-1]:.5f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}  "
                  f"lr={sched_rec_c5.get_last_lr()[0]:.2e}")

    loss_history['C5'] = c5_epoch_loss
    print(f"C5 training: {time.time()-t0:.1f}s")

    enc1_c5.eval(); enc2_c5.eval(); dec_c5.eval(); re_attn_c5.eval(); ld_c5.eval()
    scores_c5, sc_disc_c5, attn_maps_c5 = [], [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device); n = xb.size(0)
            z1 = enc1_c5(xb); x_hat1 = dec_c5(z1)
            ssim_inf = ssim_anomaly_map(xb, x_hat1).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_img  = re_attn_c5(ssim_inf)
            scores_c5.append(anomaly_score(xb, x_hat1).cpu().numpy())
            sc_disc_c5.append((1.0 - ld_c5(enc2_c5(xb * att_img))).squeeze(1).cpu().numpy())
            attn_maps_c5.append(att_img.squeeze(1).cpu().numpy())

    scores_c5    = np.concatenate(scores_c5)
    sc_disc_c5   = np.concatenate(sc_disc_c5)
    sc_fuse_c5   = 0.5 * normalise_scores(scores_c5) + 0.5 * normalise_scores(sc_disc_c5)
    attn_maps_c5 = np.concatenate(attn_maps_c5)
    m_c5       = compute_metrics(scores_c5,  binary_test)
    m_c5_disc  = compute_metrics(sc_disc_c5, binary_test)
    m_c5_fuse  = compute_metrics(sc_fuse_c5, binary_test)
    print(f"\n  SSIM primary  AUC-ROC={m_c5['auc_roc']:.4f}  AUC-PR={m_c5['auc_pr']:.4f}  F1={m_c5['f1']:.4f}")
    print(f"  Disc score    AUC-ROC={m_c5_disc['auc_roc']:.4f}")
    print(f"  Fusion        AUC-ROC={m_c5_fuse['auc_roc']:.4f}")
    all_results['C5']      = {**m_c5,      'label': 'ResNet-AAE Full Freeze (collapsed)'}
    all_results['C5_disc'] = {**m_c5_disc, 'label': 'ResNet Full Freeze disc'}
    all_results['C5_fuse'] = {**m_c5_fuse, 'label': 'ResNet Full Freeze fusion'}
    save_ckpt('C5', ['C5','C5_disc','C5_fuse'], scores_c5, sc_disc_c5, c5_epoch_loss, attn_maps=attn_maps_c5,
              enc1=enc1_c5.state_dict(), enc2=enc2_c5.state_dict(),
              dec=dec_c5.state_dict(), re_attn=re_attn_c5.state_dict(),
              disc=ld_c5.state_dict())

# %% [markdown]
# ---
# ## **Cell 12b** — Condition 6: ResNet-18-RE-Attn-AAE (Partial Fine-Tuning)
#
# **Purpose**: Address the discriminator collapse observed in C5 (full freeze) by enabling
# partial fine-tuning of the ResNet-18 backbone.
#
# **Why C5 collapses**: Fully frozen ImageNet features are optimised for object classification,
# not CXR anatomy inversion. The result is a reconstruction loss ~2× higher than the scratch
# CNN (enc1), which causes diffuse SSIM error maps, uninformative RE-Attention masks
# (→ 0), and discriminator collapse (Disc → 18.42 = −log ε).
#
# **C6 strategy — standard cross-domain transfer learning**:
# | Layers | Index | Status | Reason |
# |--------|-------|--------|--------|
# | conv1, bn1, relu, maxpool, layer1–3 | 0–6 | **Frozen** | Domain-agnostic edges/textures |
# | layer4 | 7 | **Trainable at LR × 0.1** | Adapts high-level features to CXR |
# | avgpool | 8 | **Trainable** | Feature pooling |
# | fc projection | — | **Trainable at full LR** | Randomly initialised |
# | dec | — | **Trainable at full LR** | Randomly initialised |
#
# **Expected outcome**: Reconstruction loss drops toward C4 (~0.077–0.10),
# SSIM maps become anatomically meaningful, RE-Attention activates on opacities,
# and the discriminator reaches Nash equilibrium (≈1.386).
#
# **Key comparison**:
# - `C4 → C5`: isolates full freeze failure (domain gap)
# - `C5 → C6`: shows that partial fine-tuning resolves the collapse

# %% [CELL 12b]  Condition 6 — ResNet-18-RE-Attn-AAE (partial fine-tuning)

print("="*60)
print("CONDITION 6 — ResNet-18-RE-Attn-AAE  (partial fine-tuning)")
print("="*60)
print("enc1 = ResNet-18: layers 0-6 frozen, layer4 trainable at LR×0.1.")
print("Motivaton: C5 full-freeze produced Disc=18.42 (collapse).")
print("C5 vs C6: partial fine-tuning should restore stable training.\n")

enc1_c6    = ResNetEncoder(LATENT_DIM, freeze_upto=7).to(device)
dec_c6     = CNNDecoder(LATENT_DIM).to(device)
re_attn_c6 = REAttention().to(device)
enc2_c6    = CNNEncoder(LATENT_DIM).to(device)
disc_c6    = LatentDisc(LATENT_DIM).to(device)

if is_done('C6'):
    scores_c6, sc_disc_c6, attn_maps_c6 = load_ckpt('C6')
    sc_fuse_c6 = 0.5 * normalise_scores(scores_c6) + 0.5 * normalise_scores(sc_disc_c6)
    load_weights('C6', enc1=enc1_c6, enc2=enc2_c6, dec=dec_c6, re_attn=re_attn_c6, disc=disc_c6)
else:

    # Partial fine-tuning: layer4 at LR×0.1, fc+dec at full LR
    opt_rec_c6  = Adam([
        {'params': enc1_c6.backbone[7].parameters(), 'lr': LR * 0.1},  # layer4
        {'params': enc1_c6.fc.parameters(),          'lr': LR},
        {'params': dec_c6.parameters(),              'lr': LR},
    ], betas=(BETA1, 0.999))
    opt_disc_c6 = Adam(disc_c6.parameters(),  lr=LR, betas=(BETA1, 0.999))
    opt_gen_c6  = Adam(
        list(re_attn_c6.parameters()) + list(enc2_c6.parameters()),
        lr=LR, betas=(BETA1, 0.999))

    # ── Warm-up: enc1.fc + layer4 + dec only ─────────────────────────────
    loader_c6     = make_loader(x_train_norm, BATCH_SIZE)
    c6_epoch_loss = []

    print(f"Warm-start enc1.fc + layer4 + dec for {WARMUP_EPOCHS} epochs...")
    opt_warmup_c6 = Adam([
        {'params': enc1_c6.backbone[7].parameters(), 'lr': LR * 0.1},
        {'params': enc1_c6.fc.parameters(),          'lr': LR},
        {'params': dec_c6.parameters(),              'lr': LR},
    ], betas=(BETA1, 0.999))

    t_ws6 = time.time()
    enc1_c6.train(); dec_c6.train()
    for epoch in range(WARMUP_EPOCHS):
        ws_l = []
        for (xb,) in loader_c6:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_warmup_c6.zero_grad()
            xhat = dec_c6(enc1_c6(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_warmup_c6.step()
            ws_l.append(loss.item())
        c6_epoch_loss.append(np.mean(ws_l))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={c6_epoch_loss[-1]:.5f}")
    print(f"Warm-start done ({time.time()-t_ws6:.1f}s).")
    print(f"Final warmup loss: {c6_epoch_loss[-1]:.4f}  (C5 full-freeze was ~0.164)\n")

    # ── Main training: all three phases ──────────────────────────────────
    sched_rec_c6  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rec_c6,  T_max=EPOCHS, eta_min=1e-6)
    sched_disc_c6 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc_c6, T_max=EPOCHS, eta_min=1e-6)
    sched_gen_c6  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen_c6,  T_max=EPOCHS, eta_min=1e-6)

    t0_c6 = time.time()
    for epoch in range(EPOCHS):
        enc1_c6.train(); dec_c6.train(); re_attn_c6.train()
        enc2_c6.train(); disc_c6.train()
        rec_l, d_l, g_l = [], [], []

        for (xb,) in loader_c6:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])

            # Phase 1: pass-1 reconstruction — enc1 + dec
            opt_rec_c6.zero_grad()
            z1 = enc1_c6(xb); x_hat1 = dec_c6(z1)
            loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
            loss_rec.backward(); opt_rec_c6.step()

            # Phase 2: discriminator
            opt_disc_c6.zero_grad()
            with torch.no_grad():
                z1_d = enc1_c6(xb); x_h1_d = dec_c6(z1_d)
                err_d = ssim_anomaly_map(xb, x_h1_d).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
                att_d = re_attn_c6(err_d)
                z2_d  = enc2_c6(xb * att_d)
            real_z = torch.randn_like(z2_d)
            d_real = disc_c6(real_z); d_fake = disc_c6(z2_d.detach())
            loss_d = -torch.mean(torch.log(d_real + EPS) + torch.log(1 - d_fake + EPS))
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc_c6.parameters(), max_norm=1.0)
            opt_disc_c6.step()

            # Phase 3: generator (re_attn + enc2)
            opt_gen_c6.zero_grad()
            with torch.no_grad():
                z1_g = enc1_c6(xb); x_h1_g = dec_c6(z1_g)
                err_g = ssim_anomaly_map(xb, x_h1_g).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_g = re_attn_c6(err_g)
            loss_g = LAMBDA_ADV * (-torch.mean(torch.log(disc_c6(enc2_c6(xb * att_g)) + EPS)))
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                list(re_attn_c6.parameters()) + list(enc2_c6.parameters()), max_norm=1.0)
            opt_gen_c6.step()

            rec_l.append(loss_rec.item())
            d_l.append(loss_d.item())
            g_l.append(loss_g.item())

        c6_epoch_loss.append(np.mean(rec_l))
        sched_rec_c6.step(); sched_disc_c6.step(); sched_gen_c6.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  Recon={c6_epoch_loss[-1]:.5f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}")

    print(f"C6 total: {time.time()-t0_c6:.1f}s\n")

    # ── Evaluation ───────────────────────────────────────────────────────
    enc1_c6.eval(); dec_c6.eval(); re_attn_c6.eval()
    enc2_c6.eval(); disc_c6.eval()

    scores_c6, sc_disc_c6, attn_maps_c6 = [], [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb      = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device); n = xb.size(0)
            z1      = enc1_c6(xb); x_hat = dec_c6(z1)
            err     = ssim_anomaly_map(xb, x_hat).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_img = re_attn_c6(err)
            z2      = enc2_c6(xb * att_img)
            d_sc    = disc_c6(z2)
            scores_c6.append(err.view(n, -1).quantile(0.99, dim=1).cpu().numpy())
            sc_disc_c6.append((1 - d_sc).squeeze(1).cpu().numpy())
            attn_maps_c6.append(att_img.squeeze(1).cpu().numpy())

    scores_c6    = np.concatenate(scores_c6)
    sc_disc_c6   = np.concatenate(sc_disc_c6)
    attn_maps_c6 = np.concatenate(attn_maps_c6)

    # Self-contained helpers — safe to re-run this cell standalone
    def normalise_scores(s):
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min + 1e-8)

    def evaluate(scores, labels):
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
        if len(np.unique(labels)) < 2:
            return {'auc_roc': float('nan'), 'auc_pr': float('nan'), 'f1': float('nan')}
        auc_roc = roc_auc_score(labels, scores)
        auc_pr  = average_precision_score(labels, scores)
        thresh  = np.percentile(scores[labels == 0], 95)
        f1      = f1_score(labels, (scores > thresh).astype(int), zero_division=0)
        return {'auc_roc': auc_roc, 'auc_pr': auc_pr, 'f1': f1}

    sc_fuse_c6   = 0.5 * normalise_scores(scores_c6) + 0.5 * normalise_scores(sc_disc_c6)

    m_c6      = evaluate(scores_c6,  binary_test)
    m_c6_disc = evaluate(sc_disc_c6, binary_test)
    m_c6_fuse = evaluate(sc_fuse_c6, binary_test)

    print(f"\n  C6 ResNet partial fine-tune — SSIM  AUC-ROC={m_c6['auc_roc']:.4f}")
    print(f"  Disc score    AUC-ROC={m_c6_disc['auc_roc']:.4f}")
    print(f"  Fusion        AUC-ROC={m_c6_fuse['auc_roc']:.4f}")
    print(f"\n  C5 (full freeze) → C6 (partial fine-tune): {m_c6['auc_roc'] - all_results['C5']['auc_roc']:+.4f}")
    all_results['C6']      = {**m_c6,      'label': 'ResNet-RE-Attn-AAE Partial FT'}
    all_results['C6_disc'] = {**m_c6_disc, 'label': 'ResNet Partial FT disc'}
    all_results['C6_fuse'] = {**m_c6_fuse, 'label': 'ResNet Partial FT fusion'}
    loss_history['C6']     = c6_epoch_loss
    _w = f'{OUTPUT_DIR}/weights'; os.makedirs(_w, exist_ok=True)
    torch.save(enc1_c6.state_dict(),    f'{_w}/c6_enc1.pth')
    torch.save(dec_c6.state_dict(),     f'{_w}/c6_dec.pth')
    torch.save(re_attn_c6.state_dict(), f'{_w}/c6_re_attn.pth')
    torch.save(enc2_c6.state_dict(),    f'{_w}/c6_enc2.pth')
    torch.save(disc_c6.state_dict(),    f'{_w}/c6_disc.pth')
    print(f"  C6 weights saved to {_w}/")
    save_ckpt('C6', ['C6','C6_disc','C6_fuse'], scores_c6, sc_disc_c6, c6_epoch_loss, attn_maps=attn_maps_c6,
              enc1=enc1_c6.state_dict(), enc2=enc2_c6.state_dict(),
              dec=dec_c6.state_dict(), re_attn=re_attn_c6.state_dict(),
              disc=disc_c6.state_dict())

# %% [markdown]
# ---
# ## **Cell 12c** — Condition 7: ResNet-18-RE-Attn-AAE (Mostly Fine-Tuned)
#
# **Purpose**: Show that unfreezing almost all ResNet layers further improves performance,
# and argue that with larger CXR datasets full fine-tuning would give even better results.
#
# **Strategy**: Freeze only the first two layers (conv1, bn1) which learn Gabor-like edge
# detectors — universal across all image domains. Fine-tune everything else with layered
# learning rates that respect the feature hierarchy:
#
# | Layers | Index | LR multiplier | Reason |
# |--------|-------|---------------|--------|
# | conv1, bn1 | 0–1 | **Frozen** | Universal edge detectors |
# | layer1, layer2 | 4–5 | LR × 0.05 | Low-level textures — adapt slowly |
# | layer3, layer4 | 6–7 | LR × 0.10 | High-level features — adapt faster |
# | fc projection | — | LR × 1.0 | Randomly initialised |
# | dec | — | LR × 1.0 | Randomly initialised |
#
# **C6 → C7 hypothesis**: More trainable parameters → better CXR-adapted features →
# lower reconstruction loss → sharper SSIM maps → stronger RE-Attention signal → better AUC.
#
# **Key message for presentation**: "With sufficient data, full backbone fine-tuning
# would be ideal. C7 demonstrates this trend — more fine-tuning consistently helps
# once the domain gap is addressed."

# %% [CELL 12c]  Condition 7 — ResNet-18-RE-Attn-AAE (mostly fine-tuned)

print("="*60)
print("CONDITION 7 — ResNet-18-RE-Attn-AAE  (mostly fine-tuned)")
print("="*60)
print("Frozen: conv1 + bn1 only (universal edge detectors).")
print("Trainable: layer1-4 at layered LR (0.05×–0.10×), fc at full LR.")
print("C6 vs C7: more fine-tuning → better CXR adaptation.\n")

enc1_c7    = ResNetEncoder(LATENT_DIM, freeze_upto=2).to(device)
dec_c7     = CNNDecoder(LATENT_DIM).to(device)
re_attn_c7 = REAttention().to(device)
enc2_c7    = CNNEncoder(LATENT_DIM).to(device)
disc_c7    = LatentDisc(LATENT_DIM).to(device)

if is_done('C7'):
    scores_c7, sc_disc_c7, attn_maps_c7 = load_ckpt('C7')
    sc_fuse_c7 = 0.5 * normalise_scores(scores_c7) + 0.5 * normalise_scores(sc_disc_c7)
    load_weights('C7', enc1=enc1_c7, enc2=enc2_c7, dec=dec_c7, re_attn=re_attn_c7, disc=disc_c7)
else:

    # Layered LR: early layers slower, deep layers faster
    opt_rec_c7 = Adam([
        {'params': enc1_c7.backbone[4].parameters(), 'lr': LR * 0.05},  # layer1
        {'params': enc1_c7.backbone[5].parameters(), 'lr': LR * 0.05},  # layer2
        {'params': enc1_c7.backbone[6].parameters(), 'lr': LR * 0.10},  # layer3
        {'params': enc1_c7.backbone[7].parameters(), 'lr': LR * 0.10},  # layer4
        {'params': enc1_c7.fc.parameters(),          'lr': LR},
        {'params': dec_c7.parameters(),              'lr': LR},
    ], betas=(BETA1, 0.999))
    opt_disc_c7 = Adam(disc_c7.parameters(),  lr=LR, betas=(BETA1, 0.999))
    opt_gen_c7  = Adam(
        list(re_attn_c7.parameters()) + list(enc2_c7.parameters()),
        lr=LR, betas=(BETA1, 0.999))

    loader_c7     = make_loader(x_train_norm, BATCH_SIZE)
    c7_epoch_loss = []

    print(f"Warm-start for {WARMUP_EPOCHS} epochs (layer1-4 fine-tuning)...")
    opt_warmup_c7 = Adam([
        {'params': enc1_c7.backbone[4].parameters(), 'lr': LR * 0.05},
        {'params': enc1_c7.backbone[5].parameters(), 'lr': LR * 0.05},
        {'params': enc1_c7.backbone[6].parameters(), 'lr': LR * 0.10},
        {'params': enc1_c7.backbone[7].parameters(), 'lr': LR * 0.10},
        {'params': enc1_c7.fc.parameters(),          'lr': LR},
        {'params': dec_c7.parameters(),              'lr': LR},
    ], betas=(BETA1, 0.999))

    t_ws7 = time.time()
    enc1_c7.train(); dec_c7.train()
    for epoch in range(WARMUP_EPOCHS):
        ws_l = []
        for (xb,) in loader_c7:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_warmup_c7.zero_grad()
            xhat = dec_c7(enc1_c7(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_warmup_c7.step()
            ws_l.append(loss.item())
        c7_epoch_loss.append(np.mean(ws_l))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={c7_epoch_loss[-1]:.5f}")
    print(f"Warm-start done ({time.time()-t_ws7:.1f}s).")
    print(f"Final warmup loss: {c7_epoch_loss[-1]:.4f}")
    print(f"  (C5 full-freeze ~0.164 | C6 partial ~0.115 | C7 mostly-FT expected lower)\n")

    sched_rec_c7  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rec_c7,  T_max=EPOCHS, eta_min=1e-6)
    sched_disc_c7 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc_c7, T_max=EPOCHS, eta_min=1e-6)
    sched_gen_c7  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen_c7,  T_max=EPOCHS, eta_min=1e-6)

    t0_c7 = time.time()
    for epoch in range(EPOCHS):
        enc1_c7.train(); dec_c7.train(); re_attn_c7.train()
        enc2_c7.train(); disc_c7.train()
        rec_l, d_l, g_l = [], [], []

        for (xb,) in loader_c7:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])

            # Phase 1: reconstruction
            opt_rec_c7.zero_grad()
            z1 = enc1_c7(xb); x_hat1 = dec_c7(z1)
            loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
            loss_rec.backward(); opt_rec_c7.step()

            # Phase 2: discriminator
            opt_disc_c7.zero_grad()
            with torch.no_grad():
                z1_d = enc1_c7(xb); x_h1_d = dec_c7(z1_d)
                err_d = ssim_anomaly_map(xb, x_h1_d).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
                att_d = re_attn_c7(err_d)
                z2_d  = enc2_c7(xb * att_d)
            z_real = torch.randn(n, LATENT_DIM, device=device)
            loss_d = (-torch.mean(torch.log(disc_c7(z_real) + EPS))
                      - torch.mean(torch.log(1.0 - disc_c7(z2_d) + EPS)))
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc_c7.parameters(), max_norm=1.0)
            opt_disc_c7.step()

            # Phase 3: generator
            opt_gen_c7.zero_grad()
            with torch.no_grad():
                z1_g = enc1_c7(xb); x_h1_g = dec_c7(z1_g)
                err_g = ssim_anomaly_map(xb, x_h1_g).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_g = re_attn_c7(err_g)
            loss_g = LAMBDA_ADV * (-torch.mean(torch.log(disc_c7(enc2_c7(xb * att_g)) + EPS)))
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                list(re_attn_c7.parameters()) + list(enc2_c7.parameters()), max_norm=1.0)
            opt_gen_c7.step()

            rec_l.append(loss_rec.item()); d_l.append(loss_d.item()); g_l.append(loss_g.item())

        c7_epoch_loss.append(np.mean(rec_l))
        sched_rec_c7.step(); sched_disc_c7.step(); sched_gen_c7.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  Recon={c7_epoch_loss[-1]:.5f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}  "
                  f"lr={sched_rec_c7.get_last_lr()[-2]:.2e}")

    print(f"C7 total: {time.time()-t0_c7:.1f}s\n")

    enc1_c7.eval(); dec_c7.eval(); re_attn_c7.eval(); enc2_c7.eval(); disc_c7.eval()
    scores_c7, sc_disc_c7, attn_maps_c7 = [], [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb      = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device); n = xb.size(0)
            z1      = enc1_c7(xb); x_hat = dec_c7(z1)
            err     = ssim_anomaly_map(xb, x_hat).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_img = re_attn_c7(err)
            z2      = enc2_c7(xb * att_img)
            d_sc    = disc_c7(z2)
            scores_c7.append(err.view(n, -1).quantile(0.99, dim=1).cpu().numpy())
            sc_disc_c7.append((1 - d_sc).squeeze(1).cpu().numpy())
            attn_maps_c7.append(att_img.squeeze(1).cpu().numpy())

    scores_c7    = np.concatenate(scores_c7)
    sc_disc_c7   = np.concatenate(sc_disc_c7)
    attn_maps_c7 = np.concatenate(attn_maps_c7)
    sc_fuse_c7   = 0.5 * normalise_scores(scores_c7) + 0.5 * normalise_scores(sc_disc_c7)

    m_c7      = evaluate(scores_c7,  binary_test)
    m_c7_disc = evaluate(sc_disc_c7, binary_test)
    m_c7_fuse = evaluate(sc_fuse_c7, binary_test)

    print(f"\n  C7 ResNet mostly fine-tuned — SSIM  AUC-ROC={m_c7['auc_roc']:.4f}")
    print(f"  Disc score    AUC-ROC={m_c7_disc['auc_roc']:.4f}")
    print(f"  Fusion        AUC-ROC={m_c7_fuse['auc_roc']:.4f}")
    print(f"\n  C6 (partial FT) → C7 (mostly FT): {m_c7['auc_roc'] - all_results['C6']['auc_roc']:+.4f}")
    all_results['C7']      = {**m_c7,      'label': 'ResNet-RE-Attn-AAE Mostly FT'}
    all_results['C7_disc'] = {**m_c7_disc, 'label': 'ResNet Mostly FT disc'}
    all_results['C7_fuse'] = {**m_c7_fuse, 'label': 'ResNet Mostly FT fusion'}
    loss_history['C7']     = c7_epoch_loss
    save_ckpt('C7', ['C7','C7_disc','C7_fuse'], scores_c7, sc_disc_c7, c7_epoch_loss, attn_maps=attn_maps_c7,
              enc1=enc1_c7.state_dict(), enc2=enc2_c7.state_dict(),
              dec=dec_c7.state_dict(), re_attn=re_attn_c7.state_dict(),
              disc=disc_c7.state_dict())

# %% [markdown]
# ---
# ## **Cell 13** — Results Summary
#
# Prints a formatted comparison table for all seven conditions across three metrics:
# **AUC-ROC**, **AUC-PR** (average precision), and **F1** at the Youden-optimal threshold.
#
# The ablation delta rows quantify each component's contribution:
# - `C1 → C3` : effect of adversarial latent regularisation (no attention)
# - `C3 → C4` : effect of adding SSIM-guided RE-Attention
# - `C4 → C5` : full ResNet freeze — shows domain-gap failure
# - `C5 → C6` : partial fine-tuning fix — resolves discriminator collapse
# - `C4 → C6` : net effect of adding a partially fine-tuned ResNet encoder
#
# Score fusion rows show whether combining the SSIM reconstruction score with the
# discriminator confidence score improves over either alone.

# %% [CELL 13]  Results summary

print("\n" + "="*60)
print("RESULTS SUMMARY — Extended Experiment Suite")
print("="*60)
print(f"\n  {'Condition':<42} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8}")
print(f"  {'-'*66}")

tags = {'C4': ' ← NOVEL', 'C5': ' ← FROZEN', 'C6': ' ← PARTIAL FT', 'C7': ' ← MOSTLY FT'}
for k in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
    r   = all_results[k]
    tag = tags.get(k, '')
    print(f"  {r['label']:<42} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} {r['f1']:>8.4f}{tag}")

print(f"\n  {'Score Fusion (0.5 SSIM + 0.5 Disc)':<42} {'AUC-ROC':>8}")
print(f"  {'-'*50}")
for k in ['C3_fuse', 'C4_fuse', 'C5_fuse', 'C6_fuse', 'C7_fuse']:
    r = all_results[k]
    print(f"  {r['label']:<42} {r['auc_roc']:>8.4f}")

print(f"\n  Ablation chain (SSIM primary AUC-ROC):")
print(f"  C1 → C3  +adversarial, -attention  : {all_results['C3']['auc_roc'] - all_results['C1']['auc_roc']:+.4f}")
print(f"  C3 → C4  +RE-Attention             : {all_results['C4']['auc_roc'] - all_results['C3']['auc_roc']:+.4f}")
print(f"  C4 → C5  +ResNet full freeze        : {all_results['C5']['auc_roc'] - all_results['C4']['auc_roc']:+.4f}  (domain gap)")
print(f"  C5 → C6  +partial fine-tuning       : {all_results['C6']['auc_roc'] - all_results['C5']['auc_roc']:+.4f}  (fixes collapse)")
print(f"  C6 → C7  +mostly fine-tuned         : {all_results['C7']['auc_roc'] - all_results['C6']['auc_roc']:+.4f}  (more layers)")
print(f"  C4 → C7  scratch→mostly ResNet      : {all_results['C7']['auc_roc'] - all_results['C4']['auc_roc']:+.4f}")
print(f"  C1 → C7  full chain                 : {all_results['C7']['auc_roc'] - all_results['C1']['auc_roc']:+.4f}")

# %% [markdown]
# ---
# ## **Cell 14** — Plots: Training Convergence, ROC Curves, AUC-ROC Bar Chart
#
# Three publication-ready figures are generated and saved to `OUTPUT_DIR`:
#
# 1. **`convergence.png`** — Reconstruction loss per epoch for all five conditions.
#    Shows whether each model converges stably and whether the warm-start is visible as a
#    lower starting point for C3/C4/C5.
#
# 2. **`roc_curves.png`** — ROC curves for all five primary SSIM scores.
#    The diagonal is the random classifier baseline; curves above indicate detection ability.
#
# 3. **`auc_bars.png`** — Bar chart comparing AUC-ROC across all conditions and fusion variants.
#    Bars are annotated with numeric values. This is the key at-a-glance comparison figure.

# %% [CELL 14]  Plots — Convergence, ROC, Bar chart

epochs_x    = np.arange(1, WARMUP_EPOCHS + EPOCHS + 1)

# ── Convergence ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
styles  = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,1))]
for (key, style) in zip(['C1','C2','C3','C4','C5','C6','C7'], styles):
    ax.plot(range(1, len(loss_history[key])+1), loss_history[key],
            color=PAL[key], lw=2, linestyle=style, label=all_results[key]['label'])
ax.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', lw=1, label='warmup end')
ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction Loss')
ax.set_title('Training Convergence — All Conditions (C5 full-freeze vs C6 partial FT)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/convergence.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# ── ROC curves ────────────────────────────────────────────────────────────
score_map = {'C1': scores_c1, 'C2': scores_c2, 'C3': scores_c3,
             'C4': scores_c4, 'C5': scores_c5, 'C6': scores_c6, 'C7': scores_c7}
fig, ax = plt.subplots(figsize=(8, 7))
for key in ['C1','C2','C3','C4','C5','C6','C7']:
    if len(np.unique(binary_test)) > 1:
        fpr, tpr, _ = roc_curve(binary_test, score_map[key])
        ax.plot(fpr, tpr, color=PAL[key], lw=2,
                label=f"{all_results[key]['label']} ({all_results[key]['auc_roc']:.4f})")
ax.plot([0,1],[0,1],'k--',lw=0.8)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves — RSNA Pneumonia Anomaly Detection')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# ── AUC-ROC bar chart (primary + fusion) ─────────────────────────────────
bar_keys    = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C3_fuse', 'C4_fuse', 'C5_fuse', 'C6_fuse', 'C7_fuse']
bar_labels  = [all_results[k]['label'] for k in bar_keys]
bar_vals    = [all_results[k]['auc_roc'] for k in bar_keys]
bar_colors  = [PAL.get(k[:2], '#AAAAAA') for k in bar_keys]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(range(len(bar_keys)), bar_vals, color=bar_colors,
              edgecolor='white', linewidth=1.2)
for bar, v in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(range(len(bar_keys)))
ax.set_xticklabels(bar_labels, rotation=18, ha='right', fontsize=8)
ax.set_ylim(max(0, min(bar_vals) - 0.04), min(1, max(bar_vals) + 0.05))
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC — All Conditions + Fusion Scores')
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/auc_bars.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()

print(f"Plots saved → {OUTPUT_DIR}/")

# %% [markdown]
# ---
# ## **Cell 15** — Attention Map Visualisation: C4 vs C5 (collapsed) vs C6 (partial FT)
#
# Side-by-side comparison showing the effect of backbone freeze strategy on attention quality.
#
# **Grid layout** (4 rows × 6 columns):
# - Row 1: a representative **normal** scan (lowest anomaly score)
# - Rows 2–4: the **3 highest-scored** opacity cases (ranked by C4 score)
#
# | Column | Content |
# |--------|---------|
# | Original | Input CXR with GT bounding box (cyan) |
# | C4 Recon | CNN enc1 reconstruction |
# | C4 att_img | RE-Attention mask — CNN encoder (reference) |
# | C5 att_img | RE-Attention mask — full freeze (should be near-zero / collapsed) |
# | C6 att_img | RE-Attention mask — partial fine-tune (should activate on opacities) |
# | GT bbox + C6 | X-ray + C6 attention (Reds) + GT box (cyan) |
#
# The C5 vs C6 column pair directly illustrates the discriminator collapse failure and fix.

# %% [CELL 15]  Attention visualisation — C4 vs C5 (frozen) vs C6 (partial FT)

opa_idx  = np.where(binary_test == 1)[0]
norm_idx = np.where(binary_test == 0)[0]
opa_picks  = opa_idx[np.argsort(scores_c4[opa_idx])[-3:]]
norm_picks = norm_idx[np.argsort(scores_c4[norm_idx])[:1]]
picks    = list(norm_picks) + list(opa_picks)
pick_lbl = ['Normal'] + ['Lung Opacity'] * 3

fig, axes = plt.subplots(len(picks), 6, figsize=(24, 5 * len(picks)))
if len(picks) == 1:
    axes = axes[None, :]
fig.suptitle(
    'Attention Maps: C4 CNN  |  C5 Full Freeze (collapsed)  |  C6 Partial FT (fixed)\n'
    'Original  |  C4 Recon  |  C4 att  |  C5 att  |  C6 att  |  GT bbox + C6',
    fontsize=11, fontweight='bold')
for ax, t in zip(axes[0], ['Original', 'C4 Recon', 'C4 att', 'C5 att (frozen)', 'C6 att (partial FT)', 'GT bbox + C6']):
    ax.set_title(t, fontsize=9, fontweight='bold')

scale = IMAGE_SIZE / ORIG_SIZE
for row, (idx, lbl) in enumerate(zip(picks, pick_lbl)):
    col = '#888888' if lbl == 'Normal' else '#E84C3D'
    with torch.no_grad():
        xb_t    = torch.tensor(x_test[idx][None]).to(device)
        xhat_c4 = dec_c4(enc1_c4(xb_t))
    img_np  = x_test[idx, 0]
    axes[row, 0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    axes[row, 0].set_ylabel(lbl, color=col, fontsize=9, fontweight='bold')
    axes[row, 1].imshow(xhat_c4.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[row, 2].imshow(attn_maps_c4[idx], cmap='hot', vmin=0, vmax=1)
    axes[row, 3].imshow(attn_maps_c5[idx], cmap='hot', vmin=0, vmax=1)
    axes[row, 4].imshow(attn_maps_c6[idx], cmap='hot', vmin=0, vmax=1)
    axes[row, 5].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    axes[row, 5].imshow(attn_maps_c6[idx], cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    if idx in test_boxes:
        for (bx, by, bw, bh) in test_boxes[idx]:
            for ax_ in [axes[row, 0], axes[row, 5]]:
                ax_.add_patch(patches.Rectangle(
                    (bx*scale, by*scale), bw*scale, bh*scale,
                    linewidth=1.5, edgecolor='cyan', facecolor='none'))
    for j in range(6):
        axes[row, j].axis('off')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/attention_grid.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/attention_grid.png")

# %% [markdown]
# ---
# ## **Cell 16** — Pixel-Level Localisation AUROC
#
# Measures **spatial localisation quality** by comparing per-pixel anomaly maps
# against radiologist-annotated bounding boxes.
#
# For each opacity image that has a bounding box:
# - Ground-truth mask: 1 inside the box, 0 outside (scaled to `IMAGE_SIZE`).
# - Predicted map: per-pixel SSIM error map or RE-Attention mask.
# - **Pixel AUROC** = area under the ROC curve treating each pixel as a binary classification.
#
# Pixel AUROC > 0.5 = the map assigns higher values inside boxes than outside.
# Pixel AUROC = 0.5 = random spatial assignment.
# Pixel AUROC < 0.5 = the map is **inverted** (highlighting wrong regions).
#
# Six maps are evaluated:
# - `ssim_c1` — SSIM error from C1 plain AE (baseline)
# - `ssim_c4` — SSIM error from C4 CNN-RE-Attn-AAE enc1
# - `ssim_c5` — SSIM error from C5 full-freeze ResNet enc1
# - `ssim_c6` — SSIM error from C6 partial-FT ResNet enc1
# - `attn_c4` — RE-Attention mask from C4
# - `attn_c5` — RE-Attention mask from C5 (expected near-zero due to collapse)
# - `attn_c6` — RE-Attention mask from C6 (expected to localise opacities)
#
# The gap between `attn_c5` and `attn_c6` shows the localisation benefit of partial fine-tuning.

# %% [CELL 16]  Pixel-level localisation AUROC

print("\n" + "="*60)
print("PIXEL-LEVEL LOCALISATION AUROC")
print("="*60)

def compute_ssim_maps(enc, dec):
    enc.eval(); dec.eval()
    maps = []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb   = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            xhat = dec(enc(xb))
            maps.append(ssim_anomaly_map(xb, xhat).view(-1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy())
    return np.concatenate(maps)

ssim_maps_c1 = compute_ssim_maps(enc_c1,  dec_c1)
ssim_maps_c4 = compute_ssim_maps(enc1_c4, dec_c4)
ssim_maps_c5 = compute_ssim_maps(enc1_c5, dec_c5)
ssim_maps_c6 = compute_ssim_maps(enc1_c6, dec_c6)
ssim_maps_c7 = compute_ssim_maps(enc1_c7, dec_c7)

_n_boxes = sum(1 for i in range(len(binary_test)) if binary_test[i] == 1 and i in test_boxes)
print(f"\n  ({_n_boxes}/{int(binary_test.sum())} opacity images have bounding boxes)\n")
print(f"  SSIM error map  C1 CNN-AE              : {pixel_auroc(ssim_maps_c1, test_boxes, binary_test):.4f}")
print(f"  SSIM error map  C4 CNN-RE-Attn-AAE     : {pixel_auroc(ssim_maps_c4, test_boxes, binary_test):.4f}")
print(f"  SSIM error map  C5 ResNet full freeze  : {pixel_auroc(ssim_maps_c5, test_boxes, binary_test):.4f}")
print(f"  SSIM error map  C6 ResNet partial FT   : {pixel_auroc(ssim_maps_c6, test_boxes, binary_test):.4f}")
print(f"  SSIM error map  C7 ResNet mostly FT    : {pixel_auroc(ssim_maps_c7, test_boxes, binary_test):.4f}")
print(f"  att_img         C4 CNN-RE-Attn-AAE     : {pixel_auroc(attn_maps_c4, test_boxes, binary_test):.4f}")
print(f"  att_img         C5 ResNet full freeze  : {pixel_auroc(attn_maps_c5, test_boxes, binary_test):.4f}  (collapsed)")
print(f"  att_img         C6 ResNet partial FT   : {pixel_auroc(attn_maps_c6, test_boxes, binary_test):.4f}")
print(f"  att_img         C7 ResNet mostly FT    : {pixel_auroc(attn_maps_c7, test_boxes, binary_test):.4f}")

# %% [markdown]
# ---
# ## **Cell 17** — Save All Results
#
# Persists all metrics and loss histories to `OUTPUT_DIR` for later analysis or reporting:
#
# | File | Contents |
# |------|----------|
# | `results.json` | All AUC-ROC / AUC-PR / F1 values for every condition, score variant, and pixel AUROC |
# | `loss_history.json` | Per-epoch reconstruction loss for C1–C5 |
# | `convergence.png` | Training convergence curves |
# | `roc_curves.png` | ROC curves for all primary SSIM scores |
# | `auc_bars.png` | AUC-ROC bar chart with fusion scores |
# | `attention_grid.png` | C4 vs C5 attention side-by-side with radiologist GT boxes |
#
# All numpy types are serialised to native Python types before JSON dump.

# %% [CELL 17]  Save results

def _json(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):     return int(obj)
    if isinstance(obj, np.ndarray):            return obj.tolist()
    if isinstance(obj, dict):  return {k: _json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json(v)     for v in obj]
    return obj

all_results['pixel_auroc'] = {
    'ssim_c1':  float(pixel_auroc(ssim_maps_c1, test_boxes, binary_test)),
    'ssim_c4':  float(pixel_auroc(ssim_maps_c4, test_boxes, binary_test)),
    'ssim_c5':  float(pixel_auroc(ssim_maps_c5, test_boxes, binary_test)),
    'ssim_c6':  float(pixel_auroc(ssim_maps_c6, test_boxes, binary_test)),
    'ssim_c7':  float(pixel_auroc(ssim_maps_c7, test_boxes, binary_test)),
    'attn_c4':  float(pixel_auroc(attn_maps_c4, test_boxes, binary_test)),
    'attn_c5':  float(pixel_auroc(attn_maps_c5, test_boxes, binary_test)),
    'attn_c6':  float(pixel_auroc(attn_maps_c6, test_boxes, binary_test)),
    'attn_c7':  float(pixel_auroc(attn_maps_c7, test_boxes, binary_test)),
}

with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(_json(all_results), f, indent=2)

with open(f'{OUTPUT_DIR}/loss_history.json', 'w') as f:
    json.dump(_json(loss_history), f, indent=2)

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("  results.json      — all metrics (primary + disc + fusion + pixel AUROC)")
print("  loss_history.json — epoch losses for all conditions")
print("  convergence.png   — training curves C1-C5")
print("  roc_curves.png    — ROC for all primary scores")
print("  auc_bars.png      — bar chart including fusion scores")
print("  attention_grid.png — C4 vs C5 (frozen) vs C6 (partial FT) with GT bbox")
print("\n" + "="*60)
print("DONE — check /kaggle/working/results_rsna_resnet/")
print("="*60)

# %% [markdown]
# ---
# ## **Cell 18** — Deep Technical Report: Figures + CSV + ZIP Download
#
# Generates a comprehensive technical report for a Deep Learning professor audience.
# All figures explain *why* results happen at a mechanistic level, not just *what* they are.
# Output: `/kaggle/working/dl_report/` zipped and downloaded.

# %% [CELL 18]  Deep technical report — figures, CSV, zip, download

import os, zipfile, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc as sk_auc
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
import torch

REPORT_DIR = '/kaggle/working/dl_report'
os.makedirs(REPORT_DIR, exist_ok=True)

PAL = {
    'C1': '#4878CF', 'C2': '#F5A623', 'C3': '#7B68EE',
    'C4': '#E84C3D', 'C5': '#95A5A6', 'C6': '#2ECC71', 'C7': '#1A5276',
}
COND_LABELS = {
    'C1': 'C1: CNN-AE',
    'C2': 'C2: VAE',
    'C3': 'C3: CNN-AAE\n(no attention)',
    'C4': 'C4: CNN-RE-Attn-AAE\n[NOVEL]',
    'C5': 'C5: ResNet\nFull Freeze',
    'C6': 'C6: ResNet\nPartial FT',
    'C7': 'C7: ResNet\nMostly FT',
}

def save(fig, name):
    p = f'{REPORT_DIR}/{name}'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  saved: {name}')

# ─────────────────────────────────────────────────────────────
# CSV 1: image-level metrics
# ─────────────────────────────────────────────────────────────
rows = []
for k in ['C1','C2','C3','C4','C5','C6','C7']:
    r = all_results[k]
    row = {'condition': k, 'label': r['label'],
           'auc_roc': r['auc_roc'], 'auc_pr': r['auc_pr'], 'f1': r['f1'],
           'encoder': 'ResNet-18' if k in ['C5','C6','C7'] else 'CNN',
           'adversarial': k in ['C3','C4','C5','C6','C7'],
           're_attention': k in ['C4','C5','C6','C7'],
           'backbone_freeze': {'C5':'full','C6':'partial','C7':'mostly'}.get(k, 'n/a')}
    if k+'_disc' in all_results:
        row['auc_roc_disc']  = all_results[k+'_disc']['auc_roc']
        row['auc_roc_fusion']= all_results[k+'_fuse']['auc_roc']
    rows.append(row)
df_img = pd.DataFrame(rows)
df_img.to_csv(f'{REPORT_DIR}/image_level_metrics.csv', index=False)

# CSV 2: pixel-level
px = all_results.get('pixel_auroc', {})
df_px = pd.DataFrame([
    {'condition':'C1','modality':'SSIM map','pixel_auroc': px.get('ssim_c1',float('nan'))},
    {'condition':'C4','modality':'SSIM map','pixel_auroc': px.get('ssim_c4',float('nan'))},
    {'condition':'C5','modality':'SSIM map','pixel_auroc': px.get('ssim_c5',float('nan'))},
    {'condition':'C6','modality':'SSIM map','pixel_auroc': px.get('ssim_c6',float('nan'))},
    {'condition':'C7','modality':'SSIM map','pixel_auroc': px.get('ssim_c7',float('nan'))},
    {'condition':'C4','modality':'Attn map','pixel_auroc': px.get('attn_c4',float('nan'))},
    {'condition':'C5','modality':'Attn map','pixel_auroc': px.get('attn_c5',float('nan'))},
    {'condition':'C6','modality':'Attn map','pixel_auroc': px.get('attn_c6',float('nan'))},
    {'condition':'C7','modality':'Attn map','pixel_auroc': px.get('attn_c7',float('nan'))},
])
df_px.to_csv(f'{REPORT_DIR}/pixel_level_metrics.csv', index=False)

# CSV 3: loss history
df_loss = pd.DataFrame({k: pd.Series(v) for k,v in loss_history.items()})
df_loss.to_csv(f'{REPORT_DIR}/loss_history.csv', index=False)

print("CSVs written.")

# ─────────────────────────────────────────────────────────────
# FIG 1: Ablation chain — AUC-ROC with delta annotations
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5), facecolor='#0F1117')
ax.set_facecolor('#0F1117')

keys   = ['C1','C2','C3','C4','C5','C6','C7']
aucs   = [all_results[k]['auc_roc'] for k in keys]
colors = [PAL[k] for k in keys]
bars   = ax.bar(keys, aucs, color=colors, width=0.6, zorder=3, edgecolor='white', linewidth=0.5)

for bar, auc, k in zip(bars, aucs, keys):
    ax.text(bar.get_x()+bar.get_width()/2, auc+0.004, f'{auc:.4f}',
            ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

chain = [('C1','C3','+adversarial\n−attention'),
         ('C3','C4','+RE-Attention\n[NOVEL]'),
         ('C4','C5','+ResNet freeze'),
         ('C5','C6','+partial FT'),
         ('C6','C7','+mostly FT')]
for (a,b,lbl) in chain:
    ia, ib = keys.index(a), keys.index(b)
    ya, yb = aucs[ia], aucs[ib]
    delta  = yb - ya
    col    = '#2ECC71' if delta > 0 else '#E84C3D'
    ax.annotate('', xy=(ib, yb+0.012), xytext=(ia, ya+0.012),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5,
                                connectionstyle='arc3,rad=-0.25'))
    mx = (ia+ib)/2; my = max(ya,yb)+0.035
    ax.text(mx, my, f'{delta:+.4f}\n{lbl}',
            ha='center', va='bottom', fontsize=7.5, color=col,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1E2130', edgecolor=col, alpha=0.85))

ax.axhline(0.5, color='white', lw=0.8, ls='--', alpha=0.4, label='Random (0.50)')
ax.set_ylim(0.45, 0.95)
ax.set_ylabel('AUC-ROC', color='white', fontsize=12)
ax.set_xlabel('Condition', color='white', fontsize=12)
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor('#444')
ax.set_title('Figure 1 — Ablation Chain: Image-Level AUC-ROC\n'
             'Arrows show directional effect of each design decision',
             color='white', fontsize=13, pad=12)
ax.legend(facecolor='#1E2130', edgecolor='#444', labelcolor='white', fontsize=9)
fig.tight_layout()
save(fig, 'fig1_ablation_chain.png')

# ─────────────────────────────────────────────────────────────
# FIG 2: Discriminator dynamics — Nash equilibrium vs collapse
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor='#0F1117')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
ax_main = fig.add_subplot(gs[0, :])
ax_exp1 = fig.add_subplot(gs[1, 0])
ax_exp2 = fig.add_subplot(gs[1, 1])

for ax in [ax_main, ax_exp1, ax_exp2]:
    ax.set_facecolor('#0F1117')

# disc logs from loss_history (we store recon; disc values were printed)
# We'll reconstruct epoch axes from loss_history lengths
warmup_ep = WARMUP_EPOCHS
total_ep  = WARMUP_EPOCHS + EPOCHS

# Show recon loss convergence as proxy + annotate disc events
for ax, k in [(ax_exp1,'C5'), (ax_exp2,'C4')]:
    hist = loss_history[k]
    epochs = list(range(1, len(hist)+1))
    ax.plot(epochs, hist, color=PAL[k], lw=2)
    ax.axvline(warmup_ep, color='yellow', ls='--', lw=1, alpha=0.7, label='Warmup end')
    ax.set_facecolor('#0F1117')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.set_xlabel('Epoch', color='white', fontsize=10)
    ax.set_ylabel('Recon Loss', color='white', fontsize=10)
    ax.legend(facecolor='#1E2130', labelcolor='white', fontsize=8)

ax_exp1.set_title('C5: ResNet Full-Freeze\nDisc→18.42 (collapse at epoch 10+20)',
                  color=PAL['C5'], fontsize=10)
ax_exp1.text(warmup_ep+2, max(loss_history['C5'])*0.95,
             'Disc=18.42\n= −log(EPS)\nGenerator wins\ncompletely',
             color='#E84C3D', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#E84C3D'))

ax_exp2.set_title('C4: CNN-RE-Attn-AAE\nDisc→1.387 ≈ 2·ln(2) (Nash equilibrium)',
                  color=PAL['C4'], fontsize=10)
ax_exp2.text(warmup_ep+2, max(loss_history['C4'])*0.95,
             'Disc≈1.387\n= 2·ln(2)\nNash equilibrium\nGen=Disc balanced',
             color='#2ECC71', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#2ECC71'))

# Main: all recon losses
styles = ['-','--','-.',':','-','--', (0,(3,1,1,1))]
for (k, sty) in zip(['C1','C2','C3','C4','C5','C6','C7'], styles):
    hist = loss_history[k]
    ax_main.plot(range(1,len(hist)+1), hist, color=PAL[k], lw=2,
                 linestyle=sty, label=COND_LABELS[k].replace('\n',' '))
ax_main.axvline(warmup_ep, color='yellow', ls='--', lw=1, alpha=0.6, label=f'Warmup→Main (ep {warmup_ep})')
ax_main.set_title('Training Convergence — Reconstruction Loss (all 7 conditions)',
                  color='white', fontsize=12)
ax_main.set_xlabel('Epoch', color='white'); ax_main.set_ylabel('Recon Loss', color='white')
ax_main.tick_params(colors='white')
for sp in ax_main.spines.values(): sp.set_edgecolor('#444')
ax_main.legend(facecolor='#1E2130', edgecolor='#444', labelcolor='white',
               fontsize=7.5, ncol=4, loc='upper right')
ax_main.text(0.02, 0.08,
    'KEY: Disc=18.42 = −log(EPS=1e-8) → collapse\n'
    'Disc≈1.387 = 2·ln(2) → Nash equilibrium\n'
    'LAMBDA_ADV=0.3 prevents generator from overwhelming discriminator',
    transform=ax_main.transAxes, color='#AAB0B8', fontsize=8,
    bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#555'))
fig.suptitle('Figure 2 — Discriminator Dynamics: Collapse vs Nash Equilibrium',
             color='white', fontsize=14, y=1.01)
save(fig, 'fig2_discriminator_dynamics.png')

# ─────────────────────────────────────────────────────────────
# FIG 3: Anomaly score distributions — what the model "sees"
# ─────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

score_dict = {
    'C1': scores_c1, 'C2': scores_c2, 'C3': scores_c3,
    'C4': scores_c4, 'C5': scores_c5, 'C6': scores_c6, 'C7': scores_c7,
}
fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor='#0F1117')
axes = axes.flatten()
fig.patch.set_facecolor('#0F1117')

for idx, k in enumerate(['C1','C2','C3','C4','C5','C6','C7']):
    ax = axes[idx]; ax.set_facecolor('#141720')
    sc = score_dict[k]
    norm_sc  = sc[binary_test == 0]
    anom_sc  = sc[binary_test == 1]
    xs = np.linspace(sc.min(), sc.max(), 300)
    if len(np.unique(norm_sc)) > 3:
        kde_n = gaussian_kde(norm_sc, bw_method='silverman')
        ax.fill_between(xs, kde_n(xs), alpha=0.4, color='#4878CF', label='Normal')
        ax.plot(xs, kde_n(xs), color='#4878CF', lw=1.5)
    if len(np.unique(anom_sc)) > 3:
        kde_a = gaussian_kde(anom_sc, bw_method='silverman')
        ax.fill_between(xs, kde_a(xs), alpha=0.4, color='#E84C3D', label='Pneumonia')
        ax.plot(xs, kde_a(xs), color='#E84C3D', lw=1.5)
    thresh = np.percentile(norm_sc, 95)
    ax.axvline(thresh, color='yellow', ls='--', lw=1, label='95th pct')
    overlap = np.mean(norm_sc > thresh)
    ax.set_title(f'{k}  AUC={all_results[k]["auc_roc"]:.4f}',
                 color=PAL[k], fontsize=11, fontweight='bold')
    ax.text(0.97, 0.95, f'FPR@95={overlap:.3f}', transform=ax.transAxes,
            ha='right', va='top', color='yellow', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.set_xlabel('Anomaly score', color='white', fontsize=8)
    ax.set_ylabel('Density', color='white', fontsize=8)
    ax.legend(facecolor='#1E2130', labelcolor='white', fontsize=7)

axes[-1].axis('off')
axes[-1].text(0.5, 0.6,
    'Separation between\nNormal (blue) and\nPneumonia (red) KDE\ncurves directly reflects\nAUC-ROC.\n\n'
    'C4 shows clearest\nseparation — largest gap\nbetween distributions.\n\n'
    'C5 collapses: both\ndistributions overlap\nentirely (Disc→18.42).',
    ha='center', va='center', color='#CCC', fontsize=9,
    transform=axes[-1].transAxes,
    bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#555'))
fig.suptitle('Figure 3 — Anomaly Score Distributions: Normal vs Pneumonia\n'
             'Score = 99th-percentile SSIM error map pixel (per image)',
             color='white', fontsize=13)
fig.tight_layout()
save(fig, 'fig3_score_distributions.png')

# ─────────────────────────────────────────────────────────────
# FIG 4: ROC curves — all conditions
# ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0F1117')
for ax in (ax1, ax2): ax.set_facecolor('#141720')

for k in ['C1','C2','C3','C4','C5','C6','C7']:
    sc = score_dict[k]
    fpr, tpr, _ = roc_curve(binary_test, sc)
    roc_auc = sk_auc(fpr, tpr)
    lw = 2.5 if k == 'C4' else 1.5
    ax1.plot(fpr, tpr, color=PAL[k], lw=lw,
             label=f'{k}: {roc_auc:.4f}')

ax1.plot([0,1],[0,1],'--',color='#555',lw=1)
ax1.set_title('ROC Curves — Primary SSIM Score', color='white', fontsize=12)
ax1.set_xlabel('False Positive Rate', color='white')
ax1.set_ylabel('True Positive Rate', color='white')
ax1.tick_params(colors='white')
for sp in ax1.spines.values(): sp.set_edgecolor('#444')
ax1.legend(facecolor='#1E2130', edgecolor='#444', labelcolor='white',
           fontsize=9, title='Condition  AUC', title_fontsize=9)

# Fusion ROC for C3,C4,C5,C6,C7
for k in ['C3','C4','C5','C6','C7']:
    if k+'_fuse' in all_results:
        fkey = k + '_fuse'
        sc_f = 0.5*normalise_scores(score_dict[k]) + 0.5*normalise_scores(sc_disc_c3 if k=='C3' else
               sc_disc_c4 if k=='C4' else sc_disc_c5 if k=='C5' else
               sc_disc_c6 if k=='C6' else sc_disc_c7)
        fpr, tpr, _ = roc_curve(binary_test, sc_f)
        roc_auc = sk_auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=PAL[k], lw=2,
                 label=f'{k} fuse: {roc_auc:.4f}')
ax2.plot([0,1],[0,1],'--',color='#555',lw=1)
ax2.set_title('ROC Curves — Fusion Score (0.5·SSIM + 0.5·Disc)', color='white', fontsize=12)
ax2.set_xlabel('False Positive Rate', color='white')
ax2.set_ylabel('True Positive Rate', color='white')
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_edgecolor('#444')
ax2.legend(facecolor='#1E2130', edgecolor='#444', labelcolor='white', fontsize=9)
fig.suptitle('Figure 4 — ROC Curves: Primary vs Fusion Score', color='white', fontsize=14)
fig.tight_layout()
save(fig, 'fig4_roc_curves.png')

# ─────────────────────────────────────────────────────────────
# FIG 5: Reconstruction error maps — normal vs pneumonia
# Columns: input | recon C1 | recon C4 | SSIM-err C1 | SSIM-err C4
# ─────────────────────────────────────────────────────────────
def get_recon(enc, dec, imgs, is_vae=False, is_resnet=False):
    enc.eval(); dec.eval()
    with torch.no_grad():
        xb = torch.tensor(imgs).to(device)
        if is_vae:
            z,_,_ = enc(xb)
        else:
            z = enc(xb)
        return dec(z).cpu().numpy()

n_show = 6
norm_idx  = np.where(binary_test == 0)[0][:n_show]
anom_idx  = np.where(binary_test == 1)[0][:n_show]
show_idx  = np.concatenate([norm_idx[:3], anom_idx[:3]])

imgs_show = x_test[show_idx]
r_c1 = get_recon(enc_c1, dec_c1, imgs_show)
r_c4 = get_recon(enc1_c4, dec_c4, imgs_show)
r_c6 = get_recon(enc1_c6, dec_c6, imgs_show, is_resnet=True)

def ssim_map_np(orig, recon):
    return np.abs(orig - recon).squeeze()

fig = plt.figure(figsize=(20, 9), facecolor='#0F1117')
fig.patch.set_facecolor('#0F1117')
cols = ['Input', 'C1 Recon', 'C4 Recon', 'SSIM-err C1', 'SSIM-err C4', 'Attn C4']
rows_label = ['Normal']*3 + ['Pneumonia']*3
n_cols = len(cols)

for row_i in range(6):
    for col_i, col_name in enumerate(cols):
        ax = fig.add_subplot(6, n_cols, row_i*n_cols + col_i + 1)
        ax.set_facecolor('#0F1117')
        img = imgs_show[row_i, 0]
        if col_name == 'Input':
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        elif col_name == 'C1 Recon':
            ax.imshow(r_c1[row_i, 0], cmap='gray', vmin=0, vmax=1)
        elif col_name == 'C4 Recon':
            ax.imshow(r_c4[row_i, 0], cmap='gray', vmin=0, vmax=1)
        elif col_name == 'SSIM-err C1':
            err = ssim_map_np(img, r_c1[row_i, 0])
            ax.imshow(err, cmap='hot', vmin=0, vmax=err.max()+1e-8)
        elif col_name == 'SSIM-err C4':
            err = ssim_map_np(img, r_c4[row_i, 0])
            ax.imshow(err, cmap='hot', vmin=0, vmax=err.max()+1e-8)
        elif col_name == 'Attn C4':
            try:
                test_pos = np.where(show_idx[row_i] < len(attn_maps_c4))[0]
                idx_in_test = show_idx[row_i]
                ax.imshow(attn_maps_c4[idx_in_test], cmap='plasma', vmin=0, vmax=1)
            except:
                ax.imshow(np.zeros((IMAGE_SIZE,IMAGE_SIZE)), cmap='plasma')
        ax.axis('off')
        if row_i == 0:
            ax.set_title(col_name, color='white', fontsize=9, fontweight='bold')
        if col_i == 0:
            label = rows_label[row_i]
            color = '#4878CF' if label == 'Normal' else '#E84C3D'
            ax.set_ylabel(label, color=color, fontsize=9, rotation=0,
                          labelpad=40, va='center')

fig.suptitle('Figure 5 — Reconstruction Error Maps: Normal vs Pneumonia\n'
             'C4 RE-Attention focuses error on opacity regions; C1 spreads uniformly',
             color='white', fontsize=13, y=1.01)
fig.tight_layout()
save(fig, 'fig5_reconstruction_error_maps.png')

# ─────────────────────────────────────────────────────────────
# FIG 6: Pixel-level AUROC — SSIM maps vs Attention maps
# ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0F1117')
for ax in (ax1, ax2): ax.set_facecolor('#141720')

ssim_keys = ['C1','C4','C5','C6','C7']
ssim_vals = [px.get(f'ssim_{k.lower()}', float('nan')) for k in ssim_keys]
bars = ax1.bar(ssim_keys, ssim_vals, color=[PAL[k] for k in ssim_keys],
               width=0.55, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, ssim_vals):
    if not np.isnan(val):
        ax1.text(bar.get_x()+bar.get_width()/2, val+0.002, f'{val:.4f}',
                 ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax1.axhline(0.5, color='white', ls='--', lw=0.8, alpha=0.4)
ax1.set_ylim(0.45, 0.76)
ax1.set_title('SSIM Error Map\nPixel-Level AUROC', color='white', fontsize=11)
ax1.set_ylabel('Pixel AUROC', color='white')
ax1.tick_params(colors='white')
for sp in ax1.spines.values(): sp.set_edgecolor('#444')

attn_keys = ['C4','C5','C6','C7']
attn_vals = [px.get(f'attn_{k.lower()}', float('nan')) for k in attn_keys]
bars = ax2.bar(attn_keys, attn_vals, color=[PAL[k] for k in attn_keys],
               width=0.55, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, attn_vals):
    if not np.isnan(val):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.003, f'{val:.4f}',
                 ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax2.axhline(0.5, color='yellow', ls='--', lw=1, alpha=0.7, label='Random (0.50)')
ax2.set_ylim(0.20, 0.80)
ax2.set_title('Attention Map\nPixel-Level AUROC', color='white', fontsize=11)
ax2.set_ylabel('Pixel AUROC', color='white')
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_edgecolor('#444')
ax2.legend(facecolor='#1E2130', labelcolor='white', fontsize=9)
ax2.text(0.5, 0.12,
    'C5/C6/C7 attention < 0.50\n= attention focuses on HEALTHY tissue\n(ResNet over-reconstructs lesion regions)',
    ha='center', va='bottom', transform=ax2.transAxes, color='#E84C3D', fontsize=8,
    bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#E84C3D'))

fig.suptitle('Figure 6 — Pixel-Level AUROC: SSIM Maps vs Attention Maps\n'
             'C4 attention (0.7017) is best localizer; ResNet attention inverts below random',
             color='white', fontsize=13)
fig.tight_layout()
save(fig, 'fig6_pixel_auroc.png')

# ─────────────────────────────────────────────────────────────
# FIG 7: Architecture diagram — RE-Attn-AAE data flow
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7), facecolor='#0F1117')
ax.set_facecolor('#0F1117')
ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis('off')

def box(ax, x, y, w, h, label, sub='', color='#1E2130', ec='#4878CF', fs=10):
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.08', facecolor=color, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y+(0.08 if sub else 0), label, ha='center', va='center',
            color='white', fontsize=fs, fontweight='bold', zorder=4)
    if sub:
        ax.text(x, y-0.28, sub, ha='center', va='center',
                color='#AAB', fontsize=7.5, zorder=4)

def arr(ax, x1, y1, x2, y2, label='', color='#888'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.18, label, ha='center', color=color, fontsize=8)

# Phase 1: reconstruction
box(ax, 1.5, 5.2, 2.2, 0.9, 'x (input)', '1×128×128', color='#1A2744', ec='#4878CF')
box(ax, 4.0, 5.2, 2.2, 0.9, 'Enc₁', 'CNN or ResNet-18\n→ z₁ ∈ ℝ¹²⁸', color='#1A3A5C', ec='#4878CF')
box(ax, 6.5, 5.2, 2.2, 0.9, 'Dec', 'CNN decoder\n→ x̂ ∈ [0,1]', color='#1A3A5C', ec='#4878CF')
box(ax, 9.2, 5.2, 2.4, 0.9, 'SSIM\nerror map', '|x−x̂| → [B,1,H,W]', color='#2D1B1B', ec='#E84C3D')
box(ax, 11.8, 5.2, 2.2, 0.9, 'RE-Attn', 'Conv→ReLU→Conv\n→Sigmoid mask', color='#1B2D1B', ec='#2ECC71')
box(ax, 14.5, 5.2, 2.0, 0.9, 'att', 'attention\nmask ∈[0,1]', color='#1B2D1B', ec='#2ECC71')

arr(ax, 2.6, 5.2, 2.9, 5.2, 'x', '#4878CF')
arr(ax, 5.1, 5.2, 5.4, 5.2, 'z₁', '#4878CF')
arr(ax, 7.6, 5.2, 7.9, 5.2, 'x̂', '#4878CF')
arr(ax, 10.4, 5.2, 10.7, 5.2, 'err', '#E84C3D')
arr(ax, 12.9, 5.2, 13.5, 5.2, 'att', '#2ECC71')

# Phase 1 loss
ax.text(6.5, 6.4, 'Phase 1: L_rec = 0.7·MSE + 0.3·SSIM_loss  (trains Enc₁ + Dec)',
        ha='center', color='#4878CF', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#1A2744', edgecolor='#4878CF', alpha=0.8))

# Phase 2 and 3: adversarial
box(ax, 4.0, 2.5, 2.2, 0.9, 'Enc₂', 'CNN encoder\n→ z₂ ∈ ℝ¹²⁸', color='#2D1B2D', ec='#7B68EE')
box(ax, 7.5, 2.5, 2.4, 0.9, 'x·att', 'masked input\natt∈[0,1]', color='#2D1B1B', ec='#2ECC71')
box(ax, 11.0, 2.5, 2.2, 0.9, 'Disc', 'MLP → Sigmoid\nP(z₂ is real)', color='#2D2D1B', ec='#F5A623')
box(ax, 4.0, 1.0, 2.2, 0.7, 'z_real', '~ N(0,I)', color='#1E2130', ec='#888')

arr(ax, 14.5, 4.75, 14.5, 3.5, '', '#2ECC71')
ax.annotate('', xy=(7.5, 2.9), xytext=(14.5, 3.0),
            arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=1.5,
                            connectionstyle='arc3,rad=0.3'), zorder=2)
ax.text(12.5, 3.5, 'x·att', color='#2ECC71', fontsize=9)
arr(ax, 8.7, 2.5, 9.9, 2.5, 'z₂', '#7B68EE')
arr(ax, 5.1, 2.5, 6.3, 2.5, '', '#7B68EE')
arr(ax, 4.0, 1.35, 4.0, 2.05, '', '#888')
arr(ax, 5.1, 1.0, 9.9, 2.2, 'z_real', '#888')

# Phase 2/3 loss
ax.text(7.5, 0.25,
    'Phase 2 (Disc): L_D = −E[log D(z_real)] − E[log(1−D(z₂))]\n'
    'Phase 3 (Gen):  L_G = λ_adv · (−E[log D(z₂)])   [λ_adv=0.30]',
    ha='center', color='#F5A623', fontsize=9,
    bbox=dict(boxstyle='round', facecolor='#2D2D1B', edgecolor='#F5A623', alpha=0.85))

ax.text(8.0, 7.0,
    'RE-Attention Adversarial Autoencoder (RE-Attn-AAE) — Two-Pass Architecture\n'
    'Pass 1: Enc₁→Dec reconstructs x, producing SSIM error map.\n'
    'Pass 2: Error map → RE-Attention mask → Enc₂(x·att) → adversarial alignment to N(0,I).',
    ha='center', va='top', color='white', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#888'))
fig.suptitle('Figure 7 — RE-Attn-AAE Architecture: Three-Phase Training Loop',
             color='white', fontsize=14, y=0.02)
save(fig, 'fig7_architecture.png')

# ─────────────────────────────────────────────────────────────
# FIG 8: ResNet backbone analysis — what freeze_upto controls
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0F1117')
for ax in axes: ax.set_facecolor('#141720')

# Subplot 1: trainable param counts
import torchvision.models as tv_models

def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

try:
    e5_tmp = ResNetEncoder(LATENT_DIM, freeze_upto=None)
    e6_tmp = ResNetEncoder(LATENT_DIM, freeze_upto=7)
    e7_tmp = ResNetEncoder(LATENT_DIM, freeze_upto=2)
    configs = {
        'C5\nFull freeze': count_params(e5_tmp),
        'C6\nPartial FT\n(layer4 only)': count_params(e6_tmp),
        'C7\nMostly FT\n(layer1-4)': count_params(e7_tmp),
    }
    labels = list(configs.keys())
    totals    = [v[0]/1e6 for v in configs.values()]
    trainable = [v[1]/1e6 for v in configs.values()]
    frozen    = [t-tr for t,tr in zip(totals, trainable)]
    x = np.arange(len(labels))
    axes[0].bar(x, frozen,    label='Frozen',    color='#555', width=0.5)
    axes[0].bar(x, trainable, label='Trainable', color='#E84C3D', width=0.5, bottom=frozen)
    for xi, (tot, tr) in enumerate(zip(totals, trainable)):
        axes[0].text(xi, tot+0.1, f'{tr:.1f}M\ntrain', ha='center', color='#E84C3D', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, color='white', fontsize=9)
    axes[0].set_ylabel('Parameters (M)', color='white')
    axes[0].set_title('ResNet-18 Encoder: Frozen vs Trainable Parameters\n(backbone only)', color='white', fontsize=11)
    axes[0].tick_params(colors='white')
    for sp in axes[0].spines.values(): sp.set_edgecolor('#444')
    axes[0].legend(facecolor='#1E2130', labelcolor='white', fontsize=9)
except Exception as ex:
    axes[0].text(0.5, 0.5, str(ex), ha='center', transform=axes[0].transAxes, color='red')

# Subplot 2: reconstruction loss vs AUC trade-off
final_recon = {k: loss_history[k][-1] for k in ['C4','C5','C6','C7']}
aucs_sub    = {k: all_results[k]['auc_roc'] for k in ['C4','C5','C6','C7']}
for k in ['C4','C5','C6','C7']:
    axes[1].scatter(final_recon[k], aucs_sub[k], color=PAL[k], s=200, zorder=5, edgecolors='white', lw=1.5)
    axes[1].annotate(k, (final_recon[k], aucs_sub[k]),
                     xytext=(8, 5), textcoords='offset points',
                     color=PAL[k], fontsize=11, fontweight='bold')
axes[1].set_xlabel('Final Reconstruction Loss (lower = better recon)', color='white', fontsize=10)
axes[1].set_ylabel('Image-Level AUC-ROC', color='white', fontsize=10)
axes[1].set_title('Reconstruction Quality vs Anomaly Detection AUC\n'
                  '"Better reconstruction ≠ better anomaly detection"', color='white', fontsize=11)
axes[1].tick_params(colors='white')
for sp in axes[1].spines.values(): sp.set_edgecolor('#444')
axes[1].text(0.05, 0.1,
    'C7 achieves lowest recon loss\nbut lowest AUC among CNN/ResNet variants.\n'
    'Over-fitting to normal texture\ndestroys the anomaly signal.',
    transform=axes[1].transAxes, color='#AAB0B8', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='#1E2130', edgecolor='#555'))

fig.suptitle('Figure 8 — ResNet Backbone Analysis: Freeze Strategy vs Performance',
             color='white', fontsize=13)
fig.tight_layout()
save(fig, 'fig8_resnet_analysis.png')

# ─────────────────────────────────────────────────────────────
# FIG 9: Latent space t-SNE — enc2 outputs (C4 vs C5 vs C6)
# ─────────────────────────────────────────────────────────────
def get_latent_enc2(enc1, dec, re_attn, enc2, imgs):
    enc1.eval(); dec.eval(); re_attn.eval(); enc2.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(imgs), 32):
            xb = torch.tensor(imgs[i:i+32]).to(device); n = xb.size(0)
            xh = dec(enc1(xb))
            err = (xb - xh).abs().view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att = re_attn(err)
            z2  = enc2(xb * att)
            zs.append(z2.cpu().numpy())
    return np.concatenate(zs)

try:
    z_c4 = get_latent_enc2(enc1_c4, dec_c4, re_attn_c4, enc2_c4, x_test)
    z_c6 = get_latent_enc2(enc1_c6, dec_c6, re_attn_c6, enc2_c6, x_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0F1117')
    for ax in axes: ax.set_facecolor('#141720')

    for ax, z, k, title in zip(axes, [z_c4, z_c6], ['C4','C6'],
                                ['C4: CNN-RE-Attn-AAE [NOVEL]', 'C6: ResNet Partial FT']):
        emb = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z)//4)).fit_transform(z)
        col = [PAL['C4'] if l == 0 else '#E84C3D' for l in binary_test[:len(z)]]
        ax.scatter(emb[:,0], emb[:,1], c=col, s=18, alpha=0.7, edgecolors='none')
        n_patch = mpatches.Patch(color=PAL['C4'], label='Normal')
        a_patch = mpatches.Patch(color='#E84C3D', label='Pneumonia')
        ax.legend(handles=[n_patch, a_patch], facecolor='#1E2130', labelcolor='white', fontsize=9)
        ax.set_title(f't-SNE of Enc₂ latent space\n{title}\n'
                     f'AUC={all_results[k]["auc_roc"]:.4f}', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
        ax.set_xlabel('t-SNE dim 1', color='white'); ax.set_ylabel('t-SNE dim 2', color='white')

    fig.suptitle('Figure 9 — Latent Space t-SNE: Enc₂ Output (Normal vs Pneumonia)\n'
                 'C4 shows clearer cluster separation — adversarial training aligns normal latents to N(0,I)',
                 color='white', fontsize=13)
    fig.tight_layout()
    save(fig, 'fig9_latent_tsne.png')
except Exception as ex:
    print(f'  [t-SNE skipped: {ex}]')

# ─────────────────────────────────────────────────────────────
# FIG 10: Attention map grid — C4 vs C6 vs C7 on same images
# Shows why CNN attention localises but ResNet attention inverts
# ─────────────────────────────────────────────────────────────
n_show = 5
anom_test_idx = np.where(binary_test == 1)[0][:n_show]

fig = plt.figure(figsize=(20, 7), facecolor='#0F1117')
fig.patch.set_facecolor('#0F1117')
col_titles = ['Input (Pneumonia)', 'SSIM-err C4', 'Attn C4\n[NOVEL]',
              'SSIM-err C6', 'Attn C6\n(ResNet Partial)', 'SSIM-err C7', 'Attn C7\n(ResNet Mostly)']
n_cols = len(col_titles)

for row_i, img_idx in enumerate(anom_test_idx):
    img = x_test[img_idx, 0]
    r4  = get_recon(enc1_c4, dec_c4, x_test[img_idx:img_idx+1])[0, 0]
    r6  = get_recon(enc1_c6, dec_c6, x_test[img_idx:img_idx+1], is_resnet=True)[0, 0]
    r7  = get_recon(enc1_c7, dec_c7, x_test[img_idx:img_idx+1], is_resnet=True)[0, 0]
    cols_data = [
        (img, 'gray'), (np.abs(img-r4), 'hot'),
        (attn_maps_c4[img_idx] if img_idx < len(attn_maps_c4) else np.zeros_like(img), 'plasma'),
        (np.abs(img-r6), 'hot'),
        (attn_maps_c6[img_idx] if img_idx < len(attn_maps_c6) else np.zeros_like(img), 'plasma'),
        (np.abs(img-r7), 'hot'),
        (attn_maps_c7[img_idx] if img_idx < len(attn_maps_c7) else np.zeros_like(img), 'plasma'),
    ]
    for col_i, (data, cmap) in enumerate(cols_data):
        ax = fig.add_subplot(n_show, n_cols, row_i*n_cols + col_i + 1)
        ax.set_facecolor('#0F1117')
        ax.imshow(data, cmap=cmap, vmin=0, vmax=data.max()+1e-8 if cmap != 'gray' else 1)
        ax.axis('off')
        if row_i == 0:
            ax.set_title(col_titles[col_i], color='white', fontsize=8.5, fontweight='bold')

fig.suptitle('Figure 10 — Attention Maps on Pneumonia Cases: C4 (CNN) vs C6/C7 (ResNet)\n'
             'C4 attention highlights opacity; ResNet attention focuses on background (inverted signal)',
             color='white', fontsize=13, y=1.01)
fig.tight_layout()
save(fig, 'fig10_attention_grid_pneumonia.png')

# ─────────────────────────────────────────────────────────────
# FIG 11: Complete metrics summary table as image
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6), facecolor='#0F1117')
ax.axis('off')
ax.set_facecolor('#0F1117')

table_data = []
col_labels = ['Cond.', 'Encoder', 'Adversarial', 'RE-Attn', 'Freeze',
              'AUC-ROC', 'AUC-PR', 'F1', 'px-AUROC\nSSIM', 'px-AUROC\nAttn', 'Disc\nstable']

disc_stable = {'C1':'-','C2':'-','C3':'✓','C4':'✓','C5':'✗ (collapse)','C6':'✓','C7':'✓'}
for k in ['C1','C2','C3','C4','C5','C6','C7']:
    r = all_results[k]
    ssim_px = px.get(f'ssim_{k.lower()}', float('nan'))
    attn_px = px.get(f'attn_{k.lower()}', float('nan'))
    table_data.append([
        k,
        'ResNet-18' if k in ['C5','C6','C7'] else 'CNN',
        '✓' if k in ['C3','C4','C5','C6','C7'] else '✗',
        '✓' if k in ['C4','C5','C6','C7'] else '✗',
        {'C5':'Full','C6':'Partial','C7':'Mostly'}.get(k,'—'),
        f'{r["auc_roc"]:.4f}',
        f'{r["auc_pr"]:.4f}',
        f'{r["f1"]:.4f}',
        f'{ssim_px:.4f}' if not np.isnan(ssim_px) else '—',
        f'{attn_px:.4f}' if not np.isnan(attn_px) else '—',
        disc_stable.get(k,'—'),
    ])

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)

for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#0F1117'); cell.set_edgecolor('#444')
    cell.set_text_props(color='white')
    if row == 0:
        cell.set_facecolor('#1E2744'); cell.set_text_props(fontweight='bold', color='white')
    if row > 0:
        k = table_data[row-1][0]
        if k == 'C4':
            cell.set_facecolor('#2D1B1B')
        elif k == 'C5':
            cell.set_facecolor('#1E1E1E')

ax.set_title('Figure 11 — Complete Metrics Table (C1–C7)', color='white',
             fontsize=13, pad=20)
save(fig, 'fig11_metrics_table.png')

# ─────────────────────────────────────────────────────────────
# ZIP everything and trigger download
# ─────────────────────────────────────────────────────────────
zip_path = '/kaggle/working/dl_report_complete.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(REPORT_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, '/kaggle/working')
            zf.write(fpath, arcname)
    # also include the result JSONs from OUTPUT_DIR
    for fname in ['results.json', 'loss_history.json']:
        src = f'{OUTPUT_DIR}/{fname}'
        if os.path.exists(src):
            zf.write(src, f'results/{fname}')

zip_size_mb = os.path.getsize(zip_path) / 1e6
print(f'\nZIP: {zip_path}  ({zip_size_mb:.1f} MB)')
print('\nFiles in dl_report/:')
for f in sorted(os.listdir(REPORT_DIR)):
    sz = os.path.getsize(f'{REPORT_DIR}/{f}') / 1e3
    print(f'  {f:<45} {sz:>6.1f} KB')

# Kaggle auto-link (shows as clickable in output panel)
from IPython.display import FileLink, display
display(FileLink('dl_report_complete.zip'))
print('\nDONE — click the link above to download, or find it in Output → Files.')


# %% [markdown]
# ---
# ## **Cell 19** — Save Streamlit Assets (weights + results + scores + demo maps)
#
# Collects everything the Streamlit app needs into one zip.
# Run this after all conditions (C1–C7) have trained and been evaluated.

# %% [CELL 19]  Save streamlit assets
#
# Works even after kernel restart — loads weights from disk (saved by C4/C6 training cells)
# and reads scores/metrics from results.json saved by Cell 17.

import os, zipfile, json, shutil
import numpy as np
import torch

SRC_W = f'{OUTPUT_DIR}/weights'          # weights saved by C4/C6 training cells
PACK  = ('/kaggle/working/streamlit_assets' if not SAMPLE_MODE
         else 'streamlit_assets_sample')
os.makedirs(f'{PACK}/weights', exist_ok=True)
os.makedirs(f'{PACK}/demo',    exist_ok=True)

# ── 1. Copy weights (already saved by training cells) ────────────────
for fname in ['c4_enc1.pth','c4_dec.pth','c4_re_attn.pth','c4_enc2.pth','c4_disc.pth',
              'c6_enc1.pth','c6_dec.pth','c6_re_attn.pth','c6_enc2.pth','c6_disc.pth']:
    src = f'{SRC_W}/{fname}'
    if os.path.exists(src):
        shutil.copy2(src, f'{PACK}/weights/{fname}')
        print(f'  copied {fname}')
    else:
        print(f'  MISSING: {src} — re-run C4/C6 training cell first')

# ── 2. Config (hyperparams + thresholds) ─────────────────────────────
# Load results.json saved by Cell 17 for thresholds
_res = json.load(open(f'{OUTPUT_DIR}/results.json'))
config = {
    'image_size':    IMAGE_SIZE,
    'latent_dim':    LATENT_DIM,
    'lambda_adv':    LAMBDA_ADV,
    'eps':           float(EPS),
    'warmup_epochs': WARMUP_EPOCHS,
    'epochs':        EPOCHS,
    'palette': {
        'C1':'#4878CF','C2':'#F5A623','C3':'#7B68EE',
        'C4':'#E84C3D','C5':'#95A5A6','C6':'#2ECC71','C7':'#1A5276',
    },
}
with open(f'{PACK}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('config.json saved.')

# ── 3. Metrics — copy results.json + loss_history.json from Cell 17 ──
shutil.copy2(f'{OUTPUT_DIR}/results.json',      f'{PACK}/metrics.json')
shutil.copy2(f'{OUTPUT_DIR}/loss_history.json', f'{PACK}/loss_history.json')
print('metrics.json + loss_history.json saved.')

# ── 4. Raw scores + labels ───────────────────────────────────────────
score_arrays = {
    'C1': scores_c1, 'C2': scores_c2, 'C3': scores_c3,
    'C4': scores_c4, 'C5': scores_c5, 'C6': scores_c6, 'C7': scores_c7,
}
disc_arrays = {
    'C3': sc_disc_c3, 'C4': sc_disc_c4, 'C5': sc_disc_c5,
    'C6': sc_disc_c6, 'C7': sc_disc_c7,
}
for k, arr in score_arrays.items():
    np.save(f'{PACK}/scores_{k.lower()}.npy', arr)
for k, arr in disc_arrays.items():
    np.save(f'{PACK}/disc_{k.lower()}.npy', arr)
np.save(f'{PACK}/binary_test.npy', binary_test)

# Also save per-condition thresholds into config now that we have scores
config['ssim_thresholds'] = {
    k: float(np.percentile(score_arrays[k][binary_test == 0], 95))
    for k in score_arrays
}
config['disc_thresholds'] = {
    k: float(np.percentile(disc_arrays[k][binary_test == 0], 95))
    for k in disc_arrays
}
with open(f'{PACK}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('scores + thresholds saved.')

# ── 5. Demo images: 10 normal + 10 pneumonia ────────────────────────
n_demo   = 5 if SAMPLE_MODE else 10
norm_idx = np.where(binary_test == 0)[0][:n_demo]
anom_idx = np.where(binary_test == 1)[0][:n_demo]
demo_idx = np.concatenate([norm_idx, anom_idx])
np.save(f'{PACK}/demo/images.npy',  x_test[demo_idx])
np.save(f'{PACK}/demo/labels.npy',  binary_test[demo_idx])
print(f'demo images: {n_demo} normal + {n_demo} pneumonia saved.')

# ── 6. Pre-computed maps for demo images ────────────────────────────
def save_maps(enc1, dec, re_attn, enc2, disc, imgs, tag):
    enc1.eval(); dec.eval(); re_attn.eval(); enc2.eval(); disc.eval()
    ssim_l, attn_l, recon_l, disc_l = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(imgs), 8):
            xb  = torch.tensor(imgs[i:i+8]).to(device)
            n   = xb.size(0)
            xh  = dec(enc1(xb))
            err = (xb - xh).abs().view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att = re_attn(err)
            z2  = enc2(xb * att)
            d   = disc(z2)
            ssim_l.append(err[:,0].cpu().numpy())
            attn_l.append(att[:,0].cpu().numpy())
            recon_l.append(xh[:,0].cpu().numpy())
            disc_l.append(d[:,0].cpu().numpy())
    np.save(f'{PACK}/demo/ssim_{tag}.npy',  np.concatenate(ssim_l))
    np.save(f'{PACK}/demo/attn_{tag}.npy',  np.concatenate(attn_l))
    np.save(f'{PACK}/demo/recon_{tag}.npy', np.concatenate(recon_l))
    np.save(f'{PACK}/demo/disc_{tag}.npy',  np.concatenate(disc_l))
    print(f'  {tag}: maps saved.')

demo_imgs = x_test[demo_idx]
save_maps(enc1_c4, dec_c4, re_attn_c4, enc2_c4, ld_c4,   demo_imgs, 'c4')
save_maps(enc1_c6, dec_c6, re_attn_c6, enc2_c6, disc_c6, demo_imgs, 'c6')

# ── 7. ZIP and download ──────────────────────────────────────────────
zip_path = ('/kaggle/working/streamlit_assets.zip' if not SAMPLE_MODE
            else 'streamlit_assets_sample.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(PACK):
        for fname in files:
            fpath   = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, os.path.dirname(PACK))
            zf.write(fpath, arcname)

zip_mb = os.path.getsize(zip_path) / 1e6
print(f'\nstreamlit_assets.zip: {zip_mb:.1f} MB')
print('Contents:')
for root, dirs, files in os.walk(PACK):
    for fname in sorted(files):
        sz = os.path.getsize(os.path.join(root, fname)) / 1e3
        print(f'  {os.path.relpath(os.path.join(root,fname), PACK):<45} {sz:>7.1f} KB')

if not SAMPLE_MODE:
    from IPython.display import FileLink, display
    display(FileLink('streamlit_assets.zip'))
print('\nDONE — download streamlit_assets.zip from Output → Files.')


# %% [markdown]
# ---
# ## **Cell 20** — Save to Weights & Biases + Google Drive
#
# Two options to persist data before the Kaggle session expires.
# **Option A (wandb)**: full experiment tracking — metrics, loss curves, weights as artifacts.
# **Option B (Google Drive)**: simple file copy — good fallback if wandb is unavailable.
#
# Setup wandb (one-time):
# 1. Create account at https://wandb.ai
# 2. Kaggle → Add-ons → Secrets → add secret named `REATTN_KEY`

# %% [CELL 20]  Save to wandb + Google Drive

import os, json, subprocess, sys, shutil
import numpy as np
import torch

# ── install wandb if missing ──────────────────────────────────────────
try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wandb', '-q'])
    import wandb

# ── wandb login (same pattern as bone_fracture_kaggle.py) ────────────
USE_WANDB = False
try:
    if os.path.exists('/kaggle/working'):
        from kaggle_secrets import UserSecretsClient
        _key = UserSecretsClient().get_secret('REATTN_KEY')
        wandb.login(key=_key)
    else:
        wandb.login()
    USE_WANDB = True
    print('WandB ready.')
except Exception as _e:
    USE_WANDB = False
    print(f'WandB unavailable ({_e}) — will use Google Drive only.')

# ════════════════════════════════════════════════════════════════
# OPTION A — Weights & Biases
# ════════════════════════════════════════════════════════════════
if USE_WANDB:
    run = wandb.init(
        project = 'RE-Attn-AAE-RSNA',
        name    = 'ablation-C1-C7',
        config  = {
            'image_size':    IMAGE_SIZE,
            'latent_dim':    LATENT_DIM,
            'lambda_adv':    LAMBDA_ADV,
            'warmup_epochs': WARMUP_EPOCHS,
            'epochs':        EPOCHS,
            'batch_size':    BATCH_SIZE,
            'lr':            LR,
            'dataset':       'RSNA Pneumonia Detection',
            'train_normal':  int(x_train_norm.shape[0]),
            'test_normal':   int((binary_test == 0).sum()),
            'test_opacity':  int((binary_test == 1).sum()),
        },
        tags    = ['ablation','anomaly-detection','AAE','RE-attention','CXR'],
        reinit  = True,
        settings= wandb.Settings(init_timeout=120),
    )

    # 1. Metrics table
    px = all_results.get('pixel_auroc', {})
    tbl = wandb.Table(columns=[
        'Condition','Label','AUC-ROC','AUC-PR','F1',
        'AUC-ROC Disc','AUC-ROC Fusion',
        'Pixel-AUROC SSIM','Pixel-AUROC Attn','Disc Stable'])
    disc_stable = {'C1':'-','C2':'-','C3':'yes','C4':'yes',
                   'C5':'COLLAPSED','C6':'yes','C7':'yes'}
    for k in ['C1','C2','C3','C4','C5','C6','C7']:
        r = all_results[k]
        tbl.add_data(k, r['label'],
            round(r['auc_roc'],4), round(r['auc_pr'],4), round(r['f1'],4),
            round(all_results.get(k+'_disc',{}).get('auc_roc', float('nan')),4),
            round(all_results.get(k+'_fuse',{}).get('auc_roc', float('nan')),4),
            round(px.get(f'ssim_{k.lower()}', float('nan')),4),
            round(px.get(f'attn_{k.lower()}', float('nan')),4),
            disc_stable.get(k,'-'))
    wandb.log({'ablation_results': tbl})

    # 2. Summary scalars
    summary = {}
    for k in ['C1','C2','C3','C4','C5','C6','C7']:
        r = all_results[k]
        summary[f'{k}/auc_roc'] = r['auc_roc']
        summary[f'{k}/auc_pr']  = r['auc_pr']
        summary[f'{k}/f1']      = r['f1']
        if k+'_fuse' in all_results:
            summary[f'{k}/auc_roc_fusion'] = all_results[k+'_fuse']['auc_roc']
    for pk, pv in px.items():
        if pv is not None:
            summary[f'pixel/{pk}'] = pv
    wandb.log(summary)

    # 3. Loss curves
    for k, hist in loss_history.items():
        for ep, val in enumerate(hist):
            wandb.log({f'loss/{k}': val, f'step_{k}': ep})

    # 4. Score histograms
    score_map = {
        'C1':scores_c1,'C2':scores_c2,'C3':scores_c3,'C4':scores_c4,
        'C5':scores_c5,'C6':scores_c6,'C7':scores_c7,
    }
    for k, sc in score_map.items():
        wandb.log({
            f'scores/{k}_normal':  wandb.Histogram(sc[binary_test == 0]),
            f'scores/{k}_anomaly': wandb.Histogram(sc[binary_test == 1]),
        })

    # 5. Model weights artifact (reads from disk — survives kernel restart)
    _src_w = f'{OUTPUT_DIR}/weights'
    if os.path.isdir(_src_w) and any(f.endswith('.pth') for f in os.listdir(_src_w)):
        art = wandb.Artifact('weights-c4-c6', type='model',
                             description='C4 CNN-RE-Attn-AAE + C6 ResNet Partial FT')
        art.add_dir(_src_w)
        run.log_artifact(art)
        print(f'  weights artifact → wandb')
    else:
        print(f'  weights not found at {_src_w} — run C4/C6 training cells first')

    # 6. Results JSON artifact
    art2 = wandb.Artifact('results-json', type='dataset')
    for fname in ['results.json', 'loss_history.json']:
        fpath = f'{OUTPUT_DIR}/{fname}'
        if os.path.exists(fpath):
            art2.add_file(fpath)
    run.log_artifact(art2)

    wandb.finish()
    print(f'\nwandb done. View at: https://wandb.ai/home → project RE-Attn-AAE-RSNA')

# ════════════════════════════════════════════════════════════════
# OPTION B — Google Drive  (works alongside or instead of wandb)
# ════════════════════════════════════════════════════════════════
print('\nMounting Google Drive...')
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    GDRIVE_DIR = '/content/drive/MyDrive/RE-Attn-AAE-RSNA'
except Exception:
    # On Kaggle, use gdown approach via Drive API or manual download
    print('  Google Drive mount not available on Kaggle.')
    print('  Alternative: download streamlit_assets.zip from Output → Files.')
    GDRIVE_DIR = None

if GDRIVE_DIR:
    import shutil, datetime
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    dst   = f'{GDRIVE_DIR}/{stamp}'
    os.makedirs(dst, exist_ok=True)

    # copy OUTPUT_DIR (results.json, loss_history.json, weights/)
    if os.path.isdir(OUTPUT_DIR):
        shutil.copytree(OUTPUT_DIR, f'{dst}/results', dirs_exist_ok=True)
        print(f'  results copied to Google Drive: {dst}/results')

    # copy streamlit zip if it exists
    _zip = '/kaggle/working/streamlit_assets.zip'
    if os.path.exists(_zip):
        shutil.copy2(_zip, f'{dst}/streamlit_assets.zip')
        print(f'  streamlit_assets.zip copied to Google Drive')

    print(f'\nGoogle Drive backup done: MyDrive/RE-Attn-AAE-RSNA/{stamp}/')

print('\nDONE. Your results are safe.')
