# %% [markdown]
# RE-Attn-AAE: A Reconstruction-Error-Guided Attention Adversarial Autoencoder for Dual-Domain Unsupervised Anomaly Detection
# This is  a PyTorch 
# implementation of the RE-Attn-AAE model for unsupervised anomaly detection in dual-domain data. The model leverages reconstruction error to guide attention mechanisms, 
# enhancing the detection of anomalies in complex datasets.

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


# %% [CELL 1.4]  define the configuration for the project  this includes
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
            art.tags = [_group]        # only settable on an already-logged, waited-on artifact
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
# ## **Cell 3.0** — Model architectures
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
