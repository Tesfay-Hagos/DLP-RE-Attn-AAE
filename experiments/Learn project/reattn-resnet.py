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
TEST_NORMAL   = 2000 if not SAMPLE_MODE else 10
TEST_OPACITY  = 2000 if not SAMPLE_MODE else 5

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



# %% [CELL 6]  Model architecturesWSSSSS
class CNNEncoder(nn.Module):
    """3-block conv encoder → flatten → Linear. down: 'max' | 'avg' | 'stride'."""
    def __init__(self, latent_dim, image_size=IMAGE_SIZE, down='max'):
        super().__init__()
        s = image_size // 8
        def block(cin, cout):
            if down == 'stride':
                return [nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                        nn.BatchNorm2d(cout), nn.ReLU()]
            pool = nn.MaxPool2d(2) if down == 'max' else nn.AvgPool2d(2)
            return [nn.Conv2d(cin, cout, 3, padding=1),
                    nn.BatchNorm2d(cout), nn.ReLU(), pool]
        self.conv = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128))
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
    """Same CNN backbone as CNNEncoder with dual mu / log-var projection heads.
    down: 'max' | 'avg' | 'stride' — same knob as CNNEncoder, defaults to 'max'
    (unchanged behavior) since the VAE's KL-regularized latent means the
    encoder-sweep result for the plain CNN-AE isn't assumed to transfer here.
    """
    def __init__(self, latent_dim, image_size=IMAGE_SIZE, down='max'):
        super().__init__()
        s = image_size // 8
        def block(cin, cout):
            if down == 'stride':
                return [nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                        nn.BatchNorm2d(cout), nn.ReLU()]
            pool = nn.MaxPool2d(2) if down == 'max' else nn.AvgPool2d(2)
            return [nn.Conv2d(cin, cout, 3, padding=1),
                    nn.BatchNorm2d(cout), nn.ReLU(), pool]
        self.conv = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128))
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
    """Conv error-guided attention: (B,2,H,W) → soft mask (B,1,H,W) ∈[0,1].
    2 input channels: SSIM error map + high-frequency residual of the raw image.
    The high-freq channel gives the module an explicit edge signal, so it doesn't
    have to infer "thin rib edge vs. broad opacity blob" purely from error shape."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  1, 1),            nn.Sigmoid(),
        )

    def forward(self, e):
        return self.net(e)


def high_freq(x, ksize=5):
    """(B,1,H,W) -> (B,1,H,W) high-pass residual: |x - blur(x)|. Highlights sharp
    structural edges (ribs, heart border) as an explicit feature for REAttention."""
    blur = F.avg_pool2d(x, ksize, stride=1, padding=ksize // 2)
    return (x - blur).abs()


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
# ## **Cell 3.0.1** — Evaluation utilities
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

# %% [CELL 3.0.1]  Evaluation utilities

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
# ## **Cell 3.1-sweep** — Encoder downsampling comparison (max / avg / stride)
#
# **Question:** does the choice of downsampling operator inside `CNNEncoder`
# change reconstruction/anomaly-detection quality enough to matter, before
# committing to one for C1/C3/C4?
#
# **Design — 3 variants × 3 seeds = 9 independent C1-style runs:**
# - `max`    — current default: `Conv3x3 → MaxPool2d(2)`
# - `avg`    — same, but `AvgPool2d(2)` instead of max
# - `stride` — no pooling at all: `Conv4x4(stride=2)` learns the downsampling
#
# **Why 3 seeds, not 1:** weight init, batch shuffling, and flip augmentation
# are all random. A single run's AUC is one sample from a noisy distribution
# (typically ±0.005–0.01 spread run-to-run on this setup) — comparing single
# runs risks mistaking training noise for a real architectural difference.
#
# **Why the split seed (`SPLIT_SEED`) is fixed and separate from `TRAIN_SEED`:**
# the train/test patient split is also randomised (`np.random.shuffle` in Cell
# 2.0). If seeding it together with training, changing the seed to get
# variance would *also* reshuffle which images are in train vs. test —
# making the 9 runs incomparable, since they'd be scored on different data.
# `SPLIT_SEED` never changes; only `TRAIN_SEED` varies across the 9 runs.
#
# **Isolation from the real study:** each run is saved under its own
# `cond_id = 'ENCSWEEP_{down}_s{seed}'`, never `'C1'` — so this sweep cannot
# overwrite or interfere with the real ablation chain's C1 checkpoint.
#
# **Metrics recorded per run:** image AUC-ROC / AUC-PR / F1 (detection
# quality), pixel-AUROC (localisation quality against radiologist boxes),
# final training loss, wall-clock time, and encoder parameter count (so a
# win isn't just "more capacity").
#
# **How to read the result (Cell 3.1-sweep-summary):** collapse each
# variant's 3 seeds to mean ± std. If the gap between two variants' mean
# AUC-ROC is bigger than ~2× the pooled std, treat the difference as real;
# otherwise they're statistically indistinguishable here — in that case
# pick whichever is faster / has fewer parameters, which is a legitimate
# conclusion, not a non-answer.


# %% [CELL 3.1-sweep]  Encoder downsample comparison (max / avg / stride), 3 seeds each
encoder_sweep_results = []

for down in ['max', 'avg', 'stride']:
    for seed in [42, 1337, 2024]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cond_id = f'ENCSWEEP_{down}_s{seed}'
        print(f"\n--- {cond_id} ---")

        enc = CNNEncoder(LATENT_DIM, down=down).to(device)
        dec = CNNDecoder(LATENT_DIM).to(device)
        n_params = sum(p.numel() for p in enc.parameters())

        ensure_local(cond_id)
        if is_done(cond_id):
            scores, pix_maps, _ = load_ckpt(cond_id)   # pix_maps was saved into the disc_scores slot
            load_weights(cond_id, enc1=enc, dec=dec)
            epoch_loss = loss_history[cond_id]
            train_time = np.nan   # not meaningful for a skipped/reloaded run
        else:
            opt   = Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
            loader = make_loader(x_train_norm, BATCH_SIZE)

            t0 = time.time()
            epoch_loss = []
            for epoch in range(EPOCHS):
                enc.train(); dec.train()
                losses = []
                for (xb,) in loader:
                    xb = xb.to(device)
                    flip = torch.rand(xb.size(0), device=device) > 0.5
                    xb[flip] = xb[flip].flip(dims=[3])
                    opt.zero_grad()
                    xhat = dec(enc(xb))
                    loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
                    loss.backward(); opt.step()
                    losses.append(loss.item())
                sched.step()
                epoch_loss.append(np.mean(losses))
            train_time = time.time() - t0

            enc.eval(); dec.eval()
            scores, pix_maps = [], []
            with torch.no_grad():
                for i in range(0, len(x_test), BATCH_SIZE):
                    xb   = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
                    xhat = dec(enc(xb))
                    scores.append(anomaly_score(xb, xhat).cpu().numpy())
                    pix_maps.append(ssim_anomaly_map(xb, xhat).cpu().numpy())
            scores   = np.concatenate(scores)
            pix_maps = np.concatenate(pix_maps)

        m = compute_metrics(scores, binary_test)
        pix_auroc = pixel_auroc(pix_maps, test_boxes, binary_test)

        print(f"  AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}  F1={m['f1']:.4f}  "
              f"pixel-AUROC={pix_auroc:.4f}  params={n_params:,}  time={train_time:.0f}s")

        encoder_sweep_results.append({
            'down': down, 'seed': seed, 'auc_roc': m['auc_roc'], 'auc_pr': m['auc_pr'],
            'f1': m['f1'], 'pixel_auroc': pix_auroc, 'final_loss': epoch_loss[-1],
            'params': n_params, 'time_s': train_time,
        })

        all_results[cond_id] = {**m, 'pixel_auroc': pix_auroc, 'label': f'ENCSWEEP {down} s{seed}'}
        if not is_done(cond_id):
            save_ckpt(cond_id, [cond_id], scores, pix_maps, epoch_loss,
                      enc1=enc.state_dict(), dec=dec.state_dict())


# %% [markdown]
# ---
# ## **Cell 3.1** — Exp1: CNN-AE Baseline
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


# %% [CELL 3.1]  Exp1 — CNN-AE Baseline

print("\n" + "="*60)
print("CONDITION 1 — CNN-AE Baseline")
print("="*60)

enc_c1 = CNNEncoder(LATENT_DIM,down='stride').to(device)
dec_c1 = CNNDecoder(LATENT_DIM).to(device)
ensure_local('C1')
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
# ## **Cell 3.2** — Exp2: VAE Baseline
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

# %% [CELL 3.2]  Exp2 — VAE Baseline

print("\n" + "="*60)
print("CONDITION 2 — VAE Baseline")
print("="*60)
print("Probabilistic AE: same CNN backbone + reparameterised latent + KL term.")
print("Scored with SSIM 99th-pct on reconstruction (consistent with C1/C3/C4/C5).\n")

enc_vae = VAEEncoder(LATENT_DIM,down='stride').to(device)
dec_vae = CNNDecoder(LATENT_DIM).to(device)

ensure_local('C2')
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
# ## **Cell 3.3-sweep** — C3: CNN-AAE Ablation (adversarial regularisation, no attention)
#
# **Architecture:** `CNNEncoder (enc1) → CNNDecoder + LatentDisc`
# Executes fro multiple `lambda_adv` values to check 
# if the C3 < C1 result is consistent or a config artifact.
# **LAMBDA_VALUES = [0.05, 0.1, 0.3, 0.6, 1.0]  LAMBDA_SEEDS  = [42, 1337]   

# %% [CELL 3.3-sweep]  LAMBDA_ADV robustness check — is C3 < C1 a config artifact or consistent?
# Retrains C3's exact architecture at several lambda_adv values (current default: 0.3),
# a couple of seeds each. Isolated cond_id namespace ('C3_LADV{lam}_s{seed}') — never
# touches the real 'C3' checkpoint/all_results entry used in the main ablation table.
LAMBDA_VALUES = [0.05, 0.1, 0.3, 0.6, 1.0]
LAMBDA_SEEDS  = [42, 1337]   # bump to 3 seeds later if a value looks borderline

c3_lambda_sweep_results = []

for lam in LAMBDA_VALUES:
    for seed in LAMBDA_SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cond_id = f'C3_LADV{lam}_s{seed}'
        print(f"\n--- {cond_id} ---")

        enc1 = CNNEncoder(LATENT_DIM, down='stride').to(device)
        dec  = CNNDecoder(LATENT_DIM).to(device)
        ld   = LatentDisc(LATENT_DIM).to(device)

        ensure_local(cond_id)
        if is_done(cond_id):
            scores, sc_disc, _ = load_ckpt(cond_id)
            load_weights(cond_id, enc1=enc1, dec=dec, disc=ld)
            epoch_loss = loss_history[cond_id]
        else:
            opt_rec  = Adam(list(enc1.parameters()) + list(dec.parameters()), lr=LR, betas=(BETA1, 0.999))
            opt_disc = Adam(ld.parameters(), lr=LR, betas=(BETA1, 0.999))
            opt_gen  = Adam(enc1.parameters(), lr=LR, betas=(BETA1, 0.999))
            sched_rec  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rec,  T_max=EPOCHS, eta_min=1e-6)
            sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=EPOCHS, eta_min=1e-6)
            sched_gen  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen,  T_max=EPOCHS, eta_min=1e-6)
            loader = make_loader(x_train_norm, BATCH_SIZE)

            opt_warmup = Adam(list(enc1.parameters()) + list(dec.parameters()), lr=LR, betas=(BETA1, 0.999))
            for epoch in range(WARMUP_EPOCHS):
                enc1.train(); dec.train()
                for (xb,) in loader:
                    xb = xb.to(device)
                    flip = torch.rand(xb.size(0), device=device) > 0.5
                    xb[flip] = xb[flip].flip(dims=[3])
                    opt_warmup.zero_grad()
                    xhat = dec(enc1(xb))
                    loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
                    loss.backward(); opt_warmup.step()

            epoch_loss = []
            for epoch in range(EPOCHS):
                enc1.train(); dec.train(); ld.train()
                rec_l = []
                for (xb,) in loader:
                    xb = xb.to(device); n = xb.size(0)
                    flip = torch.rand(n, device=device) > 0.5
                    xb[flip] = xb[flip].flip(dims=[3])
                    opt_rec.zero_grad()
                    z1 = enc1(xb); x_hat1 = dec(z1)
                    loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
                    loss_rec.backward(); opt_rec.step()
                    opt_disc.zero_grad()
                    with torch.no_grad():
                        z_fake = enc1(xb)
                    z_real = torch.randn(n, LATENT_DIM, device=device)
                    loss_d = (-torch.mean(torch.log(ld(z_real) + EPS))
                              - torch.mean(torch.log(1.0 - ld(z_fake) + EPS)))
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(ld.parameters(), max_norm=1.0)
                    opt_disc.step()
                    opt_gen.zero_grad()
                    loss_g = lam * (-torch.mean(torch.log(ld(enc1(xb)) + EPS)))   # <-- swept value
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(enc1.parameters(), max_norm=1.0)
                    opt_gen.step()
                    rec_l.append(loss_rec.item())
                sched_rec.step(); sched_disc.step(); sched_gen.step()
                epoch_loss.append(np.mean(rec_l))

            enc1.eval(); dec.eval(); ld.eval()
            scores, sc_disc = [], []
            with torch.no_grad():
                for i in range(0, len(x_test), BATCH_SIZE):
                    xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
                    z1 = enc1(xb); xhat = dec(z1)
                    scores.append(anomaly_score(xb, xhat).cpu().numpy())
                    sc_disc.append((1.0 - ld(z1)).squeeze(1).cpu().numpy())
            scores  = np.concatenate(scores)
            sc_disc = np.concatenate(sc_disc)

        sc_fuse = 0.5 * normalise_scores(scores) + 0.5 * normalise_scores(sc_disc)
        m       = compute_metrics(scores,  binary_test)
        m_disc  = compute_metrics(sc_disc, binary_test)
        m_fuse  = compute_metrics(sc_fuse, binary_test)

        print(f"  lambda={lam}  SSIM AUC-ROC={m['auc_roc']:.4f}  disc AUC-ROC={m_disc['auc_roc']:.4f}  "
              f"fuse AUC-ROC={m_fuse['auc_roc']:.4f}")

        c3_lambda_sweep_results.append({
            'lambda_adv': lam, 'seed': seed,
            'ssim_auc_roc': m['auc_roc'], 'disc_auc_roc': m_disc['auc_roc'], 'fuse_auc_roc': m_fuse['auc_roc'],
        })

        all_results[cond_id] = {**m, 'label': f'C3 lambda={lam} s{seed}'}
        if not is_done(cond_id):
            save_ckpt(cond_id, [cond_id], scores, sc_disc, epoch_loss,
                      enc1=enc1.state_dict(), dec=dec.state_dict(), disc=ld.state_dict())

# %% [CELL 3.3-sweep-summary]  Does C3 < C1 hold across lambda_adv, or was 0.3 just unlucky?
df_lambda = pd.DataFrame(c3_lambda_sweep_results)
summary_lambda = df_lambda.groupby('lambda_adv')[['ssim_auc_roc', 'disc_auc_roc', 'fuse_auc_roc']].agg(['mean', 'std'])
print(summary_lambda)

c1_auc = all_results['C1']['auc_roc']
print(f"\nC1 baseline AUC-ROC = {c1_auc:.4f}\n")
for lam in sorted(df_lambda['lambda_adv'].unique()):
    ssim_mean = df_lambda[df_lambda.lambda_adv == lam]['ssim_auc_roc'].mean()
    verdict = 'C3 >= C1' if ssim_mean >= c1_auc else 'C3 <  C1'
    print(f"  lambda_adv={lam:<5}  C3 SSIM mean={ssim_mean:.4f}  vs C1={c1_auc:.4f}  -> {verdict}")

# Best lambda by MEAN across seeds (not max single run — avoids cherry-picking noise).
# CELL 3.3 below reads this to train the one canonical C3 checkpoint.
best_by_mean    = df_lambda.groupby('lambda_adv')['ssim_auc_roc'].mean()
BEST_LAMBDA_ADV = float(best_by_mean.idxmax())
print(f"\nBEST_LAMBDA_ADV = {BEST_LAMBDA_ADV}  (mean SSIM AUC-ROC = {best_by_mean.max():.4f})")


# %% [markdown]
# ---
# ## **Cell 3.3** — C3: CNN-AAE Ablation (adversarial regularisation, no attention)
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

# %% [CELL 3.3]  C3 — CNN-AAE Ablation (adversarial regularisation, no attention)

print("\n" + "="*60)
print("CONDITION 3 — CNN-AAE  [ablation: adversarial without RE-Attention]")
print("="*60)
print("Adds latent discriminator to C1. enc1 latents pushed toward N(0,I).")
print("No re_attn, no enc2 — tests adversarial regularisation alone.")
print("C1→C3: does AAE help?  C3→C4: does RE-Attention add value?\n")

enc1_c3 = CNNEncoder(LATENT_DIM,down='stride').to(device)
dec_c3  = CNNDecoder(LATENT_DIM).to(device)
ld_c3   = LatentDisc(LATENT_DIM).to(device)
ensure_local('C3')
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
            loss_g = BEST_LAMBDA_ADV * (-torch.mean(torch.log(ld_c3(enc1_c3(xb)) + EPS)))
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
# ## **Cell 3.4** — C3.4: Denoising-AE control (required — isolates the denoising gain from the attention gain)
#
# C3.5's pass 1 trains on `synth → clean`, not `clean → clean` like C1 — that's a **denoising
# autoencoder**, and DAEs are not a neutral change: Kascenas, Pugeault & O'Neil (MIDL 2022)
# show a coarse-noise DAE alone reaches SOTA unsupervised tumour detection in brain MRI,
# beating VAEs outright. Without this control, C3.5's number conflates "denoising training
# helps" with "attention helps" — exactly the confound we just fixed for the discriminator.
# `enc1 + dec` only, same corruption as C3.4b, no attention, no second encoder.

# %% [CELL 3.4]  C3.4 — Denoising-AE control (no attention)

print("\n" + "="*60)
print("CONDITION 3.4 — Denoising-AE control  [no attention]")
print("="*60)
print("Isolates the denoising-training gain from the attention gain in C3.4b.\n")

enc1_c34 = CNNEncoder(LATENT_DIM, down='stride').to(device)
dec_c34  = CNNDecoder(LATENT_DIM).to(device)

ensure_local('C3.4')
if is_done('C3.4'):
    scores_c34, _, _ = load_ckpt('C3.4')
    load_weights('C3.4', enc1=enc1_c34, dec=dec_c34)
else:
    opt_c34   = Adam(list(enc1_c34.parameters()) + list(dec_c34.parameters()), lr=LR, betas=(BETA1, 0.999))
    sched_c34 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c34, T_max=EPOCHS, eta_min=1e-6)
    loader_c34     = make_loader(x_train_norm, BATCH_SIZE)
    c34_epoch_loss = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc1_c34.train(); dec_c34.train()
        losses = []
        for (xb,) in loader_c34:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            apply = (torch.rand(n, device=device) < 0.5).view(-1, 1, 1, 1).float()
            x_synth, _ = make_synthetic_anomaly(xb)
            x_in = xb * (1 - apply) + x_synth * apply       # p=0.5 corruption, matches C3.4b
            opt_c34.zero_grad()
            x_hat = dec_c34(enc1_c34(x_in))
            loss = 0.7 * mse_fn(x_hat, xb) + 0.3 * ssim_loss_fn(x_hat, xb)   # target ALWAYS clean
            loss.backward(); opt_c34.step()
            losses.append(loss.item())
        sched_c34.step()
        c34_epoch_loss.append(np.mean(losses))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  loss={c34_epoch_loss[-1]:.5f}  "
                  f"lr={sched_c34.get_last_lr()[0]:.2e}")
    loss_history['C3.4'] = c34_epoch_loss
    print(f"C3.4 training: {time.time()-t0:.1f}s")
    enc1_c34.eval(); dec_c34.eval()
    scores_c34 = []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)   # REAL images, no corruption
            scores_c34.append(anomaly_score(xb, dec_c34(enc1_c34(xb))).cpu().numpy())
    scores_c34 = np.concatenate(scores_c34)
    m_c34 = compute_metrics(scores_c34, binary_test)
    print(f"\n  AUC-ROC={m_c34['auc_roc']:.4f}  AUC-PR={m_c34['auc_pr']:.4f}  F1={m_c34['f1']:.4f}  "
          f"(compare to C1={all_results.get('C1', {}).get('auc_roc', float('nan')):.4f} to see the denoising-alone effect)")
    all_results['C3.4'] = {**m_c34, 'label': 'Denoising-AE control (no attention)'}
    save_ckpt('C3.4', ['C3.4'], scores_c34, None, c34_epoch_loss,
              enc1=enc1_c34.state_dict(), dec=dec_c34.state_dict())


# %% [markdown]
# ---
# ## **Cell 3.4b** — C3.4b: RE-Attn iterative refinement, anatomy-restricted synthetic supervision
# *(Ours — extends C3.4 — Full Novel Method)*
#
# **Three changes from C3.5, each grounded in a specific published result:**
#
# 1. **Synthetic anomalies restricted to a lung-field prior**, not pasted anywhere in the frame —
#    AnatPaste (Sato et al., *iScience* 2023) showed anatomy-restricted synthetic lesions on CXR
#    beat unrestricted placement, the highest-performing UAD model in their comparison, precisely
#    because an unrestricted generator teaches "detect any sharp/bright region" instead of
#    "detect a plausible pathology location."
# 2. **Score from the mask, not the reconstruction residual**: `s = quantile(1 - a_K, 0.99)`.
#    DRAEM (Zavrtanik et al., ICCV 2021) showed a discriminatively-trained segmentation head
#    outperforms residual-based scoring "by a large margin" — once `re_attn` has real
#    supervision, throwing its output away and scoring a noisier downstream residual instead
#    (what C3.5 does) discards the most direct signal available.
# 3. **Iterative shared-weight refinement**, `K` steps, same `re_attn`/`enc2`/`dec` every step,
#    deep-supervised at each step — the learned counterpart to IterMask² (Liang et al., MICCAI
#    2024), which performs the same error→mask→reconstruct cycle with a hand-designed
#    thresholding rule instead of a learned operator. `K` is swept at inference (no retraining)
#    to show whether refinement converges.
#
# **Class imbalance fix:** a synthetic blob covers ~5–15% of pixels, so unweighted BCE is
# dominated by the majority "keep open" class and cheaply satisfied by `a → 1` — the same
# collapse C3.5 risked, just slower. `focal_bce` (Lin et al., ICCV 2017) down-weights the
# easy majority class instead.

# %% [CELL 3.4b]  Anatomy-restricted synthesis + focal loss for C3.4b

def _make_lung_prior(size=IMAGE_SIZE, device='cpu'):
    """Two overlapping ellipses approximating a left/right lung field at the fixed
    128x128 frame this pipeline always uses. Simplification of true lung segmentation
    (no segmentation model available in this pipeline) — a fixed anatomical prior,
    same spirit as AnatPaste's threshold-based lung mask, cheaper to compute."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
    left  = (((xx - 0.32) / 0.22) ** 2 + ((yy - 0.52) / 0.38) ** 2) <= 1.0
    right = (((xx - 0.68) / 0.22) ** 2 + ((yy - 0.52) / 0.38) ** 2) <= 1.0
    prior = (left | right).float().to(device)
    return prior.view(1, 1, size, size)

def make_synthetic_anomaly_anat(xb, lung_prior):
    """Same blob generator as C3.5's make_synthetic_anomaly, but restricted to
    lung_prior — the AnatPaste-style anatomy restriction."""
    n, c, h, w = xb.shape
    noise = torch.rand(n, 1, 8, 8, device=xb.device)
    blob  = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=False)
    mask  = (blob > blob.mean(dim=[2, 3], keepdim=True)).float()
    mask  = F.avg_pool2d(mask, 9, stride=1, padding=4)
    mask  = mask * lung_prior
    bright  = (xb + 0.35).clamp(0, 1)
    x_synth = xb * (1 - mask) + bright * mask
    return x_synth, mask

def focal_bce(pred, target, gamma=2.0, alpha=0.25, eps=1e-8):
    """Lin et al. 2017 (RetinaNet), adapted to dense per-pixel mask supervision.
    Down-weights the easy majority class (att≈1 on the ~90%+ non-anomalous pixels)
    so the rare positive (mask=1, 'close here') class isn't drowned out."""
    pred = pred.clamp(eps, 1 - eps)
    pt      = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, torch.as_tensor(alpha, device=pred.device),
                                        torch.as_tensor(1 - alpha, device=pred.device))
    return (-alpha_t * (1 - pt).pow(gamma) * torch.log(pt)).mean()


# %% [CELL 3.4b]  C3.4b — RE-Attn iterative refinement  [NOVEL — FLAGSHIP, extends C3.4]

print("\n" + "="*60)
print("CONDITION 3.4b — RE-Attn iterative refinement, anatomy-restricted synthesis  [NOVEL]")
print("="*60)
print("Extends C3.4: adds K-step shared-weight mask refinement, scored on the mask itself.\n")

K_TRAIN = 3
K_EVAL_MAX = 4

enc1_c34b    = CNNEncoder(LATENT_DIM, down='stride').to(device)
enc2_c34b    = CNNEncoder(LATENT_DIM, down='stride').to(device)
dec_c34b     = CNNDecoder(LATENT_DIM).to(device)
re_attn_c34b = REAttention().to(device)
LUNG_PRIOR   = _make_lung_prior(IMAGE_SIZE, device)

ensure_local('C3.4b')
if is_done('C3.4b'):
    scores_c34b, _, attn_maps_c34b = load_ckpt('C3.4b')
    load_weights('C3.4b', enc1=enc1_c34b, enc2=enc2_c34b, dec=dec_c34b, re_attn=re_attn_c34b)
else:
    opt_c34b   = Adam(list(enc1_c34b.parameters()) + list(enc2_c34b.parameters())
                       + list(dec_c34b.parameters()) + list(re_attn_c34b.parameters()),
                       lr=LR, betas=(BETA1, 0.999))
    sched_c34b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c34b, T_max=EPOCHS, eta_min=1e-6)
    loader_c34b     = make_loader(x_train_norm, BATCH_SIZE)
    c34b_epoch_loss = []

    print(f"Warm-start enc1+dec for {WARMUP_EPOCHS} epochs (clean reconstruction)...")
    opt_warmup_c34b = Adam(list(enc1_c34b.parameters()) + list(dec_c34b.parameters()), lr=LR, betas=(BETA1, 0.999))
    t_ws = time.time()
    for epoch in range(WARMUP_EPOCHS):
        enc1_c34b.train(); dec_c34b.train()
        ws_l = []
        for (xb,) in loader_c34b:
            xb = xb.to(device)
            flip = torch.rand(xb.size(0), device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt_warmup_c34b.zero_grad()
            xhat = dec_c34b(enc1_c34b(xb))
            loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
            loss.backward(); opt_warmup_c34b.step()
            ws_l.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={np.mean(ws_l):.5f}")
    print(f"Warm-start done ({time.time()-t_ws:.1f}s). Activating iterative refinement.\n")

    c34b_diagnostics = {'rec0': [], 'rec_iter': [], 'seg': [], 'att_mean': [], 'att_std': []}
    COLLAPSE_STD_THRESHOLD = 0.02

    t0 = time.time()
    for epoch in range(EPOCHS):
        enc1_c34b.train(); enc2_c34b.train(); dec_c34b.train(); re_attn_c34b.train()
        rec0_l, rec_iter_l, seg_l, att_mean_l, att_std_l = [], [], [], [], []
        for (xb,) in loader_c34b:
            xb = xb.to(device); n = xb.size(0)
            flip = torch.rand(n, device=device) > 0.5
            xb[flip] = xb[flip].flip(dims=[3])

            apply = (torch.rand(n, device=device) < 0.5).view(-1, 1, 1, 1).float()
            x_synth, m_full = make_synthetic_anomaly_anat(xb, LUNG_PRIOR)
            x_in = xb * (1 - apply) + x_synth * apply     # p=0.5: half the batch stays clean
            m    = m_full * apply                          # mask=0 everywhere on the clean half

            opt_c34b.zero_grad()
            z1 = enc1_c34b(x_in); x_hat0 = dec_c34b(z1)
            loss_rec0 = 0.7 * mse_fn(x_hat0, xb) + 0.3 * ssim_loss_fn(x_hat0, xb)

            e = ssim_anomaly_map(x_in, x_hat0).detach().view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            seg_terms, rec_terms = [], []
            att = None
            for t in range(K_TRAIN):
                hf  = high_freq(x_in)
                att = re_attn_c34b(torch.cat([e, hf], dim=1))
                seg_terms.append(focal_bce(att, 1.0 - m))
                x_hat_t = dec_c34b(enc2_c34b(x_in * att))
                rec_terms.append(0.7 * mse_fn(x_hat_t, xb) + 0.3 * ssim_loss_fn(x_hat_t, xb))
                e = ssim_anomaly_map(x_in, x_hat_t).detach().view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss_seg      = torch.stack(seg_terms).mean()       # deep supervision: every step
            loss_rec_iter = torch.stack(rec_terms).mean()

            (loss_rec0 + LAMBDA_REC2 * loss_rec_iter + LAMBDA_SEG * loss_seg).backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc1_c34b.parameters()) + list(enc2_c34b.parameters())
                + list(dec_c34b.parameters()) + list(re_attn_c34b.parameters()), max_norm=1.0)
            opt_c34b.step()

            rec0_l.append(loss_rec0.item()); rec_iter_l.append(loss_rec_iter.item()); seg_l.append(loss_seg.item())
            att_mean_l.append(att.mean().item()); att_std_l.append(att.std().item())
        sched_c34b.step()
        c34b_epoch_loss.append(np.mean(rec0_l))
        for key, vals in [('rec0', rec0_l), ('rec_iter', rec_iter_l), ('seg', seg_l),
                           ('att_mean', att_mean_l), ('att_std', att_std_l)]:
            c34b_diagnostics[key].append(float(np.mean(vals)))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  Rec0={c34b_diagnostics['rec0'][-1]:.5f}  "
                  f"RecIter={c34b_diagnostics['rec_iter'][-1]:.5f}  Seg={c34b_diagnostics['seg'][-1]:.4f}  "
                  f"att_mean={c34b_diagnostics['att_mean'][-1]:.3f}  att_std={c34b_diagnostics['att_std'][-1]:.3f}  "
                  f"lr={sched_c34b.get_last_lr()[0]:.2e}")
        if c34b_diagnostics['att_std'][-1] < COLLAPSE_STD_THRESHOLD:
            print(f"  ⚠ WARNING epoch {epoch+1}: att.std()={c34b_diagnostics['att_std'][-1]:.4f} "
                  f"< {COLLAPSE_STD_THRESHOLD} — mask may be collapsing toward a constant.")
    loss_history['C3.4b'] = c34b_epoch_loss
    print(f"C3.4b training: {time.time()-t0:.1f}s")

    enc1_c34b.eval(); enc2_c34b.eval(); dec_c34b.eval(); re_attn_c34b.eval()
    scores_c34b, scores_c34b_resid, attn_maps_c34b = [], [], []
    k_curve_scores = {k: [] for k in range(1, K_EVAL_MAX + 1)}   # per-K s_mask, collected across all batches
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device); n = xb.size(0)   # REAL images, no corruption
            z1 = enc1_c34b(xb); x_hat = dec_c34b(z1)
            e = ssim_anomaly_map(xb, x_hat).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att = None
            for t in range(1, K_EVAL_MAX + 1):
                hf  = high_freq(xb)
                att = re_attn_c34b(torch.cat([e, hf], dim=1))
                x_hat = dec_c34b(enc2_c34b(xb * att))
                e = ssim_anomaly_map(xb, x_hat).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
                s_mask_t = torch.quantile((1.0 - att).view(n, -1), 0.99, dim=1)
                k_curve_scores[t].append(s_mask_t.cpu().numpy())
                if t == K_TRAIN:
                    scores_c34b.append(s_mask_t.cpu().numpy())
                    scores_c34b_resid.append(anomaly_score(xb, x_hat).cpu().numpy())
                    attn_maps_c34b.append(att.squeeze(1).cpu().numpy())
    scores_c34b       = np.concatenate(scores_c34b)
    scores_c34b_resid = np.concatenate(scores_c34b_resid)
    attn_maps_c34b    = np.concatenate(attn_maps_c34b)
    m_c34b       = compute_metrics(scores_c34b, binary_test)
    m_c34b_resid = compute_metrics(scores_c34b_resid, binary_test)
    print(f"\n  s_mask (K={K_TRAIN}, PRIMARY)  AUC-ROC={m_c34b['auc_roc']:.4f}  AUC-PR={m_c34b['auc_pr']:.4f}  F1={m_c34b['f1']:.4f}")
    print("  Convergence curve (s_mask AUC-ROC vs. K, no retraining needed):")
    k_curve_aucs = {}
    for k in range(1, K_EVAL_MAX + 1):
        auc_k = compute_metrics(np.concatenate(k_curve_scores[k]), binary_test)['auc_roc']
        k_curve_aucs[k] = auc_k
        print(f"    K={k}:  AUC-ROC={auc_k:.4f}" + ("  <- trained K" if k == K_TRAIN else ""))
    print(f"  s_resid (K={K_TRAIN})           AUC-ROC={m_c34b_resid['auc_roc']:.4f}")
    all_results['C3.4b']       = {**m_c34b,       'label': f'RE-Attn iterative refinement K={K_TRAIN} (Ours)'}
    all_results['C3.4b_resid'] = {**m_c34b_resid, 'label': f'RE-Attn iterative refinement residual score K={K_TRAIN}'}
    save_ckpt('C3.4b', ['C3.4b','C3.4b_resid'], scores_c34b, scores_c34b_resid, c34b_epoch_loss,
              attn_maps=attn_maps_c34b,
              enc1=enc1_c34b.state_dict(), enc2=enc2_c34b.state_dict(),
              dec=dec_c34b.state_dict(), re_attn=re_attn_c34b.state_dict())
    with open(f'{CKPT_DIR}/C3.4b_diagnostics.json', 'w') as f:
        json.dump(c34b_diagnostics, f, indent=2)
    with open(f'{CKPT_DIR}/C3.4b_k_curve.json', 'w') as f:
        json.dump({str(k): v for k, v in k_curve_aucs.items()}, f, indent=2)