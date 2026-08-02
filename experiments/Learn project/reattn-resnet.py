# %% [markdown]
# RE-Attn-AAE: A Reconstruction-Error-Guided Attention Adversarial Autoencoder for Dual-Domain Unsupervised Anomaly Detection
# ## This is  a PyTorch 
# ## implementation of the RE-Attn-AAE model for unsupervised anomaly detection in dual-domain data. The model leverages reconstruction error to guide attention mechanisms, 
# ## enhancing the detection of anomalies in complex datasets.
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

# %% [CELL 1.6]  Wandb setup and login
# USE_WANDB is the flag every later cell should check before calling wandb.*
# — that's what makes wandb fully optional (see save_ckpt above).
USE_WANDB = False
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

# %% [CELL 1.7]  Sanity-test the checkpoint/wandb helpers with dummy data
# Uses a throwaway condition name ('CTEST') so it can never collide with C1-C7.
# Covers: is_done before/after, save with no optional args, save with all
# optional args, load_ckpt round-trip, load_weights hit + miss.
TEST_COND = 'CTEST'

# 1. Nothing saved yet -> should be False
print('is_done (before save):', is_done(TEST_COND))

# 2. Fake metrics/loss, as if a training loop had just finished this condition
all_results[TEST_COND]            = {'auc_roc': 0.91, 'auc_pr': 0.87, 'f1': 0.80}
all_results[f'{TEST_COND}_disc']  = {'auc_roc': 0.60, 'auc_pr': 0.55, 'f1': 0.50}
loss_history[TEST_COND] = [1.0, 0.8, 0.6, 0.5]

dummy_scores      = np.random.rand(20).astype(np.float32)
dummy_disc_scores = np.random.rand(20).astype(np.float32)
dummy_attn_maps   = np.random.rand(4, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
dummy_model       = nn.Linear(4, 4)     # stand-in for a real enc1/dec1 module

# 3. Case A: minimal call — no disc_scores, no attn_maps, no model weights
#    wandb_group='test-runs' tags this save separately from real ablation-v2 results.
save_ckpt(TEST_COND + '_min', result_keys=[TEST_COND], scores=dummy_scores,
          disc_scores=None, epoch_loss=loss_history[TEST_COND],
          wandb_group='test-runs')

# 4. Case B: full call — disc_scores + attn_maps + one model's weights
save_ckpt(TEST_COND, result_keys=[TEST_COND, f'{TEST_COND}_disc'], scores=dummy_scores,
          disc_scores=dummy_disc_scores, epoch_loss=loss_history[TEST_COND],
          attn_maps=dummy_attn_maps, enc1=dummy_model.state_dict(),
          wandb_group='test-runs')

# 5. is_done should flip to True now that the checkpoint file exists
print('is_done (after save): ', is_done(TEST_COND))

# 6. Simulate a fresh kernel session: wipe the in-memory dicts, reload from disk
all_results.pop(TEST_COND, None); all_results.pop(f'{TEST_COND}_disc', None)
loss_history.pop(TEST_COND, None)
scores_back, disc_back, attn_back = load_ckpt(TEST_COND)
print('scores match:', np.allclose(scores_back, dummy_scores))
print('disc match:  ', np.allclose(disc_back, dummy_disc_scores))
print('attn match:  ', np.allclose(attn_back, dummy_attn_maps))
print('reloaded all_results keys:', list(all_results.keys()))

# 7. load_weights: hit case (enc1 was saved) and miss case (never saved a name like this)
fresh_model = nn.Linear(4, 4)
load_weights(TEST_COND, enc1=fresh_model)
print('weights match:', torch.allclose(fresh_model.weight, dummy_model.weight))
load_weights(TEST_COND, decoder_never_saved=fresh_model)   # expect "weight file missing" print, no crash

# %% [CELL 1.8]  Delete everything CELL 1.7 wrote (local files + in-memory entries only)
import glob
for _cond in [TEST_COND, TEST_COND + '_min']:
    for _f in glob.glob(f'{CKPT_DIR}/{_cond}_*'):
        os.remove(_f)
        print('removed:', _f)
    all_results.pop(_cond, None)
    all_results.pop(f'{_cond}_disc', None)
    loss_history.pop(_cond, None)
print('Local CTEST checkpoint files removed; all_results/loss_history entries cleared.')

# %% [CELL 1.9]  (optional, manual) Delete the wandb-side 'test-runs' artifacts
# CELL 1.8 above only removes local files. If USE_WANDB was True, CELL 1.7 also
# logged artifacts tagged group='test-runs' (see save_ckpt's wandb_group param).
# This cell finds and deletes ALL artifact versions in that group across your
# whole wandb project — not just today's CTEST run — so only run it on purpose.
# It uses the read/write wandb.Api(), not wandb.log(), so it works even outside
# an active wandb.init() run.
RUN_WANDB_CLEANUP = False   # flip to True and re-run this cell to actually delete
if RUN_WANDB_CLEANUP:
    api = wandb.Api()
    for coll in api.artifact_collections(project_name=WANDB_PROJECT, type_name='checkpoint'):
        for art in coll.versions():
            if 'test-runs' in art.tags:
                print('deleting wandb artifact:', art.name)
                art.delete(delete_aliases=True)
    print('wandb "test-runs" cleanup done.')

# %% [CELL 1.10]  DEMO ONLY — how to give each experiment its own separate wandb
# Run, versioned/tagged so you can later fetch "every experiment" or "every
# version of one scenario". This is a sandbox: it opens throwaway runs under
# group='demo-runs' and never touches CELL 1.6's real ablation-v2 run or its
# metric history. It's a reference for how you'd structure real experiment
# runs later — nothing here is wired into save_ckpt or the real init.
if USE_WANDB:
    _demo_prev_run = wandb.run   # remember the real run so we can hand control back after

    def demo_start_run(cond, run_version):
        """One wandb Run per (condition, run_version) pair.
        group='demo-runs' + tags=[run_version, cond] is what makes both
        "fetch all experiments" and "fetch all versions of one scenario" possible."""
        return wandb.init(
            project=WANDB_PROJECT, group='demo-runs', name=f'demo-{cond}-{run_version}',
            id=f'demo-{cond}-{run_version}', resume='allow',
            tags=['demo-runs', run_version, cond],
            config={'condition': cond, 'run_version': run_version, 'demo': True},
            reinit=True, settings=wandb.Settings(init_timeout=120),
        )

    def demo_fetch_all():
        """Every demo run, any scenario, any version."""
        api = wandb.Api()
        return list(api.runs(f'{api.default_entity}/{WANDB_PROJECT}', filters={'tags': 'demo-runs'}))

    def demo_fetch_by_scenario(cond):
        """All versions of ONE scenario, e.g. every 'SIM_A' run regardless of run_version."""
        api = wandb.Api()
        return list(api.runs(f'{api.default_entity}/{WANDB_PROJECT}',
                              filters={'$and': [{'tags': 'demo-runs'}, {'tags': cond}]}))

    def demo_fetch_by_version(run_version):
        """Every scenario logged under ONE version, e.g. everything under 'demo-v2'."""
        api = wandb.Api()
        return list(api.runs(f'{api.default_entity}/{WANDB_PROJECT}',
                              filters={'$and': [{'tags': 'demo-runs'}, {'tags': run_version}]}))

    # Simulate 2 scenarios, each run under 2 different versions -> 4 separate runs total.
    for _cond in ['SIM_A', 'SIM_B']:
        for _ver in ['demo-v1', 'demo-v2']:
            _run = demo_start_run(_cond, _ver)
            wandb.log({'dummy_metric': float(np.random.rand())})
            _run.finish()

    print('all demo experiments:      ', [r.name for r in demo_fetch_all()])
    print('only SIM_A, any version:   ', [r.name for r in demo_fetch_by_scenario('SIM_A')])
    print('only demo-v2, any scenario:', [r.name for r in demo_fetch_by_version('demo-v2')])

    if _demo_prev_run is not None:
        # Hand control back to the real run so later real cells keep logging there.
        wandb.init(id=_demo_prev_run.id, project=WANDB_PROJECT, resume='must', reinit=True)
        print('handed control back to real run:', wandb.run.id)
else:
    print('USE_WANDB is False — log in first (CELL 1.6) to try this demo.')

# %% [CELL 1.11]  (optional, manual) delete the 4 demo runs CELL 1.10 created
RUN_DEMO_CLEANUP = False   # flip to True and re-run this cell to actually delete
if RUN_DEMO_CLEANUP and USE_WANDB:
    api = wandb.Api()
    for r in api.runs(f'{api.default_entity}/{WANDB_PROJECT}', filters={'tags': 'demo-runs'}):
        print('deleting run:', r.name)
        r.delete()
    print('demo runs deleted.')
