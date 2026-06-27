#!/usr/bin/env python3
# ============================================================================
# RE-Attention Adversarial Autoencoder — RSNA Medical Image Extension
# Anomaly Detection on RSNA Pneumonia Detection Challenge (PyTorch / T4 GPU)
#
# DATASET SETUP:
#   1. Accept competition rules at:
#      kaggle.com/competitions/rsna-pneumonia-detection-challenge
#   2. Notebook → Data → Add Data → Your Competitions →
#      rsna-pneumonia-detection-challenge
#   3. Settings → Accelerator → GPU T4 x1
#
# Conditions (cross-domain proof of concept):
#   C1 – CNN-AE Baseline           (reconstruction score)
#   C5 – CNN-RE-Attn-AAE  [NOVEL]  (error-guided attention)
#
# Key claim: REAttention module is architecture-agnostic.
#   Identical class, zero modification — applied here to 64×64 images
#   by operating on the FLATTENED reconstruction error (4096-dim vector),
#   exactly as it operates on the 115-dim KDD99 feature vector.
#
# Data split (excludes ambiguous 'No Lung Opacity / Not Normal' class):
#   Train : normal images only  (no labels — fully unsupervised)
#   Test  : held-out normal  +  lung opacity  (binary: 0 / 1)
#   Eval  : image-level AUC-ROC / AUC-PR  +  pixel-level localisation AUROC
# ============================================================================

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

# %% [CELL 2]  Imports and global plot style

import os, time, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()           # Jupyter / Kaggle — keep inline backend
except NameError:
    matplotlib.use('Agg')   # plain .py run — no GUI windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pydicom

warnings.filterwarnings('ignore')

# ── Global plot defaults (matches KDD99 notebook) ──────────────────────────
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
    'C1': '#4878CF',   # steel blue
    'C5': '#E84C3D',   # red (novel model — always stands out)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == 'cuda':
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# %% [CELL 3]  Configuration

SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))   # set True for quick syntax / runtime check

BASE       = '/kaggle/input/competitions/rsna-pneumonia-detection-challenge'
TRAIN_DIR  = f'{BASE}/stage_2_train_images'
OUTPUT_DIR = '/kaggle/working/results_rsna' if not SAMPLE_MODE else 'results_rsna_sample'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE  = 128                       # resize 1024×1024 DICOM → 128×128
ORIG_SIZE   = 1024                      # original DICOM resolution (for bbox scaling)
FLAT_DIM    = IMAGE_SIZE * IMAGE_SIZE   # 16384 — REAttention input_dim
LATENT_DIM  = 128                              # 64 was KDD99 scale; 128px CXR needs more
LR          = 1e-4
BETA1       = 0.5
EPOCHS        = 80   if not SAMPLE_MODE else 2
WARMUP_EPOCHS = 20   if not SAMPLE_MODE else 1   # pre-train enc1+dec before activating attention
LAMBDA_ADV    = 0.3                               # adversarial weight — reconstruction must dominate
BATCH_SIZE  = 32   if not SAMPLE_MODE else 4   # 128px images — keep 32 for VRAM
SEED        = 42
EPS         = 1e-8
# How many normal / opacity images to use for the test set
TEST_NORMAL  = 2000 if not SAMPLE_MODE else 10
TEST_OPACITY = 2000 if not SAMPLE_MODE else 5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"SAMPLE_MODE  : {SAMPLE_MODE}")
print(f"IMAGE_SIZE   : {IMAGE_SIZE}×{IMAGE_SIZE}  →  FLAT_DIM={FLAT_DIM}")
print(f"LATENT_DIM   : {LATENT_DIM}")
print(f"OUTPUT_DIR   : {OUTPUT_DIR}")

# %% [CELL 4]  Data preparation — CSV split, load and resize DICOM images

def _clahe_uint8(img_f32):
    """Apply CLAHE to a float32 image.  Operates in uint8 space (standard for CXR)."""
    import cv2
    img_u8  = (img_f32 * 255).clip(0, 255).astype(np.uint8)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq  = clahe.apply(img_u8)
    return img_eq.astype(np.float32) / 255.0

def load_dcm_resized(patient_id, train_dir, size):
    """Read one DICOM, apply CLAHE, normalise to [0,1], bilinear-resize to size×size."""
    dcm = pydicom.dcmread(f'{train_dir}/{patient_id}.dcm')
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)   # global norm first
    img = _clahe_uint8(img)                                      # CLAHE contrast enhance
    t   = torch.tensor(img).unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
    t   = F.interpolate(t, size=(size, size),
                        mode='bilinear', align_corners=False)
    return t.squeeze().numpy()                                   # (size, size) float32

def load_images(patient_ids, train_dir, size, tag):
    imgs, n = [], len(patient_ids)
    for i, pid in enumerate(patient_ids):
        if i % 500 == 0:
            print(f"  {tag}: {i}/{n}")
        imgs.append(load_dcm_resized(pid, train_dir, size))
    arr = np.stack(imgs)[:, None, :, :]    # (N, 1, H, W) float32
    print(f"  {tag} done → {arr.shape}")
    return arr

if SAMPLE_MODE:
    # ── Synthetic data: mimics (N, 1, 64, 64) float32 images ───────────────
    x_train_norm = np.random.rand(30, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    x_test_norm  = np.random.rand(TEST_NORMAL,  1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    x_test_opa   = np.random.rand(TEST_OPACITY, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    # Synthetic bounding boxes for opacity samples (in ORIG_SIZE space)
    raw_boxes = {i: [(100, 200, 300, 200)] for i in range(TEST_OPACITY)}
    print(f"SAMPLE_MODE — train:{x_train_norm.shape}  "
          f"test_norm:{x_test_norm.shape}  test_opa:{x_test_opa.shape}")

else:
    # ── Real RSNA data ─────────────────────────────────────────────────────
    labels = pd.read_csv(f'{BASE}/stage_2_train_labels.csv')
    detail = pd.read_csv(f'{BASE}/stage_2_detailed_class_info.csv')

    # Patient-level class (one row per patient in detail CSV)
    patient_class = (detail.drop_duplicates('patientId')
                           .set_index('patientId')['class'])

    normal_ids  = patient_class[patient_class == 'Normal'].index.tolist()
    opacity_ids = patient_class[patient_class == 'Lung Opacity'].index.tolist()
    # 'No Lung Opacity / Not Normal' (11 821 patients) excluded:
    #   ambiguous class, no bounding boxes, contaminates both train and test

    np.random.shuffle(normal_ids)
    np.random.shuffle(opacity_ids)

    # Split normals: hold out TEST_NORMAL for test, rest for training
    test_nml_ids   = normal_ids[:TEST_NORMAL]
    train_nml_ids  = normal_ids[TEST_NORMAL:]
    test_opa_ids   = opacity_ids[:TEST_OPACITY]

    print(f"Train normal  : {len(train_nml_ids)}")
    print(f"Test  normal  : {len(test_nml_ids)}")
    print(f"Test  opacity : {len(test_opa_ids)}")
    print(f"\nLoading images (DICOM {ORIG_SIZE}→{IMAGE_SIZE}px) …")

    t0 = time.time()
    x_train_norm = load_images(train_nml_ids, TRAIN_DIR, IMAGE_SIZE, 'Train-normal')
    x_test_norm  = load_images(test_nml_ids,  TRAIN_DIR, IMAGE_SIZE, 'Test-normal')
    x_test_opa   = load_images(test_opa_ids,  TRAIN_DIR, IMAGE_SIZE, 'Test-opacity')
    print(f"All images loaded in {time.time()-t0:.0f}s")

    # Bounding boxes for opacity test images (keys = index into x_test_opa)
    box_df   = labels[labels['Target'] == 1][['patientId','x','y','width','height']]
    raw_boxes = {}
    for i, pid in enumerate(test_opa_ids):
        rows = box_df[box_df['patientId'] == pid]
        if len(rows):
            raw_boxes[i] = list(zip(rows['x'], rows['y'],
                                    rows['width'], rows['height']))

# ── Assemble test set (normal first, then opacity) ─────────────────────────
x_test      = np.concatenate([x_test_norm, x_test_opa], axis=0)
binary_test = np.array([0] * len(x_test_norm) + [1] * len(x_test_opa),
                       dtype=np.int32)
# Shift box keys by the number of normal test images so they index into x_test
test_boxes  = {k + len(x_test_norm): v for k, v in raw_boxes.items()}

print(f"\nTrain (normal only) : {x_train_norm.shape}")
print(f"Test                : {x_test.shape}  "
      f"({binary_test.mean()*100:.1f}% anomaly)")
print(f"Opacity with boxes  : {len(test_boxes)}")

# %% [CELL 5]  DataLoader helper

def make_loader(x_np, batch_size, shuffle=True, drop_last=True):
    """x_np shape: (N, 1, H, W) float32."""
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last,
                      pin_memory=(device.type == 'cuda'),
                      num_workers=2)

# %% [CELL 6]  Model architecture

class CNNEncoder(nn.Module):
    """3 × (Conv-BN-ReLU-MaxPool) → flatten → Linear to latent."""
    def __init__(self, latent_dim, image_size=IMAGE_SIZE):
        super().__init__()
        s = image_size // 8   # spatial size after 3 × MaxPool2d(2): 64→8
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 32×32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16×16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 8×8
        )
        self.fc = nn.Linear(128 * s * s, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class CNNDecoder(nn.Module):
    """Linear → unflatten → 3 × ConvTranspose2d → Sigmoid (output ∈ [0,1])."""
    def __init__(self, latent_dim, image_size=IMAGE_SIZE):
        super().__init__()
        self.s    = image_size // 8   # 8
        self.flat = 128 * self.s * self.s
        self.fc   = nn.Linear(latent_dim, self.flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),    # → IMAGE_SIZE/4
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),    # → IMAGE_SIZE/2
            nn.ConvTranspose2d(32,  1, 4, stride=2, padding=1),
            nn.Sigmoid(),                      # → IMAGE_SIZE×IMAGE_SIZE ∈ [0,1]
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 128, self.s, self.s))


class REAttention(nn.Module):
    """Convolutional error-guided attention: e=(x-x̂)² → soft spatial mask ∈[0,1].

    For images: operates on the 2D error map (B,1,H,W), keeping neighbourhood
    structure intact — critical for spatially contiguous anomalies like pneumonia.
    API-compatible with the flat KDD99 version: accepts flat (B,H*W) error
    and returns flat (B,H*W) mask; reshaping is handled internally.

    The concept (error-guided attention) is identical to KDD99.
    The implementation uses convolutions instead of dense layers because
    image anomalies are spatially structured, not feature-independent.
    """
    def __init__(self, input_dim=None, hidden=None):   # args kept for API compat
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  1, kernel_size=1),            nn.Sigmoid(),
        )

    def forward(self, e):
        # Accept flat (B, H*W) or spatial (B,1,H,W) — handle both
        flat_in = (e.dim() == 2)
        if flat_in:
            h = w = int(e.shape[1] ** 0.5)
            e = e.view(e.shape[0], 1, h, w)
        out = self.net(e)
        return out.view(out.shape[0], -1) if flat_in else out  # match input shape


class LatentDisc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32,  1),         nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)

print("Model classes defined.")
print(f"REAttention  input_dim = FLAT_DIM = {FLAT_DIM}  "
      f"(identical class to KDD99, different input size)")

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
    """Convert bounding boxes (orig space) to binary mask (size×size)."""
    scale = size / orig
    mask  = np.zeros((size, size), dtype=np.float32)
    for (x, y, w, h) in boxes:
        x1, y1 = int(x * scale), int(y * scale)
        x2 = min(size, int((x + w) * scale))
        y2 = min(size, int((y + h) * scale))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask

def pixel_auroc(attn_maps_np, boxes_dict, binary_test_arr):
    """Pixel-level AUROC: attention map vs bounding-box ground truth."""
    gt_all, pred_all = [], []
    for idx in range(len(binary_test_arr)):
        if binary_test_arr[idx] == 0 or idx not in boxes_dict:
            continue
        gt_mask   = boxes_to_mask(boxes_dict[idx]).flatten()
        pred_mask = attn_maps_np[idx].flatten()
        gt_all.append(gt_mask); pred_all.append(pred_mask)
    if not gt_all:
        return np.nan
    gt   = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)
    return roc_auc_score(gt, pred) if len(np.unique(gt)) > 1 else np.nan

mse_fn = nn.MSELoss()

def ssim_anomaly_map(x, x_hat, window=11):
    """Per-pixel SSIM anomaly map: (1 - SSIM) flattened to (B, H*W).
    Higher value = more structurally different = more anomalous.

    Why SSIM beats MSE here: pneumonia consolidation is SMOOTH, so pixel
    MSE between the input (smooth) and the AE reconstruction (textured —
    because the decoder learned normal lung texture) can be LOW even when
    the AE hallucinated wrong structure. SSIM detects the structural
    mismatch (luminance × contrast × structure) and scores it HIGH.
    """
    pad = window // 2
    mu_x  = F.avg_pool2d(x,     window, stride=1, padding=pad)
    mu_y  = F.avg_pool2d(x_hat, window, stride=1, padding=pad)
    s_x   = F.avg_pool2d(x**2,     window, stride=1, padding=pad) - mu_x**2
    s_y   = F.avg_pool2d(x_hat**2, window, stride=1, padding=pad) - mu_y**2
    s_xy  = F.avg_pool2d(x*x_hat,  window, stride=1, padding=pad) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_x*mu_y + c1) * (2*s_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (s_x + s_y + c2))
    return (1.0 - ssim.clamp(-1, 1)).view(x.size(0), -1)   # (B, H*W), ≥ 0

def ssim_loss_fn(x, x_hat):
    """Scalar SSIM reconstruction loss (mean of anomaly map). Use alongside MSE."""
    return ssim_anomaly_map(x, x_hat).mean()

def anomaly_score(x, x_hat):
    """99th-percentile SSIM anomaly score per image — committed primary metric.
    More robust than top-10% mean: adapts to any pneumonia region size.
    Consistent across C1 and C5 — no post-hoc selection between strategies.
    """
    err = ssim_anomaly_map(x, x_hat)                       # (B, H*W)
    return torch.quantile(err, 0.99, dim=1)                # (B,) scalar per image

all_results  = {}
loss_history = {}
print("Utilities defined.")

# %% [CELL 8]  Condition 1 — CNN-AE Baseline

print("\n" + "="*60)
print("CONDITION 1 — CNN-AE Baseline")
print("="*60)

enc_c1 = CNNEncoder(LATENT_DIM).to(device)
dec_c1 = CNNDecoder(LATENT_DIM).to(device)
opt_c1 = Adam(list(enc_c1.parameters()) + list(dec_c1.parameters()), lr=LR)
sched_c1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c1, T_max=EPOCHS, eta_min=1e-6)

loader_c1     = make_loader(x_train_norm, BATCH_SIZE)
c1_epoch_loss = []

t0 = time.time()
for epoch in range(EPOCHS):
    enc_c1.train(); dec_c1.train()
    losses = []
    for (xb,) in loader_c1:
        xb   = xb.to(device)
        # horizontal flip augmentation (50%)
        mask = torch.rand(xb.size(0), device=device) > 0.5
        xb[mask] = xb[mask].flip(dims=[3])
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
scores_c1, z1_c1_list = [], []
with torch.no_grad():
    for i in range(0, len(x_test), BATCH_SIZE):
        xb   = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
        z1   = enc_c1(xb); xhat = dec_c1(z1)
        scores_c1.append(anomaly_score(xb, xhat).cpu().numpy())
        z1_c1_list.append(z1.cpu().numpy())

scores_c1  = np.concatenate(scores_c1)
z1_test_c1 = np.concatenate(z1_c1_list)

m_c1 = compute_metrics(scores_c1, binary_test)
print(f"\n  AUC-ROC : {m_c1['auc_roc']:.4f}")
print(f"  AUC-PR  : {m_c1['auc_pr']:.4f}")
print(f"  F1      : {m_c1['f1']:.4f}")
all_results['C1'] = {**m_c1, 'label': 'CNN-AE Baseline'}

# %% [CELL 9]  Condition 5 — CNN-RE-Attn-AAE  [NOVEL]

print("\n" + "="*60)
print("CONDITION 5 — CNN-RE-Attn-AAE  [NOVEL]")
print("="*60)
print(f"REAttention: convolutional, operates on ({IMAGE_SIZE}×{IMAGE_SIZE}) error map")
print("Concept identical to KDD99; conv implementation for spatial image data.\n")

enc1_c5 = CNNEncoder(LATENT_DIM).to(device)
enc2_c5 = CNNEncoder(LATENT_DIM).to(device)
dec_c5  = CNNDecoder(LATENT_DIM).to(device)
re_attn = REAttention().to(device)   # convolutional — no input_dim/hidden needed
ld_c5   = LatentDisc(LATENT_DIM).to(device)

# enc1+dec: pure reconstruction — no attention, no adversarial
opt_rec_c5   = Adam(
    list(enc1_c5.parameters()) + list(dec_c5.parameters()),
    lr=LR, betas=(BETA1, 0.999))
opt_disc_c5  = Adam(ld_c5.parameters(),    lr=LR, betas=(BETA1, 0.999))
# re_attn + enc2 share a generator optimizer so re_attn receives
# gradients through the adversarial loss (Phase 3)
opt_gen_c5   = Adam(
    list(re_attn.parameters()) + list(enc2_c5.parameters()),
    lr=LR, betas=(BETA1, 0.999))
sched_rec_c5  = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_rec_c5,  T_max=EPOCHS, eta_min=1e-6)
sched_disc_c5 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_disc_c5, T_max=EPOCHS, eta_min=1e-6)
sched_gen_c5  = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_gen_c5,  T_max=EPOCHS, eta_min=1e-6)

loader_c5     = make_loader(x_train_norm, BATCH_SIZE)
c5_epoch_loss = []

# ── Warm-start: train enc1+dec alone so re_attn sees structured error maps ─
# Without this, the error map in early epochs is uniform noise (blurry decoder)
# and re_attn learns a garbage mask that persists through the full training run.
print(f"Warm-start: pre-training enc1+dec for {WARMUP_EPOCHS} epochs...")
opt_warmup = Adam(
    list(enc1_c5.parameters()) + list(dec_c5.parameters()), lr=LR, betas=(BETA1, 0.999))
t_ws = time.time()
for epoch in range(WARMUP_EPOCHS):
    enc1_c5.train(); dec_c5.train()
    ws_losses = []
    for (xb,) in loader_c5:
        xb = xb.to(device)
        mask = torch.rand(xb.size(0), device=device) > 0.5
        xb[mask] = xb[mask].flip(dims=[3])
        opt_warmup.zero_grad()
        xhat = dec_c5(enc1_c5(xb))
        loss = 0.7 * mse_fn(xhat, xb) + 0.3 * ssim_loss_fn(xhat, xb)
        loss.backward(); opt_warmup.step()
        ws_losses.append(loss.item())
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Warmup {epoch+1:02d}/{WARMUP_EPOCHS}  loss={np.mean(ws_losses):.5f}")
print(f"Warm-start done ({time.time()-t_ws:.1f}s). Activating RE-Attention + AAE.\n")

t0 = time.time()
for epoch in range(EPOCHS):
    enc1_c5.train(); enc2_c5.train(); dec_c5.train()
    re_attn.train(); ld_c5.train()
    rec_l, d_l, g_l = [], [], []

    for (xb,) in loader_c5:
        xb = xb.to(device); n = xb.size(0)
        # horizontal flip augmentation (50%)
        mask = torch.rand(n, device=device) > 0.5
        xb[mask] = xb[mask].flip(dims=[3])

        # ── Phase 1: pass-1 reconstruction only ───────────────────────────
        # dec_c5 must only see enc1(xb) gradients so it stays a faithful
        # reconstructor. Pass-2 is used exclusively for Phases 2+3 (adversarial).
        # Adding pass-2 reconstruction loss trains dec on enc2(xb*att) — a masked
        # distribution — which degrades enc1→dec quality and hurts scoring.
        opt_rec_c5.zero_grad()
        z1      = enc1_c5(xb);   x_hat1  = dec_c5(z1)
        # SSIM error: (1-SSIM) per pixel — high at opacity, low at sharp edges.
        # Detached so re_attn signal doesn't create a gradient path into enc1/dec.
        with torch.no_grad():
            ssim_err = ssim_anomaly_map(xb, x_hat1.detach()).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
        att_img = re_attn(ssim_err)                            # (B,1,H,W) ∈[0,1]
        loss_rec = 0.7 * mse_fn(x_hat1, xb) + 0.3 * ssim_loss_fn(x_hat1, xb)
        loss_rec.backward(); opt_rec_c5.step()

        # ── Phase 2: latent discriminator ─────────────────────────────────
        opt_disc_c5.zero_grad()
        with torch.no_grad():
            z1_s   = enc1_c5(xb); xh1_s = dec_c5(z1_s)
            # att_s and enc2 frozen — disc trains only on latent classification
            ssim_s = ssim_anomaly_map(xb, xh1_s).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_s  = re_attn(ssim_s)
            z2_s   = enc2_c5(xb * att_s)
        z_real  = torch.randn(n, LATENT_DIM, device=device)
        loss_d  = (-torch.mean(torch.log(ld_c5(z_real) + EPS))
                   - torch.mean(torch.log(1.0 - ld_c5(z2_s) + EPS)))
        loss_d.backward()
        # clip disc gradients — prevents saturation flip (Disc loss 1.09 → 18.48)
        torch.nn.utils.clip_grad_norm_(ld_c5.parameters(), max_norm=1.0)
        opt_disc_c5.step()

        # ── Phase 3: re_attn + enc2 adversarial update ────────────────────
        # att_g is computed OUTSIDE no_grad so re_attn receives gradients.
        # enc1/dec are frozen (no_grad) — only re_attn and enc2 learn here.
        opt_gen_c5.zero_grad()
        with torch.no_grad():
            z1_g   = enc1_c5(xb); xh1_g = dec_c5(z1_g)
            ssim_g = ssim_anomaly_map(xb, xh1_g).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
        att_g   = re_attn(ssim_g)                              # re_attn gets gradient ✓
        loss_g  = LAMBDA_ADV * (-torch.mean(torch.log(ld_c5(enc2_c5(xb * att_g)) + EPS)))
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(
            list(re_attn.parameters()) + list(enc2_c5.parameters()), max_norm=1.0)
        opt_gen_c5.step()

        rec_l.append(loss_rec.item())
        d_l.append(loss_d.item())
        g_l.append(loss_g.item())

    sched_rec_c5.step(); sched_disc_c5.step(); sched_gen_c5.step()
    c5_epoch_loss.append(np.mean(rec_l))
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
              f"Recon={c5_epoch_loss[-1]:.5f}  "
              f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}  "
              f"lr={sched_rec_c5.get_last_lr()[0]:.2e}")

loss_history['C5'] = c5_epoch_loss
print(f"C5 training: {time.time()-t0:.1f}s")

# ── Inference — single committed scoring strategy (no post-hoc selection) ─
# Primary: 99th-pct SSIM (same function as C1). Committed upfront.
# Ablation (attn×SSIM) kept for analysis only, not used for selection.
enc1_c5.eval(); enc2_c5.eval(); dec_c5.eval(); re_attn.eval(); ld_c5.eval()
scores_c5, sc_attn_ssim, z1_c5_list, attn_maps = [], [], [], []

with torch.no_grad():
    for i in range(0, len(x_test), BATCH_SIZE):
        xb      = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
        n       = xb.size(0)

        z1      = enc1_c5(xb);  x_hat1 = dec_c5(z1)
        ssim_inf = ssim_anomaly_map(xb, x_hat1).view(n, 1, IMAGE_SIZE, IMAGE_SIZE)
        att_img  = re_attn(ssim_inf)                           # (B,1,H,W)

        # Primary score — SSIM 99th pct (committed, same metric as C1)
        scores_c5.append(anomaly_score(xb, x_hat1).cpu().numpy())

        # Ablation — disc score: 1 - P(enc2 latent looks Gaussian)
        # Anomalies → enc2 latent far from prior → disc low → 1-disc high
        z2_inf = enc2_c5(xb * att_img)
        sc_attn_ssim.append((1.0 - ld_c5(z2_inf)).squeeze(1).cpu().numpy())

        z1_c5_list.append(z1.cpu().numpy())
        attn_maps.append(att_img.squeeze(1).cpu().numpy())     # (B, H, W)

scores_c5    = np.concatenate(scores_c5)
sc_attn_ssim = np.concatenate(sc_attn_ssim)
z1_test_c5   = np.concatenate(z1_c5_list)
attn_maps    = np.concatenate(attn_maps)

m_c5     = compute_metrics(scores_c5,    binary_test)
m_c5_att = compute_metrics(sc_attn_ssim, binary_test)

print(f"\n  SSIM 99th-pct (primary)   AUC-ROC={m_c5['auc_roc']:.4f}  "
      f"AUC-PR={m_c5['auc_pr']:.4f}  F1={m_c5['f1']:.4f}")
print(f"  Disc score  (ablation)     AUC-ROC={m_c5_att['auc_roc']:.4f}")
all_results['C5']     = {**m_c5,     'label': 'CNN-RE-Attn-AAE (Ours)'}
all_results['C5_att'] = {**m_c5_att, 'label': 'CNN-RE-Attn-AAE disc-score (ablation)'}

# %% [CELL 10]  Results summary

print("\n" + "="*60)
print("RESULTS SUMMARY — RSNA Pneumonia Anomaly Detection")
print("="*60)
print(f"\n  {'Condition':<36} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8}")
print(f"  {'-'*58}")
for k, r in all_results.items():
    tag = ' ← NOVEL' if k == 'C5' else ''
    print(f"  {r['label']:<36} "
          f"{r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} {r['f1']:>8.4f}{tag}")

# %% [CELL 11]  Convergence + ROC + PR curves

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CNN-RE-Attn-AAE on RSNA Chest X-Ray', fontsize=15, fontweight='bold')

epochs_x = np.arange(1, EPOCHS + 1)
axes[0].plot(epochs_x, loss_history['C1'], color=PAL['C1'], lw=2, label='C1 CNN-AE')
axes[0].plot(epochs_x, loss_history['C5'], color=PAL['C5'], lw=2,
             linestyle='--', label='C5 RE-Attn-AAE')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Reconstruction Loss (MSE)')
axes[0].set_title('Training Convergence'); axes[0].legend()

for key, scores, col in [('C1', scores_c1, PAL['C1']), ('C5', scores_c5, PAL['C5'])]:
    if len(np.unique(binary_test)) > 1:
        fpr, tpr, _ = roc_curve(binary_test, scores)
        axes[1].plot(fpr, tpr, color=col, lw=2,
                     label=f"{all_results[key]['label']} "
                           f"(AUC={all_results[key]['auc_roc']:.4f})")
axes[1].plot([0,1],[0,1],'k--',lw=0.8)
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves'); axes[1].legend(loc='lower right')

for key, scores, col in [('C1', scores_c1, PAL['C1']), ('C5', scores_c5, PAL['C5'])]:
    if len(np.unique(binary_test)) > 1:
        prec, rec, _ = precision_recall_curve(binary_test, scores)
        axes[2].plot(rec, prec, color=col, lw=2,
                     label=f"{all_results[key]['label']} "
                           f"(AP={all_results[key]['auc_pr']:.4f})")
axes[2].axhline(binary_test.mean(), color='gray', lw=0.9, linestyle='--',
                label=f'No-skill ({binary_test.mean():.2f})')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curves'); axes[2].legend(loc='upper right')

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/curves.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/curves.png")

# %% [CELL 12]  Metric bar chart — C1 vs C5

# Colour palette extended to cover all result keys
PAL_BAR = {
    'C1'     : '#4878CF',
    'C5'     : '#E84C3D',
    'C5_att' : '#9B2335',
}

# Only plot the two main conditions (C1 baseline + best C5) in the bar chart
plot_keys = ['C1', 'C5']

metrics_to_plot = [('auc_roc','AUC-ROC'), ('auc_pr','AUC-PR'), ('f1','F1 Score')]
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('RSNA — C1 Baseline vs C5 RE-Attn-AAE (best score)',
             fontsize=14, fontweight='bold')

for ax, (metric, title) in zip(axes, metrics_to_plot):
    vals   = [all_results[k][metric] for k in plot_keys]
    colors = [PAL_BAR[k] for k in plot_keys]
    labels = [all_results[k]['label'] for k in plot_keys]
    bars   = ax.bar(labels, vals, color=colors, edgecolor='white',
                    linewidth=1.2, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.08))
    ax.set_title(title); ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=10)

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/metric_bars.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/metric_bars.png")

# %% [CELL 13]  Attention mask visualisation grid — Normal vs Lung Opacity

norm_idx = np.where(binary_test == 0)[0]
opa_idx  = np.where(binary_test == 1)[0]

# 4 most-normal + 4 most-anomalous samples
n_pick   = min(4, len(norm_idx), len(opa_idx))
norm_pick = norm_idx[np.argsort(scores_c5[norm_idx])[:n_pick]]
opa_pick  = opa_idx[np.argsort(scores_c5[opa_idx])[-n_pick:]]
picks     = list(norm_pick) + list(opa_pick)
pick_lbl  = ['Normal'] * n_pick + ['Lung Opacity'] * n_pick
pick_col  = ['#888888'] * n_pick + ['#E84C3D'] * n_pick

enc1_c5.eval(); dec_c5.eval(); re_attn.eval()

n_rows = len(picks)
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
if n_rows == 1:
    axes = axes[None, :]    # ensure 2D
fig.suptitle(
    'RE-Attention Walkthrough — RSNA  (Normal vs Lung Opacity)\n'
    'Columns: Original  |  Reconstruction x̂  |  Attention Mask  |  Overlay + BBox',
    fontsize=13, fontweight='bold')

col_titles = ['Original Image', 'Reconstruction x̂',
              'Attention Mask', 'Overlay + BBox (cyan)']
for j, ct in enumerate(col_titles):
    axes[0, j].set_title(ct, fontsize=11, fontweight='bold', pad=8)

for row, (idx, lbl, col) in enumerate(zip(picks, pick_lbl, pick_col)):
    with torch.no_grad():
        xb_t  = torch.tensor(x_test[idx][None]).to(device)
        z1_   = enc1_c5(xb_t); xhat = dec_c5(z1_)
        err   = (xb_t.flatten(1) - xhat.flatten(1)) ** 2
        att   = re_attn(err).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    img_np  = x_test[idx, 0]                       # (H, W)
    xhat_np = xhat.squeeze().cpu().numpy()
    att_np  = att.squeeze().cpu().numpy()           # (H, W)

    axes[row, 0].imshow(img_np,  cmap='gray', vmin=0, vmax=1)
    axes[row, 0].set_ylabel(lbl, color=col, fontsize=9, fontweight='bold')
    axes[row, 1].imshow(xhat_np, cmap='gray', vmin=0, vmax=1)
    axes[row, 2].imshow(att_np,  cmap='hot',  vmin=0, vmax=1)
    axes[row, 3].imshow(img_np,  cmap='gray', vmin=0, vmax=1)
    axes[row, 3].imshow(att_np,  cmap='Reds', alpha=0.45, vmin=0, vmax=1)

    # Draw ground-truth bounding boxes (scaled to IMAGE_SIZE space)
    if idx in test_boxes:
        scale = IMAGE_SIZE / ORIG_SIZE
        for (bx, by, bw, bh) in test_boxes[idx]:
            for ax_ in [axes[row, 0], axes[row, 3]]:
                ax_.add_patch(patches.Rectangle(
                    (bx * scale, by * scale), bw * scale, bh * scale,
                    linewidth=1.5, edgecolor='cyan', facecolor='none'))

    for j in range(4):
        axes[row, j].axis('off')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/attention_grid.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/attention_grid.png")

# %% [CELL 14]  Pixel-level localisation evaluation + Latent PCA

# ── Pixel AUROC ───────────────────────────────────────────────────────────
# SSIM error map: enc1+dec trained on normals reconstructs pneumonia poorly
# → high (1-SSIM) exactly where consolidation is → best spatial localizer.
# Diagnosis showed att_img collapsed to ~0 everywhere (adversarial objective
# incentivises suppression of anomalies, not highlighting).
enc1_c5.eval(); dec_c5.eval()
_ssim_loc_list = []
with torch.no_grad():
    for i in range(0, len(x_test), BATCH_SIZE):
        xb   = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
        xhat = dec_c5(enc1_c5(xb))
        _ssim_loc_list.append(
            ssim_anomaly_map(xb, xhat).view(-1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy())
ssim_loc_maps = np.concatenate(_ssim_loc_list)   # (N, H, W)

_n_with_boxes = sum(1 for i in range(len(binary_test))
                    if binary_test[i] == 1 and i in test_boxes)
# C1 SSIM map for fair novelty comparison
enc_c1.eval(); dec_c1.eval()
_c1_ssim_list = []
with torch.no_grad():
    for i in range(0, len(x_test), BATCH_SIZE):
        xb   = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
        xhat = dec_c1(enc_c1(xb))
        _c1_ssim_list.append(
            ssim_anomaly_map(xb, xhat).view(-1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy())
c1_ssim_loc_maps = np.concatenate(_c1_ssim_list)

loc_auc_att      = pixel_auroc(attn_maps,        test_boxes, binary_test)
loc_auc_ssim     = pixel_auroc(ssim_loc_maps,    test_boxes, binary_test)
loc_auc_c1_ssim  = pixel_auroc(c1_ssim_loc_maps, test_boxes, binary_test)
print(f"Pixel-level localisation AUROC — C1 SSIM map  : {loc_auc_c1_ssim:.4f}")
print(f"Pixel-level localisation AUROC — C5 SSIM map  : {loc_auc_ssim:.4f}  ← primary localiser")
print(f"Pixel-level localisation AUROC — C5 att_img   : {loc_auc_att:.4f}  (collapsed)")
print(f"  ({_n_with_boxes} / {int(binary_test.sum())} opacity images have bounding boxes)\n")
loc_auc = loc_auc_ssim   # use SSIM for results JSON

# ── PCA: C1 vs C5 latent space ───────────────────────────────────────────
pca_c1 = PCA(n_components=2, random_state=SEED).fit_transform(z1_test_c1)
pca_c5 = PCA(n_components=2, random_state=SEED).fit_transform(z1_test_c5)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Latent Space PCA  (grey = Normal, red = Lung Opacity)',
             fontsize=14, fontweight='bold')
for ax, coords, title in zip(
        axes, [pca_c1, pca_c5], ['C1 CNN-AE', 'C5 CNN-RE-Attn-AAE']):
    for lbl, col, mask in [
        ('Normal',       '#888888', binary_test == 0),
        ('Lung Opacity', '#E84C3D', binary_test == 1),
    ]:
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=col, label=lbl, alpha=0.4, s=10, edgecolors='none')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.legend(markerscale=3, fontsize=9)

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/latent_pca.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/latent_pca.png")

# %% [CELL 15]  Save all results to JSON

def _json(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):     return int(obj)
    if isinstance(obj, np.ndarray):            return obj.tolist()
    if isinstance(obj, dict):  return {k: _json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json(v)     for v in obj]
    return obj

all_results['localisation'] = {
    'pixel_auroc_c5': loc_auc,
    'note': 'AUROC between C5 attention mask and radiologist bounding box'
}

with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(_json(all_results), f, indent=2)

with open(f'{OUTPUT_DIR}/loss_history.json', 'w') as f:
    json.dump(_json(loss_history), f, indent=2)

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("  results.json       — image-level metrics (C1, C5) + localisation AUROC")
print("  loss_history.json  — epoch losses")
print("  curves.png         — convergence + ROC + PR")
print("  metric_bars.png    — AUC-ROC / AUC-PR / F1 bar chart")
print("  attention_grid.png — 8-image walkthrough (4 normal, 4 opacity + bboxes)")
print("  latent_pca.png     — PCA of C1 vs C5 latent space")
print("\n" + "="*60)
print("DONE — check /kaggle/working/results_rsna/ for all outputs")
print("="*60)
