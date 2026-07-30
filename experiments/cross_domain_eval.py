# %% [markdown]
# # Cross-domain evaluation: RSNA-trained models -> NIH ChestX-ray14 / CheXpert
#
# Loads encoder/decoder weights already trained on RSNA (conditions C1, C3, C4 from
# `re_attn_aae_kaggle-RSNA-ResNet.py`) and scores them, **inference only, no retraining**,
# on a stratified subset of a second chest X-ray dataset. Answers: does the anomaly
# detector generalise to a different hospital/scanner/label-source, or was the RSNA
# AUC-ROC specific to that dataset's acquisition characteristics.
#
# Preprocessing mirrors the RSNA script exactly (min-max normalise -> CLAHE -> resize to
# IMAGE_SIZE) so any AUC drop reflects domain shift in the images/labels, not a pipeline
# mismatch. Model architectures (CNNEncoder/CNNDecoder/REAttention/LatentDisc) are copied
# verbatim from `re_attn_aae_kaggle-RSNA-ResNet.py` -- keep them in sync if that file changes.
#
# Default target dataset is NIH ChestX-ray14 (openly downloadable, no credentialing wait).
# CheXpert is supported via --dataset chexpert if/when institutional access is granted --
# see build_chexpert_subset() for the access-agreement caveat.
#
# NOT executed in this environment (no GPU, no NIH/CheXpert data, no restored RSNA
# checkpoints locally -- only C6/ResNet weights exist in this repo). Run this on the
# same machine/notebook where the RSNA C1/C3/C4 checkpoints live.

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score

IMAGE_SIZE = 128
LATENT_DIM = 128
SEED = 42

# ────────────────────────────────────────────────────────────────────────────
# Preprocessing -- must match load_dcm_resized/_clahe_uint8 in the RSNA script
# ────────────────────────────────────────────────────────────────────────────

def _clahe_uint8(img_f32):
    img_u8 = (img_f32 * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_u8).astype(np.float32) / 255.0


def load_png_resized(path, size=IMAGE_SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = _clahe_uint8(img)
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def load_image_set(paths, size, tag):
    imgs = []
    for i, p in enumerate(paths):
        if i % 500 == 0:
            print(f"  {tag}: {i}/{len(paths)}")
        imgs.append(load_png_resized(p, size))
    arr = np.stack(imgs)[:, None, :, :]
    print(f"  {tag} done -> {arr.shape}")
    return arr


# ────────────────────────────────────────────────────────────────────────────
# Dataset-specific subset builders
# ────────────────────────────────────────────────────────────────────────────

def build_nih_subset(data_dir, csv_path, n_per_class, seed=SEED,
                      positive_labels=('Infiltration', 'Consolidation')):
    """Stratified single-label subset of NIH ChestX-ray14.

    NIH-14 is entirely frontal (PA/AP), so no view filtering is needed (unlike CheXpert).
    Restricting to single-label rows (no '|' in Finding Labels) avoids ambiguous
    co-occurring pathologies contaminating the anomaly class.
    """
    df = pd.read_csv(csv_path)
    single_label = ~df['Finding Labels'].str.contains(r'\|', regex=True)
    normal_df = df[single_label & (df['Finding Labels'] == 'No Finding')]
    positive_mask = single_label & df['Finding Labels'].isin(positive_labels)
    positive_df = df[positive_mask]

    rng = np.random.default_rng(seed)
    normal_ids = rng.choice(normal_df['Image Index'].values,
                             size=min(n_per_class, len(normal_df)), replace=False)
    positive_ids = rng.choice(positive_df['Image Index'].values,
                               size=min(n_per_class, len(positive_df)), replace=False)

    print(f"NIH pool: {len(normal_df)} No-Finding, {len(positive_df)} "
          f"{'/'.join(positive_labels)} (single-label) -> sampling {len(normal_ids)}/{len(positive_ids)}")

    def _resolve(image_ids):
        # NIH images ship split across images_001..images_012/images/ subfolders.
        found = []
        for name in image_ids:
            matches = [os.path.join(root, name)
                       for root in _iter_image_subdirs(data_dir)
                       if os.path.exists(os.path.join(root, name))]
            if matches:
                found.append(matches[0])
        return found

    normal_paths = _resolve(normal_ids)
    positive_paths = _resolve(positive_ids)
    return normal_paths, positive_paths


def _iter_image_subdirs(data_dir):
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry, 'images')
        if os.path.isdir(sub):
            yield sub
    # also allow a flat layout (all pngs directly under data_dir)
    yield data_dir


def build_chexpert_subset(data_dir, csv_path, n_per_class, seed=SEED,
                           positive_labels=('Consolidation', 'Pneumonia')):
    """Stratified frontal-view subset of CheXpert.

    CAVEAT: CheXpert requires a Stanford research-use agreement (not pip/kaggle
    installable like RSNA/NIH) -- budget lead time for approval before this path
    is usable. Labels are NLP-mined from reports (1=positive, 0=negative, -1=uncertain);
    we keep only confident positives/negatives to avoid inheriting that noise.
    """
    df = pd.read_csv(csv_path)
    frontal = df['Frontal/Lateral'] == 'Frontal'
    normal_mask = frontal & (df['No Finding'] == 1)
    positive_mask = frontal & (df[list(positive_labels)] == 1).any(axis=1)

    rng = np.random.default_rng(seed)
    normal_df = df[normal_mask]
    positive_df = df[positive_mask]
    normal_paths = rng.choice(normal_df['Path'].values,
                               size=min(n_per_class, len(normal_df)), replace=False)
    positive_paths = rng.choice(positive_df['Path'].values,
                                 size=min(n_per_class, len(positive_df)), replace=False)

    print(f"CheXpert pool: {len(normal_df)} No-Finding, {len(positive_df)} "
          f"{'/'.join(positive_labels)} -> sampling {len(normal_paths)}/{len(positive_paths)}")

    root = os.path.dirname(data_dir.rstrip('/'))
    normal_full = [os.path.join(root, p) for p in normal_paths]
    positive_full = [os.path.join(root, p) for p in positive_paths]
    return normal_full, positive_full


# ────────────────────────────────────────────────────────────────────────────
# Model classes -- copied verbatim from re_attn_aae_kaggle-RSNA-ResNet.py
# ────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
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
    def __init__(self, latent_dim, image_size=IMAGE_SIZE):
        super().__init__()
        self.s = image_size // 8
        self.flat = 128 * self.s * self.s
        self.fc = nn.Linear(latent_dim, self.flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 128, self.s, self.s))


class REAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, e):
        return self.net(e)


class LatentDisc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


def ssim_anomaly_map(x, x_hat, window=11):
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(x_hat, window, stride=1, padding=pad)
    s_x = F.avg_pool2d(x ** 2, window, stride=1, padding=pad) - mu_x ** 2
    s_y = F.avg_pool2d(x_hat ** 2, window, stride=1, padding=pad) - mu_y ** 2
    s_xy = F.avg_pool2d(x * x_hat, window, stride=1, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * s_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (s_x + s_y + c2))
    return (1 - ssim_map).flatten(1)


def anomaly_score(x, x_hat):
    return torch.quantile(ssim_anomaly_map(x, x_hat), 0.99, dim=1)


def normalise_scores(s):
    s_min, s_max = s.min(), s.max()
    return (s - s_min) / (s_max - s_min + 1e-8)


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap CI / significance -- same as compute_metrics additions in the RSNA/KDD scripts
# ────────────────────────────────────────────────────────────────────────────

def bootstrap_auc(scores, binary_labels, n_boot=1000, seed=SEED):
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


def compute_metrics(scores, binary_labels, n_boot=1000, seed=SEED):
    if len(np.unique(binary_labels)) < 2:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'f1': np.nan,
                'auc_std': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    auc_roc = roc_auc_score(binary_labels, scores)
    auc_pr = average_precision_score(binary_labels, scores)
    fpr, tpr, thresh = roc_curve(binary_labels, scores)
    best = np.argmax(tpr - fpr)
    pred = (scores >= thresh[best]).astype(int)
    boot = bootstrap_auc(scores, binary_labels, n_boot=n_boot, seed=seed)
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'f1': f1_score(binary_labels, pred, zero_division=0),
            'auc_std': boot['auc_std'], 'ci_lo': boot['ci_lo'], 'ci_hi': boot['ci_hi']}


# ────────────────────────────────────────────────────────────────────────────
# Condition loading + scoring -- C1 (CNN-AE) and C4 (CNN-RE-Attn-AAE) are the
# headline scratch-CNN comparison; C3 (ablation, no attention) included for the
# same C3->C4 delta the RSNA in-domain report leads with.
# ────────────────────────────────────────────────────────────────────────────

def load_condition(cond, ckpt_dir, device):
    enc1 = CNNEncoder(LATENT_DIM).to(device)
    dec = CNNDecoder(LATENT_DIM).to(device)
    enc1.load_state_dict(torch.load(f'{ckpt_dir}/{cond}_enc1.pth', map_location=device))
    dec.load_state_dict(torch.load(f'{ckpt_dir}/{cond}_dec.pth', map_location=device))
    enc1.eval(); dec.eval()

    modules = {'enc1': enc1, 'dec': dec}
    if cond in ('C3', 'C4'):
        ld = LatentDisc(LATENT_DIM).to(device)
        ld.load_state_dict(torch.load(f'{ckpt_dir}/{cond}_disc.pth', map_location=device))
        ld.eval()
        modules['disc'] = ld
    if cond == 'C4':
        enc2 = CNNEncoder(LATENT_DIM).to(device)
        re_attn = REAttention().to(device)
        enc2.load_state_dict(torch.load(f'{ckpt_dir}/{cond}_enc2.pth', map_location=device))
        re_attn.load_state_dict(torch.load(f'{ckpt_dir}/{cond}_re_attn.pth', map_location=device))
        enc2.eval(); re_attn.eval()
        modules['enc2'] = enc2
        modules['re_attn'] = re_attn
    return modules


@torch.no_grad()
def score_condition(cond, modules, x_test, device, batch_size=32):
    scores = []
    for i in range(0, len(x_test), batch_size):
        xb = torch.tensor(x_test[i:i + batch_size], dtype=torch.float32).to(device)
        z1 = modules['enc1'](xb)
        x_hat1 = modules['dec'](z1)
        if cond == 'C4':
            ssim_inf = ssim_anomaly_map(xb, x_hat1).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            att_img = modules['re_attn'](ssim_inf)
            scores.append(anomaly_score(xb, x_hat1).cpu().numpy())
        else:
            scores.append(anomaly_score(xb, x_hat1).cpu().numpy())
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['nih', 'chexpert'], default='nih')
    ap.add_argument('--data-dir', required=True, help='image root directory')
    ap.add_argument('--csv', required=True, help='Data_Entry_2017.csv or CheXpert train.csv')
    ap.add_argument('--ckpt-dir', required=True, help='directory with RSNA C1/C3/C4 *.pth weights')
    ap.add_argument('--conditions', default='C1,C4', help='comma-separated, subset of C1,C3,C4')
    ap.add_argument('--n-per-class', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--output', default='cross_domain_results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conditions = args.conditions.split(',')

    if args.dataset == 'nih':
        normal_paths, positive_paths = build_nih_subset(
            args.data_dir, args.csv, args.n_per_class, seed=args.seed)
    else:
        normal_paths, positive_paths = build_chexpert_subset(
            args.data_dir, args.csv, args.n_per_class, seed=args.seed)

    print(f"\nLoading + preprocessing {len(normal_paths)} normal + "
          f"{len(positive_paths)} anomaly images ...")
    x_normal = load_image_set(normal_paths, IMAGE_SIZE, 'normal')
    x_positive = load_image_set(positive_paths, IMAGE_SIZE, 'anomaly')
    x_test = np.concatenate([x_normal, x_positive], axis=0)
    binary_test = np.array([0] * len(x_normal) + [1] * len(x_positive), dtype=np.int32)

    results = {}
    for cond in conditions:
        print(f"\nScoring {cond} on {args.dataset} ...")
        modules = load_condition(cond, args.ckpt_dir, device)
        scores = score_condition(cond, modules, x_test, device)
        m = compute_metrics(scores, binary_test)
        results[cond] = m
        print(f"  {cond}  AUC-ROC={m['auc_roc']:.4f}  95% CI=[{m['ci_lo']:.4f},{m['ci_hi']:.4f}]  "
              f"AUC-PR={m['auc_pr']:.4f}  F1={m['f1']:.4f}")

    results['_meta'] = {
        'dataset': args.dataset,
        'n_normal': len(x_normal),
        'n_anomaly': len(x_positive),
        'image_size': IMAGE_SIZE,
        'seed': args.seed,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()
