r"""Build a 128px MedIAnomaly dataset for the resolution test, from the full-res source.

Why this exists: MedIAnomaly-Data ships at 512x512 (RSNA, VinDr-CXR) and 500x500 (LAG),
which is 1.4 GB -- slow to upload to Kaggle when the experiment only ever looks at 128px.
MedIAnomaly-Data-64 cannot be used: dl_res128.py resizes whatever it finds UP to 128,
which would measure bilinear interpolation instead of resolution.

This pre-applies exactly the resize the loader would do:

    Image.open(f).convert('L').resize((128, 128), Image.BILINEAR)

The loader then re-resizes 128 -> 128, which is the identity, so the arrays it produces
are bit-identical to reading the 512px source directly. That equivalence is asserted per
dataset below on a random sample, not assumed.

Two details worth knowing:
  * Images are written as PNG (lossless) under their ORIGINAL filenames, including LAG's
    ".jpg" names. PIL detects format from content, not extension, so data.json -- the
    benchmark's own split file -- stays valid byte for byte. Re-encoding LAG as JPEG
    would have added compression artifacts on top of the resize.
  * Only files named in data.json are copied. Anything else in images/ is not part of
    the benchmark's split and has no business in the bundle.

    python3 prepare_data_128.py                    # build + verify
    python3 prepare_data_128.py --zip              # also write the Kaggle zip
    python3 prepare_data_128.py --size 256         # any target resolution
"""
import argparse, json, os, random, shutil, sys, zipfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.abspath(os.path.join(HERE, "../../../MedIAnomaly-Data"))
DATASETS = ("RSNA", "VinCXR", "LAG")
VERIFY_N = 25


def image_names(dd):
    """Every filename the split references, train and test, in a stable order."""
    names = list(dd["train"]["0"])
    for k in sorted(dd["test"]):
        names += list(dd["test"][k])
    return names


def build(src, dst, size, verify_n=VERIFY_N):
    os.makedirs(dst, exist_ok=True)
    summary = []
    for ds in DATASETS:
        s_dir = os.path.join(src, ds)
        if not os.path.isdir(s_dir):
            print(f"  {ds}: MISSING under {src} -- skipped")
            continue
        d_dir = os.path.join(dst, ds)
        os.makedirs(os.path.join(d_dir, "images"), exist_ok=True)

        with open(os.path.join(s_dir, "data.json")) as f:
            dd = json.load(f)
        shutil.copy(os.path.join(s_dir, "data.json"), os.path.join(d_dir, "data.json"))

        names = image_names(dd)
        src_sizes = set()
        for i, nm in enumerate(names):
            if i % 1000 == 0:
                print(f"  {ds}: {i}/{len(names)}", flush=True)
            sp = os.path.join(s_dir, "images", nm)
            im = Image.open(sp)
            src_sizes.add(im.size)
            if min(im.size) < size:
                raise SystemExit(
                    f"{ds}/{nm} is {im.size}, smaller than the {size}px target. "
                    f"Refusing to upsample -- point --src at the full-resolution data.")
            im.convert("L").resize((size, size), Image.BILINEAR).save(
                os.path.join(d_dir, "images", nm), format="PNG", optimize=True)

        # The whole point: reading the downsampled copy must equal reading the source.
        rng = random.Random(0)
        for nm in rng.sample(names, min(verify_n, len(names))):
            a = np.asarray(Image.open(os.path.join(s_dir, "images", nm))
                           .convert("L").resize((size, size), Image.BILINEAR))
            b = np.asarray(Image.open(os.path.join(d_dir, "images", nm))
                           .convert("L").resize((size, size), Image.BILINEAR))
            if not np.array_equal(a, b):
                raise SystemExit(f"{ds}/{nm}: downsampled copy differs from the source")

        n_tr = len(dd["train"]["0"])
        n_te = sum(len(v) for v in dd["test"].values())
        summary.append((ds, len(names), n_tr, n_te, sorted(src_sizes)[0]))
        print(f"  {ds}: {len(names)} images  train {n_tr} + test {n_te}  "
              f"{sorted(src_sizes)[0][0]}px -> {size}px  "
              f"[{min(verify_n, len(names))} verified identical]")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.abspath(os.path.join(a.src, f"../MedIAnomaly-Data-{a.size}"))

    if not os.path.isdir(a.src):
        raise SystemExit(f"source not found: {a.src}")
    print(f"source : {a.src}")
    print(f"target : {out}   ({a.size}px)\n")

    summary = build(a.src, out, a.size)
    if not summary:
        raise SystemExit("nothing built")

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    print(f"\nbuilt {sum(s[1] for s in summary)} images, {total/1e6:.0f} MB")

    if a.zip:
        zp = out + ".zip"
        print(f"zipping -> {zp}")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for r, _, fs in os.walk(out):
                for f in sorted(fs):
                    p = os.path.join(r, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(out)))
        print(f"  {os.path.getsize(zp)/1e6:.0f} MB")
        print(f"\nUpload {os.path.basename(zp)} as a Kaggle Dataset, attach it to the\n"
              f"notebook, and it will be found automatically. The archive expands to\n"
              f"MedIAnomaly-Data-{a.size}/<dataset>/{{images/,data.json}}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
