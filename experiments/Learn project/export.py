# %% [CELL 9]  Export a progressive Streamlit app data bundle (only finished conditions)
# Run any time — includes whatever conditions are currently is_done(), no more.
# Re-run again later as more conditions finish; the bundle just grows.
# Output: zip it and unzip into experiments/Learn project/app/data/ locally.
import shutil, zipfile

APP_BUNDLE_DIR = f'{OUTPUT_DIR}/app_data_bundle'
os.makedirs(f'{APP_BUNDLE_DIR}/scores',  exist_ok=True)
os.makedirs(f'{APP_BUNDLE_DIR}/weights', exist_ok=True)
os.makedirs(f'{APP_BUNDLE_DIR}/demo',    exist_ok=True)  # left empty — no per-sample recon/ssim export yet

ALL_CONDS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
finished  = [c for c in ALL_CONDS if is_done(c)]
print(f'Exporting finished conditions: {finished}')

bundle_metrics = {}
bundle_loss    = {}
for cond in finished:
    ensure_local(cond)
    scores, disc_sc, attn_maps = load_ckpt(cond)   # also refreshes all_results[cond]/loss_history[cond]
    bundle_metrics[cond] = all_results[cond]
    bundle_loss[cond]    = loss_history[cond]
    np.save(f'{APP_BUNDLE_DIR}/scores/scores_{cond.lower()}.npy', scores)
    if disc_sc is not None:
        np.save(f'{APP_BUNDLE_DIR}/scores/disc_{cond.lower()}.npy', disc_sc)
    for wname in ['enc1', 'dec', 'enc2', 're_attn', 'disc']:
        wp = f'{CKPT_DIR}/{cond}_{wname}.pth'
        if os.path.exists(wp):
            shutil.copy2(wp, f'{APP_BUNDLE_DIR}/weights/{cond.lower()}_{wname}.pth')

if 'binary_test' in dir():
    np.save(f'{APP_BUNDLE_DIR}/scores/binary_test.npy', binary_test)

with open(f'{APP_BUNDLE_DIR}/metrics.json', 'w') as f:
    json.dump(bundle_metrics, f, indent=2)
with open(f'{APP_BUNDLE_DIR}/loss_history.json', 'w') as f:
    json.dump(bundle_loss, f, indent=2)
with open(f'{APP_BUNDLE_DIR}/config.json', 'w') as f:
    json.dump({'image_size': IMAGE_SIZE, 'latent_dim': LATENT_DIM, 'lambda_adv': LAMBDA_ADV,
               'epochs': EPOCHS, 'warmup_epochs': WARMUP_EPOCHS, 'palette': PAL}, f, indent=2)

zip_path = f'{OUTPUT_DIR}/app_data_bundle.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(APP_BUNDLE_DIR):
        for fname in files:
            fpath   = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, APP_BUNDLE_DIR)
            zf.write(fpath, arcname)
print(f'app_data_bundle.zip ready ({os.path.getsize(zip_path)/1e3:.1f} KB) '
      '— download from Output tab, unzip into app/data/ (overwrite/merge).')