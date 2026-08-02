# %% [C 1] RE-Attn-AAE: A Reconstruction-Error-Guided Attention Adversarial Autoencoder for Dual-Domain Unsupervised Anomaly Detection
# ## This is  a PyTorch 
# ## implementation of the RE-Attn-AAE model for unsupervised anomaly detection in dual-domain data. The model leverages reconstruction error to guide attention mechanisms, 
# ## enhancing the detection of anomalies in complex datasets.
# %%
# %% [CELL 1.1]  Install / verify packages (version-aware, single source of truth)

import subprocess, sys
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

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
# %%
