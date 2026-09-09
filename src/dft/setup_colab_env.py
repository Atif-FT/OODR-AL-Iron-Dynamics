#!/usr/bin/env python3
"""
setup_colab_env.py
==================
Run once at the beginning of each new Colab session:

    %run /content/drive/MyDrive/RISETT/.DESENTRALISASI\ 2026/dft_qe_setup/setup_colab_env.py

This script:
1. Install Quantum ESPRESSO (pw.x) via apt-get
2. Verify pw.x can be found
3. Verify Fe pseudopotential is available
4. Display environment readiness summary
"""

import subprocess
import shutil
import os
import sys

SEP = "=" * 55

def run_cmd(cmd, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=True)
    return r.returncode, r.stdout, r.stderr

print(SEP)
print("  SETUP COLAB ENVIRONMENT")
print(SEP)

# -----------------------------------------------------------------------
# 1. Install Quantum ESPRESSO
# -----------------------------------------------------------------------
print("\n[1/4] Installing Quantum ESPRESSO via apt-get...")

# Check first if already installed
pw_existing = shutil.which("pw.x")
if pw_existing:
    print(f"  ✅ pw.x already exists: {pw_existing} — skip install.")
else:
    print("  Running: apt-get install -y quantum-espresso")
    code, out, err = run_cmd(["apt-get", "install", "-y", "-q", "quantum-espresso"])
    if code == 0:
        print("  ✅ Quantum ESPRESSO successfully installed.")
    else:
        print(f"  ❌ apt-get failed (code={code}):")
        print(f"     {err[-400:]}")
        print("\n  Trying alternative method: conda install...")
        # Fallback: conda-forge
        code2, _, err2 = run_cmd(
            ["conda", "install", "-y", "-c", "conda-forge", "qe"],
            capture=True
        )
        if code2 == 0:
            print("  ✅ QE successful via conda-forge.")
        else:
            print(f"  ❌ conda also failed: {err2[-200:]}")
            print("  ⚠️  Please install manually: !apt-get install -y quantum-espresso")

# -----------------------------------------------------------------------
# 2. Verifikasi pw.x
# -----------------------------------------------------------------------
print("\n[2/4] Verifying pw.x...")
pw_path = shutil.which("pw.x")
if pw_path:
    code, ver_out, ver_err = run_cmd([pw_path, "--version"])
    ver_str = (ver_out or ver_err).strip().split("\n")[0]
    print(f"  ✅ pw.x found       : {pw_path}")
    print(f"     Version          : {ver_str}")
    QE_EXEC = pw_path
else:
    # Check common Colab paths
    candidates = ["/usr/bin/pw.x", "/usr/local/bin/pw.x", "/opt/conda/bin/pw.x"]
    QE_EXEC = None
    for c in candidates:
        if os.path.exists(c):
            QE_EXEC = c
            print(f"  ✅ pw.x found (manual) : {c}")
            break
    if not QE_EXEC:
        print("  ❌ pw.x not found after installation.")
        print("     Run manually: !apt-get install -y quantum-espresso")
        QE_EXEC = "/usr/bin/pw.x"  # Set default, will error when used

# -----------------------------------------------------------------------
# 3. Verifikasi pseudopotensial
# -----------------------------------------------------------------------
print("\n[3/4] Verifying pseudopotential...")
PSEUDO_DIR = "/content/drive/MyDrive/RISETT/.DESENTRALISASI 2026/dft_qe_setup/pseudo"
FE_PSEUDO  = "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF"
fe_pseudo_path = os.path.join(PSEUDO_DIR, FE_PSEUDO)

if os.path.exists(fe_pseudo_path):
    size_kb = os.path.getsize(fe_pseudo_path) / 1024
    print(f"  ✅ {FE_PSEUDO} ({size_kb:.1f} KB)")
else:
    print(f"  ❌ Pseudopotential not found: {fe_pseudo_path}")
    print(f"     Download from: https://pseudopotentials.quantum-espresso.org/")
    print(f"     Or use: !wget -P {PSEUDO_DIR} \\")
    print(f"         https://pseudopotentials.quantum-espresso.org/upf_files/{FE_PSEUDO}")

# -----------------------------------------------------------------------
# 4. Ringkasan
# -----------------------------------------------------------------------
print(f"\n[4/4] Environment Summary")
print(f"  Python      : {sys.version.split()[0]}")

try:
    import ase
    print(f"  ASE         : {ase.__version__}")
except ImportError:
    print("  ASE         : ❌ not installed (pip install ase)")

try:
    import mace
    print(f"  MACE-torch  : {mace.__version__}")
except ImportError:
    print("  MACE-torch  : ❌ not installed (pip install mace-torch)")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    cuda_str = torch.version.cuda if cuda_ok else "not available"
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA        : {cuda_str} | GPU: {'✅' if cuda_ok else '❌'}")
except ImportError:
    print("  PyTorch     : ❌ not installed")

print(f"\n  pw.x path   : {QE_EXEC}")
pseudo_ok = os.path.exists(fe_pseudo_path)
print(f"  Pseudo Fe   : {'✅' if pseudo_ok else '❌'} {fe_pseudo_path}")

print(f"\n{SEP}")
if pw_path and pseudo_ok:
    print("  ✅ Environment is READY for DFT calculations!")
else:
    print("  ⚠️  Some components are not ready. See messages above.")
print(SEP)

# Export to global namespace so it can be used in the notebook
import builtins
builtins.QE_EXEC_AUTO = QE_EXEC
print(f"\n💡 Variable QE_EXEC_AUTO = '{QE_EXEC}' is available in the notebook.")
print("   Use: QE_EXEC = QE_EXEC_AUTO")
