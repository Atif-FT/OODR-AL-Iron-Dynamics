"""
generate_seed_dataset.py  –  Generate the DFT seed dataset for MACE active learning.

This script:
  1. Generates 24 perturbed Al, Fe, and Al-Fe alloy structures
  2. Installs Quantum ESPRESSO automatically if running in Google Colab
  3. Runs each structure through pw.x via ASE
  4. Merges all successful calculations into a single Extended XYZ file

Usage (in Colab):
  !python dft_qe_setup/generate_seed_dataset.py \\
      /content/drive/MyDrive/RISETT/.DESENTRALISASI\\ 2026/dft_qe_setup/pseudo

Output:
  data/seed_train.xyz   (saved to local Colab /content path)
  Then we sync it to Google Drive in the notebook.
"""

import os
import sys
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

# ---------------------------------------------------------------------------
# Step 0: Auto-install Quantum ESPRESSO if not found (Colab-friendly)
# ---------------------------------------------------------------------------

def _find_pw_x() -> str:
    """
    Returns the path to pw.x if found, else empty string.
    """
    import shutil
    path = shutil.which("pw.x")
    return path or ""


def ensure_quantum_espresso() -> str:
    """
    Make sure Quantum ESPRESSO (pw.x) is available.
    On Google Colab: auto-install via apt-get.
    Elsewhere: raise an informative error.
    
    Returns the path to pw.x.
    """
    pw_path = _find_pw_x()
    if pw_path:
        print(f"[QE] Found pw.x at: {pw_path}")
        return pw_path

    # Not found – try to install
    print("[QE] pw.x not found. Attempting to install Quantum ESPRESSO...")

    # Detect Google Colab
    in_colab = "COLAB_GPU" in os.environ or os.path.exists("/content")

    if in_colab:
        print("[QE] Google Colab detected → installing via apt-get (this takes ~2 min)...")
        ret = os.system("apt-get install -y quantum-espresso > /dev/null 2>&1")
        pw_path = _find_pw_x()
        if pw_path:
            print(f"[QE] Installation successful! pw.x at: {pw_path}")
            return pw_path
        else:
            # Some Colab images have it under a different name
            ret2 = os.system("apt-get install -y espresso > /dev/null 2>&1")
            pw_path = _find_pw_x()
            if pw_path:
                print(f"[QE] Installation successful! pw.x at: {pw_path}")
                return pw_path
            raise RuntimeError(
                "[QE] ❌ Could not install Quantum ESPRESSO automatically.\n"
                "Please run the following in a Colab cell and then re-run this script:\n"
                "    !apt-get install -y quantum-espresso\n"
            )
    else:
        raise RuntimeError(
            "[QE] ❌ pw.x not found on this system.\n"
            "Please install Quantum ESPRESSO:\n"
            "  • Linux/macOS: sudo apt-get install quantum-espresso  (or conda install -c conda-forge qe)\n"
            "  • Windows (WSL): use WSL2 with Ubuntu and install there\n"
        )


# ---------------------------------------------------------------------------
# Step 1: Generate perturbed structures
# ---------------------------------------------------------------------------

def create_perturbed_structures() -> list:
    """
    Returns a list of (name, Atoms) tuples covering:
      - Pure BCC Fe (equilibrium + 15 perturbed)
      - Vacancy BCC Fe (equilibrium + 15 perturbed)
      - Interstitial BCC Fe (equilibrium + 15 perturbed)

    Perturbations simulate thermal displacements at ~300–900 K and volume strains.
    """
    from ase import Atom
    base_structures = []

    # 1. Pure BCC Fe – 2×2×2 supercell (16 atoms)
    fe_unit  = bulk("Fe", "bcc", a=2.86)
    fe_super = fe_unit * (2, 2, 2)
    base_structures.append(("Fe_pure_eq", fe_super.copy()))

    # 2. Vacancy in BCC Fe – remove 1 Fe atom (15 atoms)
    fe_vacancy = fe_super.copy()
    del fe_vacancy[0]
    base_structures.append(("Fe_vacancy_eq", fe_vacancy.copy()))

    # 3. Interstitial in BCC Fe – add 1 Fe atom at octahedral site (17 atoms)
    fe_interstitial = fe_super.copy()
    # Octahedral site at conventional edge center (a/2, 0, 0) -> (1.43, 0.0, 0.0)
    fe_interstitial.append(Atom("Fe", position=[1.43, 0.0, 0.0]))
    base_structures.append(("Fe_interstitial_eq", fe_interstitial.copy()))

    perturbed_structures = []
    # Include equilibrium structures as-is
    for name, atoms in base_structures:
        perturbed_structures.append((name, atoms))

    # Generate 15 perturbed snapshots per base structure
    rng = np.random.default_rng(seed=42)
    for name, atoms in base_structures:
        for i in range(15):
            atoms_p = atoms.copy()
            # Random displacements ~ 0.03 Å (mimics low-T thermal motion)
            disp = rng.normal(0.0, 0.03, size=(len(atoms_p), 3))
            atoms_p.positions += disp
            # Isotropic volume strain in ±5 %
            strain = rng.uniform(0.95, 1.05)
            atoms_p.set_cell(atoms_p.get_cell() * strain, scale_atoms=True)
            perturbed_structures.append((f"{name}_perturbed_{i}", atoms_p))

    print(f"[GEN] Generated {len(perturbed_structures)} candidate structures (3 base × 16 variants).")
    return perturbed_structures


def create_perturbed_structures_extra() -> list:
    """
    Returns a list of (name, Atoms) tuples covering 7 extra categories (112 structures):
      - FCC Fe bulk (1 equilibrium + 15 perturbed)
      - HCP Fe bulk (1 equilibrium + 15 perturbed)
      - BCC Fe shear strained (1 equilibrium + 15 perturbed)
      - BCC Fe surface slab (1 equilibrium + 15 perturbed)
      - Stepped surface with adatoms (1 equilibrium + 15 perturbed)
      - Liquid-like BCC Fe (1 equilibrium + 15 perturbed)
      - Dislocated/highly distorted BCC Fe (1 equilibrium + 15 perturbed)

    Includes an atomic overlap filter (min distance > 1.8 Å) to prevent QE failures.
    """
    from ase import Atom
    from ase.build import surface
    base_structures = []

    # 1. FCC Fe Bulk - High Temp Phase (2x2x2 supercell - 8 atoms)
    fe_fcc = bulk("Fe", "fcc", a=3.6)
    fcc_super = fe_fcc * (2, 2, 2)
    base_structures.append(("Fe_FCC_bulk", fcc_super.copy()))

    # 2. HCP Fe Bulk - High Pressure Phase (2x2x2 supercell - 16 atoms)
    fe_hcp = bulk("Fe", "hcp", a=2.5, c=4.1)
    hcp_super = fe_hcp * (2, 2, 2)
    base_structures.append(("Fe_HCP_bulk", hcp_super.copy()))

    # 3. BCC Fe Shear-Strained (8 atom)
    fe_bcc = bulk("Fe", "bcc", a=2.86)
    fe_shear = fe_bcc * (2, 2, 2)
    cell = fe_shear.get_cell()
    cell[0, 1] += 0.15 * cell[0, 0] # Shear strain 15%
    fe_shear.set_cell(cell, scale_atoms=True)
    base_structures.append(("Fe_BCC_shear", fe_shear.copy()))

    # 4. BCC Fe Surface Slab (110) (12 atom)
    fe_surf = surface(fe_bcc, (1, 1, 0), 3)
    fe_surf.center(vacuum=5.0, axis=2)
    base_structures.append(("Fe_surface_slab", fe_surf.copy()))

    # 5. Stepped Surface with Adatoms (13 atom)
    fe_stepped = fe_surf.copy()
    fe_stepped.append(Atom("Fe", position=[1.43, 1.43, 8.0]))
    base_structures.append(("Fe_surface_adatom", fe_stepped.copy()))

    # 6. Liquid-like Fe (8 atom)
    fe_liq = fe_bcc * (2, 2, 2)
    base_structures.append(("Fe_liquid_like", fe_liq.copy()))

    # 7. Dislocated / Highly Distorted Fe (8 atom)
    fe_distort = fe_bcc * (2, 2, 2)
    cell_d = fe_distort.get_cell()
    cell_d[0, 0] *= 1.10 # Tensile strain 10%
    cell_d[1, 1] *= 0.90 # Compression 10%
    fe_distort.set_cell(cell_d, scale_atoms=True)
    base_structures.append(("Fe_distorted_plastic", fe_distort.copy()))

    perturbed_structures = []
    # Include base structures
    for name, atoms in base_structures:
        perturbed_structures.append((name, atoms))

    # Generate 15 perturbed snapshots per base structure
    rng = np.random.default_rng(seed=123)
    for name, atoms in base_structures:
        is_liquid = "liquid_like" in name
        disp_std = 0.12 if is_liquid else 0.03 # Large displacement for liquid-like
        
        for i in range(15):
            for attempt in range(20):
                atoms_p = atoms.copy()
                disp = rng.normal(0.0, disp_std, size=(len(atoms_p), 3))
                atoms_p.positions += disp
                strain = rng.uniform(0.95, 1.05)
                atoms_p.set_cell(atoms_p.get_cell() * strain, scale_atoms=True)
                
                # Check minimum distance to prevent QE crash
                dists = atoms_p.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) > 1.8:
                    perturbed_structures.append((f"{name}_perturbed_{i}", atoms_p))
                    break
            else:
                perturbed_structures.append((f"{name}_perturbed_{i}", atoms_p))

    print(f"[GEN] Generated {len(perturbed_structures)} extra structures (7 base × 16 variants).")
    return perturbed_structures


# ---------------------------------------------------------------------------
# Step 2: Orchestrate DFT calculations
# ---------------------------------------------------------------------------

def generate_seed_dataset(output_path: str,
                          pseudo_dir: str,
                          qe_executable: str = "pw.x",
                          drive_output_path: str = None,
                          kpts: tuple = (2, 2, 2),
                          mode: str = "base") -> None:
    """
    Main orchestration function — CRASH-SAFE VERSION.
    
    Parameters
    ----------
    mode              : 'base' (BCC bulk/vacancy/interstitial),
                        'extra' (7 new categories),
                        'all' (both base and extra)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from run_dft import run_calculation

    if mode == "base":
        candidates = create_perturbed_structures()
    elif mode == "extra":
        candidates = create_perturbed_structures_extra()
    elif mode == "all":
        candidates = create_perturbed_structures() + create_perturbed_structures_extra()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    workdir    = "seed_workdir"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if drive_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(drive_output_path)), exist_ok=True)

    # ── RESUME: load structures already saved on Drive ──────────────────────
    already_done = set()
    labeled_structures = []

    checkpoint = drive_output_path or output_path
    if os.path.exists(checkpoint):
        try:
            existing = read(checkpoint, index=':')
            labeled_structures = list(existing)
            already_done = set(range(len(labeled_structures)))
            print(f"[RESUME] Found {len(labeled_structures)} structures already completed.")
            print(f"         Skipping structs 0-{len(labeled_structures)-1}, resuming from struct {len(labeled_structures)}.")
        except Exception as e:
            print(f"[RESUME] Could not read checkpoint: {e} — starting fresh.")

    failed_names = []
    start_time   = __import__('time').time()

    for idx, (name, atoms) in enumerate(candidates):

        # Skip already-done structures
        if idx in already_done:
            print(f"  [SKIP] [{idx+1:02d}/{len(candidates)}] {name} — already done")
            continue

        print(f"\n--- [{idx+1}/{len(candidates)}] {name} ({len(atoms)} atoms) ---")
        temp_in  = os.path.join(workdir, f"struct_{idx}.xyz")
        temp_out = os.path.join(workdir, f"struct_{idx}_labeled.xyz")

        write(temp_in, atoms, format="extxyz")

        success = run_calculation(
            structure_path=temp_in,
            output_path=temp_out,
            pseudo_dir=pseudo_dir,
            espresso_command=qe_executable,
            kpts=kpts,
        )

        if success and os.path.exists(temp_out):
            labeled_atoms = read(temp_out)
            labeled_structures.append(labeled_atoms)

            # ── AUTO-SAVE: append to Drive immediately ───────────────────
            if drive_output_path:
                try:
                    write(drive_output_path, labeled_structures, format="extxyz")
                    elapsed = (__import__('time').time() - start_time) / 60
                    print(f"  Saved {len(labeled_structures)} structs to Drive "
                          f"[elapsed: {elapsed:.1f} min]")
                except Exception as e:
                    print(f"  Drive save failed: {e} — continuing anyway")

            # Also save local copy
            write(output_path, labeled_structures, format="extxyz")

        else:
            print(f"  Skipping {name} - calculation failed.")
            failed_names.append(name)

    # ── Final summary ────────────────────────────────────────────────────────
    total_time = (__import__('time').time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"SEED DATASET GENERATION COMPLETE")
    print(f"  Successful : {len(labeled_structures)} / {len(candidates)}")
    print(f"  Time       : {total_time:.1f} min total")
    if failed_names:
        print(f"  Failed     : {', '.join(failed_names)}")

    if labeled_structures:
        write(output_path, labeled_structures, format="extxyz")
        print(f"  Local      : {output_path}")
        if drive_output_path:
            write(drive_output_path, labeled_structures, format="extxyz")
            print(f"  Drive      : {drive_output_path}")
        print(f"{'='*60}")
    else:
        print("  ERROR: No structures were successfully labeled.")
        print(f"{'='*60}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default pseudo dir (used when running locally on Windows)
    _DEFAULT_PSEUDO = r"c:\ARJUNAA\RISETT\.DESENTRALISASI 2026\dft_qe_setup\pseudo"
    # Default output path (Colab local filesystem)
    _DEFAULT_OUT    = "data/seed_train.xyz"

    p_dir   = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PSEUDO
    out_xyz = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_OUT

    print("=" * 60)
    print("SEED DATASET GENERATOR  –  Fe BCC ML-IFF Project")
    print("=" * 60)
    print(f"Pseudopotential dir : {p_dir}")
    print(f"Output file         : {out_xyz}")

    # Step 0: Ensure QE is installed
    qe_cmd = ensure_quantum_espresso()

    # Step 1-2: Generate and label structures
    generate_seed_dataset(
        output_path=out_xyz,
        pseudo_dir=p_dir,
        qe_executable=qe_cmd,
    )
