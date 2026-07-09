"""
run_dft.py  –  ASE wrapper for Quantum ESPRESSO pw.x calculations

Compatibility matrix:
  ASE < 3.23   → ASE_ESPRESSO_COMMAND env var (old API)
  ASE 3.23-3.27→ EspressoProfile(argv=[pw.x])  (intermediate API)
  ASE >= 3.28  → EspressoProfile(command=pw.x, pseudo_dir=...)  (new API)

This file uses a try/except cascade to support all three APIs automatically.
"""

import os
import sys
import inspect
import numpy as np
from ase import Atoms
from ase.io import read, write

# ---------------------------------------------------------------------------
# Detect ASE version and import Espresso classes
# ---------------------------------------------------------------------------
import ase
_ase_version = tuple(int(x) for x in ase.__version__.split(".")[:2])

if _ase_version >= (3, 23):
    from ase.calculators.espresso import Espresso, EspressoProfile
    _USE_PROFILE = True
else:
    from ase.calculators.espresso import Espresso
    _USE_PROFILE = False


def _make_espresso_profile(espresso_command: str, pseudo_dir: str):
    """
    Create an EspressoProfile object that works across ASE 3.23 – 3.28+.

    ASE changed the EspressoProfile constructor multiple times:
      • 3.23-3.27: EspressoProfile(argv=['pw.x'])
      • 3.28+    : EspressoProfile(command='pw.x', pseudo_dir='/path')

    We inspect the constructor signature at runtime to pick the right call.
    """
    sig = inspect.signature(EspressoProfile.__init__)
    params = list(sig.parameters.keys())  # ['self', ...]

    if 'command' in params and 'pseudo_dir' in params:
        # ASE >= 3.28 new API
        return EspressoProfile(command=espresso_command, pseudo_dir=pseudo_dir)
    elif 'argv' in params:
        # ASE 3.23 – 3.27 intermediate API
        return EspressoProfile(argv=[espresso_command])
    else:
        # Unknown future API – attempt keyword-free positional call
        try:
            return EspressoProfile(espresso_command, pseudo_dir)
        except Exception:
            return EspressoProfile(argv=[espresso_command])


def _auto_kpts(atoms) -> tuple:
    """
    Automatically choose a Monkhorst-Pack k-grid based on cell geometry.
    Caps at 4 per direction to prevent memory explosion.
    Returns k=1 for any direction with vacuum (L > 8 Å in slab context).
    """
    cell   = atoms.get_cell()
    lengths = np.array([np.linalg.norm(cell[i]) for i in range(3)])
    target_density = 10.0   # Å  (BCC Fe standard: 4×4×4 on 2×2×2 supercell ~5Å)

    kpts_auto = []
    for L in lengths:
        k = max(1, round(target_density / L))
        k = min(k, 4)
        kpts_auto.append(k)

    # Slab detection: one axis >= 3x mean of the other two -> set k=1 that axis
    # (3x threshold avoids false positives on 2x2x4 supercells)
    avg_other = lambda i: np.mean([lengths[j] for j in range(3) if j != i])
    for i in range(3):
        if lengths[i] >= 3.0 * avg_other(i):
            kpts_auto[i] = 1

    return tuple(kpts_auto)


def _get_density_g_cm3(atoms) -> float:
    """
    Calculate mass density of the structure in g/cm^3.

    Fe BCC bulk  : ~7.87 g/cm3  → accepted (> 5.0)
    Fe FCC bulk  : ~8.0  g/cm3  → accepted
    Fe HCP bulk  : ~8.2  g/cm3  → accepted
    Fe slab+vacuum: ~2-4  g/cm3  → REJECTED (contains vacuum space)

    Threshold: MIN_BULK_DENSITY = 5.0 g/cm3
    """
    FE_MASS_AMU              = 55.845
    AMU_A3_TO_G_CM3          = 1.66054  # conversion: 1 amu/A^3 = 1.66054 g/cm^3
    cell     = atoms.get_cell()
    vol_ang3 = abs(float(np.dot(cell[0], np.cross(cell[1], cell[2]))))
    if vol_ang3 < 1e-6:
        return 0.0
    # assume all atoms are Fe (valid for this system)
    mass_amu = len(atoms) * FE_MASS_AMU
    return mass_amu * AMU_A3_TO_G_CM3 / vol_ang3


def _build_calculator(pseudo_dir: str,
                      elements: set,
                      espresso_command: str = "pw.x",
                      kpts: tuple = (4, 4, 4),
                      ecutwfc: float = 50.0,
                      ecutrho: float = 400.0) -> Espresso:
    """
    Build and return a configured ASE Espresso calculator.

    Parameters
    ----------
    pseudo_dir      : absolute path to the folder containing UPF pseudopotentials
    elements        : set of chemical symbol strings present in the structure
    espresso_command: path / name of the pw.x executable
    kpts            : Monkhorst-Pack k-point grid
    ecutwfc         : plane-wave cutoff energy (Ry)
    ecutrho         : charge-density cutoff energy (Ry)
    """
    pseudopotentials = {
        "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    }
    # Only include pseudopotentials for elements in this structure
    active_pseudo = {el: pseudopotentials[el] for el in elements if el in pseudopotentials}

    # -----------------------------------------------------------------------
    # Build starting_magnetization dict (1-indexed, alphabetical species order)
    # -----------------------------------------------------------------------
    sorted_elements = sorted(list(elements))
    starting_mag = {}
    if "Fe" in sorted_elements:
        fe_idx = sorted_elements.index("Fe") + 1  # QE uses 1-based index
        starting_mag[f"starting_magnetization({fe_idx})"] = 0.5

    # In ASE >= 3.28, pseudo_dir is passed via EspressoProfile constructor.
    # In older ASE, it must be inside input_data['control'].
    # We always keep it in input_data as a safety net – QE will use whichever it sees.
    input_data = {
        "control": {
            "calculation":  "scf",
            "restart_mode": "from_scratch",
            "pseudo_dir":   pseudo_dir,  # kept for compatibility with all QE/ASE combos
            "tstress":      True,   # output stress tensor  (required for MACE)
            "tprnfor":      True,   # output atomic forces  (required for MACE)
            "disk_io":      "none", # skip wavefunction dumps → faster on Colab
        },
        "system": {
            "ecutwfc":    ecutwfc,
            "ecutrho":    ecutrho,
            "occupations": "smearing",
            "smearing":   "cold",   # Marzari-Vanderbilt cold smearing
            "degauss":    0.03,     # [INCREASED] from 0.02 to 0.03 Ry to ease convergence of chaotic OOD structures
            "nspin":      2,        # spin-polarised (Fe is magnetic)
            **starting_mag,
        },
        "electrons": {
            "diagonalization":  "david",
            "mixing_beta":      0.4,          # [INCREASED] from 0.3 to 0.4 for slightly more aggressive convergence
            "mixing_mode":      "local-TF",   # better for metallic Fe (vs plain)
            "mixing_ndim":      15,           # [ADDED] Store 15 iterations of mixing history (default 8), helps difficult SCF
            "conv_thr":         1.0e-6,
            "electron_maxstep": 80,           # [DECREASED] from 200 to 80. "Fail Fast" principle: if >80 steps fail, abandon it rather than wasting 1.5 hours.
        },
    }

    if _USE_PROFILE:
        # ASE >= 3.23 – use EspressoProfile (API auto-detected at runtime)
        profile = _make_espresso_profile(espresso_command, pseudo_dir)
        print(f"  [ASE {ase.__version__}] Using EspressoProfile: {type(profile).__name__} | params={list(inspect.signature(EspressoProfile.__init__).parameters.keys())}")
        calc = Espresso(
            profile=profile,
            pseudopotentials=active_pseudo,
            input_data=input_data,
            kpts=kpts,
            crystal_coordinates=True,
        )
    else:
        # ASE < 3.23 – set environment variable (old way)
        os.environ["ASE_ESPRESSO_COMMAND"] = f"{espresso_command} -in PREFIX.pwi > PREFIX.pwo"
        calc = Espresso(
            pseudopotentials=active_pseudo,
            input_data=input_data,
            kpts=kpts,
            crystal_coordinates=True,
        )

    return calc


def run_calculation(structure_path: str,
                    output_path: str,
                    pseudo_dir: str,
                    espresso_command: str = "pw.x",
                    kpts: tuple = (4, 4, 4)) -> bool:
    """
    Read one atomic structure, run a Quantum ESPRESSO SCF calculation,
    extract energy / forces / stress, and write an extended-XYZ file that
    is directly consumable by MACE training.

    Returns True on success, False on failure (so the caller can skip).
    """
    print(f"Reading structure from {structure_path}...")
    atoms = read(structure_path)
    elements = set(atoms.get_chemical_symbols())
    n_atoms  = len(atoms)
    print(f"Elements present: {elements}  |  N atoms: {n_atoms}  |  ASE {ase.__version__}")

    # == Sanity check 1: Minimum atomic distance ================================
    try:
        dists = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(np.min(dists))
        print(f"  Min interatomic distance : {min_dist:.4f} A")
        if min_dist < 1.8:
            print(f"  SKIP: Atomic overlap ({min_dist:.4f} A < 1.8 A).")
            return False
    except Exception as d_err:
        print(f"  (Could not check min distance: {d_err})")
        min_dist = 99.0

    # == Sanity check 2: Filter slab/vacuum via MASS DENSITY ====================
    # Fe bulk: ~7.87 g/cm3  |  Fe slab + vakum: ~3.5 g/cm3
    # Threshold 5.0 g/cm3 separates both with a safe margin.
    # NOT dependent on cell dimensions (robust for non-orthogonal cells).
    MIN_BULK_DENSITY = 5.0   # g/cm3
    density = _get_density_g_cm3(atoms)
    cell_lengths = [float(np.linalg.norm(atoms.get_cell()[i])) for i in range(3)]
    print(f"  Cell lengths (A)         : {[f'{l:.2f}' for l in cell_lengths]}")
    print(f"  Structure density        : {density:.2f} g/cm3")

    if density < MIN_BULK_DENSITY:
        print(f"  SKIP: rho = {density:.2f} g/cm3 < {MIN_BULK_DENSITY} g/cm3.")
        print(f"        Structure contains vacuum (surface slab).")
        print(f"        Stress tensor on vacuum cells is INVALID for bulk Fe FF.")
        return False

    # == Choose adaptive ecutwfc based on supercell size ========================
    #   <= 16 atoms : 50 Ry  (standard BCC Fe, accurate)
    #   >  16 atoms : 40 Ry  (large supercell, still accurate, saves RAM)
    if n_atoms > 16:
        ecutwfc_eff = 40.0
        ecutrho_eff = 320.0
    else:
        ecutwfc_eff = 50.0
        ecutrho_eff = 400.0

    # == Choose adaptive k-grid =================================================
    effective_kpts = _auto_kpts(atoms)
    print(f"  ecutwfc / ecutrho        : {ecutwfc_eff:.0f} / {ecutrho_eff:.0f} Ry")
    print(f"  Effective k-grid         : {effective_kpts}")

    calc = _build_calculator(
        pseudo_dir=pseudo_dir,
        elements=elements,
        espresso_command=espresso_command,
        kpts=effective_kpts,
        ecutwfc=ecutwfc_eff,
        ecutrho=ecutrho_eff,
    )
    atoms.calc = calc

    print("Starting Quantum ESPRESSO SCF calculation...")
    try:
        energy = atoms.get_potential_energy()       # eV
        forces = atoms.get_forces()                  # eV/Å,  shape (N, 3)
        stress = atoms.get_stress(voigt=False)       # eV/Å³, shape (3, 3)

        print(f"  ✓ Energy = {energy:.6f} eV  ({len(atoms)} atoms)")

        # Store in atoms.info / atoms.arrays for Extended XYZ
        atoms.info["energy"] = energy
        # Flatten 3×3 stress to 9-element array (MACE convention)
        atoms.info["stress"] = stress.flatten()
        atoms.new_array("forces", forces)

        # Detach calculator before writing (avoids recursion)
        atoms.calc = None

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        write(output_path, atoms, format="extxyz")
        print(f"  ✓ Written to {output_path}")
        return True

    except Exception as exc:
        print(f"  ✗ QE calculation FAILED: {exc}")
        
        # Try to read and print the last 30 lines of espresso.pwo to show the actual error
        pwo_path = "espresso.pwo"
        if calc and hasattr(calc, 'directory') and calc.directory:
            pwo_path = os.path.join(calc.directory, "espresso.pwo")
        
        if os.path.exists(pwo_path):
            print(f"\n=== LAST 30 LINES OF {pwo_path} ===")
            try:
                with open(pwo_path, "r", encoding="utf-8", errors="ignore") as f_pwo:
                    lines = f_pwo.readlines()
                    last_lines = lines[-30:]
                    for line in last_lines:
                        print(line, end="")
            except Exception as read_err:
                print(f"Could not read {pwo_path}: {read_err}")
            print("=======================================\n")
        else:
            print(f"  (Output file {pwo_path} was not found)")
            
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# CLI entry-point (standalone use)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_dft.py <input_structure.xyz> <output_labeled.xyz> [pseudo_dir] [pw.x path]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    default_pseudo = "/content/drive/MyDrive/RISETT/.DESENTRALISASI 2026/dft_qe_setup/pseudo"
    p_dir       = sys.argv[3] if len(sys.argv) > 3 else default_pseudo
    qe_cmd      = sys.argv[4] if len(sys.argv) > 4 else "pw.x"

    ok = run_calculation(input_file, output_file, p_dir, espresso_command=qe_cmd)
    sys.exit(0 if ok else 1)
