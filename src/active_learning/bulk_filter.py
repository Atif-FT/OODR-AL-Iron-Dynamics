"""
bulk_filter.py - Utility to filter bulk vs slab/vacuum structures for Fe.

Used in:
  - Step 3 (MD): filter seed_configs before using as MD starting point
  - Step 4 (OOD): filter ood_candidates before FPS
  - run_dft.py: filter before DFT calculation

Principle: Mass Density
  Fe BCC bulk  : ~7.87 g/cm3
  Fe FCC bulk  : ~8.0  g/cm3
  Fe HCP bulk  : ~8.2  g/cm3
  Fe slab+vacuum: ~2-4 g/cm3  (contains empty space)

Threshold: density > MIN_DENSITY_BULK = 5.0 g/cm3
"""
import numpy as np

FE_MASS_AMU = 55.845
AMU_PER_A3_TO_G_PER_CM3 = 1.66054
MIN_DENSITY_BULK = 5.0


def get_density_g_cm3(atoms) -> float:
    """Calculate the mass density of the structure (g/cm3)."""
    cell = atoms.get_cell()
    vol_ang3 = abs(float(np.dot(cell[0], np.cross(cell[1], cell[2]))))
    if vol_ang3 < 1e-6:
        return 0.0
    mass_amu = len(atoms) * FE_MASS_AMU
    return mass_amu * AMU_PER_A3_TO_G_PER_CM3 / vol_ang3


def is_bulk_structure(atoms, min_density: float = MIN_DENSITY_BULK) -> bool:
    """Return True if atoms is a bulk structure (not slab/vacuum) AND has no atomic overlap."""
    cell = atoms.get_cell()
    lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]
    if min(lengths) < 2.0:
        return False
        
    # Check for atomic overlap (Unphysical structures that will crash QE or MD)
    try:
        dists = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1.8:
            return False
    except Exception:
        pass # If we can't compute distances, let it pass and fail elsewhere
        
    return get_density_g_cm3(atoms) >= min_density


def filter_bulk_configs(configs: list, min_density: float = MIN_DENSITY_BULK,
                        verbose: bool = True) -> list:
    """
    Filter a list of Atoms objects to keep only the bulk-like structures.
    Returns a list of bulk-like Atoms objects.
    """
    bulk_configs = []
    n_slab = 0
    for atoms in configs:
        if is_bulk_structure(atoms, min_density):
            bulk_configs.append(atoms)
        else:
            n_slab += 1
    if verbose:
        print(f"  [bulk_filter] {len(bulk_configs)}/{len(configs)} bulk "
              f"| {n_slab} slab/vacuum discarded "
              f"(threshold: rho >= {min_density} g/cm3)")
    return bulk_configs
