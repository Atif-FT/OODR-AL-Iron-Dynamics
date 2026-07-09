import os
import sys
import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

class MACEEnsembleDetector:
    """
    Ensemble-based uncertainty estimator and OOD detector for MACE.
    Computes disagreement among ensemble members to quantify model uncertainty.
    """
    def __init__(self, model_paths, device='cuda', default_dtype='float64'):
        self.device = device
        self.default_dtype = default_dtype
        self.models = []
        
        print(f"Initializing MACE Ensemble with {len(model_paths)} models...")
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
            calc = MACECalculator(
                model_paths=path,
                device=device,
                default_dtype=default_dtype
            )
            self.models.append(calc)
        print("Ensemble successfully loaded.")
        
    def evaluate_structure(self, atoms):
        """
        Runs prediction for all ensemble members.
        Returns:
            mean_energy: float (eV)
            std_energy: float (eV)
            mean_forces: np.ndarray of shape (N_atoms, 3) (eV/Angstrom)
            std_forces: np.ndarray of shape (N_atoms, 3) (eV/Angstrom)
            mean_stress: np.ndarray of shape (6,) or (9,) if available (eV/Angstrom^3)
        """
        energies = []
        forces = []
        stresses = []
        
        # Temp atoms object to avoid overriding main calc
        temp_atoms = atoms.copy()
        
        for calc in self.models:
            temp_atoms.calc = calc
            energies.append(temp_atoms.get_potential_energy())
            forces.append(temp_atoms.get_forces())
            try:
                # MACE-MP might output stresses, let's catch if not available
                stresses.append(temp_atoms.get_stress())
            except Exception:
                pass
                
        energies = np.array(energies)
        forces = np.stack(forces, axis=0)  # (N_models, N_atoms, 3)
        
        mean_energy = float(np.mean(energies))
        std_energy = float(np.std(energies))
        
        mean_forces = np.mean(forces, axis=0)
        std_forces = np.std(forces, axis=0)
        
        mean_stress = None
        if stresses:
            stresses = np.stack(stresses, axis=0)
            mean_stress = np.mean(stresses, axis=0)
            
        return mean_energy, std_energy, mean_forces, std_forces, mean_stress

    def is_ood(self, atoms, force_threshold=0.15, energy_threshold_per_atom=0.01):
        """
        Evaluates whether a structure is Out-of-Distribution (OOD).
        OOD Criteria:
        - Max standard deviation of force components across all atoms > force_threshold (eV/A)
        - Standard deviation of energy per atom > energy_threshold_per_atom (eV/atom)
        """
        mean_e, std_e, mean_f, std_f, _ = self.evaluate_structure(atoms)
        
        # 1. Force component uncertainty
        # Take the standard deviation of each force component, find the max per atom, then max over all atoms
        per_atom_force_std = np.max(std_f, axis=-1)  # (N_atoms,)
        max_force_std = np.max(per_atom_force_std)
        
        # 2. Energy uncertainty per atom
        energy_std_per_atom = std_e / len(atoms)
        
        # Flagging OOD
        force_ood = max_force_std > force_threshold
        energy_ood = energy_std_per_atom > energy_threshold_per_atom
        
        is_ood_flag = force_ood or energy_ood
        
        metrics = {
            "max_force_std": max_force_std,
            "energy_std_per_atom": energy_std_per_atom,
            "force_ood": force_ood,
            "energy_ood": energy_ood,
            "is_ood": is_ood_flag
        }
        
        return is_ood_flag, metrics

if __name__ == "__main__":
    # Small test sequence
    if len(sys.argv) < 3:
        print("Usage: python ood_detector.py <structure.xyz> <model1.model> [model2.model] ...")
        sys.exit(1)
        
    struct_path = sys.argv[1]
    models = sys.argv[2:]
    
    atoms = read(struct_path) if 'ase.io' in sys.modules else None
    if not atoms:
        from ase.io import read
        atoms = read(struct_path)
        
    detector = MACEEnsembleDetector(models, device='cpu')  # Run on CPU for test
    is_ood, metrics = detector.is_ood(atoms)
    
    print(f"Structure path: {struct_path}")
    print(f"OOD Status: {is_ood}")
    print(f"Metrics: {metrics}")
