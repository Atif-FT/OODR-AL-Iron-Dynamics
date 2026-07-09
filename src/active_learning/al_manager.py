import os
import sys
import shutil
import torch
import numpy as np
from ase.io import read, write
from ood_detector import MACEEnsembleDetector
from coreset_selection import select_coreset


def find_ensemble_models(mace_train_dir, n_members=4, prefer_stage="stage1"):
    """
    Locates the trained ensemble model files from the mace_training directory.

    Args:
        mace_train_dir: path to the mace_training/ directory
        n_members: expected number of ensemble members
        prefer_stage: 'stage1' returns best-epoch models (lower force RMSE, better
                      for MD trajectory generation in active learning);
                      'stage2' returns SWA models (lower energy RMSE, for
                      thermodynamic evaluation after AL converges).

    Returns:
        List of absolute model paths, one per ensemble member.

    Stage 1 model naming: checkpoints/{name}_run-{seed}.model
    Stage 2 model naming: checkpoints/{name}_run-{seed}_stagetwo.model

    For active learning we recommend stage1 because:
      - RMSE_F (stage1) ≈ 69 meV/Å  < RMSE_F (stage2) ≈ 88 meV/Å
      - Accurate forces are critical for stable MD trajectories.
    """
    import glob
    ckpt_dir = os.path.join(mace_train_dir, "checkpoints")
    seeds = [42, 123, 999, 777]
    members = [f"Fe_Ensemble_Member_{i}" for i in range(n_members)]

    paths = []
    for member, seed in zip(members, seeds):
        if prefer_stage == "stage1":
            # Best-epoch checkpoint from Stage 1
            pattern = os.path.join(ckpt_dir, f"{member}_run-{seed}.model")
        else:
            # SWA / Stage 2 checkpoint
            pattern = os.path.join(ckpt_dir, f"{member}_run-{seed}_stagetwo.model")

        matches = glob.glob(pattern)
        if matches:
            paths.append(matches[0])
        else:
            # Fallback: search without seed in name
            fallback = glob.glob(os.path.join(ckpt_dir, f"{member}_run-*.model"))
            if prefer_stage == "stage1":
                fallback = [f for f in fallback if "_stagetwo" not in f]
            else:
                fallback = [f for f in fallback if "_stagetwo" in f]
            if fallback:
                paths.append(sorted(fallback)[-1])
            else:
                print(f"WARNING: No model found for {member} ({prefer_stage}). Skipping.")

    print(f"[find_ensemble_models] Found {len(paths)}/{n_members} {prefer_stage} models:")
    for p in paths:
        print(f"  {os.path.relpath(p, mace_train_dir)}")
    return paths

class ActiveLearningManager:
    """
    Orchestrates the active learning loop:
    1. Scan MD trajectories for OOD configurations.
    2. Filter candidates based on ensemble disagreement.
    3. Run FPS to select diverse coreset.
    4. Trigger DFT calculations on the coreset.
    5. Append new labeled data to the training set.
    6. Trigger model retraining/fine-tuning.
    """
    def __init__(self, ensemble_paths, train_db_path, valid_db_path, pseudo_dir, work_dir="al_workdir"):
        self.ensemble_paths = ensemble_paths
        self.train_db_path = train_db_path
        self.valid_db_path = valid_db_path
        self.pseudo_dir = pseudo_dir
        self.work_dir = work_dir
        
        os.makedirs(work_dir, exist_ok=True)
        self.detector = MACEEnsembleDetector(ensemble_paths, device='cuda' if torch.cuda.is_available() else 'cpu')

    def scan_trajectory_for_ood(self, traj_path, force_threshold=0.15, max_candidates=200):
        """Reads a MD trajectory and extracts structures flagged as OOD."""
        print(f"Scanning trajectory {traj_path} for OOD structures...")
        frames = read(traj_path, index=":")
        print(f"Total trajectory frames: {len(frames)}")
        
        # Ensure bulk_filter.py is accessible (same directory)
        import sys
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from bulk_filter import is_bulk_structure
        
        candidates = []
        for idx, frame in enumerate(frames):
            is_ood, metrics = self.detector.is_ood(frame, force_threshold=force_threshold)
            
            # CRITICAL: Only process BULK structures. If MD accidentally
            # generates surface slab/vacuum (rho < 5.0), ignore it.
            if is_ood and is_bulk_structure(frame):
                # Store frame and save UQ metrics in atoms.info for traceability
                frame_copy = frame.copy()
                frame_copy.info['max_force_std'] = metrics['max_force_std']
                frame_copy.info['energy_std_per_atom'] = metrics['energy_std_per_atom']
                frame_copy.info['traj_frame_idx'] = idx
                candidates.append(frame_copy)

                
            if len(candidates) >= max_candidates:
                print(f"Reached maximum candidate limit ({max_candidates}). Stopping scan.")
                break
                
        print(f"Found {len(candidates)} OOD candidates out of {len(frames)} frames.")
        
        # Save candidates
        candidates_path = os.path.join(self.work_dir, "ood_candidates.xyz")
        write(candidates_path, candidates, format="extxyz")
        return candidates_path

    def run_dft_on_coreset(self, coreset_path, output_path, run_dft_script):
        """Runs the run_dft.py script for each structure in the coreset."""
        coreset = read(coreset_path, index=":")
        print(f"Starting DFT calculations for {len(coreset)} structures in the coreset...")
        
        labeled_structures = []
        os.makedirs(os.path.join(self.work_dir, "dft_calculations"), exist_ok=True)
        
        for idx, atoms in enumerate(coreset):
            # Save temporary structure file
            temp_struct = os.path.join(self.work_dir, "dft_calculations", f"struct_{idx}.xyz")
            temp_labeled = os.path.join(self.work_dir, "dft_calculations", f"struct_{idx}_labeled.xyz")
            
            # Resume check: skip if structure already computed successfully
            if os.path.exists(temp_labeled):
                try:
                    labeled_atoms = read(temp_labeled)
                    energy = labeled_atoms.get_potential_energy()
                    labeled_structures.append(labeled_atoms)
                    print(f"[{idx+1}/{len(coreset)}] ✅ (Using cache) Energy = {energy:.6f} eV")
                    continue
                except Exception:
                    print(f"  ⚠️ Cache for struct_{idx} invalid, recalculating...")

            write(temp_struct, atoms, format="extxyz")
            
            # Execute run_dft.py via subprocess
            cmd = [
                sys.executable,
                run_dft_script,
                temp_struct,
                temp_labeled,
                self.pseudo_dir
            ]
            
            print(f"[{idx+1}/{len(coreset)}] Calculating DFT...")
            import subprocess
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode == 0 and os.path.exists(temp_labeled):
                # Read labeled structure
                labeled_atoms = read(temp_labeled)
                labeled_structures.append(labeled_atoms)
                # In ASE, read() loads energy into a SinglePointCalculator instead of info
                energy = labeled_atoms.get_potential_energy()
                print(f"  Success: Energy = {energy:.6f} eV")
            else:
                print(f"  Failed for structure {idx}: \n--- STDOUT ---\n{res.stdout}\n--- STDERR ---\n{res.stderr}")
                
        # Write all successfully labeled structures
        write(output_path, labeled_structures, format="extxyz")
        print(f"DFT labelling finished. {len(labeled_structures)} / {len(coreset)} successfully labeled.")
        return labeled_structures

    def merge_datasets(self, new_data_path):
        """Appends new labeled data to the training database."""
        # Backup training database
        backup_path = self.train_db_path + ".bak"
        shutil.copy2(self.train_db_path, backup_path)
        print(f"Created backup of training set at {backup_path}")
        
        old_train = read(self.train_db_path, index=":")
        new_data = read(new_data_path, index=":")
        
        merged = old_train + new_data
        write(self.train_db_path, merged, format="extxyz")
        print(f"Merged training dataset. Total configurations: {len(merged)} (Added {len(new_data)})")

    def retrain_ensemble(self, base_model, finetune_script, train_sh_path):
        """Triggers training of the ensemble with the updated dataset."""
        print("Starting ensemble retraining on Google Colab...")
        cmd = [
            "bash",
            train_sh_path,
            base_model,
            self.train_db_path,
            self.valid_db_path
        ]
        import subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Ensemble retraining failed.")
        print("Ensemble retraining finished successfully.")

def run_al_iteration(traj_path, force_threshold, n_select, manager_config):
    """Executes a single active learning iteration."""
    manager = ActiveLearningManager(
        ensemble_paths=manager_config['ensemble_paths'],
        train_db_path=manager_config['train_db_path'],
        valid_db_path=manager_config['valid_db_path'],
        pseudo_dir=manager_config['pseudo_dir'],
        work_dir=manager_config['work_dir']
    )
    
    # 1. Scan trajectory for OOD candidates
    candidates_xyz = manager.scan_trajectory_for_ood(
        traj_path=traj_path, 
        force_threshold=force_threshold
    )
    
    # 2. Select diverse coreset via FPS
    coreset_xyz = os.path.join(manager.work_dir, "coreset.xyz")
    select_coreset(
        input_xyz_path=candidates_xyz,
        output_xyz_path=coreset_xyz,
        n_select=n_select,
        mace_calc=manager.detector.models[0]  # Use first model for MACE latent space
    )
    
    # 3. Label coreset using Quantum ESPRESSO
    labeled_xyz = os.path.join(manager.work_dir, "coreset_labeled.xyz")
    manager.run_dft_on_coreset(
        coreset_path=coreset_xyz,
        output_path=labeled_xyz,
        run_dft_script=manager_config['run_dft_script']
    )
    
    # 4. Merge new data
    manager.merge_datasets(labeled_xyz)
    
    # 5. Retrain model
    manager.retrain_ensemble(
        base_model=manager_config['base_model_path'],
        finetune_script=manager_config['finetune_script'],
        train_sh_path=manager_config['train_sh_path']
    )
    print("=== Active Learning Iteration Completed ===")

if __name__ == "__main__":
    # Import torch here to avoid slowing down import when running --help
    import torch
    
    # --------------------------------------------------------------------------
    # Template configuration — driven by Colab notebook in production.
    # Use Stage 1 models for AL loop (forces more accurate → stable MD).
    # Use Stage 2 / SWA models for final thermodynamic evaluation only.
    # --------------------------------------------------------------------------
    MACE_TRAIN_DIR = os.path.abspath("../mace_training")
    ensemble_paths_stage1 = find_ensemble_models(MACE_TRAIN_DIR, n_members=4, prefer_stage="stage1")

    config = {
        # Stage 1 models: RMSE_F ≈ 69 meV/Å — preferred for MD + UQ in AL loop
        'ensemble_paths': ensemble_paths_stage1,
        'train_db_path': '../data/train.xyz',
        'valid_db_path': '../data/valid.xyz',
        'pseudo_dir': '../dft_qe_setup/pseudo',
        'work_dir': 'iteration_1',
        'run_dft_script': '../dft_qe_setup/run_dft.py',
        'base_model_path': '/content/mace-mp-0-medium.model',  # Foundation model on Colab
        'finetune_script': '../mace_training/finetune_mace.py',
        'train_sh_path': '../mace_training/train_ensemble.sh'
    }

    print("al_manager.py compiled and validated.")
    print(f"Ensemble ({len(config['ensemble_paths'])} members): {config['ensemble_paths']}")
