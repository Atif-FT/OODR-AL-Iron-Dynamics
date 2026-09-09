import os
import sys
import json
import shutil
import numpy as np
from ase.io import read, write

# ==============================================================================
# HYPERPARAMETER & DIRECTORY CONFIGURATION (ADAPT TO COLAB)
# ==============================================================================
# In Colab, run this script from the project root folder:
# cd "/content/drive/MyDrive/RISETT/.DESENTRALISASI 2026"
PROJECT_ROOT = os.getcwd()
AL_DIR       = os.path.join(PROJECT_ROOT, "active_learning_loop")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MACE_DIR     = os.path.join(PROJECT_ROOT, "mace_training")
PSEUDO_DIR   = os.path.join(PROJECT_ROOT, "dft_qe_setup", "pseudo")

sys.path.insert(0, AL_DIR)
try:
    from al_manager import ActiveLearningManager, find_ensemble_models
    from coreset_selection import select_coreset
except ImportError:
    raise RuntimeError("Ensure script is executed from the project root directory (.DESENTRALISASI 2026)")

# --- Active Learning Parameters ---
MAX_ITERATIONS   = 5
CORESET_SIZE     = 30       # Constant: 30 new samples per iteration
FORCE_OOD_THR    = 0.12     # Initial uncertainty threshold
MD_TEMPERATURES  = [300, 900, 1500, 1800]
N_MD_STEPS       = 10000

# State File for Auto-Resume
STATE_FILE = os.path.join(PROJECT_ROOT, "al_state.json")

# ==============================================================================
# STATE SYSTEM & AUTO-RESUME
# ==============================================================================
def load_state():
    """Load the last status from Google Drive to resume progress."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        print(f"🔄 [AUTO-RESUME] Resuming from Iteration {state['current_iteration']}")
        return state
    else:
        print("▶️ [START] No state found. Starting from Iteration 2.")
        return {'current_iteration': 2} # We start from 2 because Iteration 1 was manual

def save_state(iteration):
    """Save state to Google Drive to prevent data loss on Colab Disconnect."""
    state = {'current_iteration': iteration}
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
    print(f"💾 [AUTO-BACKUP] State for iteration {iteration} saved to Google Drive.")

# ==============================================================================
# OODR-AL LOOP LOGIC
# ==============================================================================
def run_automated_al():
    state = load_state()
    start_iter = state['current_iteration']

    for current_iter in range(start_iter, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"🚀 STARTING ACTIVE LEARNING ITERATION {current_iter}")
        print(f"{'='*60}")

        # 0. Set Work Directory (Directly on Drive for automatic partial backup)
        iter_workdir = os.path.join(PROJECT_ROOT, f"al_iter_{current_iter}")
        os.makedirs(iter_workdir, exist_ok=True)

        # 1. Load the Latest Ensemble Models
        ensemble_paths = find_ensemble_models(MACE_DIR, prefer_stage="stage1")
        if len(ensemble_paths) < 4:
            raise RuntimeError("Ensemble models incomplete. Ensure previous training succeeded.")

        # Initialize Manager
        train_xyz = os.path.join(DATA_DIR, "train.xyz")
        valid_xyz = os.path.join(DATA_DIR, "valid.xyz")
        
        manager = ActiveLearningManager(
            ensemble_paths=ensemble_paths,
            train_db_path=train_xyz,
            valid_db_path=valid_xyz,
            pseudo_dir=PSEUDO_DIR,
            work_dir=iter_workdir
        )

        # Check if Coreset was already generated
        coreset_path = os.path.join(iter_workdir, "coreset.xyz")
        if os.path.exists(coreset_path):
            print(f"\n[CACHE] Found file {coreset_path} from previous session.")
            print(f"⏩ SKIPPING [STEP 1] to [STEP 3] and proceeding directly to [STEP 4] (DFT)...")
        else:
            # 2. GENERATE MD TRAJECTORY (EXPLORATION)
            print(f"\n[STEP 1] Running MD Exploration (OOD Search)...")
            from ase.md.langevin import Langevin
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            from ase import units
            from mace.calculators import MACECalculator

            ensemble_calc = MACECalculator(model_paths=ensemble_paths, device='cuda', default_dtype='float32')
            # USE train.xyz SO MD CAN START FROM NEW STRUCTURES (DEEP EXPLORATION)
            seed_configs_raw = read(os.path.join(DATA_DIR, "train.xyz"), index=":")
            
            # CRITICAL: Filter seed configs to ONLY use bulk (density >= 5.0 g/cm3)
            # If MD starts from a surface slab (vacuum), DFT will reject it in Step 4.
            import sys
            sys.path.append(AL_DIR)
            from bulk_filter import filter_bulk_configs
            seed_configs = filter_bulk_configs(seed_configs_raw, verbose=True)

            rng = np.random.default_rng(current_iter * 100)
            start_indices = rng.choice(len(seed_configs), size=len(MD_TEMPERATURES), replace=False)


            all_traj_frames = []
            for temp_K, seed_idx in zip(MD_TEMPERATURES, start_indices):
                start_atoms = seed_configs[seed_idx].copy()
                
                # Dynamic Shear Deformation to force OOD state
                cell = start_atoms.get_cell()
                shear_amount = rng.uniform(0.05, 0.10)
                cell[0, 1] += shear_amount * cell[0, 0]
                cell[1, 2] += shear_amount * cell[1, 1]
                start_atoms.set_cell(cell, scale_atoms=True)

                start_atoms.calc = ensemble_calc
                MaxwellBoltzmannDistribution(start_atoms, temperature_K=temp_K)
                dyn = Langevin(start_atoms, timestep=1.0*units.fs, temperature_K=temp_K, friction=0.02, loginterval=1000)
                
                saved_frames = []
                def save_frame():
                    if dyn.nsteps % 10 == 0:
                        saved_frames.append(start_atoms.copy())
                dyn.attach(save_frame, interval=1)
                dyn.run(N_MD_STEPS)
                all_traj_frames.extend(saved_frames)

            traj_path = os.path.join(iter_workdir, "trajectory_all.xyz")
            write(traj_path, all_traj_frames, format="extxyz")

            # 3. OOD SCANNING
            print(f"\n[STEP 2] Scanning Trajectory for OOD structures...")
            dynamic_thr = max(0.05, FORCE_OOD_THR - (current_iter - 2) * 0.02)
            print(f"Using dynamic threshold: {dynamic_thr:.3f} eV/Å")
            
            candidates_path = manager.scan_trajectory_for_ood(traj_path, force_threshold=dynamic_thr, max_candidates=500)
            candidates = read(candidates_path, index=":")
            
            if not candidates:
                print("🎉 NO OOD FOUND! Model is highly robust.")
                if current_iter >= 3: 
                    print("Stopping AL loop as convergence is reached.")
                    break
                else:
                    print("Proceeding to next iteration to verify...")
                    save_state(current_iter + 1)
                    continue

            # 4. CORESET SELECTION
            print(f"\n[STEP 3] Farthest Point Sampling (FPS) SELECTION...")
            select_coreset(candidates_path, coreset_path, n_select=CORESET_SIZE)

        # 5. DFT LABELLING (with bypass if already completed)
        run_dft_script = os.path.join(PROJECT_ROOT, "dft_qe_setup", "run_dft.py")
        labeled_path = os.path.join(iter_workdir, "coreset_labeled.xyz")

        if os.path.exists(labeled_path):
            print(f"\n[CACHE] ⏩ coreset_labeled.xyz found. SKIPPING [STEP 4] (DFT)...")
            labeled_structures = read(labeled_path, index=":")
            print(f"  Loading {len(labeled_structures)} labeled structures from cache.")
        else:
            print(f"\n[STEP 4] Quantum ESPRESSO (DFT) Calculation...")
            labeled_structures = manager.run_dft_on_coreset(coreset_path, labeled_path, run_dft_script)

        MIN_LABELED = 5  # Minimum labeled structures required to retrain
        if len(labeled_structures) < MIN_LABELED:
            raise RuntimeError(f"DFT only succeeded {len(labeled_structures)}/{CORESET_SIZE}. "
                               f"Needs at least {MIN_LABELED} structures. Check pseudopotential.")
        print(f"\n✅ DFT succeeded: {len(labeled_structures)}/{CORESET_SIZE} structures. Proceeding to retraining...")

        # 6. MERGE DATASET (with bypass if already done)
        merge_flag = os.path.join(iter_workdir, "merge_done.flag")
        if os.path.exists(merge_flag):
            print(f"\n[CACHE] ⏩ merge_done.flag found. SKIPPING [STEP 5] (Merge)...")
            print(f"  Dataset train.xyz has already been merged in a previous session.")
        else:
            print(f"\n[STEP 5] Merging new data into training set...")
            manager.merge_datasets(labeled_path)
            # Mark merge as complete to prevent duplicates in next run
            open(merge_flag, 'w').close()
            print(f"  💾 merge_done.flag written to GDrive indicating merge is complete.")


        # 7. RETRAIN ENSEMBLE WITH DYNAMIC EPOCHS
        print(f"\n[STEP 6] Retraining MACE Ensemble Models...")
        dynamic_epochs = 200 + ((current_iter - 1) * 50)
        print(f"Hyperparameter Strategy: Using {dynamic_epochs} Epochs for Iteration {current_iter}")
        
        train_sh_path = os.path.join(MACE_DIR, "train_ensemble.sh")
        finetune_py_path = os.path.join(MACE_DIR, "finetune_mace.py")
        with open(finetune_py_path, 'r') as file:
            script_data = file.read()
        
        import re
        script_data = re.sub(r'max_num_epochs=\d+', f'max_num_epochs={dynamic_epochs}', script_data)
        
        # SWA start set to 80% of total epochs
        swa_start = int(dynamic_epochs * 0.8)
        script_data = re.sub(r'swa_start_epoch=\d+', f'swa_start_epoch={swa_start}', script_data)
        
        with open(finetune_py_path, 'w') as file:
            file.write(script_data)

        # Ensure foundation model is available in local Colab storage (faster than GDrive).
        # Check GDrive first, then copy to /content/ for optimal training speed.
        foundation_model_gdrive = os.path.join(PROJECT_ROOT, "mace-mp-0-medium.model")
        foundation_model_local  = "/content/mace-mp-0-medium.model"

        if os.path.exists(foundation_model_local):
            foundation_model = foundation_model_local
            print(f"  [MODEL] Foundation model already present in local Colab: {foundation_model}")
        elif os.path.exists(foundation_model_gdrive):
            print(f"  [MODEL] Copying foundation model from GDrive to local Colab...")
            shutil.copy2(foundation_model_gdrive, foundation_model_local)
            foundation_model = foundation_model_local
            print(f"  [MODEL] ✅ Copied to {foundation_model_local}")
        else:
            raise FileNotFoundError(
                f"Foundation model not found in GDrive or local Colab!\n"
                f"  GDrive : {foundation_model_gdrive}\n"
                f"  Local  : {foundation_model_local}\n"
                f"Run the model download cell first."
            )

        # SAFE: Backup old checkpoint BEFORE deletion.
        # If retrain fails, the old model can be restored from this backup.
        ckpt_dir = os.path.join(MACE_DIR, "checkpoints")
        ckpt_backup_dir = os.path.join(MACE_DIR, f"checkpoints_iter{current_iter-1}_backup")
        if os.path.exists(ckpt_dir):
            shutil.copytree(ckpt_dir, ckpt_backup_dir, dirs_exist_ok=True)
            print(f"  [BACKUP] Old model backed up to {ckpt_backup_dir}")
            shutil.rmtree(ckpt_dir)
            print(f"  [CLEAN] Old checkpoints/ folder deleted, ready for new model.")
        
        # CRITICAL: Also delete flat model files in MACE_DIR root.
        # If not deleted, train_ensemble.sh will assume training is already completed
        # and skip all members (because *_stagetwo.model from previous iteration still exists).
        import glob as _glob
        flat_models = _glob.glob(os.path.join(MACE_DIR, "Fe_Ensemble_Member_*.model"))
        for fm in flat_models:
            os.remove(fm)
            print(f"  [CLEAN] Flat model file deleted: {os.path.basename(fm)}")
            
        # CRITICAL: Move to MACE_DIR first so train_ensemble.sh can find
        # finetune_mace.py located in the same directory.
        original_cwd = os.getcwd()
        os.chdir(MACE_DIR)
        print(f"  [CWD] Moving to {MACE_DIR} to execute training...")
        try:
            manager.retrain_ensemble(foundation_model, train_sh_path, train_sh_path)
            # Retrain succeeded: delete old backup to save GDrive space
            if os.path.exists(ckpt_backup_dir):
                shutil.rmtree(ckpt_backup_dir)
                print(f"  [CLEAN] Old model backup deleted (retrain successful).")
        except Exception as e:
            # Retrain failed: RESTORE old model from backup!
            print(f"  ❌ [ERROR] Retrain failed: {e}")
            print(f"  🔄 [RECOVERY] Restoring old model from backup...")
            if os.path.exists(ckpt_backup_dir):
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                shutil.copytree(ckpt_backup_dir, ckpt_dir)
                print(f"  ✅ [RECOVERY] Old model successfully restored to checkpoints/")
            raise RuntimeError(f"Ensemble retraining failed: {e}")
        finally:
            os.chdir(original_cwd)
            print(f"  [CWD] Returning to {original_cwd}")

        # 8. ITERATION COMPLETE
        print(f"\n✅ ITERATION {current_iter} COMPLETE!")
        save_state(current_iter + 1)

if __name__ == "__main__":
    run_automated_al()
