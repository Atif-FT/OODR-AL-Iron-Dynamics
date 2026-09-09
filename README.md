# OODR-AL: Active Learning of Equivariant Interatomic Potentials for Extreme Non-Equilibrium Iron Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22671770.svg)](https://doi.org/10.5281/zenodo.22671770)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete source code, active-learning orchestration pipeline, DFT calculation scripts, and LAMMPS molecular dynamics production inputs for our study:

> **Active learning of equivariant interatomic potentials for extreme non equilibrium iron dynamics**  
> *Scientific Reports* (Springer Nature)  
> Authors: Ahmad Atif Fikri, Heru Suryanto, Avita Ayu Permanasari, Rio Anugrah Vidyanto, Ahmad Al Kafi, Poespitasari Hazanah Ndaru, Mohd Sayuti Ab Karim, Dani Harmanto

---

## Key Features

1. **Equivariant Graph Neural Network Potentials:** Built upon the higher-order equivariant MACE architecture with multi-body message passing (nu=3).
2. **Deep Ensemble Uncertainty Trigger:** Autonomous detection of out-of-distribution (OOD) configurations using inter-model force variance across 4 independent MACE models.
3. **Latent-Space Farthest Point Sampling (FPS) & Physics Filters:** Maximizes configurational diversity in invariant feature space while rejecting unphysical structures (density limits 5.0 to 9.5 g/cm^3, interatomic distance >= 1.8 A).
4. **Stress-Weighted & Static Anchor Training:** Prevents catastrophic forgetting of 0 K equilibrium bulk/shear elastic constants while achieving sub-chemical accuracy on extreme 1800 K non-equilibrium dynamics.

---

## Directory Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
└── src/
    ├── active_learning/        # OOD detection, ensemble management, coreset FPS
    │   ├── al_manager.py
    │   ├── auto_al_loop.py
    │   ├── bulk_filter.py
    │   ├── coreset_selection.py
    │   ├── ood_detector.py
    │   └── train_ensemble.sh
    ├── dft/                    # Quantum ESPRESSO spin-polarized DFT templates & runners
    │   ├── run_dft.py
    │   ├── scf_template.in
    │   └── generate_seed_dataset.py
    ├── lammps/                 # LAMMPS MD scripts for extreme shock & shear simulations
    │   ├── thermal_shock.in    # 1800 K thermal shock NVE relaxation
    │   └── shear_deform.in     # 1800 K high strain-rate shear (10^9 s^-1)
    └── visualization/          # Standalone publication figure generator
        └── generate_perfected_figures.py
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Atif-FT/OODR-AL-Iron-Dynamics.git
   cd OODR-AL-Iron-Dynamics
   ```

2. **Create a conda environment and install dependencies:**
   ```bash
   conda create -n oodr_al python=3.10 -y
   conda activate oodr_al
   pip install -r requirements.txt
   pip install mace-torch ase pymatgen
   ```

---

## Data and Trained Models

Due to size considerations, the complete Extended XYZ datasets (234 DFT training configurations and 20 OOD validation snapshots) and trained PyTorch/MACE model weights are permanently archived on **Zenodo**:

- **Zenodo DOI:** [https://doi.org/10.5281/zenodo.22671770](https://doi.org/10.5281/zenodo.22671770)

Download the dataset and models from Zenodo and place them in `datasets/` and `models/` respectively.

---

## Reproducing Figures

To regenerate all 8 publication-grade figures (300 DPI):
```bash
python src/visualization/generate_perfected_figures.py
```

---

## Citation

If you find this codebase or dataset useful in your research, please cite:
```bibtex
@article{fikri2026active,
  title={Active learning of equivariant interatomic potentials for extreme non equilibrium iron dynamics},
  author={Fikri, Ahmad Atif and Suryanto, Heru and Permanasari, Avita Ayu and Vidyanto, Rio Anugrah and Al Kafi, Ahmad and Ndaru, Poespitasari Hazanah and Ab Karim, Mohd Sayuti and Harmanto, Dani},
  journal={Scientific Reports},
  year={2026}
}
```

---

## License

This project is open-source under the [MIT License](LICENSE).
