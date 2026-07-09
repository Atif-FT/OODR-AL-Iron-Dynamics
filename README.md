# OODR-AL: Out-of-Distribution Rejection Active Learning for MACE

This repository contains the source code, Active Learning loops, and simulation input scripts used in our publication on developing highly transferable Machine Learning Interatomic Potentials (ML-IPs) for BCC Iron using the **MACE (Message Passing Neural Network)** architecture coupled with an Out-of-Distribution Rejection Active Learning (OODR-AL) strategy.

## Repository Structure

- `src/active_learning/`: Contains the Python scripts and shell scripts used to orchestrate the OODR-AL loop, including `auto_al_loop.py` which interfaces with SLURM/LAMMPS and MACE.
- `src/lammps/`: Contains the Molecular Dynamics input scripts for LAMMPS used to validate the model under extreme conditions (e.g., `shear_deform.in` for high strain-rate shear and `thermal_shock.in` for 1800K thermal shock).
- `src/dft/`: Contains the Quantum ESPRESSO setup scripts and templates for evaluating OOD configurations.

## Requirements

The codebase relies on the following key dependencies:
- Python >= 3.9
- PyTorch >= 2.0
- MACE (`mace-torch`)
- ASE (Atomic Simulation Environment)
- LAMMPS (compiled with MACE plugin)
- Quantum ESPRESSO (for DFT labeling)

See `requirements.txt` for the Python environment details.

## Data and Pre-Trained Models

Due to file size limitations, the trained MACE models (including the baseline and the final OODR-AL comprehensive model) along with the massive DFT-labeled `.extxyz` datasets are hosted on **Zenodo**. 

Please refer to our Zenodo repository at **[https://doi.org/10.5281/zenodo.21275017](https://doi.org/10.5281/zenodo.21275017)** to download the models and place them in a `models/` directory before running the validation scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
