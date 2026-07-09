import os
import sys
from ase import Atoms
from ase.io import write

try:
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    print("Error: mp-api and pymatgen are required for this script.")
    print("Please install them using: pip install mp-api pymatgen")
    sys.exit(1)

def download_fe_structures(api_key, output_dir="mp_structures"):
    """
    Downloads crystal structures of Fe compounds from Materials Project.
    Note: These are relaxed structures. They must be perturbed and run through
    Quantum ESPRESSO to get forces and stresses for MACE training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Connecting to Materials Project using API key...")
    with MPRester(api_key) as mpr:
        # Search for pure Fe systems
        print("Searching for pure Fe systems...")
        results = mpr.materials.summary.search(
            chemsys="Fe", 
            fields=["material_id", "formula_pretty", "structure"]
        )
        
        print(f"Found {len(results)} materials in Fe system.")
        
        for idx, doc in enumerate(results):
            mp_id = doc.material_id
            formula = doc.formula_pretty
            pmg_struct = doc.structure
            
            # Convert pymatgen structure to ASE Atoms
            ase_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
            
            # Save structure as XYZ
            filename = f"{output_dir}/{formula}_{mp_id}.xyz"
            write(filename, ase_atoms, format="extxyz")
            print(f"[{idx+1}/{len(results)}] Saved {formula} ({mp_id}) to {filename}")
            
    print(f"Download complete. Structures saved to {output_dir}/")
    print("\nNext step: Run perturbation and Quantum ESPRESSO SCF on these structures to generate forces and stresses.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_mp_structures.py <YOUR_MATERIALS_PROJECT_API_KEY>")
        sys.exit(1)
        
    api_key = sys.argv[1]
    download_fe_structures(api_key)
