import numpy as np
import torch
from ase import Atoms
from ase.io import read, write

def farthest_point_sampling(features, n_select):
    """
    Performs Farthest Point Sampling (FPS) on a set of features.
    Args:
        features: np.ndarray of shape (N_samples, D)
        n_select: int, number of samples to select
    Returns:
        indices: list of selected sample indices
    """
    n_samples = len(features)
    if n_select >= n_samples:
        return list(range(n_samples))
        
    selected_indices = [np.random.randint(n_samples)]  # Start with random point
    dists = np.full(n_samples, np.inf)
    
    for _ in range(n_select - 1):
        last_selected = features[selected_indices[-1]]
        # Compute Euclidean distance from all points to the last selected point
        d = np.linalg.norm(features - last_selected, axis=1)
        # Update minimum distances to any selected point
        dists = np.minimum(dists, d)
        # Select the point that is farthest from all currently selected points
        next_index = np.argmax(dists)
        selected_indices.append(next_index)
        
    return selected_indices

def get_simple_structural_descriptors(atoms_list, r_max=6.0, n_bins=50):
    """
    Computes a self-contained structural descriptor (Radial Distribution Function - RDF)
    for each configuration. This is a robust, zero-dependency structural representation.
    """
    descriptors = []
    
    for atoms in atoms_list:
        n_atoms = len(atoms)
        distances = []
        # Get all pairwise distances (considering periodic boundary conditions)
        for i in range(n_atoms):
            # atoms.get_distances returns distances of atom i to all others
            d = atoms.get_distances(i, list(range(n_atoms)), mic=True)
            # Exclude self-distance (which is 0)
            distances.extend(d[d > 1e-5])
            
        distances = np.array(distances)
        # Compute RDF histogram as the descriptor
        hist, _ = np.histogram(distances, bins=n_bins, range=(0, r_max), density=True)
        descriptors.append(hist)
        
    return np.stack(descriptors)

def get_mace_latent_descriptors(atoms_list, mace_calculator):
    """
    Advanced descriptor extractor: extracts latent representations (node embeddings)
    from the MACE model for each configuration.
    """
    model = mace_calculator.model.to(mace_calculator.device)
    dtype = mace_calculator.models[0].dtype if hasattr(mace_calculator, 'models') else torch.float64
    
    latent_features = []
    
    # We use a hook to extract output of the interaction layers
    features_holder = {}
    
    def hook_fn(module, input, output):
        # output is usually (n_nodes, n_features) or similar
        features_holder['feats'] = output.detach().cpu().numpy()
        
    # Register hook on the final interaction layer or readout block
    # MACE models typically have 'products' or 'readouts' or 'interaction_layers'
    hook = None
    if hasattr(model, 'products'):
        # In typical MACE architectures, 'products' holds the many-body features
        hook = model.products[-1].register_forward_hook(hook_fn)
    elif hasattr(model, 'interaction_layers'):
        hook = model.interaction_layers[-1].register_forward_hook(hook_fn)
        
    for atoms in atoms_list:
        try:
            # Create MACE batch input
            batch = mace_calculator._atoms_to_batch(atoms) # Internal utility in MACECalculator
            # Run forward pass (hook will trigger)
            _ = model(batch, compute_force=False, compute_virial=False, compute_stress=False)
            
            # Extract features from holder
            feats = features_holder.get('feats')
            if feats is not None:
                # Mean-pool over atoms to get one feature vector per configuration
                config_feat = np.mean(feats, axis=0)
                latent_features.append(config_feat)
            else:
                raise ValueError("Hook did not capture features.")
        except Exception as e:
            # Fallback to RDF descriptors if MACE internal interface changes
            print(f"Warning: Failed to extract MACE latent features: {e}. Falling back to RDF.")
            hook.remove() if hook else None
            return get_simple_structural_descriptors(atoms_list)
            
    if hook:
        hook.remove()
        
    return np.stack(latent_features)

def select_coreset(input_xyz_path, output_xyz_path, n_select, mace_calc=None):
    """
    Loads candidates, extracts descriptors, performs FPS, and writes the coreset.
    """
    print(f"Loading candidate structures from {input_xyz_path}...")
    candidates = read(input_xyz_path, index=":")
    print(f"Total candidates: {len(candidates)}")
    
    if len(candidates) <= n_select:
        print("Number of candidates is smaller than n_select. Copying all to output.")
        write(output_xyz_path, candidates, format="extxyz")
        return candidates
        
    if mace_calc is not None:
        print("Extracting MACE latent descriptors...")
        features = get_mace_latent_descriptors(candidates, mace_calc)
    else:
        print("No MACE calculator provided. Extracting self-contained RDF descriptors...")
        features = get_simple_structural_descriptors(candidates)
        
    print(f"Running Farthest Point Sampling to select {n_select} configurations...")
    selected_indices = farthest_point_sampling(features, n_select)
    
    coreset = [candidates[i] for i in selected_indices]
    write(output_xyz_path, coreset, format="extxyz")
    print(f"Successfully wrote {len(coreset)} coreset structures to {output_xyz_path}")
    return coreset

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python coreset_selection.py <input.xyz> <output.xyz> <n_select>")
        sys.exit(1)
        
    in_xyz = sys.argv[1]
    out_xyz = sys.argv[2]
    num_select = int(sys.argv[3])
    
    select_coreset(in_xyz, out_xyz, num_select)
