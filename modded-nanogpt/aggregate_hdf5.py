#!/usr/bin/env python3
"""
Aggregate evaluation results into an HDF5 file optimized for per-token queries.

Usage:
    python aggregate_hdf5.py                           # Create aggregated.h5
    python aggregate_hdf5.py --output results.h5       # Custom filename

Structure:
    /metadata/doc_names         - string array
    /metadata/doc_lengths       - int array  
    /metadata/eval_steps        - int array
    /metadata/seeds             - int array
    /losses/{doc_name}/token_{idx}  - float16 array [n_seeds, n_steps]

Querying:
    import h5py
    with h5py.File('aggregated.h5', 'r') as f:
        # Get losses for a specific token across all seeds and steps
        losses = f['losses/arithmetic.txt/token_5'][:]  # [n_seeds, n_steps]
"""

import argparse
import glob
import os
import numpy as np
import h5py
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Aggregate results into HDF5")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing log files")
    parser.add_argument("--output", default="aggregated.h5", help="Output filename")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 299], help="Seed range")
    args = parser.parse_args()
    
    start_seed, end_seed = args.seeds
    
    # Find all eval_losses files
    pattern = os.path.join(args.logs_dir, "seed*_eval_losses.pt")
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} result files")
    
    if not files:
        print("No results found.")
        return
    
    # Filter by seed range and extract seed numbers
    seed_files = []
    for fpath in files:
        fname = os.path.basename(fpath)
        seed = int(fname.split("_")[0].replace("seed", ""))
        if start_seed <= seed <= end_seed:
            seed_files.append((seed, fpath))
    
    seed_files.sort(key=lambda x: x[0])
    seeds = [s for s, _ in seed_files]
    
    print(f"Processing {len(seeds)} seeds (range {min(seeds)}-{max(seeds)})")
    
    # Load first file to get metadata
    first_data = torch.load(seed_files[0][1], weights_only=False)
    doc_names = first_data['doc_names']
    doc_lengths = first_data['doc_lengths']
    eval_steps = sorted(first_data['eval_steps'])
    num_docs = len(doc_names)
    num_steps = len(eval_steps)
    num_seeds = len(seeds)
    
    # Create seed index mapping (seed number -> index in array)
    seed_to_idx = {s: i for i, s in enumerate(seeds)}
    
    print(f"Documents: {doc_names}")
    print(f"Steps: {num_steps}, Seeds: {num_seeds}")
    
    # Pre-allocate arrays for each (doc, token) -> [n_seeds, n_steps]
    # This avoids multiple passes through the data
    print("\nAllocating arrays...")
    token_losses = {}
    for doc_idx, (doc_name, doc_len) in enumerate(zip(doc_names, doc_lengths)):
        token_losses[doc_name] = np.zeros((doc_len, num_seeds, num_steps), dtype=np.float16)
    
    # Process each seed file
    print("Loading data...")
    for seed, fpath in tqdm(seed_files, desc="Processing seeds"):
        data = torch.load(fpath, weights_only=False)
        results = data['results']
        seed_idx = seed_to_idx[seed]
        
        for step_idx, step in enumerate(eval_steps):
            if step not in results:
                continue
            for doc_idx, doc_name in enumerate(doc_names):
                if doc_idx >= len(results[step]):
                    continue
                doc_losses = results[step][doc_idx].numpy().astype(np.float16)
                actual_len = min(len(doc_losses), doc_lengths[doc_idx])
                token_losses[doc_name][:actual_len, seed_idx, step_idx] = doc_losses[:actual_len]
    
    # Create HDF5 file
    output_path = os.path.join(args.logs_dir, args.output)
    print(f"\nWriting to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        meta = f.create_group('metadata')
        meta.create_dataset('seeds', data=np.array(seeds, dtype=np.int32))
        meta.create_dataset('eval_steps', data=np.array(eval_steps, dtype=np.int32))
        meta.create_dataset('doc_lengths', data=np.array(doc_lengths, dtype=np.int32))
        
        # Store doc names as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        meta.create_dataset('doc_names', data=doc_names, dtype=dt)
        
        # Create losses group organized by doc_name/token_idx
        losses_grp = f.create_group('losses')
        
        for doc_name, doc_len in zip(doc_names, doc_lengths):
            doc_grp = losses_grp.create_group(doc_name)
            
            for token_idx in range(doc_len):
                # Shape: [n_seeds, n_steps]
                doc_grp.create_dataset(
                    f'token_{token_idx}',
                    data=token_losses[doc_name][token_idx],
                    compression='gzip',
                    compression_opts=4
                )
    
    # Print file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")
    
    # Check for missing seeds
    expected = set(range(start_seed, end_seed + 1))
    found = set(seeds)
    missing = sorted(expected - found)
    if missing:
        print(f"Missing seeds ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main()
