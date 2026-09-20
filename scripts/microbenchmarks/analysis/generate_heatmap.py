#!/usr/bin/env python3
import os
import subprocess
import re
import glob
import sys
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError:
    print("Please install required libraries: pip install matplotlib seaborn numpy")
    sys.exit(1)

def get_file_targets(filepath):
    """Run beegfs-ctl to get the storage targets (OSTs) for a given file."""
    try:
        # Run beegfs-ctl --getentryinfo
        result = subprocess.run(
            ['beegfs-ctl', '--getentryinfo', filepath],
            capture_output=True, text=True, check=True
        )
        
        targets = []
        in_targets_section = False
        for line in result.stdout.splitlines():
            if "Storage Targets:" in line or "Storage targets:" in line:
                in_targets_section = True
                continue
            if in_targets_section:
                # Match lines like "+ 2" or "+ 4" which indicate the target ID
                m = re.search(r'\+\s*(\d+)', line)
                if m:
                    targets.append(int(m.group(1)))
                elif line.strip() == "" or not line.strip().startswith("+"):
                    # Reached end of targets block if there's text without a '+'
                    if targets:
                        break
        return targets
    except subprocess.CalledProcessError as e:
        print(f"Error querying {filepath}: {e}")
        return []
    except FileNotFoundError:
        print("Error: beegfs-ctl command not found. Ensure you are on a BeeGFS client.")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_heatmap.py <directory_with_test_files>")
        sys.exit(1)
        
    target_dir = sys.argv[1]
    files = glob.glob(os.path.join(target_dir, '**', '*'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    if not files:
        print(f"No files found in {target_dir}")
        sys.exit(1)
        
    print(f"Found {len(files)} files. Querying BeeGFS for target locations...")
    
    # Store data: { filename: [target_id1, target_id2, ...] }
    file_targets = {}
    all_known_targets = set()
    
    for f in files:
        t = get_file_targets(f)
        if t:
            file_targets[os.path.basename(f)] = t
            all_known_targets.update(t)
            
    if not file_targets:
        print("No target data could be extracted. Are these files on BeeGFS?")
        sys.exit(1)
        
    # Prepare matrix for heatmap
    sorted_targets = sorted(list(all_known_targets))
    sorted_files = sorted(list(file_targets.keys()))
    
    matrix = np.zeros((len(sorted_files), len(sorted_targets)))
    
    for i, fname in enumerate(sorted_files):
        targets = file_targets[fname]
        for t in targets:
            j = sorted_targets.index(t)
            matrix[i, j] = 1  # 1 indicating the file uses this OST
            
    # Generate Heatmap
    plt.figure(figsize=(10, max(6, len(sorted_files) * 0.3)))
    sns.heatmap(matrix, 
                xticklabels=[f"OST {t}" for t in sorted_targets],
                yticklabels=sorted_files,
                cmap="YlGnBu", 
                cbar=False,
                linewidths=.5)
                
    plt.title("BeeGFS File-to-OST Stripe Distribution Map")
    plt.xlabel("Storage Targets (OSTs)")
    plt.ylabel("Files")
    
    out_img = "ost_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"Heatmap successfully generated and saved to {out_img}")

if __name__ == '__main__':
    main()