#!/usr/bin/env python3
"""
run_pipeline.py

Complete pipeline for BeeGFS storage pool analysis:
1. Run workloads on HDD pool → generate Darshan logs
2. Parse Darshan logs → darshan_output_hdd/global.csv
3. Run workloads on SSD pool → generate Darshan logs
4. Parse Darshan logs → darshan_output_ssd/global.csv
5. Run analysis on both datasets → compare HDD vs SSD behavior

Usage:
    python3 run_pipeline.py --runs 5
    python3 run_pipeline.py --runs 5 --hdd-only
    python3 run_pipeline.py --runs 5 --ssd-only
    python3 run_pipeline.py --analyze-only  # Skip workloads, just analyze existing data
"""

import argparse
import os
import subprocess
import sys
import time
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Storage pool IDs (from beegfs-ctl --liststoragepools)
POOL_DEFAULT = 1
POOL_HDD = 2
POOL_SSD = 3

# Base directories
BASE_DIR = Path(__file__).parent.absolute()
BEEGFS_BASE = Path("/mnt/beegfs/advay")

# Workload directories (files are created here)
WORKLOAD_DIR_HDD = BEEGFS_BASE / "hdd" / "workloads" / "tmp"
WORKLOAD_DIR_SSD = BEEGFS_BASE / "ssd" / "workloads" / "tmp"

# Output directories (Darshan CSVs and Analysis results)
OUTPUT_BASE = BASE_DIR / "output"
DARSHAN_OUTPUT_HDD = OUTPUT_BASE / "hdd" / "darshan"
DARSHAN_OUTPUT_SSD = OUTPUT_BASE / "ssd" / "darshan"

# Analysis output directories
ANALYSIS_OUTPUT_HDD = OUTPUT_BASE / "hdd" / "analysis"
ANALYSIS_OUTPUT_SSD = OUTPUT_BASE / "ssd" / "analysis"

# Scripts
SCRIPTS_DIR = BASE_DIR / "scripts"
RUN_WORKLOADS_SCRIPT = SCRIPTS_DIR / "run_workloads.py"
PARSE_DARSHAN_SCRIPT = SCRIPTS_DIR / "parse_darshan.py"
ANALYSIS_SCRIPT = SCRIPTS_DIR / "analysis.py"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and optionally capture output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"ERROR: Command failed with exit code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            sys.exit(1)
        return result
    else:
        result = subprocess.run(cmd)
        if check and result.returncode != 0:
            print(f"ERROR: Command failed with exit code {result.returncode}")
            sys.exit(1)
        return result


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Ensured directory exists: {path}")


def set_storage_pool(directory, pool_id):
    """
    Set BeeGFS storage pool for a directory.
    This ensures all files created in this directory go to the specified pool.
    """
    directory = Path(directory)
    ensure_directory(directory)
    
    # Set storage pool using beegfs-ctl
    cmd = ["sudo", "beegfs-ctl", "--setpattern", 
           f"--storagepoolid={pool_id}", str(directory)]
    
    print(f"\nSetting storage pool {pool_id} for: {directory}")
    result = run_command(cmd, check=False, capture_output=True)
    
    if result.returncode != 0:
        print(f"WARNING: Failed to set storage pool: {result.stderr}")
        print(f"You may need to run this script with sudo or configure permissions")
        return False
    
    # Verify the pool was set
    verify_cmd = ["beegfs-ctl", "--getentryinfo", str(directory)]
    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
    
    # Check for "Storage Pool:" (capital P) in output
    if f"Storage Pool: {pool_id}" in verify_result.stdout or f"Storage pool: {pool_id}" in verify_result.stdout:
        print(f"✓ Storage pool {pool_id} set successfully")
        # Print relevant line
        for line in verify_result.stdout.split('\n'):
            if 'Storage Pool' in line or 'Storage pool' in line:
                print(f"  {line.strip()}")
        return True
    else:
        print(f"WARNING: Could not verify storage pool setting")
        print(f"Output: {verify_result.stdout}")
        return False


def verify_pools_configured():
    """Verify that HDD and SSD pools exist."""
    print("\nVerifying BeeGFS storage pools...")
    cmd = ["sudo", "beegfs-ctl", "--liststoragepools"]
    result = run_command(cmd, capture_output=True, check=False)
    
    if result.returncode != 0:
        print("ERROR: Failed to list storage pools")
        print("Make sure you have permission to run beegfs-ctl")
        return False
    
    output = result.stdout
    has_hdd = f"{POOL_HDD}" in output and "hdd" in output.lower()
    has_ssd = f"{POOL_SSD}" in output and "ssd" in output.lower()
    
    print(output)
    
    if not has_hdd or not has_ssd:
        print("\nERROR: HDD or SSD pool not found!")
        print(f"Expected pool {POOL_HDD} (HDD) and pool {POOL_SSD} (SSD)")
        print("Run pooling_scripts/configure_pools.sh first")
        return False
    
    print("✓ HDD and SSD pools are configured")
    return True


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def stage_setup_directories():
    """Stage 0: Create and configure directories with storage pools."""
    print("\n" + "="*80)
    print("STAGE 0: Setting up directories and storage pools")
    print("="*80)
    
    # Verify pools exist
    if not verify_pools_configured():
        sys.exit(1)
    
    # Create workload directories and set pools
    print("\nConfiguring HDD workload directory...")
    set_storage_pool(WORKLOAD_DIR_HDD, POOL_HDD)
    
    print("\nConfiguring SSD workload directory...")
    set_storage_pool(WORKLOAD_DIR_SSD, POOL_SSD)
    
    # Create output directories (no pool setting needed, these are local/NFS)
    ensure_directory(DARSHAN_OUTPUT_HDD)
    ensure_directory(DARSHAN_OUTPUT_SSD)
    ensure_directory(ANALYSIS_OUTPUT_HDD)
    ensure_directory(ANALYSIS_OUTPUT_SSD)
    
    print("\n✓ Directory setup complete")


def stage_run_workloads_hdd(num_runs):
    """Stage 1a: Run workloads on HDD pool."""
    print("\n" + "="*80)
    print("STAGE 1a: Running workloads on HDD pool")
    print("="*80)
    
    cmd = [
        "python3", str(RUN_WORKLOADS_SCRIPT),
        "--runs", str(num_runs),
        "--output", str(DARSHAN_OUTPUT_HDD),
        "--workload-dir", str(WORKLOAD_DIR_HDD)
    ]
    
    run_command(cmd)
    print("\n✓ HDD workloads complete")


def stage_run_workloads_ssd(num_runs):
    """Stage 1b: Run workloads on SSD pool."""
    print("\n" + "="*80)
    print("STAGE 1b: Running workloads on SSD pool")
    print("="*80)
    
    cmd = [
        "python3", str(RUN_WORKLOADS_SCRIPT),
        "--runs", str(num_runs),
        "--output", str(DARSHAN_OUTPUT_SSD),
        "--workload-dir", str(WORKLOAD_DIR_SSD)
    ]
    
    run_command(cmd)
    print("\n✓ SSD workloads complete")


def stage_parse_darshan_hdd():
    """Stage 2a: Parse HDD Darshan logs."""
    print("\n" + "="*80)
    print("STAGE 2a: Parsing HDD Darshan logs")
    print("="*80)
    
    # run_workloads.py already calls parse_darshan.py
    # This stage is now integrated, but we keep it for clarity
    print("✓ Parsing handled by run_workloads.py")


def stage_parse_darshan_ssd():
    """Stage 2b: Parse SSD Darshan logs."""
    print("\n" + "="*80)
    print("STAGE 2b: Parsing SSD Darshan logs")
    print("="*80)
    
    print("✓ Parsing handled by run_workloads.py")


def stage_analyze_hdd():
    """Stage 3a: Analyze HDD data."""
    print("\n" + "="*80)
    print("STAGE 3a: Analyzing HDD data")
    print("="*80)
    
    global_csv = DARSHAN_OUTPUT_HDD / "global.csv"
    if not global_csv.exists():
        print(f"ERROR: {global_csv} not found")
        print("Run HDD workloads first")
        return False
    
    cmd = [
        "python3", str(ANALYSIS_SCRIPT),
        "--input", str(global_csv),
        "--output-dir", str(ANALYSIS_OUTPUT_HDD)
    ]
    
    run_command(cmd)
    print(f"\n✓ HDD analysis complete")
    print(f"Results in: {ANALYSIS_OUTPUT_HDD}")
    return True


def stage_analyze_ssd():
    """Stage 3b: Analyze SSD data."""
    print("\n" + "="*80)
    print("STAGE 3b: Analyzing SSD data")
    print("="*80)
    
    global_csv = DARSHAN_OUTPUT_SSD / "global.csv"
    if not global_csv.exists():
        print(f"ERROR: {global_csv} not found")
        print("Run SSD workloads first")
        return False
    
    cmd = [
        "python3", str(ANALYSIS_SCRIPT),
        "--input", str(global_csv),
        "--output-dir", str(ANALYSIS_OUTPUT_SSD)
    ]
    
    run_command(cmd)
    print(f"\n✓ SSD analysis complete")
    print(f"Results in: {ANALYSIS_OUTPUT_SSD}")
    return True


def stage_compare_results():
    """Stage 4: Compare HDD vs SSD results."""
    print("\n" + "="*80)
    print("STAGE 4: Comparing HDD vs SSD results")
    print("="*80)
    
    hdd_csv = ANALYSIS_OUTPUT_HDD / "means_only.csv"
    ssd_csv = ANALYSIS_OUTPUT_SSD / "means_only.csv"
    
    if not hdd_csv.exists() or not ssd_csv.exists():
        print("ERROR: Analysis outputs not found")
        return
    
    print("\nYou can now compare:")
    print(f"  HDD: {ANALYSIS_OUTPUT_HDD}")
    print(f"  SSD: {ANALYSIS_OUTPUT_SSD}")
    print("\nKey files to compare:")
    print("  - means_only.csv: Mean counter values per profile")
    print("  - heatmap_all_counters.png: Visual comparison")
    print("  - bar_charts_discriminative.png: Top predictors")
    print("\nTo see differences in I/O behavior between HDD and SSD pools,")
    print("compare the same workload profile across both outputs.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run complete BeeGFS storage analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: 5 runs on both HDD and SSD
  python3 run_pipeline.py --runs 5
  
  # Only HDD pool (for testing)
  python3 run_pipeline.py --runs 3 --hdd-only
  
  # Only SSD pool (for testing)
  python3 run_pipeline.py --runs 3 --ssd-only
  
  # Skip workloads, just analyze existing data
  python3 run_pipeline.py --analyze-only
  
  # Setup directories only (configure pools)
  python3 run_pipeline.py --setup-only
        """
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per workload profile (default: 5)"
    )
    
    parser.add_argument(
        "--hdd-only",
        action="store_true",
        help="Only run HDD pool workloads"
    )
    
    parser.add_argument(
        "--ssd-only",
        action="store_true",
        help="Only run SSD pool workloads"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip workloads, only run analysis on existing data"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup directories and configure storage pools"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BeeGFS STORAGE POOL ANALYSIS PIPELINE")
    print("="*80)
    print(f"Runs per profile: {args.runs}")
    print(f"HDD pool ID: {POOL_HDD}")
    print(f"SSD pool ID: {POOL_SSD}")
    print(f"HDD workload dir: {WORKLOAD_DIR_HDD}")
    print(f"SSD workload dir: {WORKLOAD_DIR_SSD}")
    
    start_time = time.time()
    
    # Stage 0: Setup
    stage_setup_directories()
    
    if args.setup_only:
        print("\n✓ Setup complete (--setup-only mode)")
        return
    
    # Stage 1: Run workloads
    if not args.analyze_only:
        if not args.ssd_only:
            stage_run_workloads_hdd(args.runs)
        
        if not args.hdd_only:
            stage_run_workloads_ssd(args.runs)
    else:
        print("\n⊙ Skipping workloads (--analyze-only mode)")
    
    # Stage 2: Parse (integrated into run_workloads.py)
    # Stage 3: Analyze
    hdd_ok = ssd_ok = False
    
    if not args.ssd_only:
        hdd_ok = stage_analyze_hdd()
    
    if not args.hdd_only:
        ssd_ok = stage_analyze_ssd()
    
    # Stage 4: Compare
    if hdd_ok and ssd_ok:
        stage_compare_results()
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"PIPELINE COMPLETE in {elapsed:.1f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()
