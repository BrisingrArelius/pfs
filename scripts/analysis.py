#!/usr/bin/env python3
"""
analysis.py

Analyzes Darshan counter data from global.csv to:
1. Compute statistics (mean, std, min, max, CV) across multiple runs per profile
2. Compare HDD vs SSD storage performance
3. Generate visualizations:
   - Interleaved heatmap (HDD and SSD profiles side-by-side)
   - I/O bandwidth and latency analysis (side-by-side HDD vs SSD)
   - Performance gain quantification
   - PCA scatter plot for workload clustering

Usage:
    # Single storage analysis
    python analysis.py --input ./output/ssd/global.csv --output ./analysis_output/ssd
    
    # HDD vs SSD comparison
    python analysis.py --hdd ./output/hdd/global.csv --ssd ./output/ssd/global.csv --output ./analysis_output/comparison

Options:
    --cv-threshold: Maximum coefficient of variation for "stable" counters (default: 0.2)
    --top-n: Number of top discriminative counters to visualize (default: 10)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

CV_THRESHOLD = 0.2  # Coefficient of variation threshold for stable counters
TOP_N_COUNTERS = 10  # Number of top counters to visualize in bar charts

# Counters to exclude from analysis (metadata only, not I/O metrics)
EXCLUDE_COUNTERS = [
    'timestamp',      # Parser-added metadata (not from Darshan)
    'label',          # Run label (e.g., "write_heavy_run1")
    'modules_used',   # Which modules were parsed (e.g., "posix")
]

# Patterns of counters to exclude (raw values without context)
# These are filtered out AFTER computing derived metrics
EXCLUDE_PATTERNS = [
    '_START_TIMESTAMP',  # Raw timestamps (we compute duration instead)
    '_END_TIMESTAMP',    # Raw timestamps (we compute duration instead)
]

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Darshan counter data and generate visualizations."
    )
    
    # Input options - either single input or HDD+SSD comparison
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to single global.csv file"
    )
    input_group.add_argument(
        "--hdd",
        help="Path to HDD global.csv file (for comparison mode)"
    )
    
    parser.add_argument(
        "--ssd",
        help="Path to SSD global.csv file (required with --hdd)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./analysis_output",
        help="Output directory for analysis results (default: ./analysis_output)"
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=CV_THRESHOLD,
        help=f"Coefficient of variation threshold for stable counters (default: {CV_THRESHOLD})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_COUNTERS,
        help=f"Number of top discriminative counters to visualize (default: {TOP_N_COUNTERS})"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.hdd and not args.ssd:
        parser.error("--ssd is required when using --hdd")
    if args.ssd and not args.hdd:
        parser.error("--hdd is required when using --ssd")
    
    if args.input and not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")
    if args.hdd and not os.path.isfile(args.hdd):
        parser.error(f"HDD file not found: {args.hdd}")
    if args.ssd and not os.path.isfile(args.ssd):
        parser.error(f"SSD file not found: {args.ssd}")
    
    return args


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def compute_derived_metrics(df):
    """
    Compute derived I/O metrics from raw Darshan counters.
    
    Derived metrics:
    - Duration windows: END_TIMESTAMP - START_TIMESTAMP for each operation type
    - Bandwidth: BYTES / duration (MB/s)
    - Latency: duration / operations (ms per operation)
    - Mean operation sizes: BYTES / OPS
    - Ratios: Sequential/total, seek rate, etc.
    
    Returns:
        df: DataFrame with new derived metric columns added
    """
    print("Computing derived metrics...")
    
    # Helper function to safely add a computed column
    def safe_compute(result_col, numerator_col, denominator_col, scale=1.0):
        if numerator_col in df.columns and denominator_col in df.columns:
            df[result_col] = (df[numerator_col] * scale) / df[denominator_col].replace(0, np.nan)
    
    # Duration windows (in seconds)
    timestamp_pairs = [
        ('POSIX_F_OPEN_START_TIMESTAMP', 'POSIX_F_OPEN_END_TIMESTAMP', 'POSIX_OPEN_DURATION'),
        ('POSIX_F_READ_START_TIMESTAMP', 'POSIX_F_READ_END_TIMESTAMP', 'POSIX_READ_DURATION'),
        ('POSIX_F_WRITE_START_TIMESTAMP', 'POSIX_F_WRITE_END_TIMESTAMP', 'POSIX_WRITE_DURATION'),
        ('POSIX_F_CLOSE_START_TIMESTAMP', 'POSIX_F_CLOSE_END_TIMESTAMP', 'POSIX_CLOSE_DURATION'),
        ('POSIX_F_META_START_TIMESTAMP', 'POSIX_F_META_END_TIMESTAMP', 'POSIX_META_DURATION'),
        ('MPIIO_F_OPEN_START_TIMESTAMP', 'MPIIO_F_OPEN_END_TIMESTAMP', 'MPIIO_OPEN_DURATION'),
        ('MPIIO_F_READ_START_TIMESTAMP', 'MPIIO_F_READ_END_TIMESTAMP', 'MPIIO_READ_DURATION'),
        ('MPIIO_F_WRITE_START_TIMESTAMP', 'MPIIO_F_WRITE_END_TIMESTAMP', 'MPIIO_WRITE_DURATION'),
    ]
    
    for start_col, end_col, duration_col in timestamp_pairs:
        if start_col in df.columns and end_col in df.columns:
            df[duration_col] = df[end_col] - df[start_col]
    
    # Constants
    MB = 1024 * 1024
    MS_PER_SEC = 1000
    
    # =========================================================================
    # BANDWIDTH METRICS (MB/s)
    # =========================================================================
    safe_compute('POSIX_READ_BW_MBps', 'POSIX_BYTES_READ', 'POSIX_READ_DURATION', scale=1/MB)
    safe_compute('POSIX_WRITE_BW_MBps', 'POSIX_BYTES_WRITTEN', 'POSIX_WRITE_DURATION', scale=1/MB)
    
    # Total I/O bandwidth
    if ('POSIX_BYTES_READ' in df.columns and 'POSIX_BYTES_WRITTEN' in df.columns and
        'POSIX_READ_DURATION' in df.columns and 'POSIX_WRITE_DURATION' in df.columns):
        total_bytes = df['POSIX_BYTES_READ'] + df['POSIX_BYTES_WRITTEN']
        total_duration = df['POSIX_READ_DURATION'] + df['POSIX_WRITE_DURATION']
        df['POSIX_TOTAL_BW_MBps'] = (total_bytes / MB) / total_duration.replace(0, np.nan)
    
    # =========================================================================
    # LATENCY METRICS (milliseconds per operation)
    # =========================================================================
    safe_compute('POSIX_READ_LATENCY_ms', 'POSIX_READ_DURATION', 'POSIX_READS', scale=MS_PER_SEC)
    safe_compute('POSIX_WRITE_LATENCY_ms', 'POSIX_WRITE_DURATION', 'POSIX_WRITES', scale=MS_PER_SEC)
    safe_compute('POSIX_OPEN_LATENCY_ms', 'POSIX_OPEN_DURATION', 'POSIX_OPENS', scale=MS_PER_SEC)
    safe_compute('POSIX_CLOSE_LATENCY_ms', 'POSIX_CLOSE_DURATION', 'POSIX_OPENS', scale=MS_PER_SEC)
    safe_compute('POSIX_META_LATENCY_ms', 'POSIX_META_DURATION', 'POSIX_STATS', scale=MS_PER_SEC)
    
    # =========================================================================
    # LEGACY METRICS (kept for compatibility)
    # =========================================================================
    safe_compute('POSIX_READ_DENSITY', 'POSIX_BYTES_READ', 'POSIX_READ_DURATION')
    safe_compute('POSIX_WRITE_DENSITY', 'POSIX_BYTES_WRITTEN', 'POSIX_WRITE_DURATION')
    safe_compute('POSIX_MEAN_READ_SIZE', 'POSIX_BYTES_READ', 'POSIX_READS')
    safe_compute('POSIX_MEAN_WRITE_SIZE', 'POSIX_BYTES_WRITTEN', 'POSIX_WRITES')
    safe_compute('POSIX_SEQ_READ_RATIO', 'POSIX_SEQ_READS', 'POSIX_READS')
    safe_compute('POSIX_SEQ_WRITE_RATIO', 'POSIX_SEQ_WRITES', 'POSIX_WRITES')
    
    # Seek rate
    if 'POSIX_SEEKS' in df.columns and 'POSIX_READS' in df.columns and 'POSIX_WRITES' in df.columns:
        total_ops = df['POSIX_READS'] + df['POSIX_WRITES']
        df['POSIX_SEEK_RATE'] = df['POSIX_SEEKS'] / total_ops.replace(0, np.nan)
    
    derived_count = sum(1 for c in df.columns if any(x in c for x in ['DURATION', 'BW_', 'LATENCY_', 'DENSITY', 'MEAN_', 'RATIO', 'RATE']))
    print(f"  Added {derived_count} derived metrics (bandwidth, latency, ratios)")
    
    return df


def load_and_preprocess(csv_path):
    """
    Load global.csv and extract profile names from labels.
    
    Returns:
        df: DataFrame with 'profile' column added
        counter_cols: List of counter column names
    """
    df = pd.read_csv(csv_path)
    
    # Extract profile name from label (format: profile_runN)
    df['profile'] = df['label'].str.rsplit('_run', n=1).str[0]
    
    # Compute derived metrics before filtering
    df = compute_derived_metrics(df)
    
    # Build set of columns to exclude for faster lookup
    exclude_set = set(EXCLUDE_COUNTERS) | {'profile'}
    
    # Single pass: filter columns efficiently
    counter_cols = [
        c for c in df.columns 
        if c not in exclude_set
        and not any(pattern in c for pattern in EXCLUDE_PATTERNS)
        and df[c].notna().any() 
        and (df[c] != 0).any()
    ]
    
    print(f"Loaded {len(df)} rows, {len(df['profile'].unique())} profiles")
    print(f"Valid counters: {len(counter_cols)}")
    
    return df, counter_cols


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(df, counter_cols):
    """
    Compute mean, std, min, max, and coefficient of variation for each counter
    across runs per profile.
    
    Returns:
        stats_df: DataFrame with MultiIndex columns (counter, statistic)
    """
    print("\nComputing statistics across runs...")
    
    # Build aggregation dict once
    agg_dict = {col: ['mean', 'std', 'min', 'max', 'count'] for col in counter_cols}
    
    # Single groupby operation
    stats = df.groupby('profile').agg(agg_dict)
    
    # Batch compute CV for all counters to avoid DataFrame fragmentation
    cv_dict = {
        (col, 'cv'): stats[(col, 'std')] / stats[(col, 'mean')].replace(0, np.nan)
        for col in counter_cols
    }
    cv_df = pd.DataFrame(cv_dict)
    stats = pd.concat([stats, cv_df], axis=1)
    
    return stats


def identify_stable_counters(stats, counter_cols, cv_threshold):
    """
    Identify counters with low coefficient of variation (stable across runs).
    
    Returns:
        stable_counters: List of counter names with CV < threshold for ALL profiles
    """
    print(f"\nIdentifying stable counters (CV < {cv_threshold})...")
    
    # Vectorized: check all at once
    stable = [
        col for col in counter_cols
        if (cv_vals := stats[(col, 'cv')].dropna()).size > 0 and cv_vals.max() < cv_threshold
    ]
    
    print(f"Stable counters: {len(stable)} out of {len(counter_cols)}")
    
    return stable


def identify_discriminative_counters(stats, counter_cols, top_n):
    """
    Identify counters with high variance across profiles (good discriminators).
    Uses coefficient of variation of the means across profiles.
    
    Returns:
        top_counters: List of top N most discriminative counter names
    """
    print(f"\nIdentifying top {top_n} discriminative counters...")
    
    # Vectorized computation
    means_df = stats.loc[:, [(col, 'mean') for col in counter_cols]]
    scores = {
        col: (std / mean if (mean := means_df[(col, 'mean')].mean()) != 0 else 0)
        for col in counter_cols
        for std in [means_df[(col, 'mean')].std()]
    }
    
    # Sort by score descending
    sorted_counters = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"Top {top_n} discriminative counters:")
    for i, (counter, score) in enumerate(sorted_counters, 1):
        print(f"  {i}. {counter}: {score:.3f}")
    
    return [c for c, _ in sorted_counters]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_heatmap(stats, counter_cols, output_path, title, cmap='YlOrRd'):
    """
    Generate heatmap of normalized counter means across workloads.
    
    Args:
        stats: Statistics DataFrame with MultiIndex columns
        counter_cols: List of counter names to include
        output_path: Full path to save the plot
        title: Plot title
        cmap: Color map (default: 'YlOrRd')
    """
    if not counter_cols:
        print(f"  No counters to plot for {title}")
        return
    
    # Extract means efficiently using dict comprehension
    print(f"\nGenerating heatmap: {title}...")
    mean_data = {col: stats[(col, 'mean')] for col in counter_cols}
    df_means = pd.DataFrame(mean_data)
    
    # Normalize to 0-1 scale (inline scaler to avoid variable)
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(df_means),
        index=df_means.index,
        columns=df_means.columns
    )
    
    # Plot heatmap
    fig_width = max(12, len(counter_cols) * 0.3)
    plt.figure(figsize=(fig_width, 8))
    sns.heatmap(
        normalized,
        cmap=cmap,
        cbar_kws={'label': 'Normalized Value (0-1)'},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Counters', fontsize=12)
    plt.ylabel('Workload Profiles', fontsize=12)
    plt.xticks(rotation=90, fontsize=max(8, min(12, 120 // len(counter_cols))))
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_bar_charts(stats, top_counters, output_dir):
    """
    Generate bar charts for top discriminative counters.
    """
    print("\nGenerating bar charts for discriminative counters...")
    
    n_counters = len(top_counters)
    n_cols = 3
    n_rows = (n_counters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_counters > 1 else [axes]
    
    for i, counter in enumerate(top_counters):
        ax = axes[i]
        
        means = stats[(counter, 'mean')]
        stds = stats[(counter, 'std')]
        
        x_pos = np.arange(len(means))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(counter, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Top Discriminative Counters Across Workloads', fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bar_charts_discriminative.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_pca(stats, counter_cols, output_dir):
    """
    Generate PCA scatter plot to visualize workload clustering.
    """
    print("\nGenerating PCA scatter plot...")
    
    # Extract means for all profiles and counters
    mean_data = {}
    for col in counter_cols:
        mean_data[col] = stats[(col, 'mean')]
    
    df_means = pd.DataFrame(mean_data).fillna(0)
    
    # Normalize
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df_means)
    
    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(normalized)
    
    # Plot
    plt.figure(figsize=(12, 8))
    profiles = df_means.index.tolist()
    
    # Use different colors for each profile
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
    
    for i, profile in enumerate(profiles):
        plt.scatter(reduced[i, 0], reduced[i, 1], 
                   color=colors[i], s=200, alpha=0.7, 
                   edgecolors='black', linewidth=1.5,
                   label=profile)
        plt.annotate(profile, (reduced[i, 0], reduced[i, 1]), 
                    fontsize=10, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('PCA: Workload Clustering Based on I/O Counters', fontsize=16, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pca_clustering.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# EXPORT STATISTICS
# =============================================================================

def export_statistics(stats, counter_cols, output_dir):
    """
    Export statistics to CSV for further analysis.
    """
    print("\nExporting statistics to CSV...")
    
    # Flatten MultiIndex columns efficiently using pandas operations
    # Reset index to make 'profile' a column
    stats_flat = stats.copy()
    stats_flat.index.name = 'profile'
    stats_flat = stats_flat.reset_index()
    
    # Flatten column MultiIndex to single level with underscore separator
    stats_flat.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in stats_flat.columns]
    
    output_path = os.path.join(output_dir, 'statistics.csv')
    stats_flat.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Export summary with just means (use dict comprehension for speed)
    mean_data = {col: stats[(col, 'mean')] for col in counter_cols}
    df_means = pd.DataFrame(mean_data)
    df_means.index.name = 'profile'
    
    output_path_means = os.path.join(output_dir, 'means_only.csv')
    df_means.to_csv(output_path_means)
    print(f"  Saved: {output_path_means}")


# =============================================================================
# HDD vs SSD COMPARISON
# =============================================================================

def _plot_side_by_side_bars(ax, stats_hdd, stats_ssd, metric, title, ylabel, color, use_log=False):
    """
    Helper function to create side-by-side bar chart for HDD vs SSD comparison.
    Returns True if plot was created, False if metric not available.
    """
    if metric not in [col for col, _ in stats_hdd.columns]:
        ax.text(0.5, 0.5, f'{metric}\nnot available', 
               ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return False
    
    hdd_vals = stats_hdd[(metric, 'mean')].dropna()
    ssd_vals = stats_ssd[(metric, 'mean')].dropna()
    
    common_profiles = sorted(set(hdd_vals.index) & set(ssd_vals.index))
    if not common_profiles:
        ax.text(0.5, 0.5, 'No common profiles', 
               ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return False
    
    hdd_data = [hdd_vals.loc[p] for p in common_profiles]
    ssd_data = [ssd_vals.loc[p] for p in common_profiles]
    
    x = np.arange(len(common_profiles))
    width = 0.35
    
    ax.bar(x - width/2, hdd_data, width, label='HDD', alpha=0.8, color='orange')
    ax.bar(x + width/2, ssd_data, width, label='SSD', alpha=0.8, color=color)
    
    ax.set_xlabel('Workload Profile', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_profiles, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Auto log-scale for wide value ranges
    if use_log and len(hdd_data + ssd_data) > 0:
        all_vals = hdd_data + ssd_data
        if max(all_vals) / min([v for v in all_vals if v > 0] + [1]) > 100:
            ax.set_yscale('log')
            ax.set_ylabel(f'{ylabel} (log scale)', fontsize=11)
    
    return True


def plot_interleaved_heatmap(stats_hdd, stats_ssd, counter_cols, output_path):
    """
    Generate interleaved heatmap with HDD and SSD profiles side-by-side.
    Format: profile_1gb_hdd, profile_1gb_ssd, profile_10gb_hdd, profile_10gb_ssd, ...
    """
    print("\nGenerating interleaved HDD vs SSD heatmap...")
    
    # Extract means
    hdd_means = pd.DataFrame({col: stats_hdd[(col, 'mean')] for col in counter_cols})
    ssd_means = pd.DataFrame({col: stats_ssd[(col, 'mean')] for col in counter_cols})
    
    # Create interleaved DataFrame
    interleaved_rows = []
    interleaved_index = []
    
    # Get all unique profiles (without storage type suffix)
    all_profiles = sorted(set(hdd_means.index) | set(ssd_means.index))
    
    for profile in all_profiles:
        if profile in hdd_means.index:
            interleaved_rows.append(hdd_means.loc[profile])
            interleaved_index.append(f"{profile}_hdd")
        if profile in ssd_means.index:
            interleaved_rows.append(ssd_means.loc[profile])
            interleaved_index.append(f"{profile}_ssd")
    
    df_interleaved = pd.DataFrame(interleaved_rows, index=interleaved_index)
    
    # Normalize each counter to 0-1 scale
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(df_interleaved),
        index=df_interleaved.index,
        columns=df_interleaved.columns
    )
    
    # Plot heatmap
    fig_width = max(16, len(counter_cols) * 0.35)
    fig_height = max(12, len(interleaved_index) * 0.4)
    plt.figure(figsize=(fig_width, fig_height))
    
    sns.heatmap(
        normalized,
        cmap='YlOrRd',
        cbar_kws={'label': 'Normalized Value (0-1)'},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5
    )
    
    plt.title('HDD vs SSD Performance: Interleaved Heatmap', fontsize=18, pad=20)
    plt.xlabel('Counters', fontsize=14)
    plt.ylabel('Workload Profiles (HDD / SSD)', fontsize=14)
    plt.xticks(rotation=90, fontsize=max(8, min(12, 150 // len(counter_cols))))
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_bandwidth_comparison(stats_hdd, stats_ssd, output_dir):
    """
    Generate side-by-side bandwidth comparison plots for HDD vs SSD.
    Shows Read BW, Write BW, and Total BW.
    """
    print("\nGenerating bandwidth comparison plots...")
    
    bw_metrics = [
        ('POSIX_READ_BW_MBps', 'Read Bandwidth', 'green'),
        ('POSIX_WRITE_BW_MBps', 'Write Bandwidth', 'blue'),
        ('POSIX_TOTAL_BW_MBps', 'Total Bandwidth', 'purple')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (metric, title, color) in enumerate(bw_metrics):
        _plot_side_by_side_bars(axes[idx], stats_hdd, stats_ssd, metric, title, 
                               'Bandwidth (MB/s)', color, use_log=False)
    
    plt.suptitle('I/O Bandwidth: HDD vs SSD Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bandwidth_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_latency_comparison(stats_hdd, stats_ssd, output_dir):
    """
    Generate side-by-side latency comparison plots for HDD vs SSD.
    Shows Read, Write, Open, and Close latencies.
    """
    print("\nGenerating latency comparison plots...")
    
    latency_metrics = [
        ('POSIX_READ_LATENCY_ms', 'Read Latency', 'green'),
        ('POSIX_WRITE_LATENCY_ms', 'Write Latency', 'blue'),
        ('POSIX_OPEN_LATENCY_ms', 'Open Latency', 'red'),
        ('POSIX_CLOSE_LATENCY_ms', 'Close Latency', 'purple')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, title, color) in enumerate(latency_metrics):
        _plot_side_by_side_bars(axes[idx], stats_hdd, stats_ssd, metric, title,
                               'Latency (ms)', color, use_log=True)
    
    plt.suptitle('Operation Latency: HDD vs SSD Comparison', fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_performance_gains(stats_hdd, stats_ssd, output_dir):
    """
    Quantify and visualize performance gains of SSD over HDD.
    Shows speedup factors for bandwidth and latency reduction.
    """
    print("\nGenerating performance gain analysis...")
    
    # Bandwidth speedup (SSD_BW / HDD_BW)
    bw_metrics = ['POSIX_READ_BW_MBps', 'POSIX_WRITE_BW_MBps', 'POSIX_TOTAL_BW_MBps']
    
    # Latency reduction (HDD_lat / SSD_lat - shows how many times faster SSD is)
    lat_metrics = ['POSIX_READ_LATENCY_ms', 'POSIX_WRITE_LATENCY_ms', 
                   'POSIX_OPEN_LATENCY_ms', 'POSIX_CLOSE_LATENCY_ms']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Bandwidth Speedup ---
    speedup_data = []
    labels = []
    
    for metric in bw_metrics:
        if metric not in [col for col, _ in stats_hdd.columns]:
            continue
        
        hdd_vals = stats_hdd[(metric, 'mean')].dropna()
        ssd_vals = stats_ssd[(metric, 'mean')].dropna()
        common = sorted(set(hdd_vals.index) & set(ssd_vals.index))
        
        for profile in common:
            if hdd_vals.loc[profile] > 0:
                speedup = ssd_vals.loc[profile] / hdd_vals.loc[profile]
                speedup_data.append(speedup)
                labels.append(f"{profile}\n{metric.replace('POSIX_', '').replace('_BW_MBps', '')}")
    
    if speedup_data:
        x = np.arange(len(speedup_data))
        colors = ['green' if s > 1 else 'red' for s in speedup_data]
        ax1.bar(x, speedup_data, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Baseline (HDD)')
        ax1.set_xlabel('Profile & Metric', fontsize=11)
        ax1.set_ylabel('Speedup Factor (SSD / HDD)', fontsize=11)
        ax1.set_title('Bandwidth Speedup: SSD vs HDD', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # --- Latency Reduction ---
    reduction_data = []
    lat_labels = []
    
    for metric in lat_metrics:
        if metric not in [col for col, _ in stats_hdd.columns]:
            continue
        
        hdd_vals = stats_hdd[(metric, 'mean')].dropna()
        ssd_vals = stats_ssd[(metric, 'mean')].dropna()
        common = sorted(set(hdd_vals.index) & set(ssd_vals.index))
        
        for profile in common:
            if ssd_vals.loc[profile] > 0:
                reduction = hdd_vals.loc[profile] / ssd_vals.loc[profile]
                reduction_data.append(reduction)
                lat_labels.append(f"{profile}\n{metric.replace('POSIX_', '').replace('_LATENCY_ms', '')}")
    
    if reduction_data:
        x = np.arange(len(reduction_data))
        colors = ['green' if r > 1 else 'red' for r in reduction_data]
        ax2.bar(x, reduction_data, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Baseline (HDD)')
        ax2.set_xlabel('Profile & Metric', fontsize=11)
        ax2.set_ylabel('Latency Reduction Factor (HDD / SSD)', fontsize=11)
        ax2.set_title('Latency Improvement: SSD vs HDD', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(lat_labels, rotation=45, ha='right', fontsize=7)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance Gains: SSD vs HDD', fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'performance_gains.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==========================================================================
    # COMPARISON MODE: HDD vs SSD
    # ==========================================================================
    if args.hdd and args.ssd:
        print("="*70)
        print("COMPARISON MODE: HDD vs SSD Analysis")
        print("="*70)
        print(f"HDD Input:  {args.hdd}")
        print(f"SSD Input:  {args.ssd}")
        print(f"Output:     {args.output_dir}")
        print(f"CV threshold: {args.cv_threshold}")
        print(f"Top N counters: {args.top_n}")
        
        # Load both datasets
        print("\n--- Loading HDD data ---")
        df_hdd, counters_hdd = load_and_preprocess(args.hdd)
        stats_hdd = compute_statistics(df_hdd, counters_hdd)
        
        print("\n--- Loading SSD data ---")
        df_ssd, counters_ssd = load_and_preprocess(args.ssd)
        stats_ssd = compute_statistics(df_ssd, counters_ssd)
        
        # Use intersection of counters (only those present in both)
        common_counters = sorted(set(counters_hdd) & set(counters_ssd))
        print(f"\nCommon counters: {len(common_counters)} "
              f"(HDD: {len(counters_hdd)}, SSD: {len(counters_ssd)})")
        
        # Generate comparison visualizations
        plot_interleaved_heatmap(
            stats_hdd, stats_ssd, common_counters,
            os.path.join(args.output_dir, 'heatmap_hdd_ssd_interleaved.png')
        )
        
        plot_bandwidth_comparison(stats_hdd, stats_ssd, args.output_dir)
        plot_latency_comparison(stats_hdd, stats_ssd, args.output_dir)
        plot_performance_gains(stats_hdd, stats_ssd, args.output_dir)
        
        # Export both datasets
        print("\n--- Exporting HDD statistics ---")
        export_statistics(stats_hdd, counters_hdd, args.output_dir)
        
        print("\n--- Exporting SSD statistics ---")
        # Rename files to avoid overwrite
        ssd_output_dir = os.path.join(args.output_dir, 'ssd_stats')
        os.makedirs(ssd_output_dir, exist_ok=True)
        export_statistics(stats_ssd, counters_ssd, ssd_output_dir)
        
        print("\n" + "="*70)
        print("HDD vs SSD Comparison Complete!")
        print(f"Results saved to: {args.output_dir}")
        print("="*70)
        
    # ==========================================================================
    # SINGLE FILE MODE
    # ==========================================================================
    else:
        print("="*70)
        print("SINGLE FILE MODE: Standard Analysis")
        print("="*70)
        print(f"Input:  {args.input}")
        print(f"Output: {args.output_dir}")
        print(f"CV threshold: {args.cv_threshold}")
        print(f"Top N counters: {args.top_n}")
        
        # Load data
        df, counter_cols = load_and_preprocess(args.input)
        
        # Compute statistics
        stats = compute_statistics(df, counter_cols)
        
        # Identify stable and discriminative counters
        stable_counters = identify_stable_counters(stats, counter_cols, args.cv_threshold)
        top_counters = identify_discriminative_counters(stats, counter_cols, args.top_n)
        
        # Generate visualizations
        print("\nGenerating heatmap (all counters)...")
        plot_heatmap(
            stats, counter_cols,
            os.path.join(args.output_dir, 'heatmap_all_counters.png'),
            'Heatmap of Normalized Counter Values Across Workloads',
            cmap='YlOrRd'
        )
        
        print("\nGenerating heatmap (stable counters only)...")
        plot_heatmap(
            stats, stable_counters,
            os.path.join(args.output_dir, 'heatmap_stable_counters.png'),
            'Stable Counters (Low CV) Heatmap',
            cmap='YlGnBu'
        )
        
        plot_bar_charts(stats, top_counters, args.output_dir)
        plot_pca(stats, counter_cols, args.output_dir)
        
        # Export statistics
        export_statistics(stats, counter_cols, args.output_dir)
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print(f"Results saved to: {args.output_dir}")
        print("="*70)


if __name__ == "__main__":
    main()
