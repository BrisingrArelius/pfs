#!/usr/bin/env python3
"""
analysis.py

Analyzes Darshan counter data from global.csv to:
1. Compute statistics (mean, std, min, max, CV) across multiple runs per profile
2. Identify stable/reliable counters (low coefficient of variation)
3. Generate visualizations:
   - Heatmap of normalized counter values across workloads
   - Bar charts for discriminative counters
   - PCA scatter plot for workload clustering

Usage:
    python analysis.py --input ./darshan_output/global.csv --output ./analysis_output

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
    parser.add_argument(
        "--input",
        required=True,
        help="Path to global.csv file"
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
    
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")
    
    return args


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def compute_derived_metrics(df):
    """
    Compute derived I/O metrics from raw Darshan counters.
    
    Derived metrics:
    - Duration windows: END_TIMESTAMP - START_TIMESTAMP for each operation type
    - I/O density: BYTES / duration (for read/write operations)
    - Mean operation sizes: BYTES / OPS (for read/write operations)
    - Ratios: Sequential/total, seek rate, etc.
    
    Returns:
        df: DataFrame with new derived metric columns added
    """
    print("Computing derived metrics...")
    
    # Duration windows (in seconds, assuming timestamps are in seconds)
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
    
    # I/O density (bytes per second)
    if 'POSIX_BYTES_READ' in df.columns and 'POSIX_READ_DURATION' in df.columns:
        df['POSIX_READ_DENSITY'] = df['POSIX_BYTES_READ'] / df['POSIX_READ_DURATION'].replace(0, np.nan)
    
    if 'POSIX_BYTES_WRITTEN' in df.columns and 'POSIX_WRITE_DURATION' in df.columns:
        df['POSIX_WRITE_DENSITY'] = df['POSIX_BYTES_WRITTEN'] / df['POSIX_WRITE_DURATION'].replace(0, np.nan)
    
    # Mean operation sizes
    if 'POSIX_BYTES_READ' in df.columns and 'POSIX_READS' in df.columns:
        df['POSIX_MEAN_READ_SIZE'] = df['POSIX_BYTES_READ'] / df['POSIX_READS'].replace(0, np.nan)
    
    if 'POSIX_BYTES_WRITTEN' in df.columns and 'POSIX_WRITES' in df.columns:
        df['POSIX_MEAN_WRITE_SIZE'] = df['POSIX_BYTES_WRITTEN'] / df['POSIX_WRITES'].replace(0, np.nan)
    
    # Sequential access ratio
    if 'POSIX_SEQ_READS' in df.columns and 'POSIX_READS' in df.columns:
        df['POSIX_SEQ_READ_RATIO'] = df['POSIX_SEQ_READS'] / df['POSIX_READS'].replace(0, np.nan)
    
    if 'POSIX_SEQ_WRITES' in df.columns and 'POSIX_WRITES' in df.columns:
        df['POSIX_SEQ_WRITE_RATIO'] = df['POSIX_SEQ_WRITES'] / df['POSIX_WRITES'].replace(0, np.nan)
    
    # Seek rate (seeks per total operations)
    if 'POSIX_SEEKS' in df.columns and 'POSIX_READS' in df.columns and 'POSIX_WRITES' in df.columns:
        total_ops = df['POSIX_READS'] + df['POSIX_WRITES']
        df['POSIX_SEEK_RATE'] = df['POSIX_SEEKS'] / total_ops.replace(0, np.nan)
    
    print(f"  Added {sum(1 for c in df.columns if 'DURATION' in c or 'DENSITY' in c or 'MEAN_' in c or 'RATIO' in c or 'RATE' in c)} derived metrics")
    
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
    
    # Identify counter columns (exclude metadata)
    all_cols = df.columns.tolist()
    counter_cols = [c for c in all_cols if c not in EXCLUDE_COUNTERS and c != 'profile']
    
    # Filter out raw timestamps and redundant stride counters
    counter_cols = [
        c for c in counter_cols 
        if not any(pattern in c for pattern in EXCLUDE_PATTERNS)
    ]
    
    # Remove counters that are all NaN or all zero
    valid_counters = []
    for col in counter_cols:
        if df[col].notna().any() and (df[col] != 0).any():
            valid_counters.append(col)
    
    print(f"Loaded {len(df)} rows, {len(df['profile'].unique())} profiles")
    print(f"Valid counters: {len(valid_counters)} out of {len(counter_cols)}")
    
    return df, valid_counters


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
    
    agg_dict = {}
    for col in counter_cols:
        agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']
    
    stats = df.groupby('profile').agg(agg_dict)
    
    # Compute coefficient of variation (CV = std / mean)
    cv_data = {}
    for col in counter_cols:
        mean_vals = stats[(col, 'mean')]
        std_vals = stats[(col, 'std')]
        # Avoid division by zero: set CV to NaN where mean is 0
        cv = std_vals / mean_vals.replace(0, np.nan)
        cv_data[col] = cv
    
    cv_df = pd.DataFrame(cv_data)
    
    # Add CV to stats as a new level
    for col in counter_cols:
        stats[(col, 'cv')] = cv_df[col]
    
    return stats


def identify_stable_counters(stats, counter_cols, cv_threshold):
    """
    Identify counters with low coefficient of variation (stable across runs).
    
    Returns:
        stable_counters: List of counter names with CV < threshold for ALL profiles
    """
    print(f"\nIdentifying stable counters (CV < {cv_threshold})...")
    
    stable = []
    for col in counter_cols:
        cv_vals = stats[(col, 'cv')].dropna()
        if cv_vals.empty:
            continue
        max_cv = cv_vals.max()
        if max_cv < cv_threshold:
            stable.append(col)
    
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
    
    discriminative_scores = {}
    for col in counter_cols:
        means = stats[(col, 'mean')]
        # Variance of means across profiles (normalized by mean)
        if means.mean() != 0:
            score = means.std() / means.mean()
        else:
            score = 0
        discriminative_scores[col] = score
    
    # Sort by score descending
    sorted_counters = sorted(discriminative_scores.items(), key=lambda x: x[1], reverse=True)
    top_counters = [c for c, s in sorted_counters[:top_n]]
    
    print(f"Top {top_n} discriminative counters:")
    for i, (counter, score) in enumerate(sorted_counters[:top_n], 1):
        print(f"  {i}. {counter}: {score:.3f}")
    
    return top_counters


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
    
    # Extract means for specified counters
    mean_data = {}
    for col in counter_cols:
        mean_data[col] = stats[(col, 'mean')]
    
    df_means = pd.DataFrame(mean_data)
    
    # Normalize each counter to 0-1 scale
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(df_means),
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
    
    # Flatten MultiIndex columns for easier reading
    export_data = []
    for profile in stats.index:
        row = {'profile': profile}
        for col in counter_cols:
            row[f'{col}_mean'] = stats.loc[profile, (col, 'mean')]
            row[f'{col}_std'] = stats.loc[profile, (col, 'std')]
            row[f'{col}_min'] = stats.loc[profile, (col, 'min')]
            row[f'{col}_max'] = stats.loc[profile, (col, 'max')]
            row[f'{col}_cv'] = stats.loc[profile, (col, 'cv')]
            row[f'{col}_count'] = stats.loc[profile, (col, 'count')]
        export_data.append(row)
    
    df_export = pd.DataFrame(export_data)
    
    output_path = os.path.join(output_dir, 'statistics.csv')
    df_export.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Also export a summary with just means
    mean_data = {}
    for col in counter_cols:
        mean_data[col] = stats[(col, 'mean')]
    df_means = pd.DataFrame(mean_data)
    df_means.index.name = 'profile'
    
    output_path_means = os.path.join(output_dir, 'means_only.csv')
    df_means.to_csv(output_path_means)
    print(f"  Saved: {output_path_means}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
