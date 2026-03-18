#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import argparse
import re
import sys
from pingouin import multivariate_normality 

def plot_graph(corr_matrix, mask_matrix, save_path=None):
    """Runs in an isolated process to keep the heatmap visible and save it if requested"""
    sns.heatmap(corr_matrix, mask=mask_matrix, annot=True, fmt='.3g', center=0, cmap="PiYG", cbar_kws={"shrink": .5})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    
    # Crucial: Save must happen BEFORE plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

if __name__ == "__main__":
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description='Calculate and visualize a correlation matrix.')
    parser.add_argument('-c', '--csv', help='Path to a local CSV file')
    parser.add_argument('-g', '--gsheet', help='URL of a public Google Sheet')
    parser.add_argument('-m', '--method', dest='corr_method', default='auto', choices=['auto', 'spearman', 'pearson'], 
    help='Correlation method (default: auto-detect based on normality)')
    parser.add_argument('-s', '--save', metavar='heatmap.png', help='Save the heatmap to an image file (e.g., heatmap.png)')
    
    # Print usage menu if run blindly
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # 2. Load the Data
    if args.csv:
        print(f"Loading local CSV: {args.csv}...")
        df = pd.read_csv(args.csv)
    elif args.gsheet:
        url = args.gsheet
        if "export?format=csv" not in url:
            url = re.sub(r'/edit.*', '/export?format=csv', url)
        
        print(f"Fetching Google Sheet...\nURL: {url}")
        try:
            df = pd.read_csv(url)
        except Exception as e:
            print("\nERROR: Could not load the Google Sheet.")
            print("Please ensure the sharing settings are set to 'Anyone with the link can view'.")
            sys.exit(1)
    else:
        # Fallback just in case someone passes flags without values
        parser.print_help()
        sys.exit(1)

    # Clean the data: Drop any text/string columns so the math doesn't crash
    df_numeric = df.select_dtypes(include=[np.number])

    # 3. Determine Correlation Method
    corr_method = args.corr_method
    if corr_method == 'auto':
        try:
            norm = multivariate_normality(df_numeric, alpha=.05)
            corr_method = "pearson" if norm.normal else "spearman"
        except Exception:
            print("Normality test failed (likely due to small sample size). Defaulting to Spearman.")
            corr_method = "spearman"

    print(f'\nCalculating {corr_method.title()} correlation...\n')

    # 4. Generate the Matrix & Heatmap Mask
    corr = df_numeric.corr(method=corr_method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 5. Build and Clean the Sorted Console List
    df_corr = corr.stack().reset_index()
    df_corr.columns = ['FEATURE_1', 'FEATURE_2', 'CORRELATION']
    
    mask_dups = (df_corr[['FEATURE_1', 'FEATURE_2']].apply(frozenset, axis=1).duplicated()) | (df_corr['FEATURE_1'] == df_corr['FEATURE_2'])
    df_corr = df_corr[~mask_dups]

    with pd.option_context('display.max_rows', 500, 'display.max_columns', 50, 'display.precision', 3):
        print(df_corr.sort_values('CORRELATION', ascending=True).to_string(index=False))

    # Let the user know what is happening with the image
    if args.save:
        print(f"\nSaving high-resolution heatmap to: {args.save}")
        
    print("Check your taskbar! The interactive heatmap has opened in a new window.")

    # 6. Spawn the GUI Process
    proc = multiprocessing.Process(target=plot_graph, args=(corr, mask, args.save))
    proc.start()
