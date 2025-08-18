#!/usr/bin/env python3
"""
Quick comparison test between old and new versions of the acoustic analysis.
This will process the same test files with both versions and compare key metrics.
"""

import sys
import os
import time
import subprocess
import pandas as pd
import numpy as np

def run_old_version(input_folder, output_folder):
    """Run the original version (sequential only)."""
    print("=" * 50)
    print("RUNNING ORIGINAL VERSION")
    print("=" * 50)
    
    # Note: The original version doesn't have CLI support
    # We'll have to adapt it or run it manually
    print("Original version requires manual GUI execution.")
    print("Please run the original version manually with:")
    print(f"  Input folder: {input_folder}")
    print(f"  Output folder: {output_folder}_original")
    print("  Mode: Dataset")
    print("  Save results and press Enter when done...")
    input("Press Enter when original version is complete...")
    
    # Try to read the results
    csv_path = os.path.join(f"{output_folder}_original", "Acoustic_Indices.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"Could not find results at {csv_path}")
        return None

def run_new_version_sequential(input_folder, output_folder):
    """Run the new version in sequential mode."""
    print("=" * 50)
    print("RUNNING NEW VERSION (SEQUENTIAL)")
    print("=" * 50)
    
    # Import and run the new version
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from processing.core_processing import (
        process_files_sequential, 
        convert_results_to_dataframes,
        parse_date_and_filename_from_filename
    )
    
    # Find WAV files
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    # Parse filenames
    date_list = []
    filename_list = []
    for file in audio_files:
        dt, filename_with_numbering = parse_date_and_filename_from_filename(file)
        if dt and filename_with_numbering:
            date_list.append(dt)
            filename_list.append(filename_with_numbering)
    
    # Test parameters
    processing_params = {
        'mode': 'dataset',
        'time_interval': 0,
        'flim_low': [0, 1500],
        'flim_mid': [1500, 8000],
        'sensitivity': -35.0,
        'gain': 0.0,
        'calculate_marine': True
    }
    
    print(f"Processing {len(audio_files)} files sequentially...")
    start_time = time.time()
    results = process_files_sequential(audio_files, processing_params)
    processing_time = time.time() - start_time
    
    print(f"Sequential processing completed in {processing_time:.2f} seconds")
    
    # Convert to DataFrame
    result_df, result_df_per_bin = convert_results_to_dataframes(results, filename_list, date_list, 'dataset')
    
    # Save results
    os.makedirs(f"{output_folder}_new_sequential", exist_ok=True)
    csv_path = os.path.join(f"{output_folder}_new_sequential", "Acoustic_Indices.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return result_df

def run_new_version_parallel(input_folder, output_folder):
    """Run the new version in parallel mode."""
    print("=" * 50)
    print("RUNNING NEW VERSION (PARALLEL)")
    print("=" * 50)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from processing.core_processing import (
        process_files_parallel, 
        convert_results_to_dataframes,
        parse_date_and_filename_from_filename
    )
    
    # Find WAV files
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    # Parse filenames
    date_list = []
    filename_list = []
    for file in audio_files:
        dt, filename_with_numbering = parse_date_and_filename_from_filename(file)
        if dt and filename_with_numbering:
            date_list.append(dt)
            filename_list.append(filename_with_numbering)
    
    # Test parameters
    processing_params = {
        'mode': 'dataset',
        'time_interval': 0,
        'flim_low': [0, 1500],
        'flim_mid': [1500, 8000],
        'sensitivity': -35.0,
        'gain': 0.0,
        'calculate_marine': True
    }
    
    print(f"Processing {len(audio_files)} files in parallel...")
    start_time = time.time()
    results = process_files_parallel(audio_files, processing_params)
    processing_time = time.time() - start_time
    
    print(f"Parallel processing completed in {processing_time:.2f} seconds")
    
    # Convert to DataFrame
    result_df, result_df_per_bin = convert_results_to_dataframes(results, filename_list, date_list, 'dataset')
    
    # Save results
    os.makedirs(f"{output_folder}_new_parallel", exist_ok=True)
    csv_path = os.path.join(f"{output_folder}_new_parallel", "Acoustic_Indices.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return result_df

def compare_results(df1, df2, name1, name2):
    """Compare two result DataFrames."""
    print("=" * 50)
    print(f"COMPARING {name1} vs {name2}")
    print("=" * 50)
    
    if df1 is None or df2 is None:
        print("Cannot compare - one or both DataFrames are None")
        return
    
    print(f"{name1} shape: {df1.shape}")
    print(f"{name2} shape: {df2.shape}")
    
    # Find common columns (excluding Date and Filename)
    cols1 = set(df1.columns) - {'Date', 'Filename'}
    cols2 = set(df2.columns) - {'Date', 'Filename'}
    common_cols = cols1.intersection(cols2)
    
    print(f"Common acoustic indices: {len(common_cols)}")
    print(f"Only in {name1}: {cols1 - cols2}")
    print(f"Only in {name2}: {cols2 - cols1}")
    
    if len(common_cols) == 0:
        print("No common columns to compare!")
        return
    
    # Compare values for common columns
    differences = {}
    for col in sorted(common_cols):
        if col in df1.columns and col in df2.columns:
            try:
                # Calculate relative difference for numeric columns
                val1 = pd.to_numeric(df1[col], errors='coerce').fillna(0)
                val2 = pd.to_numeric(df2[col], errors='coerce').fillna(0)
                
                if len(val1) == len(val2):
                    # Calculate mean absolute relative error
                    mask = (val1 != 0) | (val2 != 0)
                    if mask.any():
                        relative_diff = np.abs((val1 - val2) / (np.abs(val1) + np.abs(val2) + 1e-10))
                        mean_diff = relative_diff[mask].mean()
                        differences[col] = mean_diff
            except:
                pass
    
    # Show top differences
    if differences:
        sorted_diffs = sorted(differences.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 largest relative differences:")
        for col, diff in sorted_diffs[:10]:
            print(f"  {col}: {diff:.4f}")
        
        # Show summary statistics
        diff_values = list(differences.values())
        print(f"\nDifference statistics:")
        print(f"  Mean: {np.mean(diff_values):.6f}")
        print(f"  Median: {np.median(diff_values):.6f}")
        print(f"  Max: {np.max(diff_values):.6f}")
        print(f"  Indices with >1% difference: {sum(1 for d in diff_values if d > 0.01)}")

if __name__ == "__main__":
    input_folder = "test_wav_files"
    output_folder = "comparison_test"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    print("ACOUSTIC ANALYSIS VERSION COMPARISON")
    print("This will compare results between old and new versions")
    print(f"Input folder: {input_folder}")
    print(f"Output base: {output_folder}")
    print()
    
    # Run tests
    new_sequential_df = run_new_version_sequential(input_folder, output_folder)
    new_parallel_df = run_new_version_parallel(input_folder, output_folder)
    
    # Compare sequential vs parallel (should be identical)
    if new_sequential_df is not None and new_parallel_df is not None:
        compare_results(new_sequential_df, new_parallel_df, "NEW SEQUENTIAL", "NEW PARALLEL")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print("1. Sequential vs Parallel should be nearly identical (small numerical differences OK)")
    print("2. Marine indices may differ from original due to frequency band fixes")
    print("3. Core acoustic indices should be very similar to original version")