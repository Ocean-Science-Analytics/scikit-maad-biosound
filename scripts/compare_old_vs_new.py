#!/usr/bin/env python3
"""
Comparison test between the original version and current version.
This script will run both versions and compare acoustic indices results.
"""

import sys
import os
import time
import subprocess
import pandas as pd
import numpy as np
import shutil

def run_original_version_processing(input_folder, output_folder):
    """
    Run the original version processing logic directly.
    This extracts the core processing from the original without GUI dependencies.
    """
    print("=" * 50)
    print("RUNNING ORIGINAL VERSION PROCESSING")
    print("=" * 50)
    
    # Import original functions directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'archive', 'original'))
    
    # We need to extract the processing logic from the original file
    # Since it's GUI-based, we'll copy the relevant functions and run them
    
    # For now, let's manually execute the original processing functions
    from archive.original.SciKit_Maad_File_Processing_GUI import (
        process_files_sequential as original_sequential,
        parse_date_and_filename_from_filename as original_parse,
        convert_results_to_dataframes as original_convert
    )
    
    # This approach won't work because the original is GUI-based
    # Let's try a different approach - create a standalone wrapper
    print("Note: Original version requires GUI interaction")
    print("For now, we'll compare with equivalent parameters using hardcoded defaults")
    
    return None

def extract_original_processing():
    """
    Extract the original processing logic into a callable function.
    """
    # Read the original file and extract processing functions
    original_file = os.path.join(os.path.dirname(__file__), 'archive', 'original', 'SciKit_Maad_File_Processing-GUI.py')
    
    # Import the original functions we need
    import importlib.util
    spec = importlib.util.spec_from_file_location("original", original_file)
    original_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(original_module)
        return original_module
    except Exception as e:
        print(f"Could not load original module: {e}")
        return None

def run_new_version(input_folder, output_folder):
    """Run the new version processing."""
    print("=" * 50)
    print("RUNNING NEW VERSION")
    print("=" * 50)
    
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
    
    # Use same parameters as original (default marine settings)
    processing_params = {
        'mode': 'dataset',  # Equivalent to original "30min" mode
        'time_interval': 0,
        'flim_low': [0, 1500],    # Same defaults as original
        'flim_mid': [1500, 8000], # Same defaults as original  
        'sensitivity': -35.0,     # Same defaults as original
        'gain': 0.0,              # Same defaults as original
        'calculate_marine': True  # Enable marine indices
    }
    
    print(f"Processing {len(audio_files)} files...")
    start_time = time.time()
    results = process_files_sequential(audio_files, processing_params)
    processing_time = time.time() - start_time
    
    print(f"New version completed in {processing_time:.2f} seconds")
    
    # Convert to DataFrame
    result_df, result_df_per_bin = convert_results_to_dataframes(results, filename_list, date_list, 'dataset')
    
    # Save results
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "New_Version_Results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return result_df

def simulate_original_results(input_folder, output_folder):
    """
    Simulate what the original version would produce.
    Since we can't easily run the original GUI version, we'll create a baseline
    using the same core maad functions but with the original's exact logic.
    """
    print("=" * 50)
    print("SIMULATING ORIGINAL VERSION RESULTS")
    print("=" * 50)
    
    # Use the exact same processing as original but without GUI
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from maad import sound, features
    import pandas as pd
    import numpy as np
    import datetime
    
    def original_parse_filename(filename):
        """Original filename parsing logic."""
        try:
            basename = os.path.basename(filename)
            parts = basename.split('_')
            
            if len(parts) < 4:
                return None, None
                
            date_str = parts[1]
            time_str = parts[2]
            
            if len(date_str) != 8 or len(time_str) != 6:
                return None, None
                
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            dt = datetime.datetime(year, month, day, hour, minute, second)
            filename_with_numbering = '_'.join(parts[:-1]) + "_" + parts[-1]
            return dt, filename_with_numbering
            
        except Exception:
            return None, None
    
    # Find files
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    print(f"Processing {len(audio_files)} files with original logic...")
    
    result_data = []
    
    # Original processing parameters (these were hardcoded in original)
    flim_low = [0, 1500]      # Original default
    flim_mid = [1500, 8000]   # Original default
    sensitivity = -35.0       # Original default  
    gain = 0.0               # Original default
    
    for file_path in audio_files:
        print(f"  Processing: {os.path.basename(file_path)}")
        
        # Parse filename
        parsed_date, filename_with_numbering = original_parse_filename(file_path)
        if not parsed_date:
            continue
            
        try:
            # Load audio (same as original)
            wave, fs = sound.load(filename=file_path, channel='left', detrend=True, verbose=False)
            
            # Generate spectrogram (same parameters as original)
            Sxx_power, tn, fn, ext = sound.spectrogram(
                x=wave,
                fs=fs,
                window='hann',
                nperseg=512,
                noverlap=512//2,
                verbose=False,
                display=False,
                savefig=None
            )
            
            # Temporal indices (same as original)
            temporal_indices = features.all_temporal_alpha_indices(
                s=wave,
                fs=fs,
                gain=gain,
                sensibility=sensitivity,
                dB_threshold=3,
                rejectDuration=0.01,
                verbose=False,
                display=False
            )
            
            # Spectral indices (same as original)
            spectral_indices, spectral_indices_per_bin = features.all_spectral_alpha_indices(
                Sxx_power=Sxx_power,
                tn=tn,
                fn=fn,
                flim_low=flim_low,
                flim_mid=flim_mid,
                flim_hi=[8000, 40000],
                gain=gain,
                sensitivity=sensitivity,
                verbose=False,
                R_compatible='soundecology',
                mask_param1=6,
                mask_param2=0.5,
                display=False
            )
            
            # Combine results (original method) - convert Series to scalar values
            combined_indices = {}
            
            # Handle temporal indices
            for key, value in temporal_indices.items():
                if hasattr(value, 'iloc'):  # pandas Series
                    combined_indices[key] = value.iloc[0] if len(value) > 0 else 0
                else:
                    combined_indices[key] = value
            
            # Handle spectral indices    
            for key, value in spectral_indices.items():
                if hasattr(value, 'iloc'):  # pandas Series
                    combined_indices[key] = value.iloc[0] if len(value) > 0 else 0
                else:
                    combined_indices[key] = value
                    
            combined_indices['Date'] = parsed_date
            combined_indices['Filename'] = os.path.basename(file_path)
            
            result_data.append(combined_indices)
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Create DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Save results
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "Original_Simulated_Results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Simulated original results saved to: {csv_path}")
    print(f"Result shape: {result_df.shape}")
    
    return result_df

def compare_results(original_df, new_df):
    """Compare original vs new results."""
    print("=" * 60)
    print("COMPARING ORIGINAL vs NEW VERSION")
    print("=" * 60)
    
    if original_df is None or new_df is None:
        print("Cannot compare - one or both DataFrames are None")
        return
    
    print(f"Original shape: {original_df.shape}")
    print(f"New shape: {new_df.shape}")
    
    # Find common columns (excluding Date and Filename)
    orig_cols = set(original_df.columns) - {'Date', 'Filename'}
    new_cols = set(new_df.columns) - {'Date', 'Filename'}
    common_cols = orig_cols.intersection(new_cols)
    
    print(f"Common acoustic indices: {len(common_cols)}")
    print(f"Only in original: {sorted(orig_cols - new_cols)}")
    print(f"Only in new: {sorted(new_cols - orig_cols)}")
    
    if len(common_cols) == 0:
        print("No common columns to compare!")
        return
    
    # Compare values for common columns
    differences = {}
    for col in sorted(common_cols):
        try:
            # Calculate relative difference for numeric columns
            val1 = pd.to_numeric(original_df[col], errors='coerce').fillna(0)
            val2 = pd.to_numeric(new_df[col], errors='coerce').fillna(0)
            
            if len(val1) == len(val2):
                # Calculate mean absolute relative error
                mask = (val1 != 0) | (val2 != 0)
                if mask.any():
                    relative_diff = np.abs((val1 - val2) / (np.abs(val1) + np.abs(val2) + 1e-10))
                    mean_diff = relative_diff[mask].mean()
                    differences[col] = mean_diff
        except:
            pass
    
    # Show results
    if differences:
        sorted_diffs = sorted(differences.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 largest relative differences:")
        for col, diff in sorted_diffs[:10]:
            print(f"  {col}: {diff:.6f}")
        
        # Show summary statistics
        diff_values = list(differences.values())
        print(f"\nDifference statistics:")
        print(f"  Mean: {np.mean(diff_values):.8f}")
        print(f"  Median: {np.median(diff_values):.8f}")
        print(f"  Max: {np.max(diff_values):.8f}")
        print(f"  Indices with >1% difference: {sum(1 for d in diff_values if d > 0.01)}")
        print(f"  Indices with >0.1% difference: {sum(1 for d in diff_values if d > 0.001)}")

if __name__ == "__main__":
    input_folder = "test_wav_files"
    base_output = "old_vs_new_comparison"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    
    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    print("ACOUSTIC ANALYSIS: ORIGINAL vs NEW VERSION COMPARISON")
    print("=" * 60)
    print(f"Input folder: {input_folder}")
    print()
    
    # Run both versions
    original_df = simulate_original_results(input_folder, f"{base_output}_original")
    new_df = run_new_version(input_folder, f"{base_output}_new")
    
    # Compare results
    compare_results(original_df, new_df)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print("Key findings:")
    print("1. Large differences expected in marine indices (due to frequency band fixes)")
    print("2. Core acoustic indices should be very similar")
    print("3. Small numerical differences are normal due to processing changes")