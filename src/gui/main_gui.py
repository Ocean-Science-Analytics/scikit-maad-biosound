# This script uses the sci-kit maad package to analyze acoustic indices from wav files and generate a csv file and figures displaying the measured indices
# Phase 1 Fix: Robust error handling and graceful degradation

# Created by Jared Stephens 12/27/2023
# Revised by Jared Stephens 02/14/2023
    # Fixed date parsing error by creating "parse_date_and_filename_from_filename" function to manually extract the date and filename from each wav file.
    # Also created dataframes to store Date and Filename info
    # Then merged the Date/Filename dataframe with the acoustic indices dataframe.
    # Changed "norm=True" to "norm=False" which solved most of the issues with the figures not dispalying data
# Revised by Jared Stephens 03/11/2024
    # Fixed false color spectrogram by including "results_df_per_bin" and "indices_per_bin" dataframes
    # Increased the axis text size in all the figures
    # Built a simple GUI with folder browsing options and a time scale choice between 24 hours (average across a full day) or the original dataset timescale
# Revised by Jared Stephens 04/10/2024
    # Added the manual time interval option which measures indices based off the user defined samples (time interval)
    # Simplified the input for the six different indices for the 'individual features plot'
# Modified by M. Weirathmueller August 2025 - bug fix + documentation updates

#############################################################################################################################################################################################

### NOTE FOR THE USER: To adjust the six different indices displayed in the 'individual features plot', see 'indice_one - indice_six' below the temporal features near the top

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from maad import sound, features
from maad.util import date_parser, plot_correlation_map, plot_features_map, plot_features, false_Color_Spectro
from tkinter import Tk, filedialog, Label, Entry, Button, Frame, SOLID, StringVar, IntVar, Checkbutton, BooleanVar, messagebox
import datetime
import traceback
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.run_metadata import create_run_metadata

# Define spectral and temporal features
SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
    'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
    'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
    'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
    'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
    'AGI','ROItotal','ROIcover'
]

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
    'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
    'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount'
]

# These are the six indices displayed in the individual features plot
indice_one='Hf'
indice_two='AEI'
indice_three='NDSI'
indice_four='ACI'
indice_five='TFSD'
indice_six='ROItotal'  # Note: This may not always be computed

def calculate_marine_biophony_anthrophony(Sxx_power, fn, flim_low, flim_mid):
    """
    Calculate anthrophony and biophony for marine environments.
    
    Based on correspondence with Sylvain Haupert (scikit-maad developer):
    - Anthrophony: flim_low range (vessel noise, typically 0-1000 Hz)
    - Biophony: flim_mid range (biological sounds, typically 1000-8000 Hz)
    
    Args:
        Sxx_power: Power spectrogram
        fn: Frequency array
        flim_low: [min, max] frequency range for anthrophony (vessel noise)
        flim_mid: [min, max] frequency range for biophony (biological sounds)
    
    Returns:
        tuple: (anthrophony_energy, biophony_energy)
    """
    # Extract power in anthrophony band (low frequencies - vessel noise)
    anthro_mask = (fn >= flim_low[0]) & (fn < flim_low[1])
    anthrophony_power = Sxx_power[anthro_mask]
    anthrophony_energy = np.sum(anthrophony_power)
    
    # Extract power in biophony band (mid frequencies - biological sounds)
    bio_mask = (fn >= flim_mid[0]) & (fn < flim_mid[1])
    biophony_power = Sxx_power[bio_mask]
    biophony_energy = np.sum(biophony_power)
    
    return anthrophony_energy, biophony_energy

def calculate_marine_indices(Sxx_power, fn, flim_low, flim_mid, S=-35.0, G=0.0):
    """
    Calculate marine-specific acoustic indices with corrected frequency bands.
    
    This function specifically handles NDSI, BioEnergy, AnthroEnergy, rBA, and BI
    with marine-appropriate frequency band assignments.
    
    Args:
        Sxx_power: Power spectrogram
        fn: Frequency array
        flim_low: Anthrophony frequency range [min, max]
        flim_mid: Biophony frequency range [min, max]
        S: Sensitivity (dB)
        G: Gain (dB)
    
    Returns:
        dict: Dictionary of calculated marine indices
    """
    # Calculate anthrophony and biophony energies
    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )
    
    # Calculate marine-specific indices
    marine_indices = {}
    
    # NDSI (Normalized Difference Soundscape Index)
    if (bio_energy + anthro_energy) > 0:
        marine_indices['NDSI_marine'] = (bio_energy - anthro_energy) / (bio_energy + anthro_energy)
    else:
        marine_indices['NDSI_marine'] = 0
    
    # Energy metrics
    marine_indices['BioEnergy_marine'] = bio_energy
    marine_indices['AnthroEnergy_marine'] = anthro_energy
    
    # rBA (ratio of biophony to anthrophony)
    if anthro_energy > 0:
        marine_indices['rBA_marine'] = bio_energy / anthro_energy
    else:
        marine_indices['rBA_marine'] = np.inf if bio_energy > 0 else 0
    
    # BI (Bioacoustic Index) - calculated on biophony band
    bio_mask = (fn >= flim_mid[0]) & (fn < flim_mid[1])
    if np.any(bio_mask):
        bio_spectrum = np.mean(Sxx_power[bio_mask, :], axis=1) if len(Sxx_power.shape) > 1 else Sxx_power[bio_mask]
        marine_indices['BI_marine'] = np.sum(bio_spectrum * np.log10(bio_spectrum + 1e-10))
    else:
        marine_indices['BI_marine'] = 0
    
    return marine_indices

def process_single_file(filename_args):
    """
    Process a single audio file for parallel processing.
    
    Args:
        filename_args: Tuple containing (filename, params_dict)
        
    Returns:
        dict: Processing results or None if failed
    """
    try:
        filename, params = filename_args
        
        # Extract parameters
        mode = params['mode']
        time_interval = params.get('time_interval', 0)
        flim_low = params['flim_low']
        flim_mid = params['flim_mid']
        sensitivity = params['sensitivity']
        gain = params['gain']
        calculate_marine = params['calculate_marine']
        
        print(f"  Processing: {os.path.basename(filename)}")
        
        # Parse date and filename
        parsed_date, filename_with_numbering = parse_date_and_filename_from_filename(filename)
        if parsed_date is None:
            print(f"    Skipping {os.path.basename(filename)} - invalid filename format")
            return None
        
        # Load audio file
        wave, fs = sound.load(filename=filename, channel='left', detrend=True, verbose=False)
        
        # Process based on mode
        results = []
        
        if mode == "20min":  # Manual time intervals
            total_samples = len(wave)
            samples_per_interval = int(fs * time_interval)
            
            previous_segment_wave = None
            for start_sample in range(0, total_samples, samples_per_interval):
                end_sample = min(start_sample + samples_per_interval, total_samples)
                interval_length = (end_sample - start_sample) / fs
                
                if interval_length < 1 and previous_segment_wave is not None:
                    previous_segment_wave = np.concatenate([previous_segment_wave, wave[start_sample:end_sample]])
                    continue
                elif interval_length < 1:
                    continue
                    
                segment_wave = wave[start_sample:end_sample]
                if previous_segment_wave is not None:
                    segment_wave = np.concatenate([previous_segment_wave, segment_wave])
                previous_segment_wave = segment_wave
                
                # Calculate indices for this segment
                segment_result = calculate_indices_for_segment(
                    segment_wave, fs, sensitivity, gain, flim_low, flim_mid, 
                    calculate_marine, os.path.basename(filename)
                )
                if segment_result:
                    results.append(segment_result)
        else:
            # Process entire file
            file_result = calculate_indices_for_segment(
                wave, fs, sensitivity, gain, flim_low, flim_mid,
                calculate_marine, os.path.basename(filename)
            )
            if file_result:
                results.append(file_result)
        
        return {
            'filename': filename,
            'parsed_date': parsed_date,
            'filename_with_numbering': filename_with_numbering,
            'results': results
        }
        
    except Exception as e:
        print(f"    ERROR processing {os.path.basename(filename)}: {str(e)}")
        return None

def calculate_indices_for_segment(wave, fs, sensitivity, gain, flim_low, flim_mid, calculate_marine, filename):
    """
    Calculate indices for a single segment of audio.
    
    Returns:
        dict: Calculated indices or None if failed
    """
    try:
        # Generate spectrogram
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
        
        # Calculate temporal indices
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
        
        # Calculate spectral indices
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
        
        # Calculate marine-specific indices if requested
        if calculate_marine:
            marine_indices = calculate_marine_indices(
                Sxx_power, fn, flim_low, flim_mid, sensitivity, gain
            )
            spectral_indices.update(marine_indices)
        
        # Combine all indices
        all_indices = {**temporal_indices, **spectral_indices}
        all_indices['Filename'] = filename
        
        return {
            'indices': all_indices,
            'indices_per_bin': spectral_indices_per_bin
        }
        
    except Exception as e:
        print(f"      ERROR calculating indices: {str(e)}")
        return None

def process_files_parallel(file_paths, params):
    """
    Process files in parallel using multiprocessing.
    
    Returns:
        list: Processing results for each file
    """
    num_workers = min(cpu_count() - 1, 4)  # Leave one core free, max 4 workers
    
    # Prepare arguments for parallel processing
    file_args = [(filepath, params) for filepath in file_paths]
    
    # Process files in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, file_args)
    
    return results

def process_files_sequential(file_paths, params):
    """
    Process files sequentially (for comparison or when parallel is disabled).
    
    Returns:
        list: Processing results for each file
    """
    results = []
    for filepath in file_paths:
        result = process_single_file((filepath, params))
        results.append(result)
    
    return results

def convert_results_to_dataframes(processing_results, filename_list, date_list, mode):
    """
    Convert processing results to pandas DataFrames.
    
    Returns:
        tuple: (result_df, result_df_per_bin)
    """
    result_df = pd.DataFrame()
    result_df_per_bin = pd.DataFrame()
    
    # Create filename to date mapping
    filename_to_date = dict(zip(filename_list, date_list))
    
    for result in processing_results:
        if result is None:
            continue
            
        filename = os.path.basename(result['filename'])
        parsed_date = result['parsed_date']
        
        for segment_result in result['results']:
            if segment_result is None:
                continue
                
            # Add main indices
            indices_row = segment_result['indices'].copy()
            indices_row['Date'] = parsed_date
            result_df = pd.concat([result_df, pd.DataFrame([indices_row])], ignore_index=True)
            
            # Add per-bin indices
            if 'indices_per_bin' in segment_result:
                indices_per_bin = segment_result['indices_per_bin'].copy()
                indices_per_bin['Filename'] = filename
                indices_per_bin['Date'] = parsed_date
                result_df_per_bin = pd.concat([result_df_per_bin, indices_per_bin], ignore_index=True)
    
    return result_df, result_df_per_bin

def safe_plot_index(df, index_name, ax, mode, position_label):
    """
    Safely plot an acoustic index with error handling.
    
    Args:
        df: DataFrame containing the index data
        index_name: Name of the index to plot
        ax: Matplotlib axis to plot on
        mode: Time mode for plotting
        position_label: Label for this plot position (for error messages)
    
    Returns:
        tuple: (figure, axis) after plotting or error handling
    """
    fig = ax.figure  # Get the figure from the axis
    
    try:
        if index_name in df.columns:
            # Index exists, try to plot it
            print(f"  Plotting {index_name}...")
            fig, ax = plot_features(df[[index_name]], norm=False, mode=mode, ax=ax)
        else:
            # Index doesn't exist in dataframe
            print(f"  WARNING: Index '{index_name}' not found in data - showing placeholder")
            ax.text(0.5, 0.5, f"Index not available:\n{index_name}", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_title(f"{index_name} (Not Computed)")
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
    except Exception as e:
        # Error during plotting
        print(f"  ERROR plotting {index_name}: {str(e)}")
        ax.text(0.5, 0.5, f"Error plotting:\n{index_name}", 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='red')
        ax.set_title(f"{index_name} (Error)")
    
    return fig, ax

def run_analysis():
    """
    Main analysis function with comprehensive error handling and metadata tracking.
    """
    # Initialize metadata tracking (will be saved to metadata subfolder)
    output_folder = output_folder_var.get()
    output_metadata_path = os.path.join(output_folder, "metadata")
    os.makedirs(output_metadata_path, exist_ok=True)
    run_metadata = create_run_metadata(output_metadata_path)
    
    # Track errors and warnings for final report
    errors = []
    warnings = []
    
    print("\n" + "="*60)
    print("STARTING ACOUSTIC ANALYSIS")
    print("="*60)
    
    # Start metadata tracking
    run_metadata.start_run(
        input_folder=input_folder_var.get(),
        output_folder=output_folder_var.get(),
        run_identifier=run_identifier_var.get().strip(),
        mode_24h=mode_var_24h.get(),
        mode_30min=mode_var_30min.get(),
        mode_20min=mode_var_20min.get(),
        time_interval=time_interval_var.get(),
        flim_low_str=flim_low_var.get(),
        flim_mid_str=flim_mid_var.get(),
        sensitivity_str=sensitivity_var.get(),
        gain_str=gain_var.get(),
        parallel_enabled=parallel_var.get(),
        compare_performance=compare_performance_var.get()
    )
    
    # Validate time scale selection
    if mode_var_24h.get() and mode_var_30min.get() and mode_var_20min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
        return
    elif mode_var_24h.get() and mode_var_30min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
        return
    elif mode_var_24h.get() and mode_var_20min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
        return
    elif mode_var_30min.get() and mode_var_20min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
        return
    elif not mode_var_24h.get() and not mode_var_30min.get() and not mode_var_20min.get():
        messagebox.showerror("Error", "Please select one time scale option.")
        return
    
    # Validate folders
    input_folder = input_folder_var.get()
    output_folder = output_folder_var.get()
    
    if not input_folder:
        messagebox.showerror("Error", "Please select an input folder.")
        return
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return
    if not os.path.exists(input_folder):
        messagebox.showerror("Error", f"Input folder does not exist:\n{input_folder}")
        return
    
    # Determine mode
    if mode_var_24h.get():
        mode = "24h"
        print("Time scale: Hourly")
    elif mode_var_30min.get():
        mode = "30min"
        print("Time scale: Dataset")
    elif mode_var_20min.get():
        mode = "20min"
        print("Time scale: Manual")
        try:
            time_interval = int(time_interval_var.get())
            print(f"Time interval: {time_interval} seconds")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for time interval (seconds).")
            return
    
    # Parse optional marine acoustic parameters
    # Use defaults if not specified
    flim_low = [0, 1500]  # Default anthrophony range
    flim_mid = [1500, 8000]  # Default biophony range
    sensitivity = -35.0  # Default sensitivity
    gain = 0.0  # Default gain
    
    # Check if user provided custom frequency bands
    if flim_low_var.get().strip():
        try:
            flim_low = list(map(int, flim_low_var.get().split(',')))
            if len(flim_low) != 2 or flim_low[0] >= flim_low[1]:
                raise ValueError("Invalid frequency range")
            print(f"Custom anthrophony range: {flim_low[0]}-{flim_low[1]} Hz")
        except:
            messagebox.showerror("Error", "Invalid anthrophony range. Use format: min,max (e.g., 0,1000)")
            return
    
    if flim_mid_var.get().strip():
        try:
            flim_mid = list(map(int, flim_mid_var.get().split(',')))
            if len(flim_mid) != 2 or flim_mid[0] >= flim_mid[1]:
                raise ValueError("Invalid frequency range")
            print(f"Custom biophony range: {flim_mid[0]}-{flim_mid[1]} Hz")
        except:
            messagebox.showerror("Error", "Invalid biophony range. Use format: min,max (e.g., 1000,8000)")
            return
    
    if sensitivity_var.get().strip():
        try:
            sensitivity = float(sensitivity_var.get())
            print(f"Custom sensitivity: {sensitivity}")
        except:
            messagebox.showerror("Error", "Invalid sensitivity value. Must be a number.")
            return
    
    if gain_var.get().strip():
        try:
            gain = float(gain_var.get())
            print(f"Custom gain: {gain}")
        except:
            messagebox.showerror("Error", "Invalid gain value. Must be a number.")
            return

    # Find audio files
    print("\nSearching for WAV files...")
    df = pd.DataFrame()
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        messagebox.showerror("Error", f"No WAV files found in:\n{input_folder}")
        return
    
    print(f"Found {len(audio_files)} WAV files")
    
    # Parse filenames
    date_list = []
    filename_list = []
    for file in audio_files:
        dt, filename_with_numbering = parse_date_and_filename_from_filename(file)
        if dt and filename_with_numbering:
            date_list.append(dt)
            filename_list.append(filename_with_numbering)
        else:
            warnings.append(f"Could not parse filename: {os.path.basename(file)}")
    
    if not filename_list:
        messagebox.showerror("Error", "No files with valid naming convention found.\n\nExpected format: [prefix]_YYYYMMDD_HHMMSS_[suffix].wav")
        return
    
    print(f"Successfully parsed {len(filename_list)} filenames")
    
    df['Date'] = date_list
    df['Filename'] = filename_list
    
    result_df = pd.DataFrame()
    result_df_per_bin = pd.DataFrame()
    files_processed = 0
    files_failed = 0
    
    # Add processing info to metadata
    run_metadata.add_processing_info(
        files_found=len(filename_list),
        processing_mode="parallel" if parallel_var.get() else "sequential",
        compare_performance_enabled=compare_performance_var.get()
    )
    
    # Performance settings
    use_parallel = parallel_var.get()
    compare_performance = compare_performance_var.get()
    
    if compare_performance:
        print("\n=== PERFORMANCE COMPARISON MODE ===")
        print("Will run analysis both sequentially and in parallel for benchmarking")
        print(f"Testing with {len(filename_list)} files...")
    elif use_parallel:
        print(f"\n=== PARALLEL PROCESSING ENABLED ===")
        num_workers = min(cpu_count() - 1, 4)  # Leave one core free, max 4 workers
        print(f"Using {num_workers} worker processes for {len(filename_list)} files")
    else:
        print("\n=== SEQUENTIAL PROCESSING ===")
    
    # Prepare parameters for processing
    processing_params = {
        'mode': mode,
        'time_interval': time_interval if mode == "20min" else 0,
        'flim_low': flim_low,
        'flim_mid': flim_mid,
        'sensitivity': sensitivity,
        'gain': gain,
        'calculate_marine': flim_low_var.get().strip() or flim_mid_var.get().strip()
    }
    
    # Create full file paths
    full_file_paths = [os.path.join(input_folder, filename) for filename in filename_list]
    
    # Performance comparison mode
    if compare_performance:
        print("\n--- Running Sequential Version ---")
        start_time = time.time()
        sequential_results = process_files_sequential(full_file_paths, processing_params)
        sequential_time = time.time() - start_time
        
        print(f"\n--- Running Parallel Version ---")
        start_time = time.time()
        parallel_results = process_files_parallel(full_file_paths, processing_params)
        parallel_time = time.time() - start_time
        
        # Use parallel results for final output
        processing_results = parallel_results
        files_processed = len([r for r in processing_results if r is not None])
        files_failed = len([r for r in processing_results if r is None])
        
        # Print performance comparison
        print(f"\n=== PERFORMANCE COMPARISON RESULTS ===")
        print(f"Sequential processing: {sequential_time:.2f} seconds")
        print(f"Parallel processing:   {parallel_time:.2f} seconds")
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            print(f"Speedup: {speedup:.2f}x faster with parallel processing")
            efficiency = speedup / min(cpu_count() - 1, 4) * 100
            print(f"Parallel efficiency: {efficiency:.1f}%")
            
            # Add performance metrics to metadata
            run_metadata.add_performance_metrics(
                sequential_time=sequential_time,
                parallel_time=parallel_time,
                speedup=speedup,
                efficiency=efficiency,
                files_processed=files_processed,
                files_failed=files_failed
            )
            
            # Generate performance report
            try:
                from processing.performance_comparison_report import generate_performance_report
                additional_info = {
                    "frequency_bands": f"Anthrophony: {flim_low}, Biophony: {flim_mid}",
                    "marine_indices_enabled": processing_params['calculate_marine'],
                    "processing_mode": mode,
                    "gui_version": "Marine Acoustics with Parallel Processing"
                }
                # Save performance report to metadata folder
                perf_report_folder = os.path.join(output_folder, "metadata", "performance_reports")
                os.makedirs(perf_report_folder, exist_ok=True)
                generate_performance_report(
                    sequential_time, parallel_time, len(filename_list), 
                    min(cpu_count() - 1, 4), perf_report_folder, additional_info
                )
                print("📊 Performance comparison report saved to output folder")
            except Exception as e:
                print(f"Note: Could not generate performance report: {e}")
        
    elif use_parallel:
        # Parallel processing only
        print("Processing files in parallel...")
        start_time = time.time()
        processing_results = process_files_parallel(full_file_paths, processing_params)
        processing_time = time.time() - start_time
        files_processed = len([r for r in processing_results if r is not None])
        files_failed = len([r for r in processing_results if r is None])
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        
        # Add performance metrics to metadata
        run_metadata.add_performance_metrics(
            processing_time=processing_time,
            processing_mode="parallel",
            files_processed=files_processed,
            files_failed=files_failed
        )
        
    else:
        # Sequential processing only
        print("Processing files sequentially...")
        start_time = time.time()
        processing_results = process_files_sequential(full_file_paths, processing_params)
        processing_time = time.time() - start_time
        files_processed = len([r for r in processing_results if r is not None])
        files_failed = len([r for r in processing_results if r is None])
        print(f"Sequential processing completed in {processing_time:.2f} seconds")
        
        # Add performance metrics to metadata
        run_metadata.add_performance_metrics(
            processing_time=processing_time,
            processing_mode="sequential",
            files_processed=files_processed,
            files_failed=files_failed
        )
    
    # Convert results to DataFrames
    print(f"\nProcessing complete: {files_processed} files processed, {files_failed} files failed")
    
    # Add file information to metadata
    input_files = [os.path.join(input_folder, f) for f in filename_list]
    output_files = []
    try:
        # Try to list output files
        for f in os.listdir(output_folder):
            if f.endswith(('.csv', '.png', '.pdf', '.json', '.txt')):
                output_files.append(f)
    except:
        pass  # If we can't list output files, that's okay
    
    run_metadata.add_file_info(input_files, output_files)
    
    if files_processed == 0:
        # Finalize metadata tracking for failed run
        run_metadata.finish_run(success=False, error_message="No audio files could be processed successfully")
        messagebox.showerror("Error", "No audio files could be processed successfully.\n\nCheck the console for error details.")
        return
    
    # Process results into DataFrames (replacing the old processing logic)
    result_df, result_df_per_bin = convert_results_to_dataframes(processing_results, filename_list, date_list, mode)
    
    # Skip the old processing logic and go directly to plotting
    if False:  # This disables the old processing code below
        # Manual time interval mode
        time_interval = int(time_interval_var.get())
        for i, filename in enumerate(filename_list, 1):
            print(f"  Processing file {i}/{len(filename_list)}: {filename}")
            fullfilename = os.path.join(input_folder, filename)
            
            try:
                wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)
                # Use the parameters from GUI (or defaults)
                S = sensitivity
                G = gain
                
                total_samples = len(wave)
                samples_per_interval = int(fs * time_interval)
                
                interval_duration = time_interval
                previous_segment_wave = None
                for start_sample in range(0, total_samples, samples_per_interval):
                    end_sample = min(start_sample + samples_per_interval, total_samples)
                    interval_length = (end_sample - start_sample) / fs
                    if interval_length < 1 and previous_segment_wave is not None:
                        previous_segment_wave = np.concatenate([previous_segment_wave, wave[start_sample:end_sample]])
                        continue
                    elif interval_length < 1:
                        print(f"    Skipping last {interval_length:.2f}s (too short)")
                        continue
                    segment_wave = wave[start_sample:end_sample]
                    if previous_segment_wave is not None:
                        segment_wave = np.concatenate([previous_segment_wave, segment_wave])
                    previous_segment_wave = segment_wave
                    
                    Sxx_power, tn, fn, ext = sound.spectrogram(
                        x=segment_wave, 
                        fs=fs, 
                        window='hann', 
                        nperseg=512,
                        noverlap=512//2,
                        verbose=False, 
                        display=False, 
                        savefig=None
                    )   
                    
                    temporal_indices = features.all_temporal_alpha_indices(
                        s=segment_wave, 
                        fs=fs, 
                        gain=G, 
                        sensibility=S,
                        dB_threshold=3, 
                        rejectDuration=0.01,
                        verbose=False,
                        display=False
                    )
                    
                    spectral_indices, spectral_indices_per_bin = features.all_spectral_alpha_indices(
                        Sxx_power=Sxx_power,
                        tn=tn,
                        fn=fn,
                        flim_low=flim_low,
                        flim_mid=flim_mid,
                        flim_hi=[8000, 40000],
                        gain=G,
                        sensitivity=S,
                        verbose=False, 
                        R_compatible='soundecology',
                        mask_param1=6, 
                        mask_param2=0.5,
                        display=False
                    )
                    
                    # Calculate marine-specific indices if custom frequency bands are provided
                    if flim_low_var.get().strip() or flim_mid_var.get().strip():
                        marine_indices = calculate_marine_indices(
                            Sxx_power, fn, flim_low, flim_mid, S, G
                        )
                        # Add marine indices to the spectral indices dictionary
                        spectral_indices.update(marine_indices)
                    
                    indices_df_per_bin = pd.concat([spectral_indices_per_bin], axis=1)
                    indices_df_per_bin.insert(0, 'Filename', filename)
                    result_df_per_bin = pd.concat([result_df_per_bin, indices_df_per_bin], ignore_index=True)
                    
                    indices_df = pd.concat([temporal_indices, spectral_indices], axis=1)
                    indices_df.insert(0, 'Filename', filename)
                    result_df = pd.concat([result_df, indices_df], ignore_index=True)
                
                files_processed += 1
                
            except Exception as e:
                files_failed += 1
                error_msg = f"Failed to process '{filename}': {str(e)}"
                print(f"    ERROR: {error_msg}")
                errors.append(error_msg)
    
    elif mode_var_24h.get() or mode_var_30min.get():
        # Hourly or Dataset mode
        for i, filename in enumerate(filename_list, 1):
            print(f"  Processing file {i}/{len(filename_list)}: {filename}")
            fullfilename = os.path.join(input_folder, filename)
            
            try:
                wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)
                # Use the parameters from GUI (or defaults)
                S = sensitivity
                G = gain
                
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
                
                temporal_indices = features.all_temporal_alpha_indices(
                    s=wave,
                    fs=fs,
                    gain=G,
                    sensibility=S,
                    dB_threshold=3,
                    rejectDuration=0.01,
                    verbose=False,
                    display=False
                )
                
                spectral_indices, spectral_indices_per_bin = features.all_spectral_alpha_indices(
                    Sxx_power=Sxx_power,
                    tn=tn,
                    fn=fn,
                    flim_low=flim_low,
                    flim_mid=flim_mid,
                    flim_hi=[8000, 40000],
                    gain=G,
                    sensitivity=S,
                    verbose=False,
                    R_compatible='soundecology',
                    mask_param1=6,
                    mask_param2=0.5,
                    display=False
                )
                
                # Calculate marine-specific indices if custom frequency bands are provided
                if flim_low_var.get().strip() or flim_mid_var.get().strip():
                    marine_indices = calculate_marine_indices(
                        Sxx_power, fn, flim_low, flim_mid, S, G
                    )
                    # Add marine indices to the spectral indices dictionary
                    spectral_indices.update(marine_indices)
                
                indices_df_per_bin = pd.concat([spectral_indices_per_bin], axis=1)
                indices_df_per_bin.insert(0, 'Filename', filename)
                result_df_per_bin = pd.concat([result_df_per_bin, indices_df_per_bin], ignore_index=True)
                
                indices_df = pd.concat([temporal_indices, spectral_indices], axis=1)
                indices_df.insert(0, 'Filename', filename)
                result_df = pd.concat([result_df, indices_df], ignore_index=True)
                
                files_processed += 1
                
            except Exception as e:
                files_failed += 1
                error_msg = f"Failed to process '{filename}': {str(e)}"
                print(f"    ERROR: {error_msg}")
                errors.append(error_msg)
    
    print(f"\nFile processing complete: {files_processed} succeeded, {files_failed} failed")
    
    # Check if we have any results
    if result_df.empty:
        messagebox.showerror("Error", "No audio files could be processed successfully.\n\nCheck the console for error details.")
        return
    
    # Merge dataframes
    full_df = df.merge(result_df, how='inner', on='Filename')
    
    # Adjust dates for manual time interval mode
    if mode_var_20min.get():
        for filename in filename_list:
            filename_rows = full_df[full_df['Filename'] == filename]
            time_diff = pd.to_timedelta(np.arange(len(filename_rows)) * time_interval, unit='s')
            full_df.loc[full_df['Filename'] == filename, 'Date'] += time_diff
    
    # Create organized output directory structure first
    output_figures_path = os.path.join(output_folder, "figures")
    output_metadata_path = os.path.join(output_folder, "metadata")
    output_csv_path = os.path.join(output_folder, "data")
    
    os.makedirs(output_figures_path, exist_ok=True)
    os.makedirs(output_metadata_path, exist_ok=True)
    os.makedirs(output_csv_path, exist_ok=True)
    
    # Check for potential overwrites (always check, regardless of identifier)
    run_id = run_identifier_var.get().strip()
    
    # Determine filenames that will be created
    if run_id:
        expected_files = [
            (os.path.join(output_csv_path, f"{run_id}_Acoustic_Indices.csv"), f"{run_id}_Acoustic_Indices.csv"),
            (os.path.join(output_figures_path, f"{run_id}_correlation_map.png"), f"{run_id}_correlation_map.png"),
            (os.path.join(output_figures_path, f"{run_id}_individual_features.png"), f"{run_id}_individual_features.png"),
            (os.path.join(output_figures_path, f"{run_id}_false_color_spectrograms.png"), f"{run_id}_false_color_spectrograms.png")
        ]
    else:
        expected_files = [
            (os.path.join(output_csv_path, "Acoustic_Indices.csv"), "Acoustic_Indices.csv"),
            (os.path.join(output_figures_path, "correlation_map.png"), "correlation_map.png"),
            (os.path.join(output_figures_path, "individual_features.png"), "individual_features.png"),
            (os.path.join(output_figures_path, "false_color_spectrograms.png"), "false_color_spectrograms.png")
        ]
    
    # Check which files actually exist
    existing_files = []
    for filepath, filename in expected_files:
        if os.path.exists(filepath):
            existing_files.append(filename)
    
    if existing_files:
        # Files exist - warn user with context-appropriate message
        if run_id:
            # User provided identifier but files still exist (duplicate identifier)
            message = f"⚠️ WARNING: Run identifier '{run_id}' already exists!\n\n"
            message += "The following files will be OVERWRITTEN:\n"
            message += "\n".join(f"  • {f}" for f in existing_files)
            message += "\n\nThis could mean:\n"
            message += "1. You're intentionally re-running the same analysis\n"
            message += "2. You accidentally reused a previous identifier\n\n"
            message += "To avoid overwriting:\n"
            message += "• Change your Run Identifier (e.g., add '_v2', date, etc.)\n"
            message += "• Choose a different output folder\n\n"
            message += "Continue and overwrite existing files?"
        else:
            # No identifier provided
            message = "⚠️ WARNING: The following files will be OVERWRITTEN:\n\n"
            message += "\n".join(f"  • {f}" for f in existing_files)
            message += "\n\nTo avoid overwriting:\n"
            message += "1. Add a Run Identifier (e.g., 'Station1_2024')\n"
            message += "2. Choose a different output folder\n\n"
            message += "Continue and overwrite existing files?"
        
        result = messagebox.askyesno("Overwrite Warning", message, icon='warning')
        if not result:
            print("Analysis cancelled by user to avoid overwriting files.")
            run_metadata.finish_run(success=False, error_message="Cancelled by user to avoid overwriting")
            return
        else:
            # User chose to overwrite - record this in metadata
            run_metadata.add_processing_info(
                overwrite_warning_shown=True,
                files_overwritten=existing_files,
                user_confirmed_overwrite=True
            )
    
    # Save CSV (this should always work if we got this far)
    print("\nSaving results...")
    try:
        # Use run identifier if provided
        if run_id:
            csv_filename = f"{run_id}_Acoustic_Indices.csv"
        else:
            csv_filename = "Acoustic_Indices.csv"
        output_file_path = os.path.join(output_csv_path, csv_filename)
        full_df.to_csv(output_file_path, index=False)
        print(f"  CSV saved: {output_file_path}")
    except Exception as e:
        error_msg = f"Failed to save CSV: {str(e)}"
        errors.append(error_msg)
        print(f"  ERROR: {error_msg}")
        messagebox.showerror("Error", f"Could not save results:\n{error_msg}")
        return
    
    # Check which indices are available
    print("\nChecking available indices...")
    available_indices = [col for col in full_df.columns if col not in ['Date', 'Filename']]
    print(f"  Found {len(available_indices)} indices")
    
    # Check for the six plotting indices
    plot_indices = {
        'indice_one': indice_one,
        'indice_two': indice_two,
        'indice_three': indice_three,
        'indice_four': indice_four,
        'indice_five': indice_five,
        'indice_six': indice_six
    }
    
    missing_plot_indices = []
    for name, index in plot_indices.items():
        if index not in full_df.columns:
            missing_plot_indices.append(index)
            warnings.append(f"Index '{index}' not available for plotting")
    
    if missing_plot_indices:
        print(f"  WARNING: Some indices not available: {', '.join(missing_plot_indices)}")
    
    # Prepare dataframe for plotting
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df.set_index('Date', inplace=True)
    
    # Generate plots with error handling
    print("\nGenerating figures...")
    plots_failed = []
    
    # 1. Correlation map
    try:
        print("  Creating correlation map...")
        fig_correlation, ax_correlation = plot_correlation_map(full_df, R_threshold=0, figsize=(12, 10))
        # Use run identifier for figure naming
        correlation_filename = f"{run_id}_correlation_map.png" if run_id else "correlation_map.png"
        fig_correlation.savefig(os.path.join(output_figures_path, correlation_filename))
        print(f"    Saved: {correlation_filename}")
    except Exception as e:
        error_msg = f"Correlation map failed: {str(e)}"
        plots_failed.append("correlation_map")
        warnings.append(error_msg)
        print(f"    WARNING: {error_msg}")
    
    # 2. Individual features plot
    try:
        print("  Creating individual features plot...")
        fig, ax = plt.subplots(3, 2, sharex=True, squeeze=True, figsize=(16, 14))
        
        # Plot each index with error handling
        fig, ax[0, 0] = safe_plot_index(full_df, indice_one, ax[0, 0], mode, 'indice_one')
        fig, ax[0, 1] = safe_plot_index(full_df, indice_two, ax[0, 1], mode, 'indice_two')
        fig, ax[1, 0] = safe_plot_index(full_df, indice_three, ax[1, 0], mode, 'indice_three')
        fig, ax[1, 1] = safe_plot_index(full_df, indice_four, ax[1, 1], mode, 'indice_four')
        fig, ax[2, 0] = safe_plot_index(full_df, indice_five, ax[2, 0], mode, 'indice_five')
        fig, ax[2, 1] = safe_plot_index(full_df, indice_six, ax[2, 1], mode, 'indice_six')
        
        fig.suptitle('Individual Acoustic Features', fontsize=16)
        fig.tight_layout()
        features_filename = f"{run_id}_individual_features.png" if run_id else "individual_features.png"
        fig.savefig(os.path.join(output_figures_path, features_filename))
        print(f"    Saved: {features_filename}")
    except Exception as e:
        error_msg = f"Individual features plot failed: {str(e)}"
        plots_failed.append("individual_features")
        warnings.append(error_msg)
        print(f"    WARNING: {error_msg}")
    
    # 3. False color spectrogram
    try:
        print("  Creating false color spectrogram...")
        fcs, triplet = false_Color_Spectro(
            df=result_df_per_bin,
            indices=['KURTt_per_bin', 'EVNspCount_per_bin', 'MEANt_per_bin'],
            reverseLUT=False,
            unit='days',
            permut=False,
            display=False,
            figsize=(16, 14)
        )
        spectro_filename = f"{run_id}_false_color_spectrograms.png" if run_id else "false_color_spectrograms.png"
        plt.savefig(os.path.join(output_figures_path, spectro_filename))
        print(f"    Saved: {spectro_filename}")
    except Exception as e:
        error_msg = f"False color spectrogram failed: {str(e)}"
        plots_failed.append("false_color_spectrogram")
        warnings.append(error_msg)
        print(f"    WARNING: {error_msg}")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Files processed: {files_processed}/{len(filename_list)}")
    print(f"Results saved to: {output_folder}")
    
    if errors or warnings:
        print("\nISSUES ENCOUNTERED:")
        if errors:
            print(f"  Errors: {len(errors)}")
            for e in errors[:3]:  # Show first 3 errors
                print(f"    - {e}")
            if len(errors) > 3:
                print(f"    ... and {len(errors)-3} more")
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for w in warnings[:3]:  # Show first 3 warnings
                print(f"    - {w}")
            if len(warnings) > 3:
                print(f"    ... and {len(warnings)-3} more")
    
    # Show completion message to user
    if errors or warnings:
        message = f"Analysis completed with issues:\n\n"
        message += f"✓ Files processed: {files_processed}/{len(filename_list)}\n"
        message += f"✓ CSV saved successfully\n"
        if plots_failed:
            message += f"⚠ Some plots failed: {', '.join(plots_failed)}\n"
        if missing_plot_indices:
            message += f"⚠ Missing indices: {', '.join(missing_plot_indices)}\n"
        message += f"\nResults saved to:\n{output_folder}\n\n"
        message += "Check console for details."
        messagebox.showwarning("Analysis Complete (with warnings)", message)
        
        # Finalize metadata tracking (warnings still count as success)
        success = len(errors) == 0
        error_msg = None if len(errors) == 0 else f"{len(errors)} errors occurred during processing"
        run_metadata.finish_run(success=success, error_message=error_msg)
    else:
        message = f"Analysis completed successfully!\n\n"
        message += f"✓ Files processed: {files_processed}\n"
        message += f"✓ All plots generated\n"
        message += f"✓ All indices computed\n\n"
        message += f"Results saved to:\n{output_folder}"
        messagebox.showinfo("Success", message)
        
        # Finalize metadata tracking for successful completion
        run_metadata.finish_run(success=True)

def select_folder(var):
    """
    Open folder selection dialog.
    
    Args:
        var: StringVar to store the selected path
    """
    # Start in current working directory for easier navigation
    initial_dir = os.getcwd()
    folder_path = filedialog.askdirectory(title="Select Folder", initialdir=initial_dir)
    var.set(folder_path)

def parse_date_and_filename_from_filename(filename):
    """
    Parse date and filename from WAV file naming convention.
    
    Expected format: [prefix]_YYYYMMDD_HHMMSS_[suffix].wav
    
    Args:
        filename: Full path to the WAV file
    
    Returns:
        tuple: (datetime object, filename with numbering) or (None, None) if parsing fails
    """
    try:
        basename = os.path.basename(filename)
        parts = basename.split('_')
        
        # Need at least 4 parts: prefix, date, time, suffix
        if len(parts) < 4:
            raise ValueError("Not enough underscore-separated parts")
        
        # Parse date from second part
        date_str = parts[1]
        if len(date_str) != 8:
            raise ValueError("Date part should be 8 digits (YYYYMMDD)")
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        
        # Parse time from third part
        time_str = parts[2]
        if len(time_str) != 6:
            raise ValueError("Time part should be 6 digits (HHMMSS)")
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        dt = datetime.datetime(year, month, day, hour, minute, second)
        filename_with_numbering = '_'.join(parts[:-1]) + "_" + parts[-1]
        return dt, filename_with_numbering
        
    except Exception as e:
        print(f"    Could not parse '{os.path.basename(filename)}': {str(e)}")
        return None, None

# Create GUI
root = Tk()
root.title("Scikit-Maad Acoustic Indices (Phase 1)")
root.geometry('700x800')  # Increased height for performance controls
root.configure(bg='navy')

# Title
title_label = Label(root, text="Acoustic Indices", font=("Arial", 24, "bold"), bg="navy", fg="white")
title_label.grid(row=0, column=1, padx=10, pady=20, columnspan=4)

# Input Folder
input_folder_var = StringVar()
Label(root, text="Input Folder:", font=("Arial", 16), bg="navy", fg="white").grid(row=1, column=1, padx=10, pady=10)
Entry(root, textvariable=input_folder_var, font=("Arial", 14)).grid(row=1, column=2, padx=10, pady=10)
Button(root, text="Browse", font=("Arial", 16), command=lambda: select_folder(input_folder_var)).grid(row=1, column=3, padx=10, pady=10)

# Output Folder
output_folder_var = StringVar()
Label(root, text="Output Folder:", font=("Arial", 16), bg="navy", fg="white").grid(row=2, column=1, padx=10, pady=10)
Entry(root, textvariable=output_folder_var, font=("Arial", 14)).grid(row=2, column=2, padx=10, pady=10)
Button(root, text="Browse", font=("Arial", 16), command=lambda: select_folder(output_folder_var)).grid(row=2, column=3, padx=10, pady=10)

# Run Identifier (optional - for file naming)
run_identifier_var = StringVar()
Label(root, text="Run Identifier:", font=("Arial", 16), bg="navy", fg="white").grid(row=3, column=1, padx=10, pady=10)
identifier_entry = Entry(root, textvariable=run_identifier_var, font=("Arial", 14))
identifier_entry.grid(row=3, column=2, padx=10, pady=10)
# Helpful hint about preventing overwrites
hint_text = "(optional - prevents overwrites)"
Label(root, text=hint_text, font=("Arial", 11, "italic"), bg="navy", fg="lightgray").grid(row=3, column=3, padx=10, pady=10, sticky='w')

# Time Scale Checkboxes
mode_var_24h = BooleanVar()
mode_var_30min = BooleanVar()
mode_var_20min = BooleanVar()

Label(root, text="Time Scale:", font=("Arial", 16), bg="navy", fg="white").grid(row=5, column=1, padx=10, pady=10)
Checkbutton(root, text="Hourly  ", font=("Arial", 16), variable=mode_var_24h, bg="navy", fg="white", selectcolor="darkgrey").grid(row=4, column=2, padx=10, pady=10)
Checkbutton(root, text="Dataset", font=("Arial", 16), variable=mode_var_30min, bg="navy", fg="white", selectcolor="darkgrey").grid(row=5, column=2, padx=10, pady=10)
Checkbutton(root, text="Manual ", font=("Arial", 16), variable=mode_var_20min, bg="navy", fg="white", selectcolor="darkgrey").grid(row=6, column=2, padx=5, pady=5)

# Time Interval Input
time_interval_var = StringVar()
Label(root, text="(secs)", font=("Arial", 16), bg="navy", fg="white").grid(row=6, column=4, columnspan=2, padx=5, pady=5)
Entry(root, textvariable=time_interval_var, font=("Arial", 14), width=8).grid(row=6, column=3, columnspan=1, padx=5, pady=5)

# Separator line
Label(root, text="─" * 50, font=("Arial", 12), bg="navy", fg="gray").grid(row=7, column=1, columnspan=4, pady=10)

# Frequency Band Controls (Optional - for marine acoustics)
Label(root, text="Marine Acoustic Settings (Optional):", font=("Arial", 16, "bold"), bg="navy", fg="white").grid(row=7, column=1, columnspan=3, padx=10, pady=10)

# Anthrophony frequency range
flim_low_var = StringVar()
Label(root, text="Anthrophony Range (Hz):", font=("Arial", 14), bg="navy", fg="white").grid(row=8, column=1, padx=10, pady=5, sticky='e')
Entry(root, textvariable=flim_low_var, font=("Arial", 12), width=15).grid(row=8, column=2, padx=10, pady=5)
Label(root, text="e.g., 0,1000", font=("Arial", 10), bg="navy", fg="lightgray").grid(row=8, column=3, padx=5, pady=5, sticky='w')
flim_low_var.set("")  # Empty default - will use hardcoded values if not specified

# Biophony frequency range  
flim_mid_var = StringVar()
Label(root, text="Biophony Range (Hz):", font=("Arial", 14), bg="navy", fg="white").grid(row=9, column=1, padx=10, pady=5, sticky='e')
Entry(root, textvariable=flim_mid_var, font=("Arial", 12), width=15).grid(row=9, column=2, padx=10, pady=5)
Label(root, text="e.g., 1000,8000", font=("Arial", 10), bg="navy", fg="lightgray").grid(row=9, column=3, padx=5, pady=5, sticky='w')
flim_mid_var.set("")  # Empty default

# Sensitivity
sensitivity_var = StringVar()
Label(root, text="Sensitivity (S):", font=("Arial", 14), bg="navy", fg="white").grid(row=10, column=1, padx=10, pady=5, sticky='e')
Entry(root, textvariable=sensitivity_var, font=("Arial", 12), width=15).grid(row=10, column=2, padx=10, pady=5)
Label(root, text="default: -169.4", font=("Arial", 10), bg="navy", fg="lightgray").grid(row=10, column=3, padx=5, pady=5, sticky='w')
sensitivity_var.set("")  # Empty default

# Gain
gain_var = StringVar()
Label(root, text="Gain (G):", font=("Arial", 14), bg="navy", fg="white").grid(row=11, column=1, padx=10, pady=5, sticky='e')
Entry(root, textvariable=gain_var, font=("Arial", 12), width=15).grid(row=11, column=2, padx=10, pady=5)
Label(root, text="default: 0", font=("Arial", 10), bg="navy", fg="lightgray").grid(row=11, column=3, padx=5, pady=5, sticky='w')
gain_var.set("")  # Empty default

# Performance Settings
Label(root, text="─" * 50, font=("Arial", 12), bg="navy", fg="gray").grid(row=13, column=1, columnspan=4, pady=10)
Label(root, text="Performance Settings:", font=("Arial", 16, "bold"), bg="navy", fg="white").grid(row=14, column=1, columnspan=3, padx=10, pady=10)

# Parallel processing checkbox
parallel_var = BooleanVar()
parallel_var.set(True)  # Default to enabled
Checkbutton(root, text="Enable Parallel Processing (faster)", font=("Arial", 14), variable=parallel_var, 
           bg="navy", fg="white", selectcolor="darkgrey").grid(row=15, column=1, columnspan=2, padx=10, pady=5, sticky='w')

# Performance comparison checkbox
compare_performance_var = BooleanVar()
compare_performance_var.set(False)  # Default to disabled
Checkbutton(root, text="Compare Performance (benchmark mode)", font=("Arial", 14), variable=compare_performance_var,
           bg="navy", fg="white", selectcolor="darkgrey").grid(row=16, column=1, columnspan=2, padx=10, pady=5, sticky='w')

# Run Button
Button(root, text="Run Analysis", font=("Arial", 20), command=run_analysis, width=14).grid(row=17, column=1, columnspan=4, padx=10, pady=20)

root.mainloop()