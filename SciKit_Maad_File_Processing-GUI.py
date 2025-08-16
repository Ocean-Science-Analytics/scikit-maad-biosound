# This script uses the sci-kit maad package to analyze acoustic indices from wav files and generate a csv file and figures displaying the measured indices
# Phase 1 Fix: Robust error handling and graceful degradation

# Created by Jared Stephens 12/27/2023
# Modified by M. Weirathmueller August 2025 - bug fix + documentation updates

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from maad import sound, features
from maad.util import date_parser, plot_correlation_map, plot_features_map, plot_features, false_Color_Spectro
from tkinter import Tk, filedialog, Label, Entry, Button, Frame, SOLID, StringVar, IntVar, Checkbutton, BooleanVar, messagebox
import datetime
import traceback

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
    Main analysis function with comprehensive error handling.
    """
    # Track errors and warnings for final report
    errors = []
    warnings = []
    
    print("\n" + "="*60)
    print("STARTING ACOUSTIC ANALYSIS")
    print("="*60)
    
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
    
    # Process files based on mode
    print("\nProcessing audio files...")
    
    if not mode_var_24h.get() and not mode_var_30min.get() and mode_var_20min.get():
        # Manual time interval mode
        time_interval = int(time_interval_var.get())
        for i, filename in enumerate(filename_list, 1):
            print(f"  Processing file {i}/{len(filename_list)}: {filename}")
            fullfilename = os.path.join(input_folder, filename)
            
            try:
                wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)
                S = -185.5
                G = 20
                
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
                        flim_low=[0, 1500],
                        flim_mid=[1500, 8000],
                        flim_hi=[8000, 40000],
                        gain=G,
                        sensitivity=S,
                        verbose=False, 
                        R_compatible='soundecology',
                        mask_param1=6, 
                        mask_param2=0.5,
                        display=False
                    )
                    
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
                S = -185.5
                G = 20
                
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
                    flim_low=[0, 1500],
                    flim_mid=[1500, 8000],
                    flim_hi=[8000, 40000],
                    gain=G,
                    sensitivity=S,
                    verbose=False,
                    R_compatible='soundecology',
                    mask_param1=6,
                    mask_param2=0.5,
                    display=False
                )
                
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
    
    # Save CSV (this should always work if we got this far)
    print("\nSaving results...")
    try:
        output_file_path = os.path.join(output_folder, "Acoustic_Indices.csv")
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
    
    # Create output figures directory
    output_figures_path = os.path.join(output_folder, "output_figures")
    os.makedirs(output_figures_path, exist_ok=True)
    
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
        fig_correlation.savefig(os.path.join(output_figures_path, "correlation_map.png"))
        print("    Saved: correlation_map.png")
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
        fig.savefig(os.path.join(output_figures_path, "individual_features.png"))
        print("    Saved: individual_features.png")
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
        plt.savefig(os.path.join(output_figures_path, "false_color_spectrograms.png"))
        print("    Saved: false_color_spectrograms.png")
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
    else:
        message = f"Analysis completed successfully!\n\n"
        message += f"✓ Files processed: {files_processed}\n"
        message += f"✓ All plots generated\n"
        message += f"✓ All indices computed\n\n"
        message += f"Results saved to:\n{output_folder}"
        messagebox.showinfo("Success", message)

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
root.geometry('608x500')
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

# Time Scale Checkboxes
mode_var_24h = BooleanVar()
mode_var_30min = BooleanVar()
mode_var_20min = BooleanVar()

Label(root, text="Time Scale:", font=("Arial", 16), bg="navy", fg="white").grid(row=4, column=1, padx=10, pady=10)
Checkbutton(root, text="Hourly  ", font=("Arial", 16), variable=mode_var_24h, bg="navy", fg="white", selectcolor="darkgrey").grid(row=3, column=2, padx=10, pady=10)
Checkbutton(root, text="Dataset", font=("Arial", 16), variable=mode_var_30min, bg="navy", fg="white", selectcolor="darkgrey").grid(row=4, column=2, padx=10, pady=10)
Checkbutton(root, text="Manual ", font=("Arial", 16), variable=mode_var_20min, bg="navy", fg="white", selectcolor="darkgrey").grid(row=5, column=2, padx=5, pady=5)

# Time Interval Input
time_interval_var = StringVar()
Label(root, text="(secs)", font=("Arial", 16), bg="navy", fg="white").grid(row=5, column=4, columnspan=2, padx=5, pady=5)
Entry(root, textvariable=time_interval_var, font=("Arial", 14), width=8).grid(row=5, column=3, columnspan=1, padx=5, pady=5)

# Run Button
Button(root, text="Run Analysis", font=("Arial", 20), command=run_analysis, width=14).grid(row=8, column=1, columnspan=4, padx=10, pady=20)

root.mainloop()