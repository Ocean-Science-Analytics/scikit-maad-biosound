# This script using the sci-kit maad package to analyze acoustic indices from wav files and generate a csv file and figures displaying the measured indices

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
indice_six='ROItotal'

# Function to run the analysis
def run_analysis():
    if mode_var_24h.get() and mode_var_30min.get() and mode_var_20min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
    elif mode_var_24h.get() and mode_var_30min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
    elif mode_var_24h.get() and mode_var_20min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
    elif mode_var_30min.get() and mode_var_20min.get():
        messagebox.showerror("Error", "Please select only one time scale option.")
    elif not mode_var_24h.get() and not mode_var_30min.get() and not mode_var_20min.get():
        messagebox.showerror("Error", "Please select one time scale option.")
    else:
        input_folder = input_folder_var.get()
        output_folder = output_folder_var.get()
        if mode_var_24h.get():
            mode = "24h"
        elif mode_var_30min.get():
            mode = "30min"
        elif mode_var_20min.get():
            mode = "20min"

        df = pd.DataFrame()
        audio_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))

        date_list = []
        filename_list = []
        for file in audio_files:
            dt, filename_with_numbering = parse_date_and_filename_from_filename(file)
            if dt and filename_with_numbering:
                date_list.append(dt)
                filename_list.append(filename_with_numbering)

        df['Date'] = date_list
        df['Filename'] = filename_list

        result_df = pd.DataFrame()
        result_df_per_bin = pd.DataFrame()

        if not mode_var_24h.get() and not mode_var_30min.get() and mode_var_20min.get():
            time_interval = int(time_interval_var.get())  # Retrieve the time interval value
            for filename in filename_list:
                fullfilename = os.path.join(input_folder, filename)

                try:
                    wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)
                    S = -185.5
                    G = 20

                    # Calculate the total number of samples in the file
                    total_samples = len(wave)

                    # Calculate the number of samples per interval based on the specified time interval
                    samples_per_interval = int(fs * time_interval)  # Convert time interval to samples

                    # Iterate over the file in intervals
                    interval_duration = time_interval
                    previous_segment_wave = None
                    for start_sample in range(0, total_samples, samples_per_interval):
                        end_sample = min(start_sample + samples_per_interval, total_samples)
                        interval_length = (end_sample - start_sample) / fs
                        if interval_length < 1 and previous_segment_wave is not None:
                            previous_segment_wave = np.concatenate([previous_segment_wave, wave[start_sample:end_sample]])
                            #messagebox.showinfo("Note:", f"The remainder samples (<1sec) of {filename} were added on to last {time_interval} second sample.")
                            continue
                        elif interval_length < 1:
                            messagebox.showinfo("Note", f"The end of {filename} was skipped due to unequal time intervals")
                            continue
                        segment_wave = wave[start_sample:end_sample]
                        if previous_segment_wave is not None:
                            segment_wave = np.concatenate([previous_segment_wave, segment_wave])
                        previous_segment_wave = segment_wave

                        #if interval_length < 1:
                            #messagebox.showinfo("Note", f"The end of {filename} was skipped due to unequal time intervals")
                            #continue
                        #segment_wave = wave[start_sample:end_sample]

                        Sxx_power, tn, fn, ext = sound.spectrogram (
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

                except Exception as e:
                    print(f"Error processing file '{filename}': {e}")

        elif mode_var_24h.get() or mode_var_30min.get():
            for filename in filename_list:
                    fullfilename = os.path.join(input_folder, filename)

                    try:
                        wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)
                        S = -185.5
                        G = 20

                        Sxx_power, tn, fn, ext = sound.spectrogram (
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

                    except Exception as e:
                        print(f"Error processing file '{filename}': {e}")    

        full_df = df.merge(result_df, how='inner', on='Filename')

        # Adjust the Date column for each new filename
        if mode_var_20min.get():
            for filename in filename_list:
                filename_rows = full_df[full_df['Filename'] == filename]
                time_diff = pd.to_timedelta(np.arange(len(filename_rows)) * time_interval, unit='s')
                full_df.loc[full_df['Filename'] == filename, 'Date'] += time_diff

        output_file_path = os.path.join(output_folder, "Acoustic_Indices.csv")
        full_df.to_csv(output_file_path, index=False)

        output_figures_path = os.path.join(output_folder, "output_figures")
        os.makedirs(output_figures_path, exist_ok=True)

        full_df['Date'] = pd.to_datetime(full_df['Date'])
        full_df.set_index('Date', inplace=True)

        fig_correlation, ax_correlation = plot_correlation_map(full_df, R_threshold=0, figsize=(12, 10))
        fig_correlation.savefig(os.path.join(output_figures_path, "correlation_map.png"))

        fig, ax = plt.subplots(3, 2, sharex=True, squeeze=True, figsize=(16, 14))

        fig, ax[0, 0] = plot_features(full_df[[indice_one]], norm=False, mode=mode, ax=ax[0, 0])
        fig, ax[0, 1] = plot_features(full_df[[indice_two]], norm=False, mode=mode, ax=ax[0, 1])
        fig, ax[1, 0] = plot_features(full_df[[indice_three]], norm=False, mode=mode, ax=ax[1, 0])
        fig, ax[1, 1] = plot_features(full_df[[indice_four]], norm=False, mode=mode, ax=ax[1, 1])
        fig, ax[2, 0] = plot_features(full_df[[indice_five]], norm=False, mode=mode, ax=ax[2, 0])
        fig, ax[2, 1] = plot_features(full_df[[indice_six]], norm=False, mode=mode, ax=ax[2, 1])
        fig.savefig(os.path.join(output_figures_path, "individual_features.png"))

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
        #plt.show()

# Function to select folder
def select_folder(var):
    folder_path = filedialog.askdirectory(title="Select Folder")
    var.set(folder_path)

# Function to parse date and filename from filename
def parse_date_and_filename_from_filename(filename):
    try:
        basename = os.path.basename(filename)
        parts = basename.split('_')
        year = int(parts[1][:4])
        month = int(parts[1][4:6])
        day = int(parts[1][6:8])
        hour = int(parts[2][:2])
        minute = int(parts[2][2:4])
        second = int(parts[2][4:6])
        dt = datetime.datetime(year, month, day, hour, minute, second)
        filename_with_numbering = '_'.join(parts[:-1]) + "_" + parts[-1]
        return dt, filename_with_numbering
    except Exception as e:
        print(f"Error parsing date from filename '{filename}': {e}")
        return None, None

# Create GUI
root = Tk()
root.title("Scikit-Maad Acoustic Indices")
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

