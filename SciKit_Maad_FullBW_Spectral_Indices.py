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

# Revised by ELF with help from ChatGPT 06/03/2024
    # Only contains calculations for NDSI, BioEnergy, AnthroEnergy, rBA, BI, ACI, EPS, ROU, ADI, AEI
    # Switched anthrophony to 0-1,000 Hz, and biophony to above that. These use only flim_low and flim_med so
    # got rid of flim_hi in this script. This is ONLY for use with the indices above. Redefines calculate_biophony_anthrophony to
    # accomodate the change of flims and calculates ACI separately in full bandwidth and in 1,000 Hz bins
    # ACI now exported as mean over bandwidth and the FFT based band values in a separate column
    # GUI Updated to allow user to select anthrophony and biophony bands and also the sensitivity (S) and gain (G)

#############################################################################################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from maad import sound, features
from maad.util import date_parser
from tkinter import Tk, filedialog, Label, Entry, Button, Checkbutton, StringVar, BooleanVar, messagebox
import datetime

# Define the spectral indices to calculate
SPECIFIED_INDICES = [
    'ROU', 'ADI', 'AEI'
]

# Function to calculate ACI
def calculate_aci(Sxx_power):
    try:
        aci = features.acoustic_complexity_index(Sxx_power)
        if isinstance(aci, tuple):
            aci_2d_array = aci[0]
            aci_1d_array = aci[1]
            aci_mean_value = aci[2]
        else:
            aci_2d_array = aci
            aci_1d_array = np.mean(aci, axis=0)
            aci_mean_value = np.mean(aci_1d_array)
        return aci_2d_array, aci_1d_array, aci_mean_value
    except Exception as e:
        print(f"Error in calculate_aci: {e}")
        return None, None, None

# Function to calculate Anthrophony and Biophony
def calculate_anthrophony_biophony(Sxx_power, fn, flim_low, flim_mid):
    anthrophony_power = Sxx_power[(fn >= flim_low[0]) & (fn < flim_low[1])]
    biophony_power = Sxx_power[(fn >= flim_mid[0]) & (fn < flim_mid[1])]
    anthrophony_energy = np.sum(anthrophony_power)
    biophony_energy = np.sum(biophony_power)
    return anthrophony_energy, biophony_energy

# Function to process each segment or whole file
def process_segment(wave, fs, S, G, filename, flim_low, flim_mid):
    Sxx_power, tn, fn, ext = sound.spectrogram(
        x=wave,
        fs=fs,
        window='hann',
        nperseg=512,
        noverlap=512 // 2,
        verbose=False,
        display=False,
        savefig=None
    )

    # Calculate ACI over the full bandwidth
    aci_2d_array, aci_1d_array, aci_mean_value = calculate_aci(Sxx_power)

    # Calculate specified spectral indices
    spectral_indices, _ = features.all_spectral_alpha_indices(
        Sxx_power=Sxx_power,
        tn=tn,
        fn=fn,
        flim_low=[0, fn[-1]],  # Full bandwidth
        flim_mid=[0, fn[-1]],  # Full bandwidth for consistency
        gain=G,
        sensitivity=S,
        verbose=False,
        R_compatible='soundecology',
        mask_param1=6,
        mask_param2=0.5,
        display=False,
        indices=SPECIFIED_INDICES  # Calculate only specified indices
    )

    anthro_energy, bio_energy = calculate_anthrophony_biophony(Sxx_power, fn, flim_low, flim_mid)

    # Prepare results dictionary to ensure single value per index
    result_dict = spectral_indices.mean().to_dict()  # assuming mean to get a single value if needed
    result_dict['ACI'] = aci_mean_value
    result_dict['ACI_by_band'] = aci_1d_array
    result_dict['Filename'] = filename
    result_dict['AnthroEnergy'] = anthro_energy
    result_dict['BioEnergy'] = bio_energy

    # Calculate NDSI
    result_dict['NDSI'] = (bio_energy - anthro_energy) / (bio_energy + anthro_energy)
    # Calculate rBA (Relative Biophony-Anthrophony)
    result_dict['rBA'] = bio_energy / (bio_energy + anthro_energy)
    # Calculate BI (Biotic Index) as an example, adjust if different calculation needed
    result_dict['BI'] = bio_energy / anthro_energy

    # Add frequency resolution to the results
    frequency_resolution = fs / 512
    result_dict['FrequencyResolution'] = frequency_resolution

    return result_dict

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

        # Retrieve frequency limits and S, G values from user input
        flim_low = list(map(int, flim_low_var.get().split(',')))
        flim_mid = list(map(int, flim_mid_var.get().split(',')))
        S = float(sensitivity_var.get())
        G = float(gain_var.get())

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

        result_list = []

        for filename in filename_list:
            fullfilename = os.path.join(input_folder, filename)

            try:
                wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)

                if mode_var_20min.get():
                    time_interval = int(time_interval_var.get())  # Retrieve the time interval value
                    # Calculate the total number of samples in the file
                    total_samples = len(wave)
                    # Calculate the number of samples per interval based on the specified time interval
                    samples_per_interval = int(fs * time_interval)  # Convert time interval to samples
                    previous_segment_wave = None
                    for start_sample in range(0, total_samples, samples_per_interval):
                        end_sample = min(start_sample + samples_per_interval, total_samples)
                        interval_length = (end_sample - start_sample) / fs
                        if interval_length < 1 and previous_segment_wave is not None:
                            previous_segment_wave = np.concatenate([previous_segment_wave, wave[start_sample:end_sample]])
                            continue
                        elif interval_length < 1:
                            messagebox.showinfo("Note", f"The end of {filename} was skipped due to unequal time intervals")
                            continue
                        segment_wave = wave[start_sample:end_sample]
                        if previous_segment_wave is not None:
                            segment_wave = np.concatenate([previous_segment_wave, segment_wave])
                        previous_segment_wave = segment_wave

                        # Process the segment
                        indices_dict = process_segment(segment_wave, fs, S, G, filename, flim_low, flim_mid)
                        result_list.append(indices_dict)
                else:
                    # Process the entire file
                    indices_dict = process_segment(wave, fs, S, G, filename, flim_low, flim_mid)
                    result_list.append(indices_dict)

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

        result_df = pd.DataFrame(result_list)

        # Ensure 'Filename' column exists in both DataFrames before merging
        if 'Filename' not in df.columns:
            raise KeyError("'Filename' column is missing in df")
        if 'Filename' not in result_df.columns:
            raise KeyError("'Filename' column is missing in result_df")

        full_df = df.merge(result_df, how='inner', on='Filename')

        # Adjust the Date column for each new filename
        if mode_var_20min.get():
            for filename in filename_list:
                filename_rows = full_df[full_df['Filename'] == filename]
                time_diff = pd.to_timedelta(np.arange(len(filename_rows)) * time_interval, unit='s')
                full_df.loc[full_df['Filename'] == filename, 'Date'] += time_diff

        # Filter columns to include only the specified indices and the required columns
        columns_to_export = ['Date', 'Filename', 'ACI', 'ACI_by_band', 'FrequencyResolution', 'AnthroEnergy', 'BioEnergy', 'NDSI', 'rBA', 'BI'] + SPECIFIED_INDICES
        full_df = full_df[columns_to_export]

        output_file_path = os.path.join(output_folder, "Acoustic_Indices.csv")
        full_df.to_csv(output_file_path, index=False)

        messagebox.showinfo("Success", f"Analysis complete. Results saved to {output_file_path}")

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
root.geometry('650x700')  # Adjusted to accommodate additional input fields
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

Label(root, text="Time Scale:", font=("Arial", 16), bg="navy", fg="white").grid(row=3, column=1, padx=10, pady=10)
Checkbutton(root, text="Hourly", font=("Arial", 16), variable=mode_var_24h, bg="navy", fg="white", selectcolor="darkgrey").grid(row=3, column=2, padx=10, pady=10)
Checkbutton(root, text="Dataset", font=("Arial", 16), variable=mode_var_30min, bg="navy", fg="white", selectcolor="darkgrey").grid(row=4, column=2, padx=10, pady=10)
Checkbutton(root, text="Manual", font=("Arial", 16), variable=mode_var_20min, bg="navy", fg="white", selectcolor="darkgrey").grid(row=5, column=2, padx=5, pady=5)

# Time Interval Input
time_interval_var = StringVar()
Label(root, text="(secs)", font=("Arial", 16), bg="navy", fg="white").grid(row=5, column=4, columnspan=2, padx=5, pady=5)
Entry(root, textvariable=time_interval_var, font=("Arial", 14), width=8).grid(row=5, column=3, columnspan=1, padx=5, pady=5)

# Frequency Limits Input
flim_low_var = StringVar()
flim_mid_var = StringVar()

Label(root, text="Anthrophony Frequency Range (Hz):", font=("Arial", 16), bg="navy", fg="white").grid(row=6, column=1, padx=10, pady=10)
Entry(root, textvariable=flim_low_var, font=("Arial", 14)).grid(row=6, column=2, padx=10, pady=10)
flim_low_var.set("0,1000")  # default value

Label(root, text="Biophony Frequency Range (Hz):", font=("Arial", 16), bg="navy", fg="white").grid(row=7, column=1, padx=10, pady=10)
Entry(root, textvariable=flim_mid_var, font=("Arial", 14)).grid(row=7, column=2, padx=10, pady=10)
flim_mid_var.set("1000,8000")  # default value

# Sensitivity and Gain Input
sensitivity_var = StringVar()
gain_var = StringVar()

Label(root, text="Microphone Sensitivity (S):", font=("Arial", 16), bg="navy", fg="white").grid(row=8, column=1, padx=10, pady=10)
Entry(root, textvariable=sensitivity_var, font=("Arial", 14)).grid(row=8, column=2, padx=10, pady=10)
sensitivity_var.set("-169.4")  # default value

Label(root, text="Gain (G):", font=("Arial", 16), bg="navy", fg="white").grid(row=9, column=1, padx=10, pady=10)
Entry(root, textvariable=gain_var, font=("Arial", 14)).grid(row=9, column=2, padx=10, pady=10)
gain_var.set("0")  # default value

# Run Button
Button(root, text="Run Analysis", font=("Arial", 20), command=run_analysis, width=14).grid(row=10, column=1, columnspan=4, padx=10, pady=20)

root.mainloop()