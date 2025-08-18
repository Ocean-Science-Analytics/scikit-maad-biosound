#!/usr/bin/env python3
"""
Standalone audio file processor for multiprocessing.
Separated from GUI to avoid import issues in spawned processes.

This module contains all the processing logic extracted from the main GUI
to ensure proper multiprocessing on macOS without GUI dependencies.
"""

import os
import sys
import numpy as np
import pandas as pd
import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maad import sound, features

# Import debug configuration
try:
    from gui.debug_config import verbose_print
except ImportError:
    # Fallback if debug_config not available (for standalone use)
    def verbose_print(message):
        pass

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


def calculate_marine_corrected_indices(Sxx_power, fn, flim_low, flim_mid, S=-35.0, G=0.0):
    """
    Calculate marine-corrected versions of the 5 indices that require frequency band corrections.
    
    According to Sylvain Haupert (scikit-maad developer correspondence):
    - Only NDSI, BioEnergy, AnthroEnergy, rBA, and BI need marine-specific frequency assignments
    - These should REPLACE the original indices, not add new ones
    - Anthrophony: flim_low range (vessel noise, typically 0-1500 Hz)
    - Biophony: flim_mid range (biological sounds, typically 1500-8000 Hz)
    
    Args:
        Sxx_power: Power spectrogram
        fn: Frequency array
        flim_low: Anthrophony frequency range [min, max]
        flim_mid: Biophony frequency range [min, max]
        S: Sensitivity (dB)
        G: Gain (dB)
    
    Returns:
        dict: Dictionary of corrected marine indices (without '_marine' suffix)
    """
    # Calculate anthrophony and biophony energies
    anthrophony_energy, biophony_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )
    
    # Calculate marine-corrected indices (these will REPLACE the originals)
    corrected_indices = {}
    
    # NDSI (Normalized Difference Soundscape Index) - Marine corrected
    if (biophony_energy + anthrophony_energy) > 0:
        corrected_indices['NDSI'] = (biophony_energy - anthrophony_energy) / (biophony_energy + anthrophony_energy)
    else:
        corrected_indices['NDSI'] = 0
    
    # Energy metrics - Marine corrected
    corrected_indices['BioEnergy'] = biophony_energy
    corrected_indices['AnthroEnergy'] = anthrophony_energy
    
    # rBA (ratio of biophony to anthrophony) - Marine corrected
    if anthrophony_energy > 0:
        corrected_indices['rBA'] = biophony_energy / anthrophony_energy
    else:
        corrected_indices['rBA'] = np.inf if biophony_energy > 0 else 0
    
    # BI (Bioacoustic Index) - Marine corrected, calculated on biophony band
    bio_mask = (fn >= flim_mid[0]) & (fn < flim_mid[1])
    if np.any(bio_mask):
        bio_spectrum = np.mean(Sxx_power[bio_mask, :], axis=1) if len(Sxx_power.shape) > 1 else Sxx_power[bio_mask]
        corrected_indices['BI'] = np.sum(bio_spectrum * np.log10(bio_spectrum + 1e-10))
    else:
        corrected_indices['BI'] = 0
    
    return corrected_indices


def calculate_indices_for_segment(wave, fs, sensitivity, gain, flim_low, flim_mid, calculate_marine, filename):
    """
    Calculate indices for a single segment of audio.
    
    Returns:
        dict: Calculated indices or None if failed
    """
    try:
        verbose_print(f"    Starting spectrogram for {filename}")
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
        
        verbose_print(f"    Spectrogram complete, starting temporal indices")
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
        
        verbose_print(f"    Temporal indices complete, starting spectral indices")
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
        
        verbose_print(f"    Spectral indices complete, processing marine corrections")
        
        verbose_print(f"    Combining indices")
        # Combine all indices - ensure they are dictionaries
        # Handle case where indices might be DataFrames or Series
        if hasattr(temporal_indices, 'iloc') and len(temporal_indices) > 0:
            temporal_dict = temporal_indices.iloc[0].to_dict() if len(temporal_indices) > 0 else {}
        elif isinstance(temporal_indices, dict):
            temporal_dict = temporal_indices
        else:
            temporal_dict = {}
        
        if hasattr(spectral_indices, 'iloc') and len(spectral_indices) > 0:
            spectral_dict = spectral_indices.iloc[0].to_dict() if len(spectral_indices) > 0 else {}
        elif isinstance(spectral_indices, dict):
            spectral_dict = spectral_indices
        else:
            spectral_dict = {}
        
        # Start with temporal and spectral indices
        all_indices = {**temporal_dict, **spectral_dict}
        
        # Apply marine corrections if requested (REPLACE the 5 affected indices)
        if calculate_marine:
            verbose_print(f"    Applying marine corrections to 5 indices (NDSI, BioEnergy, AnthroEnergy, rBA, BI)")
            marine_corrections = calculate_marine_corrected_indices(
                Sxx_power, fn, flim_low, flim_mid, sensitivity, gain
            )
            # Replace the original 5 indices with marine-corrected versions
            all_indices.update(marine_corrections)
            verbose_print(f"    Marine corrections applied - 5 indices replaced")
        else:
            verbose_print(f"    No marine corrections requested - using standard calculations")
        all_indices['Filename'] = filename
        
        verbose_print(f"    Successfully calculated indices for {filename}")
        return {
            'indices': all_indices,
            'indices_per_bin': spectral_indices_per_bin
        }
        
    except Exception as e:
        verbose_print(f"      ERROR calculating indices for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_single_file_standalone(filename_args):
    """
    Process a single audio file for multiprocessing - standalone version.
    
    This function replicates the exact functionality of the GUI's process_single_file
    function but is isolated from GUI imports to work with multiprocessing on macOS.
    
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
        verbose_print(f"    Processing mode: '{mode}', time_interval: {time_interval}")
        
        # Parse date and filename
        parsed_date, filename_with_numbering = parse_date_and_filename_from_filename(filename)
        if parsed_date is None:
            print(f"    Skipping {os.path.basename(filename)} - invalid filename format")
            return None
        
        # Load audio file
        verbose_print(f"    Loading audio file: {os.path.basename(filename)}")
        wave, fs = sound.load(filename=filename, channel='left', detrend=True, verbose=False)
        verbose_print(f"    Loaded {len(wave)} samples at {fs} Hz")
        
        # Process based on mode
        results = []
        
        if mode in ["manual", "hourly"]:  # Time-based segmentation
            mode_description = "hourly (60-minute segments)" if mode == "hourly" else f"manual ({time_interval}s intervals)"
            verbose_print(f"    Processing in {mode_description}")
            total_samples = len(wave)
            samples_per_interval = int(fs * time_interval)
            
            verbose_print(f"    Total samples: {total_samples}, samples per interval: {samples_per_interval}")
            
            previous_segment_wave = None
            segment_count = 0
            
            for start_sample in range(0, total_samples, samples_per_interval):
                end_sample = min(start_sample + samples_per_interval, total_samples)
                interval_length = (end_sample - start_sample) / fs
                
                verbose_print(f"    Processing segment {segment_count + 1}: samples {start_sample}-{end_sample}, length {interval_length:.2f}s")
                
                # Handle short segments (< 1 second)
                if interval_length < 1 and previous_segment_wave is not None:
                    verbose_print(f"    Short segment ({interval_length:.2f}s), concatenating with previous")
                    previous_segment_wave = np.concatenate([previous_segment_wave, wave[start_sample:end_sample]])
                    continue
                elif interval_length < 1:
                    verbose_print(f"    Short segment ({interval_length:.2f}s), skipping")
                    continue
                    
                segment_wave = wave[start_sample:end_sample]
                if previous_segment_wave is not None:
                    verbose_print(f"    Concatenating with previous segment")
                    segment_wave = np.concatenate([previous_segment_wave, segment_wave])
                previous_segment_wave = segment_wave
                
                # Calculate indices for this segment
                segment_filename = f"{os.path.basename(filename)}_segment_{segment_count + 1}"
                segment_result = calculate_indices_for_segment(
                    segment_wave, fs, sensitivity, gain, flim_low, flim_mid, 
                    calculate_marine, segment_filename
                )
                if segment_result:
                    verbose_print(f"    Successfully processed segment {segment_count + 1}")
                    results.append(segment_result)
                else:
                    verbose_print(f"    Failed to process segment {segment_count + 1}")
                
                segment_count += 1
                
        else:
            # Process entire file (mode: "dataset")
            verbose_print(f"    Processing entire file in dataset mode (full duration)")
            file_result = calculate_indices_for_segment(
                wave, fs, sensitivity, gain, flim_low, flim_mid,
                calculate_marine, os.path.basename(filename)
            )
            if file_result:
                verbose_print(f"    Successfully processed entire file")
                results.append(file_result)
            else:
                verbose_print(f"    Failed to process file")
        
        verbose_print(f"    Processing complete: {len(results)} results generated")
        
        # Return results in the exact same format as GUI version
        return {
            'filename': filename,
            'parsed_date': parsed_date,
            'filename_with_numbering': filename_with_numbering,
            'results': results
        }
        
    except Exception as e:
        print(f"    ERROR processing {os.path.basename(filename)}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None