#!/usr/bin/env python3
"""
Test script to verify the marine indices fixes.
"""

import os
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processing.core_processing import (
    convert_results_to_dataframes,
    parse_date_and_filename_from_filename,
    process_files_sequential,
)


def test_marine_indices_fix():
    print("🧪 TESTING MARINE INDICES FIXES")
    print("=" * 60)

    # Test with sample files
    input_folder = "test_wav_files"

    # Find WAV files
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"❌ No WAV files found in {input_folder}")
        return

    print(f"📁 Found {len(audio_files)} files")

    # Parse filenames
    date_list = []
    filename_list = []
    for file in audio_files:
        dt, filename_with_numbering = parse_date_and_filename_from_filename(file)
        if dt and filename_with_numbering:
            date_list.append(dt)
            filename_list.append(filename_with_numbering)

    # Test with marine corrections enabled (should give 60 indices)
    processing_params = {
        'mode': 'dataset',
        'time_interval': 0,
        'flim_low': [0, 1500],      # Marine-appropriate frequency bands
        'flim_mid': [1500, 8000],   # Marine-appropriate frequency bands
        'sensitivity': -185.5,      # Original implementation values
        'gain': 20.0,               # Original implementation values
        'calculate_marine': True    # Enable marine corrections
    }

    print("🔬 Processing with marine corrections enabled...")
    results = process_files_sequential(audio_files, processing_params)

    # Convert to DataFrame
    result_df, result_df_per_bin = convert_results_to_dataframes(results, filename_list, date_list, 'dataset')

    print("\n📊 RESULTS:")
    print(f"   Shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns")

    # Check if we have exactly 60 columns (Date + Filename + 58 acoustic indices)
    expected_cols = 60  # Date, Filename + 58 acoustic indices
    if result_df.shape[1] == expected_cols:
        print(f"   ✅ Correct number of columns: {result_df.shape[1]} (expected {expected_cols})")
    else:
        print(f"   ❌ Wrong number of columns: {result_df.shape[1]} (expected {expected_cols})")

    # Check which marine indices are present
    acoustic_cols = [col for col in result_df.columns if col not in ['Date', 'Filename']]
    marine_affected = ['NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI']
    marine_with_suffix = ['NDSI_marine', 'rBA_marine', 'AnthroEnergy_marine', 'BioEnergy_marine', 'BI_marine']

    print("\n🌊 MARINE INDICES CHECK:")
    present_marine = [idx for idx in marine_affected if idx in acoustic_cols]
    present_marine_suffix = [idx for idx in marine_with_suffix if idx in acoustic_cols]

    print(f"   Marine indices (corrected): {present_marine}")
    print(f"   Marine indices (old suffix): {present_marine_suffix}")

    if len(present_marine) == 5 and len(present_marine_suffix) == 0:
        print("   ✅ Marine indices correctly replaced (no '_marine' suffixes)")
    elif len(present_marine_suffix) > 0:
        print("   ❌ Still has '_marine' suffixed indices - fix not complete")
    else:
        print("   ⚠️ Missing some marine indices")

    # Show some sample values to verify they're the marine-corrected versions
    print("\n📋 SAMPLE VALUES (first file):")
    for idx in marine_affected[:3]:  # Show first 3
        if idx in result_df.columns:
            val = result_df[idx].iloc[0]
            print(f"   {idx}: {val:.6f}")

    # Compare with original values if available
    if os.path.exists("regression_testing/jared_version/Acoustic_Indices.csv"):
        print("\n🔍 COMPARING WITH ORIGINAL VERSION:")
        try:
            original_df = pd.read_csv("regression_testing/jared_version/Acoustic_Indices.csv")

            for idx in marine_affected[:3]:  # Compare first 3
                if idx in result_df.columns and idx in original_df.columns:
                    new_val = result_df[idx].iloc[0]
                    orig_val = original_df[idx].iloc[0]
                    diff = abs(new_val - orig_val)
                    pct_change = (diff / abs(orig_val)) * 100 if orig_val != 0 else 0
                    print(f"   {idx}: {orig_val:.6f} → {new_val:.6f} ({pct_change:.1f}% change)")
        except Exception as e:
            print(f"   Could not compare with original: {e}")

    # Save results for inspection
    output_file = "test_fixed_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("🎉 MARINE INDICES FIX TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_marine_indices_fix()
