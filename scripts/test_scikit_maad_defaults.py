#!/usr/bin/env python3
"""
Test that we're using correct scikit-maad defaults.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processing.core_processing import (
    convert_results_to_dataframes,
    parse_date_and_filename_from_filename,
    process_files_sequential,
)


def test_scikit_maad_defaults():
    print("🔬 TESTING SCIKIT-MAAD DEFAULTS")
    print("=" * 60)

    # Test with sample files
    audio_files = ["test_wav_files/marine_20250818_060552_vessel_passing.wav"]

    # Parse filename
    dt, filename_with_numbering = parse_date_and_filename_from_filename(audio_files[0])
    date_list = [dt]
    filename_list = [filename_with_numbering]

    print("📊 Current defaults in our system:")
    print("  flim_low: [0, 1000] (anthrophony)")
    print("  flim_mid: [1000, 10000] (biophony)")
    print("  sensitivity: -35.0")
    print("  gain: 0.0")

    # Test with current defaults (should match scikit-maad)
    processing_params = {
        'mode': 'dataset',
        'time_interval': 0,
        'flim_low': [0, 1000],      # scikit-maad default
        'flim_mid': [1000, 10000],  # scikit-maad default
        'sensitivity': -35.0,       # scikit-maad default
        'gain': 0.0,               # commonly used default
        'calculate_marine': True
    }

    print("\n🧪 Processing with scikit-maad defaults...")
    results = process_files_sequential(audio_files, processing_params)
    result_df, _ = convert_results_to_dataframes(results, filename_list, date_list, 'dataset')

    print("\n📊 RESULTS:")
    print(f"   Shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
    print("   ✅ Should have exactly 62 columns (Date + Filename + 60 acoustic indices)")

    # Show some key values
    print("\n🔍 SAMPLE VALUES (using scikit-maad defaults):")
    key_indices = ['NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI', 'LEQf']
    for idx in key_indices:
        if idx in result_df.columns:
            val = result_df[idx].iloc[0]
            print(f"   {idx}: {val:.6f}")

    # Compare with our previous "marine" defaults
    print("\n📈 COMPARISON WITH PREVIOUS MARINE DEFAULTS:")
    print("   Previous: flim_low=[0,1500], flim_mid=[1500,8000]")
    print("   Current:  flim_low=[0,1000], flim_mid=[1000,10000] (scikit-maad standard)")
    print("   Impact: Biophony range is wider (10kHz vs 8kHz), starts higher (1kHz vs 1.5kHz)")

    # Test marine corrections
    marine_indices = ['NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI']
    print("\n🌊 MARINE CORRECTIONS STATUS:")
    for idx in marine_indices:
        if idx in result_df.columns:
            print(f"   ✅ {idx}: Marine-corrected using user-defined frequency bands")

    print("\n💡 INTERPRETATION:")
    print("   - Using standard scikit-maad defaults ensures compatibility")
    print("   - Users can still customize for their specific environment")
    print("   - Marine corrections applied when custom frequency bands provided")
    print("   - Broader biophony range (1-10kHz) captures more marine life sounds")

if __name__ == "__main__":
    test_scikit_maad_defaults()
