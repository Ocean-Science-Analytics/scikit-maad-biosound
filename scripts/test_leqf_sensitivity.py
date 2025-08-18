#!/usr/bin/env python3
"""
Test that LEQf properly uses user-provided sensitivity and gain values.
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processing.core_processing import process_files_sequential, convert_results_to_dataframes, parse_date_and_filename_from_filename

def test_leqf_sensitivity():
    print("🧪 TESTING LEQf SENSITIVITY DEPENDENCY")
    print("=" * 60)
    
    # Use first test file
    audio_files = ["test_wav_files/marine_20250818_060552_vessel_passing.wav"]
    
    # Parse filename
    dt, filename_with_numbering = parse_date_and_filename_from_filename(audio_files[0])
    date_list = [dt]
    filename_list = [filename_with_numbering]
    
    # Test with different sensitivity values
    test_cases = [
        {"name": "Original Jared values", "sensitivity": -185.5, "gain": 20.0},
        {"name": "Marine default values", "sensitivity": -35.0, "gain": 0.0},
        {"name": "Alternative values", "sensitivity": -100.0, "gain": 10.0}
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\n🔬 Testing: {case['name']}")
        print(f"   Sensitivity: {case['sensitivity']}, Gain: {case['gain']}")
        
        processing_params = {
            'mode': 'dataset',
            'time_interval': 0,
            'flim_low': [0, 1500],
            'flim_mid': [1500, 8000],
            'sensitivity': case['sensitivity'],
            'gain': case['gain'],
            'calculate_marine': True
        }
        
        # Process file
        processing_results = process_files_sequential(audio_files, processing_params)
        result_df, _ = convert_results_to_dataframes(processing_results, filename_list, date_list, 'dataset')
        
        # Extract LEQf value
        leqf_value = result_df['LEQf'].iloc[0]
        results[case['name']] = leqf_value
        
        print(f"   LEQf result: {leqf_value:.6f}")
    
    print(f"\n📊 SUMMARY:")
    print("=" * 40)
    for name, value in results.items():
        print(f"{name:25} LEQf: {value:.6f}")
    
    # Check if values are different (they should be!)
    values = list(results.values())
    if len(set([round(v, 6) for v in values])) > 1:
        print(f"\n✅ SUCCESS: LEQf values change with sensitivity/gain (as expected)")
        print(f"   This proves LEQf is using the user-provided parameters correctly")
    else:
        print(f"\n❌ PROBLEM: LEQf values are identical - may be using hard-coded parameters")
    
    # Show differences
    print(f"\n🔍 DIFFERENCES FROM ORIGINAL:")
    original_val = results["Original Jared values"]
    for name, value in results.items():
        if name != "Original Jared values":
            diff = value - original_val
            print(f"   {name}: {diff:+.6f} difference")

if __name__ == "__main__":
    test_leqf_sensitivity()