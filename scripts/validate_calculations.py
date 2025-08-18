#!/usr/bin/env python3
"""
Validation script to compare our implementation against direct scikit-maad calls.

This script:
1. Runs our current processing pipeline on a test file
2. Runs direct scikit-maad calls with the same parameters
3. Compares results to validate our calculations are correct

Usage:
    python scripts/validate_calculations.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from maad import sound, features
from processing.standalone_processor import process_single_file_standalone, parse_date_and_filename_from_filename

def run_direct_maad_calculation(filename, sensitivity=-35.0, gain=0.0, flim_low=[0, 1500], flim_mid=[1500, 8000]):
    """
    Run direct scikit-maad calculations on a file using standard parameters.
    
    Returns:
        dict: Results from direct scikit-maad calls
    """
    print(f"\n=== Direct scikit-maad calculation for {os.path.basename(filename)} ===")
    
    # Load audio file (same way our code does)
    wave, fs = sound.load(filename=filename, channel='left', detrend=True, verbose=False)
    print(f"Loaded {len(wave)} samples at {fs} Hz")
    
    # Generate spectrogram (same parameters as our code)
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
    print(f"Spectrogram shape: {Sxx_power.shape}")
    
    # Calculate standard indices using scikit-maad defaults
    print("Calculating standard indices...")
    
    results = {}
    
    # Temporal alpha indices (temporal/statistical)
    alpha_temporal_results = features.all_temporal_alpha_indices(
        s=wave, 
        fs=fs, 
        gain=gain, 
        sensibility=sensitivity,
        dB_threshold=3, 
        verbose=False, 
        display=False
    )
    results.update(alpha_temporal_results)
    print(f"Temporal alpha indices: {list(alpha_temporal_results.keys())}")
    
    # Spectral alpha indices  
    alpha_spectral_results, alpha_spectral_per_bin = features.all_spectral_alpha_indices(
        Sxx_power=Sxx_power,
        tn=tn,
        fn=fn,
        flim_low=flim_low,
        flim_mid=flim_mid,
        verbose=False, 
        display=False
    )
    results.update(alpha_spectral_results)
    print(f"Spectral alpha indices: {list(alpha_spectral_results.keys())}")
    
    # Calculate custom marine indices for comparison
    print("\nCalculating custom marine frequency band indices...")
    
    # Anthrophony and biophony energies
    anthro_mask = (fn >= flim_low[0]) & (fn < flim_low[1])
    bio_mask = (fn >= flim_mid[0]) & (fn < flim_mid[1])
    
    anthrophony_energy = np.sum(Sxx_power[anthro_mask])
    biophony_energy = np.sum(Sxx_power[bio_mask])
    
    # Marine NDSI
    if (biophony_energy + anthrophony_energy) > 0:
        marine_ndsi = (biophony_energy - anthrophony_energy) / (biophony_energy + anthrophony_energy)
    else:
        marine_ndsi = 0
    
    # Marine rBA
    marine_rba = biophony_energy / anthrophony_energy if anthrophony_energy > 0 else (np.inf if biophony_energy > 0 else 0)
    
    # Marine BI (on biophony band only)
    if np.any(bio_mask):
        bio_spectrum = np.mean(Sxx_power[bio_mask, :], axis=1) if len(Sxx_power.shape) > 1 else Sxx_power[bio_mask]
        marine_bi = np.sum(bio_spectrum * np.log10(bio_spectrum + 1e-10))
    else:
        marine_bi = 0
    
    # Add marine indices to results
    results.update({
        'NDSI_marine_direct': marine_ndsi,
        'BioEnergy_marine_direct': biophony_energy,
        'AnthroEnergy_marine_direct': anthrophony_energy,
        'rBA_marine_direct': marine_rba,
        'BI_marine_direct': marine_bi
    })
    
    print(f"Marine NDSI (direct): {marine_ndsi:.6f}")
    print(f"Marine BioEnergy (direct): {biophony_energy:.2f}")
    print(f"Marine AnthroEnergy (direct): {anthrophony_energy:.2f}")
    print(f"Marine rBA (direct): {marine_rba:.6f}")
    print(f"Marine BI (direct): {marine_bi:.6f}")
    
    return results

def run_our_implementation(filename, sensitivity=-35.0, gain=0.0, flim_low=[0, 1500], flim_mid=[1500, 8000]):
    """
    Run our current implementation on a file.
    
    Returns:
        dict: Results from our implementation
    """
    print(f"\n=== Our implementation for {os.path.basename(filename)} ===")
    
    # Set up parameters exactly as our GUI would
    params = {
        'mode': 'daily',  # Process entire file as one segment
        'time_interval': 0,
        'flim_low': flim_low,
        'flim_mid': flim_mid, 
        'sensitivity': sensitivity,
        'gain': gain,
        'calculate_marine': True
    }
    
    # Process the file using our implementation
    result = process_single_file_standalone((filename, params))
    
    if result is None:
        print("ERROR: Our implementation returned None!")
        return {}
    
    print(f"Our implementation returned {len(result)} top-level keys")
    print("Available top-level keys:", list(result.keys()) if result else "None")
    
    # Extract the actual indices from the 'results' key
    results_data = result.get('results', [])
    print(f"Results data type: {type(results_data)}")
    print(f"Results data length: {len(results_data) if hasattr(results_data, '__len__') else 'N/A'}")
    
    # Handle case where results is a list of dictionaries (multiple segments)
    if isinstance(results_data, list) and len(results_data) > 0:
        segment_data = results_data[0]  # Take first segment for comparison
        print(f"First segment keys: {list(segment_data.keys()) if hasattr(segment_data, 'keys') else 'Not a dict'}")
        
        # Extract indices from the segment
        if 'indices' in segment_data:
            indices = segment_data['indices']
            print(f"Found indices dict with {len(indices)} indices")
            print("Sample indices keys:", list(indices.keys())[:10] if hasattr(indices, 'keys') else str(indices)[:100])
        else:
            indices = segment_data
            print(f"Using segment data directly with {len(indices)} items")
    elif isinstance(results_data, dict):
        indices = results_data.get('indices', results_data)
        print(f"Single result dict with {len(indices)} indices")
    else:
        indices = {}
        print("No valid indices found in results")
    
    # Print key marine indices for comparison
    marine_keys = ['NDSI', 'BioEnergy', 'AnthroEnergy', 'rBA', 'BI']
    for key in marine_keys:
        if key in indices:
            print(f"{key} (our impl): {indices[key]:.6f}")
        else:
            print(f"{key}: NOT FOUND in our results")
    
    return indices  # Return the actual indices, not the wrapper dict

def compare_results(direct_results, our_results, tolerance=1e-6):
    """
    Compare results between direct scikit-maad and our implementation.
    
    Returns:
        dict: Comparison summary
    """
    print(f"\n=== COMPARISON RESULTS ===")
    
    comparison = {
        'matches': 0,
        'mismatches': 0,
        'differences': {}
    }
    
    # Compare marine indices specifically
    marine_comparisons = [
        ('NDSI', 'NDSI_marine_direct'),
        ('BioEnergy', 'BioEnergy_marine_direct'), 
        ('AnthroEnergy', 'AnthroEnergy_marine_direct'),
        ('rBA', 'rBA_marine_direct'),
        ('BI', 'BI_marine_direct')
    ]
    
    for our_key, direct_key in marine_comparisons:
        if our_key in our_results and direct_key in direct_results:
            our_val = our_results[our_key]
            direct_val = direct_results[direct_key]
            
            # Handle special cases (inf, nan)
            if np.isnan(our_val) and np.isnan(direct_val):
                print(f"✓ {our_key}: Both NaN")
                comparison['matches'] += 1
            elif np.isinf(our_val) and np.isinf(direct_val):
                print(f"✓ {our_key}: Both Inf") 
                comparison['matches'] += 1
            elif abs(our_val - direct_val) < tolerance:
                print(f"✓ {our_key}: {our_val:.8f} ≈ {direct_val:.8f} (diff: {abs(our_val - direct_val):.2e})")
                comparison['matches'] += 1
            else:
                print(f"✗ {our_key}: {our_val:.8f} ≠ {direct_val:.8f} (diff: {abs(our_val - direct_val):.2e})")
                comparison['mismatches'] += 1
                comparison['differences'][our_key] = {
                    'ours': our_val,
                    'direct': direct_val,
                    'difference': abs(our_val - direct_val)
                }
        else:
            print(f"? {our_key}: Missing from one implementation")
    
    # Compare some standard indices that should be identical
    standard_indices = ['ZCR', 'MEANf', 'VARf', 'SKEWf', 'KURTf', 'H', 'Ht', 'Hf']
    for idx in standard_indices:
        if idx in our_results and idx in direct_results:
            our_val = our_results[idx]
            direct_val = direct_results[idx]
            
            # Handle pandas Series or scalar values
            if hasattr(our_val, 'item'):
                our_val = our_val.item()
            if hasattr(direct_val, 'item'):
                direct_val = direct_val.item()
            
            # Convert to float if possible
            try:
                our_val = float(our_val)
                direct_val = float(direct_val)
                
                if abs(our_val - direct_val) < tolerance:
                    print(f"✓ {idx}: Match (both {our_val:.6f})")
                    comparison['matches'] += 1
                else:
                    print(f"✗ {idx}: {our_val:.6f} ≠ {direct_val:.6f} (diff: {abs(our_val - direct_val):.2e})")
                    comparison['mismatches'] += 1
                    comparison['differences'][idx] = {
                        'ours': our_val,
                        'direct': direct_val,
                        'difference': abs(our_val - direct_val)
                    }
            except (ValueError, TypeError):
                print(f"? {idx}: Could not compare values (types: {type(our_val)}, {type(direct_val)})")
                comparison['differences'][idx] = {
                    'ours': str(our_val),
                    'direct': str(direct_val),
                    'difference': 'type_mismatch'
                }
    
    return comparison

def save_validation_report(test_file, comparison, direct_results, our_results, sensitivity, gain, flim_low, flim_mid):
    """
    Save a detailed validation report to the scripts folder.
    """
    import json
    from datetime import datetime
    
    scripts_folder = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = scripts_folder / f"validation_report_{timestamp}.json"
    
    # Prepare report data
    report = {
        "validation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_file": str(test_file.name),
            "script_version": "1.0",
            "validation_purpose": "Compare our implementation against direct scikit-maad calls"
        },
        "test_parameters": {
            "sensitivity_db": sensitivity,
            "gain_db": gain,
            "anthrophony_band_hz": flim_low,
            "biophony_band_hz": flim_mid
        },
        "validation_results": {
            "total_comparisons": comparison['matches'] + comparison['mismatches'],
            "successful_matches": comparison['matches'],
            "mismatches": comparison['mismatches'],
            "validation_passed": comparison['mismatches'] == 0
        },
        "detailed_comparison": {},
        "summary": {}
    }
    
    # Add detailed comparison for key marine indices
    marine_comparisons = [
        ('NDSI', 'NDSI_marine_direct'),
        ('BioEnergy', 'BioEnergy_marine_direct'), 
        ('AnthroEnergy', 'AnthroEnergy_marine_direct'),
        ('rBA', 'rBA_marine_direct'),
        ('BI', 'BI_marine_direct')
    ]
    
    for our_key, direct_key in marine_comparisons:
        if our_key in our_results and direct_key in direct_results:
            our_val = float(our_results[our_key]) if not pd.isna(our_results[our_key]) else None
            direct_val = float(direct_results[direct_key]) if not pd.isna(direct_results[direct_key]) else None
            
            comparison_result = {
                "our_implementation": our_val,
                "direct_scikit_maad": direct_val,
                "matches": False,
                "difference": None,
                "notes": ""
            }
            
            if our_val is not None and direct_val is not None:
                if np.isnan(our_val) and np.isnan(direct_val):
                    comparison_result["matches"] = True
                    comparison_result["notes"] = "Both values are NaN"
                elif np.isinf(our_val) and np.isinf(direct_val):
                    comparison_result["matches"] = True
                    comparison_result["notes"] = "Both values are infinite"
                else:
                    diff = abs(our_val - direct_val)
                    comparison_result["difference"] = float(diff)
                    comparison_result["matches"] = diff < 1e-6
                    if comparison_result["matches"]:
                        comparison_result["notes"] = f"Perfect match (diff: {diff:.2e})"
                    else:
                        comparison_result["notes"] = f"Mismatch detected (diff: {diff:.2e})"
            
            report["detailed_comparison"][our_key] = comparison_result
    
    # Add summary statistics
    if our_results and direct_results:
        report["summary"] = {
            "our_implementation_indices_count": len(our_results),
            "direct_scikit_maad_indices_count": len(direct_results),
            "key_marine_indices": {
                "NDSI": {
                    "our_value": float(our_results.get('NDSI', 'N/A')) if 'NDSI' in our_results else None,
                    "direct_value": float(direct_results.get('NDSI_marine_direct', 'N/A')) if 'NDSI_marine_direct' in direct_results else None
                },
                "biophony_energy": {
                    "our_value": float(our_results.get('BioEnergy', 'N/A')) if 'BioEnergy' in our_results else None,
                    "direct_value": float(direct_results.get('BioEnergy_marine_direct', 'N/A')) if 'BioEnergy_marine_direct' in direct_results else None
                },
                "anthrophony_energy": {
                    "our_value": float(our_results.get('AnthroEnergy', 'N/A')) if 'AnthroEnergy' in our_results else None,
                    "direct_value": float(direct_results.get('AnthroEnergy_marine_direct', 'N/A')) if 'AnthroEnergy_marine_direct' in direct_results else None
                }
            }
        }
    
    # Write JSON report to file
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also create a human-readable text report
        txt_report_file = report_file.with_suffix('.txt')
        with open(txt_report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SCIKIT-MAAD BIOSOUND VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write(f"Generated: {report['validation_metadata']['timestamp']}\n")
            f.write(f"Test File: {report['validation_metadata']['test_file']}\n")
            f.write(f"Purpose: {report['validation_metadata']['validation_purpose']}\n\n")
            
            # Test parameters
            f.write("TEST PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sensitivity: {report['test_parameters']['sensitivity_db']} dB\n")
            f.write(f"Gain: {report['test_parameters']['gain_db']} dB\n")
            f.write(f"Anthrophony band: {report['test_parameters']['anthrophony_band_hz'][0]}-{report['test_parameters']['anthrophony_band_hz'][1]} Hz\n")
            f.write(f"Biophony band: {report['test_parameters']['biophony_band_hz'][0]}-{report['test_parameters']['biophony_band_hz'][1]} Hz\n\n")
            
            # Overall results
            f.write("VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total comparisons: {report['validation_results']['total_comparisons']}\n")
            f.write(f"Successful matches: {report['validation_results']['successful_matches']}\n")
            f.write(f"Mismatches: {report['validation_results']['mismatches']}\n")
            f.write(f"Validation passed: {'✅ YES' if report['validation_results']['validation_passed'] else '❌ NO'}\n\n")
            
            # Detailed comparison
            f.write("DETAILED MARINE INDICES COMPARISON:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Index':<15} {'Our Value':<15} {'Direct Value':<15} {'Match':<8} {'Difference':<12}\n")
            f.write("-" * 60 + "\n")
            
            for index_name, comparison_data in report['detailed_comparison'].items():
                our_val = comparison_data['our_implementation']
                direct_val = comparison_data['direct_scikit_maad']
                matches = "✅" if comparison_data['matches'] else "❌"
                diff = comparison_data.get('difference', 'N/A')
                
                if isinstance(our_val, (int, float)) and isinstance(direct_val, (int, float)):
                    f.write(f"{index_name:<15} {our_val:<15.6f} {direct_val:<15.6f} {matches:<8} {diff:<12.2e}\n")
                else:
                    f.write(f"{index_name:<15} {str(our_val):<15} {str(direct_val):<15} {matches:<8} {str(diff):<12}\n")
            
            f.write("\n")
            
            # Summary stats if available
            if 'summary' in report and report['summary']:
                f.write("SUMMARY STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Our implementation indices count: {report['summary'].get('our_implementation_indices_count', 'N/A')}\n")
                f.write(f"Direct scikit-maad indices count: {report['summary'].get('direct_scikit_maad_indices_count', 'N/A')}\n\n")
                
                if 'key_marine_indices' in report['summary']:
                    f.write("KEY MARINE INDICES VALUES:\n")
                    f.write("-" * 30 + "\n")
                    for key, values in report['summary']['key_marine_indices'].items():
                        f.write(f"{key}:\n")
                        f.write(f"  Our implementation: {values.get('our_value', 'N/A')}\n")
                        f.write(f"  Direct scikit-maad: {values.get('direct_value', 'N/A')}\n")
                        f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("This report validates that our marine acoustic processing implementation\n")
            f.write("produces identical results to direct scikit-maad function calls.\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n📄 Validation reports saved:")
        print(f"   📊 JSON (for computers): {report_file.name}")
        print(f"   📝 TXT (human-readable): {txt_report_file.name}")
        return str(report_file)
    except Exception as e:
        print(f"⚠️  Could not save validation report: {e}")
        return None

def main():
    """
    Main validation function.
    """
    print("Starting calculation validation...")
    
    # Use a simple test file
    test_file = Path(__file__).parent.parent / "test_wav_files" / "marine_20250818_102448_quiet_ocean.wav"
    
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        print("Available test files:")
        test_dir = test_file.parent
        if test_dir.exists():
            for f in test_dir.glob("*.wav"):
                print(f"  - {f.name}")
        return 1
    
    print(f"Using test file: {test_file.name}")
    
    # Test parameters (typical marine settings)
    sensitivity = -35.0
    gain = 0.0
    flim_low = [0, 1500]      # Anthrophony (vessel noise)
    flim_mid = [1500, 8000]   # Biophony (biological sounds)
    
    print(f"Test parameters:")
    print(f"  Sensitivity: {sensitivity} dB")
    print(f"  Gain: {gain} dB") 
    print(f"  Anthrophony band: {flim_low[0]}-{flim_low[1]} Hz")
    print(f"  Biophony band: {flim_mid[0]}-{flim_mid[1]} Hz")
    
    try:
        # Run both implementations
        direct_results = run_direct_maad_calculation(str(test_file), sensitivity, gain, flim_low, flim_mid)
        our_results = run_our_implementation(str(test_file), sensitivity, gain, flim_low, flim_mid)
        
        # Compare results
        comparison = compare_results(direct_results, our_results)
        
        # Save detailed validation report
        report_file = save_validation_report(test_file, comparison, direct_results, our_results, sensitivity, gain, flim_low, flim_mid)
        
        # Summary
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Matches: {comparison['matches']}")
        print(f"Mismatches: {comparison['mismatches']}")
        
        if comparison['mismatches'] == 0:
            print("🎉 SUCCESS: All calculations match!")
            if report_file:
                print(f"📄 Validation report: {Path(report_file).name}")
            return 0
        else:
            print("⚠️  WARNING: Some calculations differ!")
            print("\nDifferences found:")
            for key, diff_info in comparison['differences'].items():
                print(f"  {key}: ours={diff_info['ours']:.8f}, direct={diff_info['direct']:.8f}, diff={diff_info['difference']:.2e}")
            if report_file:
                print(f"📄 See detailed report: {Path(report_file).name}")
            return 1
            
    except Exception as e:
        print(f"ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)