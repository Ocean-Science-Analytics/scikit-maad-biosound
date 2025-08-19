#!/usr/bin/env python3
"""
Performance comparison test script for scikit-maad processing.
Run from terminal to compare sequential vs parallel processing performance.

Usage: python performance_test.py [input_folder] [output_folder]
"""

import sys
import os
import time
from multiprocessing import cpu_count

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.processing.core_processing import (
    process_files_parallel, 
    process_files_sequential, 
    parse_date_and_filename_from_filename
)

def run_performance_test(input_folder, output_folder=None):
    """Run performance comparison between sequential and parallel processing."""
    
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(__file__), "performance_test_output")
    
    print("=" * 60)
    print("SCIKIT-MAAD PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Find WAV files
    print("\nSearching for WAV files...")
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"ERROR: No WAV files found in {input_folder}")
        return
    
    print(f"Found {len(audio_files)} WAV files")
    
    # Parse filenames
    filename_list = []
    for file in audio_files:
        dt, filename_with_numbering = parse_date_and_filename_from_filename(file)
        if dt and filename_with_numbering:
            filename_list.append(filename_with_numbering)
    
    if not filename_list:
        print("ERROR: No files with valid naming convention found.")
        return
    
    print(f"Successfully parsed {len(filename_list)} filenames")
    
    # Test parameters
    processing_params = {
        'mode': 'dataset',  # Process entire files
        'time_interval': 0,
        'flim_low': [0, 1500],
        'flim_mid': [1500, 8000],
        'sensitivity': -35.0,
        'gain': 0.0,
        'calculate_marine': True
    }
    
    print(f"\nTesting with {len(filename_list)} files...")
    print(f"Available CPU cores: {cpu_count()}")
    
    # Sequential processing
    print("\n--- SEQUENTIAL PROCESSING ---")
    start_time = time.time()
    sequential_results = process_files_sequential(audio_files, processing_params)
    sequential_time = time.time() - start_time
    sequential_success = len([r for r in sequential_results if r is not None])
    sequential_failed = len([r for r in sequential_results if r is None])
    
    print(f"Sequential completed in {sequential_time:.2f} seconds")
    print(f"Files processed: {sequential_success}, failed: {sequential_failed}")
    
    # Parallel processing
    print("\n--- PARALLEL PROCESSING ---")
    start_time = time.time()
    parallel_results = process_files_parallel(audio_files, processing_params)
    parallel_time = time.time() - start_time
    parallel_success = len([r for r in parallel_results if r is not None])
    parallel_failed = len([r for r in parallel_results if r is None])
    
    print(f"Parallel completed in {parallel_time:.2f} seconds")
    print(f"Files processed: {parallel_success}, failed: {parallel_failed}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    print(f"Sequential processing: {sequential_time:.2f} seconds")
    print(f"Parallel processing:   {parallel_time:.2f} seconds")
    
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"Speedup: {speedup:.2f}x faster with parallel processing")
        
        num_workers = min(cpu_count() - 1, 4)
        efficiency = speedup / num_workers * 100
        print(f"Parallel efficiency: {efficiency:.1f}%")
        print(f"Workers used: {num_workers}")
    
    # Save performance report
    try:
        os.makedirs(output_folder, exist_ok=True)
        report_file = os.path.join(output_folder, f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_file, 'w') as f:
            f.write("SCIKIT-MAAD PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Folder: {input_folder}\n")
            f.write(f"Files Tested: {len(filename_list)}\n")
            f.write(f"CPU Cores: {cpu_count()}\n")
            f.write(f"Workers Used: {min(cpu_count() - 1, 4)}\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"Sequential Time: {sequential_time:.2f} seconds\n")
            f.write(f"Parallel Time: {parallel_time:.2f} seconds\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
            f.write(f"Efficiency: {efficiency:.1f}%\n\n")
            
            f.write("PROCESSING SUCCESS:\n")
            f.write(f"Sequential: {sequential_success} success, {sequential_failed} failed\n")
            f.write(f"Parallel: {parallel_success} success, {parallel_failed} failed\n")
        
        print(f"\n📊 Performance report saved: {report_file}")
        
    except Exception as e:
        print(f"Warning: Could not save performance report: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python performance_test.py <input_folder> [output_folder]")
        print("\nExample:")
        print("  python performance_test.py test_wav_files/")
        print("  python performance_test.py test_wav_files/ performance_output/")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    run_performance_test(input_folder, output_folder)