#!/usr/bin/env python3
"""
Test parallel processing performance
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial

def mock_process_file(args):
    """Mock file processing function for testing"""
    filename, params = args
    
    # Simulate processing time (normally this would be audio analysis)
    processing_time = np.random.uniform(0.1, 0.3)  # 100-300ms per file
    time.sleep(processing_time)
    
    return {
        'filename': filename,
        'processing_time': processing_time,
        'result': f"Processed {filename}"
    }

def test_sequential_vs_parallel():
    """Test sequential vs parallel processing performance"""
    
    # Create mock file list
    num_files = 8  # Test with 8 files
    file_list = [f"test_file_{i:03d}.wav" for i in range(num_files)]
    params = {'mode': 'test'}
    
    print(f"Testing with {num_files} mock files...")
    print(f"Available CPU cores: {cpu_count()}")
    
    # Test sequential processing
    print("\n--- Sequential Processing ---")
    start_time = time.time()
    
    sequential_results = []
    for filename in file_list:
        result = mock_process_file((filename, params))
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # Test parallel processing
    print("\n--- Parallel Processing ---")
    start_time = time.time()
    
    num_workers = min(cpu_count() - 1, 4)
    file_args = [(filename, params) for filename in file_list]
    
    with Pool(num_workers) as pool:
        parallel_results = pool.map(mock_process_file, file_args)
    
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.2f} seconds")
    print(f"Workers used: {num_workers}")
    
    # Calculate performance metrics
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        efficiency = speedup / num_workers * 100
        
        print(f"\n--- Performance Results ---")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Efficiency: {efficiency:.1f}%")
        
        if speedup > 1.5:
            print("✓ Parallel processing is significantly faster")
        elif speedup > 1.1:
            print("✓ Parallel processing is faster")
        else:
            print("⚠ Parallel processing may not be beneficial for this workload")
    
    # Verify results are identical
    seq_filenames = [r['filename'] for r in sequential_results]
    par_filenames = [r['filename'] for r in parallel_results]
    
    if sorted(seq_filenames) == sorted(par_filenames):
        print("✓ Both methods processed the same files")
    else:
        print("✗ Results differ between methods")
    
    return sequential_time, parallel_time, speedup if sequential_time > 0 else 0

def estimate_real_world_performance():
    """Estimate performance with realistic file processing times"""
    
    print("\n" + "="*50)
    print("REAL-WORLD PERFORMANCE ESTIMATION")
    print("="*50)
    
    # Realistic processing times for audio analysis
    file_sizes = [
        ("Small (30s)", 2.0),    # 2 seconds processing
        ("Medium (5min)", 15.0), # 15 seconds processing  
        ("Large (30min)", 90.0), # 90 seconds processing
    ]
    
    num_files_scenarios = [5, 10, 20, 50]
    num_workers = min(cpu_count() - 1, 4)
    
    print(f"Assuming {num_workers} worker processes")
    print(f"CPU cores available: {cpu_count()}")
    
    for size_name, processing_time in file_sizes:
        print(f"\n--- {size_name} files ({processing_time}s each) ---")
        
        for num_files in num_files_scenarios:
            # Sequential time
            seq_time = num_files * processing_time
            
            # Parallel time (simplified model)
            par_time = max(
                processing_time,  # At least one file's worth of time
                (num_files * processing_time) / num_workers  # Divided by workers
            )
            
            speedup = seq_time / par_time if par_time > 0 else 1
            
            print(f"  {num_files:2d} files: {seq_time/60:.1f}min → {par_time/60:.1f}min ({speedup:.1f}x faster)")

if __name__ == "__main__":
    print("Testing Parallel Processing Performance")
    print("="*50)
    
    # Run actual performance test
    test_sequential_vs_parallel()
    
    # Show realistic estimates
    estimate_real_world_performance()
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    print("- Parallel processing is most beneficial with 4+ files")
    print("- Larger files show better speedup ratios")
    print("- Use parallel processing by default for batch jobs")
    print("- Keep sequential option for debugging or single files")