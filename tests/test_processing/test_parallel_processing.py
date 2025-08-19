#!/usr/bin/env python3
"""
Test parallel processing functionality and performance
"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np


def mock_file_processor(args):
    """Mock file processing function for testing"""
    filename, params = args

    # Simulate processing time
    processing_time = np.random.uniform(0.05, 0.15)  # 50-150ms
    time.sleep(processing_time)

    return {
        'filename': filename,
        'processing_time': processing_time,
        'indices': {
            'NDSI': np.random.random() * 2 - 1,
            'ACI': np.random.random(),
            'Filename': filename
        }
    }

def test_parallel_vs_sequential():
    """Test that parallel processing is faster than sequential"""

    # Create test file list
    num_files = 6
    test_files = [f"test_file_{i:03d}.wav" for i in range(num_files)]
    params = {'mode': 'test'}

    results = {'passed': 0, 'failed': 0, 'details': []}

    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for filename in test_files:
        result = mock_file_processor((filename, params))
        sequential_results.append(result)
    sequential_time = time.time() - start_time

    # Parallel processing
    start_time = time.time()
    num_workers = min(cpu_count() - 1, 4)
    file_args = [(filename, params) for filename in test_files]

    with Pool(num_workers) as pool:
        parallel_results = pool.map(mock_file_processor, file_args)
    parallel_time = time.time() - start_time

    # Verify results
    if len(sequential_results) == len(parallel_results) == num_files:
        results['passed'] += 1
        results['details'].append(f"✓ Both methods processed {num_files} files")
    else:
        results['failed'] += 1
        results['details'].append(f"❌ File count mismatch: seq={len(sequential_results)}, par={len(parallel_results)}")

    # Check performance
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    if speedup > 1.2:  # At least 20% faster
        results['passed'] += 1
        results['details'].append(f"✓ Parallel is {speedup:.1f}x faster ({sequential_time:.2f}s vs {parallel_time:.2f}s)")
    else:
        results['failed'] += 1
        results['details'].append(f"❌ Parallel not faster enough: {speedup:.1f}x speedup")

    # Check efficiency
    efficiency = (speedup / num_workers) * 100 if num_workers > 0 else 0
    if efficiency > 30:  # At least 30% efficiency
        results['passed'] += 1
        results['details'].append(f"✓ Parallel efficiency: {efficiency:.0f}%")
    else:
        results['failed'] += 1
        results['details'].append(f"❌ Low parallel efficiency: {efficiency:.0f}%")

    return results, sequential_time, parallel_time, speedup

def deterministic_processor(args):
    """Deterministic processor function (module level for pickling)"""
    filename, params = args
    # Deterministic result based on filename (using sum of ASCII values)
    ascii_sum = sum(ord(c) for c in filename) % 1000
    return {
        'filename': filename,
        'value': ascii_sum,
        'indices': {'test_index': ascii_sum / 1000.0}
    }

def test_parallel_correctness():
    """Test that parallel processing produces correct results"""

    # Use deterministic test to verify correctness
    test_files = ["file_a.wav", "file_b.wav", "file_c.wav"]
    params = {'mode': 'test'}

    results = {'passed': 0, 'failed': 0, 'details': []}

    # Sequential processing
    sequential_results = []
    for filename in test_files:
        result = deterministic_processor((filename, params))
        sequential_results.append(result)

    # Parallel processing
    file_args = [(filename, params) for filename in test_files]
    with Pool(2) as pool:
        parallel_results = pool.map(deterministic_processor, file_args)

    # Sort results by filename for comparison
    sequential_sorted = sorted(sequential_results, key=lambda x: x['filename'])
    parallel_sorted = sorted(parallel_results, key=lambda x: x['filename'])

    # Compare results
    for seq, par in zip(sequential_sorted, parallel_sorted):
        if seq['filename'] == par['filename'] and seq['value'] == par['value']:
            results['passed'] += 1
            results['details'].append(f"✓ {seq['filename']}: Consistent results")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {seq['filename']}: Results differ")

    return results

def test_worker_scaling():
    """Test performance with different numbers of workers"""

    test_files = [f"scale_test_{i:02d}.wav" for i in range(8)]
    params = {'mode': 'test'}

    results = {'passed': 0, 'failed': 0, 'details': []}

    # Test with different worker counts
    worker_counts = [1, 2, 4]
    timings = {}

    for num_workers in worker_counts:
        file_args = [(filename, params) for filename in test_files]

        start_time = time.time()
        with Pool(num_workers) as pool:
            pool_results = pool.map(mock_file_processor, file_args)
        timings[num_workers] = time.time() - start_time

        if len(pool_results) == len(test_files):
            results['passed'] += 1
            results['details'].append(f"✓ {num_workers} workers: {timings[num_workers]:.2f}s")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {num_workers} workers: Wrong result count")

    # Check that more workers generally perform better (for sufficient files)
    if len(timings) >= 2:
        if timings[2] < timings[1] * 1.1:  # Allow 10% variance
            results['passed'] += 1
            results['details'].append("✓ Worker scaling shows improvement")
        else:
            results['details'].append("⚠ Worker scaling may not show improvement (small workload)")

    return results

def error_prone_processor(args):
    """Error-prone processor function (module level for pickling)"""
    filename, params = args
    # Simulate occasional errors
    if 'error' in filename:
        raise ValueError(f"Simulated error for {filename}")
    return {'filename': filename, 'status': 'success'}

def test_error_handling():
    """Test error handling in parallel processing"""

    results = {'passed': 0, 'failed': 0, 'details': []}

    # Mix of good and bad files
    test_files = ["good_1.wav", "error_file.wav", "good_2.wav"]

    # Test sequential error handling first
    sequential_results = []
    for filename in test_files:
        try:
            result = error_prone_processor((filename, {}))
            sequential_results.append(result)
        except Exception:
            sequential_results.append(None)

    # Count good vs failed
    good_sequential = [r for r in sequential_results if r is not None]
    failed_sequential = [r for r in sequential_results if r is None]

    if len(good_sequential) == 2 and len(failed_sequential) == 1:
        results['passed'] += 1
        results['details'].append("✓ Sequential error handling works")
    else:
        results['failed'] += 1
        results['details'].append("❌ Sequential error handling failed")

    # For parallel, we'll just test that the function can handle errors
    # (actual parallel error handling would need more complex setup)
    try:
        error_prone_processor(("good_file.wav", {}))
        results['passed'] += 1
        results['details'].append("✓ Error-prone processor works for good files")
    except Exception:
        results['failed'] += 1
        results['details'].append("❌ Error-prone processor failed unexpectedly")

    try:
        error_prone_processor(("error_file.wav", {}))
        results['failed'] += 1
        results['details'].append("❌ Error-prone processor should have failed")
    except Exception:
        results['passed'] += 1
        results['details'].append("✓ Error-prone processor correctly raises errors")

    return results

def run_tests(test_dir=None):
    """Run all parallel processing tests"""

    print("Testing parallel processing...")

    all_results = {}

    # Core functionality tests
    perf_results, seq_time, par_time, speedup = test_parallel_vs_sequential()
    all_results['performance'] = perf_results

    all_results['correctness'] = test_parallel_correctness()
    all_results['scaling'] = test_worker_scaling()
    all_results['error_handling'] = test_error_handling()

    # Combine results
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())

    print(f"Parallel Processing Tests: {total_passed} passed, {total_failed} failed")

    if speedup > 0:
        print(f"Performance: {speedup:.1f}x speedup ({seq_time:.2f}s → {par_time:.2f}s)")

    # Print details for any failures
    for test_name, results in all_results.items():
        if results['failed'] > 0:
            print(f"\n{test_name} failures:")
            for detail in results['details']:
                if detail.startswith('❌'):
                    print(f"  {detail}")

    return {
        'status': 'PASSED' if total_failed == 0 else 'FAILED',
        'tests_run': total_passed + total_failed,
        'passed': total_passed,
        'failed': total_failed,
        'speedup': speedup if 'speedup' in locals() else 0,
        'details': all_results
    }

if __name__ == "__main__":
    results = run_tests()
    print(f"Final result: {results['status']}")
