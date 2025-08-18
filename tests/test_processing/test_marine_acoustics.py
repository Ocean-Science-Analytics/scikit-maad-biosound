#!/usr/bin/env python3
"""
Test marine acoustics frequency band calculations
"""

import numpy as np

def calculate_marine_biophony_anthrophony(Sxx_power, fn, flim_low, flim_mid):
    """Copy of function for testing"""
    anthro_mask = (fn >= flim_low[0]) & (fn < flim_low[1])
    anthrophony_power = Sxx_power[anthro_mask]
    anthrophony_energy = np.sum(anthrophony_power)
    
    bio_mask = (fn >= flim_mid[0]) & (fn < flim_mid[1])
    biophony_power = Sxx_power[bio_mask]
    biophony_energy = np.sum(biophony_power)
    
    return anthrophony_energy, biophony_energy

def test_frequency_band_assignment():
    """Test correct assignment of frequency bands for marine acoustics"""
    
    # Create test spectrogram
    fn = np.linspace(0, 10000, 1000)  # 0-10kHz, 1000 bins
    Sxx_power = np.zeros(1000)
    
    # Add energy in different bands
    low_mask = (fn >= 0) & (fn < 1000)     # 0-1000 Hz
    mid_mask = (fn >= 1000) & (fn < 8000)  # 1000-8000 Hz
    high_mask = fn >= 8000                 # 8000+ Hz
    
    Sxx_power[low_mask] = 10.0   # High energy in low frequencies
    Sxx_power[mid_mask] = 5.0    # Moderate energy in mid frequencies
    Sxx_power[high_mask] = 1.0   # Low energy in high frequencies
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    # Test marine frequency band assignment
    flim_low = [0, 1000]    # Anthrophony = vessel noise
    flim_mid = [1000, 8000] # Biophony = biological sounds
    
    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )
    
    # Expected values based on our test data
    expected_anthro = np.sum(Sxx_power[low_mask])  # Should be ~1000
    expected_bio = np.sum(Sxx_power[mid_mask])     # Should be ~3500
    
    tests = [
        (anthro_energy > 0, True, "Anthrophony has energy"),
        (bio_energy > 0, True, "Biophony has energy"),
        (abs(anthro_energy - expected_anthro) < 0.1, True, f"Anthro energy correct: {anthro_energy}"),
        (abs(bio_energy - expected_bio) < 0.1, True, f"Bio energy correct: {bio_energy}"),
        (bio_energy > anthro_energy, True, "Bio energy > anthro (in test case)"),
    ]
    
    for actual, expected, description in tests:
        if actual == expected:
            results['passed'] += 1
            results['details'].append(f"✓ {description}")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {description}: {actual}")
    
    return results

def test_marine_ndsi_calculation():
    """Test NDSI calculation with marine frequency bands"""
    
    def calculate_ndsi(anthro_energy, bio_energy):
        if (bio_energy + anthro_energy) > 0:
            return (bio_energy - anthro_energy) / (bio_energy + anthro_energy)
        else:
            return 0
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    test_scenarios = [
        (1000, 3000, "Bio dominant"),      # NDSI should be positive
        (3000, 1000, "Anthro dominant"),   # NDSI should be negative
        (2000, 2000, "Equal"),             # NDSI should be 0
        (0, 1000, "Bio only"),             # NDSI should be 1
        (1000, 0, "Anthro only"),          # NDSI should be -1
        (0, 0, "No energy"),               # NDSI should be 0
    ]
    
    for anthro, bio, description in test_scenarios:
        ndsi = calculate_ndsi(anthro, bio)
        
        # Validate NDSI range
        if -1 <= ndsi <= 1:
            results['passed'] += 1
            results['details'].append(f"✓ {description}: NDSI = {ndsi:.3f}")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {description}: NDSI out of range = {ndsi:.3f}")
        
        # Validate expected direction
        if bio > anthro and ndsi <= 0:
            results['failed'] += 1
            results['details'].append(f"❌ {description}: NDSI should be positive when bio > anthro")
        elif anthro > bio and ndsi >= 0:
            results['failed'] += 1
            results['details'].append(f"❌ {description}: NDSI should be negative when anthro > bio")
    
    return results

def test_vessel_noise_scenario():
    """Test realistic vessel noise scenario"""
    
    # Simulate vessel passing by
    fn = np.linspace(0, 10000, 1000)
    Sxx_power = np.zeros(1000)
    
    # Heavy vessel noise in low frequencies
    vessel_mask = (fn >= 0) & (fn < 1500)
    Sxx_power[vessel_mask] = 50.0  # Very high energy
    
    # Light biological activity
    bio_mask = (fn >= 2000) & (fn < 8000)
    Sxx_power[bio_mask] = 2.0  # Low energy
    
    # Marine bands
    flim_low = [0, 1000]    # Anthrophony
    flim_mid = [1000, 8000] # Biophony (includes some vessel noise spillover)
    
    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    # Vessel noise should dominate anthrophony band
    anthro_only_mask = (fn >= 0) & (fn < 1000)
    expected_anthro = np.sum(Sxx_power[anthro_only_mask])
    
    if abs(anthro_energy - expected_anthro) < 0.1:
        results['passed'] += 1
        results['details'].append(f"✓ Vessel noise captured in anthrophony: {anthro_energy}")
    else:
        results['failed'] += 1
        results['details'].append(f"❌ Anthrophony energy mismatch: {anthro_energy} vs {expected_anthro}")
    
    # Bio energy should include vessel spillover + bio activity
    bio_vessel_mask = (fn >= 1000) & (fn < 1500)  # Vessel spillover
    bio_only_mask = (fn >= 2000) & (fn < 8000)    # Pure bio
    expected_bio = np.sum(Sxx_power[bio_vessel_mask]) + np.sum(Sxx_power[bio_only_mask])
    
    if abs(bio_energy - expected_bio) < 0.1:
        results['passed'] += 1
        results['details'].append(f"✓ Bio energy includes spillover: {bio_energy}")
    else:
        results['failed'] += 1
        results['details'].append(f"❌ Bio energy mismatch: {bio_energy} vs {expected_bio}")
    
    return results

def test_frequency_band_boundaries():
    """Test that frequency bands don't overlap incorrectly"""
    
    fn = np.linspace(0, 10000, 1000)
    Sxx_power = np.ones(1000)  # Uniform energy
    
    # Test different band configurations
    band_configs = [
        ([0, 1000], [1000, 8000], "Adjacent bands"),
        ([0, 500], [1000, 8000], "Gap between bands"),
        ([0, 1500], [1500, 8000], "Different split"),
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for flim_low, flim_mid, description in band_configs:
        anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
            Sxx_power, fn, flim_low, flim_mid
        )
        
        # Calculate expected energies
        anthro_bins = np.sum((fn >= flim_low[0]) & (fn < flim_low[1]))
        bio_bins = np.sum((fn >= flim_mid[0]) & (fn < flim_mid[1]))
        
        expected_anthro = anthro_bins * 1.0
        expected_bio = bio_bins * 1.0
        
        if abs(anthro_energy - expected_anthro) < 0.1 and abs(bio_energy - expected_bio) < 0.1:
            results['passed'] += 1
            results['details'].append(f"✓ {description}: A={anthro_energy:.0f}, B={bio_energy:.0f}")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {description}: Energy mismatch")
    
    return results

def run_tests(test_dir=None):
    """Run all marine acoustics tests"""
    
    print("Testing marine acoustics calculations...")
    
    all_results = {
        'frequency_assignment': test_frequency_band_assignment(),
        'ndsi_calculation': test_marine_ndsi_calculation(),
        'vessel_noise': test_vessel_noise_scenario(),
        'band_boundaries': test_frequency_band_boundaries()
    }
    
    # Combine results
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    
    print(f"Marine Acoustics Tests: {total_passed} passed, {total_failed} failed")
    
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
        'details': all_results
    }

if __name__ == "__main__":
    results = run_tests()
    print(f"Final result: {results['status']}")