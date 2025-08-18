#!/usr/bin/env python3
"""
Test marine-specific frequency band calculations (standalone)
"""

import numpy as np

def calculate_marine_biophony_anthrophony(Sxx_power, fn, flim_low, flim_mid):
    """
    Calculate anthrophony and biophony for marine environments.
    (Copy of function from GUI for testing)
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

def calculate_marine_ndsi(anthro_energy, bio_energy):
    """Calculate marine NDSI"""
    if (bio_energy + anthro_energy) > 0:
        return (bio_energy - anthro_energy) / (bio_energy + anthro_energy)
    else:
        return 0

def create_test_spectrogram():
    """Create a test spectrogram with known frequency content"""
    # Create frequency array from 0 to 10000 Hz (1000 bins)
    fn = np.linspace(0, 10000, 1000)
    
    # Create a simple power spectrogram with different energy in different bands
    Sxx_power = np.zeros(1000)
    
    # Add energy in low frequencies (0-1000 Hz) - vessel noise
    # This should be about 100 bins (1000 Hz / 10000 Hz * 1000 bins)
    low_freq_mask = (fn >= 0) & (fn < 1000)
    Sxx_power[low_freq_mask] = 10.0  # High energy in anthrophony band
    
    # Add energy in mid frequencies (1000-8000 Hz) - biological sounds  
    # This should be about 700 bins
    mid_freq_mask = (fn >= 1000) & (fn < 8000)
    Sxx_power[mid_freq_mask] = 5.0  # Moderate energy in biophony band
    
    return Sxx_power, fn

def test_frequency_band_assignment():
    """Test that frequency bands are correctly assigned for marine acoustics"""
    print("Testing marine frequency band assignment...")
    
    Sxx_power, fn = create_test_spectrogram()
    
    # Marine frequency bands (as per your colleague's research)
    flim_low = [0, 1000]  # Anthrophony = vessel noise (low freq)
    flim_mid = [1000, 8000]  # Biophony = biological sounds (mid freq)
    
    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )
    
    print(f"  ✓ Anthrophony (vessel noise, 0-1000 Hz): {anthro_energy:.1f}")
    print(f"  ✓ Biophony (biological, 1000-8000 Hz): {bio_energy:.1f}")
    
    # Verify anthrophony gets the high-energy low frequencies
    assert anthro_energy > 0, "Anthrophony should have energy"
    assert bio_energy > 0, "Biophony should have energy"
    
    # In our test, biophony should have more total energy (700 bins vs 100 bins)
    # but lower energy density (5.0 vs 10.0)
    assert bio_energy > anthro_energy, "Biophony should have more total energy in test"
    
    print("  ✓ Energy distribution correct for marine environment")

def test_marine_ndsi():
    """Test NDSI calculation with marine frequency bands"""
    print("\nTesting marine NDSI calculation...")
    
    Sxx_power, fn = create_test_spectrogram()
    
    # Test with different scenarios
    scenarios = [
        ([0, 1000], [1000, 8000], "Standard marine bands"),
        ([0, 500], [1000, 8000], "Narrow vessel noise band"),
        ([0, 1000], [2000, 8000], "High-frequency bio only"),
    ]
    
    for flim_low, flim_mid, description in scenarios:
        anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
            Sxx_power, fn, flim_low, flim_mid
        )
        
        ndsi = calculate_marine_ndsi(anthro_energy, bio_energy)
        
        print(f"  {description}:")
        print(f"    Anthro: {anthro_energy:.1f}, Bio: {bio_energy:.1f}")
        print(f"    NDSI: {ndsi:.3f}")
        
        # NDSI should be between -1 and 1
        assert -1 <= ndsi <= 1, f"NDSI out of range: {ndsi}"
        
        # With more biophony, NDSI should be positive
        if bio_energy > anthro_energy:
            assert ndsi > 0, "NDSI should be positive when bio > anthro"
        
        print("    ✓ NDSI calculation valid")

def test_vessel_noise_dominance():
    """Test scenario where vessel noise dominates (common in marine acoustics)"""
    print("\nTesting vessel noise dominance scenario...")
    
    # Create spectrogram with heavy vessel noise
    fn = np.linspace(0, 10000, 1000)
    Sxx_power = np.zeros(1000)
    
    # Heavy vessel noise (0-1000 Hz)
    vessel_mask = (fn >= 0) & (fn < 1000)
    Sxx_power[vessel_mask] = 50.0  # Very high energy
    
    # Light biological activity (1000-8000 Hz)
    bio_mask = (fn >= 1000) & (fn < 8000)
    Sxx_power[bio_mask] = 2.0  # Low energy
    
    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, [0, 1000], [1000, 8000]
    )
    
    ndsi = calculate_marine_ndsi(anthro_energy, bio_energy)
    
    print(f"  Vessel-dominated scenario:")
    print(f"    Vessel noise energy: {anthro_energy:.1f}")
    print(f"    Biological energy: {bio_energy:.1f}")
    print(f"    NDSI: {ndsi:.3f} (negative = anthrophony dominant)")
    
    assert anthro_energy > bio_energy, "Vessel noise should dominate"
    assert ndsi < 0, "NDSI should be negative with vessel dominance"
    
    print("  ✓ Correctly identifies vessel noise dominance")

def test_frequency_overlap():
    """Test that frequency bands don't overlap incorrectly"""
    print("\nTesting frequency band boundaries...")
    
    fn = np.linspace(0, 10000, 1000)
    Sxx_power = np.ones(1000)  # Uniform energy
    
    # Test adjacent bands (no overlap)
    flim_low = [0, 1000]
    flim_mid = [1000, 8000]
    
    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )
    
    # Calculate expected energies
    anthro_bins = np.sum((fn >= 0) & (fn < 1000))
    bio_bins = np.sum((fn >= 1000) & (fn < 8000))
    
    expected_anthro = anthro_bins * 1.0
    expected_bio = bio_bins * 1.0
    
    print(f"  Adjacent bands test:")
    print(f"    Anthro bins: {anthro_bins}, energy: {anthro_energy:.1f}")
    print(f"    Bio bins: {bio_bins}, energy: {bio_energy:.1f}")
    
    assert abs(anthro_energy - expected_anthro) < 0.1, "Anthro energy mismatch"
    assert abs(bio_energy - expected_bio) < 0.1, "Bio energy mismatch"
    
    print("  ✓ Frequency bands properly separated")

if __name__ == "__main__":
    print("Testing Marine Acoustic Frequency Band Calculations")
    print("=" * 60)
    
    test_frequency_band_assignment()
    test_marine_ndsi()
    test_vessel_noise_dominance()
    test_frequency_overlap()
    
    print("\n" + "=" * 60)
    print("All marine frequency band tests passed! ✓")
    print("\nKey validations:")
    print("- Anthrophony correctly captures vessel noise (low frequencies)")
    print("- Biophony correctly captures biological sounds (mid frequencies)")
    print("- NDSI calculation works with marine frequency assignments")
    print("- Vessel noise dominance scenarios handled correctly")
    print("- Frequency band boundaries are properly enforced")