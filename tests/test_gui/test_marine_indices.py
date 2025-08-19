#!/usr/bin/env python3
"""
Test marine-specific frequency band calculations
"""

import os
import sys

import numpy as np

# Add src directory to path to import the GUI module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Import the marine calculation functions from the new structure
from gui.main_gui import calculate_marine_biophony_anthrophony, calculate_marine_indices


def create_test_spectrogram():
    """Create a test spectrogram with known frequency content"""
    # Create frequency array from 0 to 10000 Hz
    fn = np.linspace(0, 10000, 1000)

    # Create a simple power spectrogram with different energy in different bands
    Sxx_power = np.zeros(1000)

    # Add energy in low frequencies (0-1000 Hz) - vessel noise
    low_freq_mask = (fn >= 0) & (fn < 1000)
    Sxx_power[low_freq_mask] = 10.0  # High energy in anthrophony band

    # Add energy in mid frequencies (1000-8000 Hz) - biological sounds
    mid_freq_mask = (fn >= 1000) & (fn < 8000)
    Sxx_power[mid_freq_mask] = 5.0  # Moderate energy in biophony band

    # Add some energy in high frequencies (8000+ Hz)
    high_freq_mask = fn >= 8000
    Sxx_power[high_freq_mask] = 1.0  # Low energy in high frequencies

    return Sxx_power, fn

def test_marine_biophony_anthrophony():
    """Test the marine biophony/anthrophony calculation"""
    print("Testing marine biophony/anthrophony calculation...")

    Sxx_power, fn = create_test_spectrogram()

    # Define marine frequency bands
    flim_low = [0, 1000]  # Anthrophony (vessel noise)
    flim_mid = [1000, 8000]  # Biophony (biological sounds)

    anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
        Sxx_power, fn, flim_low, flim_mid
    )

    # Calculate expected values
    # Anthrophony: 100 bins * 10.0 energy = 1000
    expected_anthro = np.sum(Sxx_power[(fn >= 0) & (fn < 1000)])
    # Biophony: 700 bins * 5.0 energy = 3500
    expected_bio = np.sum(Sxx_power[(fn >= 1000) & (fn < 8000)])

    assert abs(anthro_energy - expected_anthro) < 0.1, f"Anthrophony mismatch: {anthro_energy} vs {expected_anthro}"
    assert abs(bio_energy - expected_bio) < 0.1, f"Biophony mismatch: {bio_energy} vs {expected_bio}"

    print(f"  ✓ Anthrophony energy: {anthro_energy:.1f} (vessel noise 0-1000 Hz)")
    print(f"  ✓ Biophony energy: {bio_energy:.1f} (biological 1000-8000 Hz)")
    print(f"  ✓ Ratio (bio/anthro): {bio_energy/anthro_energy:.2f}")

def test_marine_indices():
    """Test calculation of marine-specific indices"""
    print("\nTesting marine-specific indices...")

    Sxx_power, fn = create_test_spectrogram()

    # Test with typical marine parameters
    flim_low = [0, 1000]  # Vessel noise
    flim_mid = [1000, 8000]  # Biological sounds

    indices = calculate_marine_indices(
        Sxx_power, fn, flim_low, flim_mid, S=-169.4, G=0
    )

    # Check that all expected indices are calculated
    expected_indices = ['NDSI_marine', 'BioEnergy_marine', 'AnthroEnergy_marine', 'rBA_marine', 'BI_marine']
    for idx in expected_indices:
        assert idx in indices, f"Missing index: {idx}"
        print(f"  ✓ {idx}: {indices[idx]:.4f}")

    # Verify NDSI calculation
    anthro = indices['AnthroEnergy_marine']
    bio = indices['BioEnergy_marine']
    expected_ndsi = (bio - anthro) / (bio + anthro)
    assert abs(indices['NDSI_marine'] - expected_ndsi) < 0.001, "NDSI calculation error"

    # Verify rBA calculation
    expected_rba = bio / anthro if anthro > 0 else 0
    assert abs(indices['rBA_marine'] - expected_rba) < 0.001, "rBA calculation error"

def test_different_frequency_bands():
    """Test with different frequency band configurations"""
    print("\nTesting different frequency band configurations...")

    Sxx_power, fn = create_test_spectrogram()

    test_configs = [
        ([0, 500], [500, 8000], "Narrow anthro, wide bio"),
        ([0, 1500], [1500, 5000], "Wider anthro, narrower bio"),
        ([0, 2000], [2000, 10000], "Very wide bands"),
    ]

    for flim_low, flim_mid, description in test_configs:
        anthro_energy, bio_energy = calculate_marine_biophony_anthrophony(
            Sxx_power, fn, flim_low, flim_mid
        )

        print(f"  Config: {description}")
        print(f"    Anthro ({flim_low[0]}-{flim_low[1]} Hz): {anthro_energy:.1f}")
        print(f"    Bio ({flim_mid[0]}-{flim_mid[1]} Hz): {bio_energy:.1f}")

        # Verify energies are positive
        assert anthro_energy >= 0, "Negative anthrophony energy"
        assert bio_energy >= 0, "Negative biophony energy"
        print("    ✓ Valid energy values")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")

    # Test with zero energy
    Sxx_power = np.zeros(1000)
    fn = np.linspace(0, 10000, 1000)

    indices = calculate_marine_indices(
        Sxx_power, fn, [0, 1000], [1000, 8000]
    )

    assert indices['NDSI_marine'] == 0, "NDSI should be 0 with no energy"
    assert indices['rBA_marine'] == 0, "rBA should be 0 with no energy"
    print("  ✓ Handles zero energy correctly")

    # Test with only anthrophony
    Sxx_power = np.zeros(1000)
    Sxx_power[(fn >= 0) & (fn < 1000)] = 10.0

    indices = calculate_marine_indices(
        Sxx_power, fn, [0, 1000], [1000, 8000]
    )

    assert indices['NDSI_marine'] < 0, "NDSI should be negative with only anthrophony"
    assert indices['rBA_marine'] == 0, "rBA should be 0 with no biophony"
    print("  ✓ Handles anthrophony-only correctly")

    # Test with only biophony
    Sxx_power = np.zeros(1000)
    Sxx_power[(fn >= 1000) & (fn < 8000)] = 10.0

    indices = calculate_marine_indices(
        Sxx_power, fn, [0, 1000], [1000, 8000]
    )

    assert indices['NDSI_marine'] > 0, "NDSI should be positive with only biophony"
    assert np.isinf(indices['rBA_marine']), "rBA should be inf with no anthrophony"
    print("  ✓ Handles biophony-only correctly")

if __name__ == "__main__":
    print("Testing Marine Acoustic Frequency Band Calculations")
    print("=" * 60)

    test_marine_biophony_anthrophony()
    test_marine_indices()
    test_different_frequency_bands()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("All marine frequency band tests passed! ✓")
    print("\nKey insights:")
    print("- Anthrophony correctly assigned to low frequencies (vessel noise)")
    print("- Biophony correctly assigned to mid frequencies (biological)")
    print("- Marine indices calculated with proper frequency assignments")
    print("- Edge cases handled gracefully")
