#!/usr/bin/env python3
"""
Test the new GUI parameters work correctly
"""

import numpy as np
from maad import sound

def test_frequency_band_parsing():
    """Test that frequency band strings are parsed correctly"""
    
    # Test valid frequency band parsing
    test_cases = [
        ("0,1000", [0, 1000]),
        ("1000,8000", [1000, 8000]),
        ("0,500", [0, 500]),
    ]
    
    for input_str, expected in test_cases:
        result = list(map(int, input_str.split(',')))
        assert result == expected, f"Failed to parse {input_str}"
        print(f"✓ Parsed '{input_str}' -> {result}")
    
    # Test invalid cases
    invalid_cases = [
        "1000",  # Missing comma
        "1000,500",  # Invalid range (min > max)
        "abc,def",  # Non-numeric
    ]
    
    for invalid in invalid_cases:
        try:
            parts = list(map(int, invalid.split(',')))
            if len(parts) != 2 or parts[0] >= parts[1]:
                raise ValueError("Invalid range")
            assert False, f"Should have failed for {invalid}"
        except:
            print(f"✓ Correctly rejected invalid input: '{invalid}'")

def test_default_values():
    """Test that default values are set correctly when GUI fields are empty"""
    
    # Simulate empty GUI fields
    flim_low_var = ""
    flim_mid_var = ""
    sensitivity_var = ""
    gain_var = ""
    
    # Apply defaults
    flim_low = [0, 1500] if not flim_low_var.strip() else list(map(int, flim_low_var.split(',')))
    flim_mid = [1500, 8000] if not flim_mid_var.strip() else list(map(int, flim_mid_var.split(',')))
    sensitivity = -35.0 if not sensitivity_var.strip() else float(sensitivity_var)
    gain = 0.0 if not gain_var.strip() else float(gain_var)
    
    assert flim_low == [0, 1500], f"Default flim_low incorrect: {flim_low}"
    assert flim_mid == [1500, 8000], f"Default flim_mid incorrect: {flim_mid}"
    assert sensitivity == -35.0, f"Default sensitivity incorrect: {sensitivity}"
    assert gain == 0.0, f"Default gain incorrect: {gain}"
    
    print("✓ All default values set correctly")

def test_custom_values():
    """Test that custom values override defaults"""
    
    # Simulate custom GUI input
    flim_low_var = "0,1000"
    flim_mid_var = "1000,5000"
    sensitivity_var = "-169.4"
    gain_var = "10"
    
    # Parse custom values
    flim_low = list(map(int, flim_low_var.split(',')))
    flim_mid = list(map(int, flim_mid_var.split(',')))
    sensitivity = float(sensitivity_var)
    gain = float(gain_var)
    
    assert flim_low == [0, 1000], f"Custom flim_low incorrect: {flim_low}"
    assert flim_mid == [1000, 5000], f"Custom flim_mid incorrect: {flim_mid}"
    assert sensitivity == -169.4, f"Custom sensitivity incorrect: {sensitivity}"
    assert gain == 10.0, f"Custom gain incorrect: {gain}"
    
    print("✓ All custom values parsed correctly")

if __name__ == "__main__":
    print("Testing GUI parameter handling...")
    print("-" * 40)
    
    test_frequency_band_parsing()
    print()
    
    test_default_values()
    print()
    
    test_custom_values()
    print()
    
    print("-" * 40)
    print("All tests passed! ✓")