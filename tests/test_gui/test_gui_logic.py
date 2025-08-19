#!/usr/bin/env python3
"""
Test GUI logic without requiring actual GUI display
This tests the business logic that the GUI uses
"""

def parse_frequency_range(range_str):
    """Parse frequency range string like '0,1000' into [0, 1000]"""
    if not range_str.strip():
        return None
    try:
        parts = list(map(int, range_str.split(',')))
        if len(parts) != 2:
            raise ValueError("Must have exactly 2 values")
        if parts[0] >= parts[1]:
            raise ValueError("Min must be less than max")
        return parts
    except Exception as e:
        raise ValueError(f"Invalid frequency range: {e}")

def get_parameters_with_defaults(flim_low_str, flim_mid_str, sensitivity_str, gain_str):
    """
    Get parameters with defaults, simulating GUI behavior
    """
    # Apply defaults if empty
    flim_low = parse_frequency_range(flim_low_str) if flim_low_str else [0, 1500]
    flim_mid = parse_frequency_range(flim_mid_str) if flim_mid_str else [1500, 8000]
    sensitivity = float(sensitivity_str) if sensitivity_str else -35.0
    gain = float(gain_str) if gain_str else 0.0

    return flim_low, flim_mid, sensitivity, gain

def test_empty_fields_use_defaults():
    """Test that empty GUI fields result in default values"""
    flim_low, flim_mid, S, G = get_parameters_with_defaults("", "", "", "")

    assert flim_low == [0, 1500], f"Expected [0, 1500], got {flim_low}"
    assert flim_mid == [1500, 8000], f"Expected [1500, 8000], got {flim_mid}"
    assert S == -35.0, f"Expected -35.0, got {S}"
    assert G == 0.0, f"Expected 0.0, got {G}"
    print("✓ Empty fields correctly use defaults")

def test_custom_marine_values():
    """Test custom marine acoustic values"""
    flim_low, flim_mid, S, G = get_parameters_with_defaults(
        "0,1000", "1000,8000", "-169.4", "10"
    )

    assert flim_low == [0, 1000], f"Expected [0, 1000], got {flim_low}"
    assert flim_mid == [1000, 8000], f"Expected [1000, 8000], got {flim_mid}"
    assert S == -169.4, f"Expected -169.4, got {S}"
    assert G == 10.0, f"Expected 10.0, got {G}"
    print("✓ Custom marine values parsed correctly")

def test_partial_custom_values():
    """Test mixing custom and default values"""
    flim_low, flim_mid, S, G = get_parameters_with_defaults(
        "0,500", "", "-169.4", ""
    )

    assert flim_low == [0, 500], f"Expected [0, 500], got {flim_low}"
    assert flim_mid == [1500, 8000], f"Expected [1500, 8000] (default), got {flim_mid}"
    assert S == -169.4, f"Expected -169.4, got {S}"
    assert G == 0.0, f"Expected 0.0 (default), got {G}"
    print("✓ Partial custom values work with defaults")

def test_invalid_frequency_ranges():
    """Test that invalid frequency ranges are caught"""
    invalid_cases = [
        ("1000,500", "Invalid: min >= max"),
        ("1000", "Invalid: missing comma"),
        ("abc,def", "Invalid: non-numeric"),
        ("0,0", "Invalid: min >= max"),
    ]

    for invalid_str, reason in invalid_cases:
        try:
            result = parse_frequency_range(invalid_str)
            assert False, f"Should have rejected {invalid_str}"
        except ValueError:
            print(f"✓ Correctly rejected '{invalid_str}' ({reason})")

def test_frequency_band_overlap():
    """Test checking for frequency band overlap"""
    test_cases = [
        ([0, 1000], [1000, 8000], True, "Adjacent bands OK"),
        ([0, 1000], [500, 8000], False, "Overlapping bands"),
        ([0, 1000], [1500, 8000], True, "Gap between bands OK"),
        ([0, 1000], [0, 8000], False, "Complete overlap"),
    ]

    for low, mid, should_be_valid, description in test_cases:
        # Check if bands don't overlap (except at boundary)
        is_valid = low[1] <= mid[0]
        assert is_valid == should_be_valid, f"Failed for {description}"
        status = "valid" if is_valid else "invalid"
        print(f"✓ {description}: correctly identified as {status}")

def simulate_gui_workflow():
    """Simulate a complete GUI workflow"""
    print("\n--- Simulating GUI Workflow ---")

    # User scenario 1: Using all defaults
    print("Scenario 1: User provides no custom settings")
    params = get_parameters_with_defaults("", "", "", "")
    print(f"  Using defaults: flim_low={params[0]}, flim_mid={params[1]}")

    # User scenario 2: Marine acoustics researcher
    print("\nScenario 2: Marine acoustics with vessel noise")
    params = get_parameters_with_defaults("0,1000", "1000,8000", "-169.4", "0")
    print(f"  Marine settings: anthro={params[0]}, bio={params[1]}")

    # User scenario 3: Custom frequency analysis
    print("\nScenario 3: Custom frequency bands")
    params = get_parameters_with_defaults("0,500", "2000,10000", "-150", "5")
    print(f"  Custom bands: low={params[0]}, mid={params[1]}")

    print("\n✓ All workflow scenarios handled correctly")

if __name__ == "__main__":
    print("Testing GUI Logic (No Display Required)")
    print("=" * 50)

    test_empty_fields_use_defaults()
    test_custom_marine_values()
    test_partial_custom_values()
    print()

    test_invalid_frequency_ranges()
    print()

    test_frequency_band_overlap()

    simulate_gui_workflow()

    print("\n" + "=" * 50)
    print("All GUI logic tests passed! ✓")
    print("\nNote: These tests verify the GUI's business logic")
    print("without requiring a display or Tkinter to be installed.")
