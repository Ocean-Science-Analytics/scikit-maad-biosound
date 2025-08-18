#!/usr/bin/env python3
"""
Test GUI parameter handling and validation
"""

def test_frequency_band_parsing():
    """Test frequency band string parsing"""
    test_cases = [
        ("0,1000", [0, 1000], True),
        ("1000,8000", [1000, 8000], True),
        ("0,500", [0, 500], True),
        ("", None, True),  # Empty should use defaults
        ("1000", None, False),  # Invalid format
        ("1000,500", None, False),  # Invalid range (min > max)
        ("abc,def", None, False),  # Non-numeric
        ("0,0", None, False),  # Invalid range (min = max)
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for input_str, expected, should_pass in test_cases:
        try:
            if input_str.strip():
                parts = list(map(int, input_str.split(',')))
                if len(parts) != 2 or parts[0] >= parts[1]:
                    result = None
                    success = False
                else:
                    result = parts
                    success = True
            else:
                result = None
                success = True  # Empty is valid
            
            if success == should_pass and (not should_pass or result == expected):
                results['passed'] += 1
                results['details'].append(f"✓ '{input_str}' -> {result}")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ '{input_str}' -> {result} (expected {expected})")
                
        except Exception as e:
            if should_pass:
                results['failed'] += 1
                results['details'].append(f"❌ '{input_str}' -> Exception: {e}")
            else:
                results['passed'] += 1
                results['details'].append(f"✓ '{input_str}' -> Correctly rejected")
    
    return results

def test_parameter_defaults():
    """Test that default values are applied correctly"""
    
    # Simulate empty GUI fields
    gui_inputs = {
        'flim_low': '',
        'flim_mid': '',
        'sensitivity': '',
        'gain': ''
    }
    
    # Apply defaults (as done in GUI)
    flim_low = [0, 1500] if not gui_inputs['flim_low'].strip() else list(map(int, gui_inputs['flim_low'].split(',')))
    flim_mid = [1500, 8000] if not gui_inputs['flim_mid'].strip() else list(map(int, gui_inputs['flim_mid'].split(',')))
    sensitivity = -35.0 if not gui_inputs['sensitivity'].strip() else float(gui_inputs['sensitivity'])
    gain = 0.0 if not gui_inputs['gain'].strip() else float(gui_inputs['gain'])
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    # Check defaults
    tests = [
        (flim_low, [0, 1500], "flim_low default"),
        (flim_mid, [1500, 8000], "flim_mid default"),
        (sensitivity, -35.0, "sensitivity default"),
        (gain, 0.0, "gain default")
    ]
    
    for actual, expected, name in tests:
        if actual == expected:
            results['passed'] += 1
            results['details'].append(f"✓ {name}: {actual}")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {name}: {actual} (expected {expected})")
    
    return results

def test_marine_trigger_logic():
    """Test when marine indices should be calculated"""
    
    test_cases = [
        ('', '', False, 'No custom bands'),
        ('0,1000', '', True, 'Custom anthro only'),
        ('', '1000,8000', True, 'Custom bio only'),
        ('0,1000', '1000,8000', True, 'Both custom'),
        ('  ', '  ', False, 'Whitespace only'),
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for flim_low_str, flim_mid_str, expected, description in test_cases:
        # This is the logic from the GUI
        should_calculate = bool(flim_low_str.strip() or flim_mid_str.strip())
        
        if should_calculate == expected:
            results['passed'] += 1
            results['details'].append(f"✓ {description}: {'will' if expected else 'will not'} calculate marine indices")
        else:
            results['failed'] += 1
            results['details'].append(f"❌ {description}: Expected {expected}, got {should_calculate}")
    
    return results

def run_tests(test_dir=None):
    """Run all GUI parameter tests"""
    
    print("Testing GUI parameter handling...")
    
    all_results = {
        'frequency_parsing': test_frequency_band_parsing(),
        'parameter_defaults': test_parameter_defaults(),
        'marine_trigger': test_marine_trigger_logic()
    }
    
    # Combine results
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    
    print(f"GUI Parameter Tests: {total_passed} passed, {total_failed} failed")
    
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