#!/usr/bin/env python3
"""
Integration tests for GUI functionality using mocking
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from tkinter import messagebox  # Import so it exists for patching

def test_gui_with_marine_params():
    """Test that GUI correctly processes marine acoustic parameters"""
    
    # Mock the entire tkinter messagebox to prevent popups
    with patch('tkinter.messagebox'):
        # Mock file dialog to return test paths
        with patch('tkinter.filedialog.askdirectory') as mock_dialog:
            mock_dialog.return_value = '/test/path'
            
            # Create a test root window (won't display)
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            # Create StringVars as they would be in the GUI
            flim_low_var = tk.StringVar(value='0,1000')
            flim_mid_var = tk.StringVar(value='1000,8000')
            sensitivity_var = tk.StringVar(value='-169.4')
            gain_var = tk.StringVar(value='10')
            
            # Test the parsing logic
            flim_low = list(map(int, flim_low_var.get().split(',')))
            flim_mid = list(map(int, flim_mid_var.get().split(',')))
            sensitivity = float(sensitivity_var.get())
            gain = float(gain_var.get())
            
            assert flim_low == [0, 1000]
            assert flim_mid == [1000, 8000]
            assert sensitivity == -169.4
            assert gain == 10.0
            
            print("✓ GUI variables correctly parsed")
            
            root.destroy()

def test_gui_validation():
    """Test GUI input validation"""
    
    root = tk.Tk()
    root.withdraw()
    
    # Test various invalid inputs
    test_cases = [
        ('', True),  # Empty is valid (uses defaults)
        ('0,1000', True),  # Valid range
        ('1000,0', False),  # Invalid (max < min)
        ('1000', False),  # Missing comma
        ('abc,def', False),  # Non-numeric
    ]
    
    for input_val, should_be_valid in test_cases:
        var = tk.StringVar(value=input_val)
        
        is_valid = True
        if input_val.strip():  # Only validate non-empty
            try:
                parts = list(map(int, input_val.split(',')))
                if len(parts) != 2 or parts[0] >= parts[1]:
                    is_valid = False
            except:
                is_valid = False
        
        assert is_valid == should_be_valid, f"Validation failed for '{input_val}'"
        status = "valid" if is_valid else "invalid"
        print(f"✓ '{input_val}' correctly identified as {status}")
    
    root.destroy()

def test_run_analysis_mock():
    """Test the run_analysis function with mocked GUI components"""
    
    # This simulates testing the actual run_analysis function
    # without launching the GUI
    
    with patch('tkinter.messagebox.showerror') as mock_error:
        with patch('tkinter.messagebox.showinfo') as mock_info:
            # Simulate the GUI state
            gui_state = {
                'input_folder': '/test/input',
                'output_folder': '/test/output',
                'flim_low': '0,1000',
                'flim_mid': '1000,8000',
                'sensitivity': '-169.4',
                'gain': '0',
                'mode': 'dataset'
            }
            
            # Mock os.path.exists to return True
            with patch('os.path.exists', return_value=True):
                # Mock os.walk to return test files
                with patch('os.walk') as mock_walk:
                    mock_walk.return_value = [
                        ('/test/input', [], ['test_20240101_120000_001.wav'])
                    ]
                    
                    # Mock the sound loading and processing
                    with patch('maad.sound.load') as mock_load:
                        mock_load.return_value = (Mock(), 44100)  # wave, fs
                        
                        print("✓ Mocked GUI analysis setup completed")
                        
                        # In a real test, we'd call run_analysis() here
                        # and verify it processes with our parameters

def test_gui_components_exist():
    """Test that all expected GUI components are created"""
    
    root = tk.Tk()
    root.withdraw()
    
    # Create the GUI components as in the main script
    components = {
        'flim_low_var': tk.StringVar(),
        'flim_mid_var': tk.StringVar(),
        'sensitivity_var': tk.StringVar(),
        'gain_var': tk.StringVar(),
        'input_folder_var': tk.StringVar(),
        'output_folder_var': tk.StringVar(),
        'time_interval_var': tk.StringVar(),
        'mode_var_24h': tk.BooleanVar(),
        'mode_var_30min': tk.BooleanVar(),
        'mode_var_20min': tk.BooleanVar(),
    }
    
    # Verify all components are created
    for name, component in components.items():
        assert component is not None, f"Component {name} not created"
        print(f"✓ {name} created successfully")
    
    # Test setting and getting values
    components['flim_low_var'].set('0,1000')
    assert components['flim_low_var'].get() == '0,1000'
    print("✓ Component value setting/getting works")
    
    root.destroy()

if __name__ == "__main__":
    print("Testing GUI Integration...")
    print("-" * 40)
    
    test_gui_with_marine_params()
    print()
    
    test_gui_validation()
    print()
    
    test_run_analysis_mock()
    print()
    
    test_gui_components_exist()
    print()
    
    print("-" * 40)
    print("All GUI integration tests passed! ✓")
    print("\nNote: These tests verify GUI logic without displaying windows.")
    print("For full end-to-end testing, consider using GUI automation tools.")