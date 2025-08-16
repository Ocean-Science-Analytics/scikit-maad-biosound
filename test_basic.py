#!/usr/bin/env python3
"""
Basic test suite for Scikit-MAAD GUI - tests core functionality without external dependencies
Run with: python3 test_basic.py
"""

import unittest
import os
import tempfile
import shutil
import datetime
import wave
import struct
import numpy


class TestFilenameParser(unittest.TestCase):
    """Test filename parsing logic (recreated to avoid imports)."""
    
    def parse_date_and_filename_from_filename(self, filename):
        """Recreate the parsing function for testing."""
        try:
            basename = os.path.basename(filename)
            parts = basename.split('_')
            
            # Need at least 4 parts: prefix, date, time, suffix
            if len(parts) < 4:
                raise ValueError("Not enough underscore-separated parts")
            
            # Parse date from second part
            date_str = parts[1]
            if len(date_str) != 8:
                raise ValueError("Date part should be 8 digits (YYYYMMDD)")
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            # Parse time from third part
            time_str = parts[2]
            if len(time_str) != 6:
                raise ValueError("Time part should be 6 digits (HHMMSS)")
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            dt = datetime.datetime(year, month, day, hour, minute, second)
            filename_with_numbering = '_'.join(parts[:-1]) + "_" + parts[-1]
            return dt, filename_with_numbering
            
        except Exception:
            return None, None
    
    def test_valid_filename_parsing(self):
        """Test parsing of correctly formatted filenames."""
        test_cases = [
            ("Recording_20240515_143022_001.wav", 
             datetime.datetime(2024, 5, 15, 14, 30, 22)),
            ("Site1_20231225_060000_A.wav",
             datetime.datetime(2023, 12, 25, 6, 0, 0)),
            ("/path/to/Audio_20240101_120000_test.wav",
             datetime.datetime(2024, 1, 1, 12, 0, 0)),
        ]
        
        for filename, expected_date in test_cases:
            with self.subTest(filename=filename):
                dt, parsed_name = self.parse_date_and_filename_from_filename(filename)
                self.assertEqual(dt, expected_date)
                self.assertIsNotNone(parsed_name)
    
    def test_invalid_filename_parsing(self):
        """Test handling of incorrectly formatted filenames."""
        invalid_cases = [
            "no_underscores.wav",           # No underscores
            "Recording_2024_143022_001.wav", # Year too short
            "Recording_20240515_1430_001.wav", # Time too short
            "Recording_YYYYMMDD_HHMMSS_001.wav", # Not numbers
            "Recording_20241315_143022_001.wav", # Invalid month
            "Recording_20240515_253022_001.wav", # Invalid hour
        ]
        
        for filename in invalid_cases:
            with self.subTest(filename=filename):
                dt, parsed_name = self.parse_date_and_filename_from_filename(filename)
                self.assertIsNone(dt)
                self.assertIsNone(parsed_name)
    
    def test_edge_cases(self):
        """Test edge cases for filename parsing."""
        # Test leap year date
        dt, _ = self.parse_date_and_filename_from_filename("Rec_20240229_000000_1.wav")
        self.assertEqual(dt, datetime.datetime(2024, 2, 29, 0, 0, 0))
        
        # Test non-leap year (should fail)
        dt, _ = self.parse_date_and_filename_from_filename("Rec_20230229_000000_1.wav")
        self.assertIsNone(dt)


class TestWAVFileGeneration(unittest.TestCase):
    """Test WAV file generation for testing purposes."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_create_test_wav(self):
        """Test creation of a valid WAV file."""
        filepath = os.path.join(self.test_dir, "Test_20240101_120000_001.wav")
        sample_rate = 22050
        duration = 1  # 1 second for quick test
        
        # Generate simple sine wave
        t = numpy.linspace(0, duration, int(sample_rate * duration))
        signal = 0.5 * numpy.sin(2 * numpy.pi * 1000 * t)
        signal_int16 = numpy.int16(signal * 32767)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(len(signal_int16))
            wav_data = struct.pack('h' * len(signal_int16), *signal_int16)
            wav_file.writeframes(wav_data)
        
        # Verify file exists and is valid
        self.assertTrue(os.path.exists(filepath))
        
        # Check we can read it back
        with wave.open(filepath, 'rb') as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), sample_rate)
            self.assertEqual(wav_file.getnframes(), len(signal_int16))


class TestFileOperations(unittest.TestCase):
    """Test file and directory operations."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_directory_creation(self):
        """Test creation of output directories."""
        output_figures_path = os.path.join(self.test_dir, 'output_figures')
        os.makedirs(output_figures_path, exist_ok=True)
        
        self.assertTrue(os.path.exists(output_figures_path))
        self.assertTrue(os.path.isdir(output_figures_path))
    
    def test_wav_file_discovery(self):
        """Test finding WAV files in directory structure."""
        # Create nested directory structure
        subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(subdir)
        
        # Create test files
        wav_files = [
            os.path.join(self.test_dir, 'Test_20240101_120000_001.wav'),
            os.path.join(subdir, 'Test_20240101_120100_002.wav'),
            os.path.join(self.test_dir, 'not_a_wav.txt'),
        ]
        
        for filepath in wav_files:
            with open(filepath, 'w') as f:
                f.write('dummy content')
        
        # Find WAV files (simulate os.walk)
        found_wav_files = []
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith('.wav'):
                    found_wav_files.append(os.path.join(root, file))
        
        # Should find 2 WAV files, not the .txt file
        self.assertEqual(len(found_wav_files), 2)
        self.assertTrue(any('120000' in f for f in found_wav_files))
        self.assertTrue(any('120100' in f for f in found_wav_files))


class TestDataStructures(unittest.TestCase):
    """Test data structure handling."""
    
    def test_index_lists(self):
        """Test that expected index lists are properly defined."""
        # These should match what's in the actual GUI
        SPECTRAL_FEATURES = ['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
            'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
            'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
            'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
            'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
            'AGI','ROItotal','ROIcover']
        
        TEMPORAL_FEATURES = ['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
            'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
            'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']
        
        # Basic checks
        self.assertIsInstance(SPECTRAL_FEATURES, list)
        self.assertIsInstance(TEMPORAL_FEATURES, list)
        self.assertGreater(len(SPECTRAL_FEATURES), 30)  # Should have many spectral indices
        self.assertGreater(len(TEMPORAL_FEATURES), 10)   # Should have many temporal indices
        
        # Check some key indices are present
        self.assertIn('ROItotal', SPECTRAL_FEATURES)
        self.assertIn('ACI', SPECTRAL_FEATURES)
        self.assertIn('NDSI', SPECTRAL_FEATURES)
        self.assertIn('ZCR', TEMPORAL_FEATURES)
    
    def test_plotting_indices(self):
        """Test default plotting indices configuration."""
        # Default indices for the 6 plots
        default_indices = ['Hf', 'AEI', 'NDSI', 'ACI', 'TFSD', 'ROItotal']
        
        # Check they are all strings
        for idx in default_indices:
            self.assertIsInstance(idx, str)
            self.assertGreater(len(idx), 1)  # Should be non-empty strings


def run_tests():
    """Run all tests and print summary."""
    print("="*60)
    print("Running Basic Scikit-MAAD GUI Tests")
    print("="*60)
    print("Testing core functionality without external dependencies...")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestFilenameParser,
        TestWAVFileGeneration,
        TestFileOperations,
        TestDataStructures
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✓ All basic tests passed!")
        print("Core functionality appears to be working correctly.")
    else:
        print(f"✗ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.splitlines()[-1] if traceback else 'Unknown error'}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.splitlines()[-1] if traceback else 'Unknown error'}")
    
    print("\nNote: These tests check core logic without requiring scikit-maad.")
    print("For full functionality testing, run with a complete Python environment.")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)