#!/usr/bin/env python3
"""
Simple wrapper to generate sample marine acoustic WAV files for testing.
Run this from the project root directory.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.sample_generator import create_test_dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate sample marine acoustic WAV files for testing the GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_samples.py                    # Create 5 files in 'test_wav_files'
  python generate_samples.py -n 10              # Create 10 sample files
  python generate_samples.py -o my_samples      # Use custom output directory
  python generate_samples.py -d 60              # Create 60-second files
  
The files will contain realistic marine acoustic scenarios:
  - Quiet ocean (minimal activity)
  - Vessel passing (ship noise with some biology)
  - Dolphin pod (echolocation clicks)
  - Fish spawning (chorus sounds)
  - Busy harbor (heavy vessel traffic)
        """
    )
    
    parser.add_argument("--output", "-o", default="test_wav_files", 
                       help="Output directory for WAV files (default: test_wav_files)")
    parser.add_argument("--num-files", "-n", type=int, default=5,
                       help="Number of files to create (default: 5, max: 5)")
    parser.add_argument("--duration", "-d", type=int, default=30,
                       help="Duration of each file in seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Create the sample files
    print("\n🎵 Marine Acoustic Sample Generator")
    print("=" * 60)
    create_test_dataset(args.output, min(args.num_files, 5), args.duration)
    print("\n✅ Sample files ready for testing!")
    print(f"Now run: python main.py")