#!/usr/bin/env python3
"""
Generate test WAV files for scikit-maad acoustic analysis
Creates small WAV files quickly for testing
"""

import numpy as np
import wave
import struct
import os
from datetime import datetime

def generate_test_wav(output_dir="test_data", duration=10, sample_rate=22050):
    """
    Generate test WAV files with various acoustic features.
    Using shorter duration and lower sample rate for quick testing.
    
    Args:
        output_dir: Directory to save test files
        duration: Duration of each file in seconds (default 10s for quick tests)
        sample_rate: Sample rate in Hz (default 22050 for smaller files)
    """
    
    # Create test_outputs directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating test WAV files in '{output_dir}' directory...")
    
    # Generate timestamps for filenames
    timestamps = [
        "20240515_090000",  # May 15, 2024, 09:00:00
        "20240515_090100",  # May 15, 2024, 09:01:00
        "20240515_090200",  # May 15, 2024, 09:02:00
    ]
    
    # Time array
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate 3 different test signals
    signals = []
    descriptions = []
    
    # Signal 1: Simple tone with harmonics (simulates bird call)
    print("  Generating signal 1: Bird-like tones...")
    signal1 = np.zeros_like(t)
    # Add some chirps
    for i in range(3):  # 3 chirps
        chirp_start = i * 3  # Every 3 seconds
        chirp_duration = 0.5
        if chirp_start + chirp_duration < duration:
            chirp_mask = (t >= chirp_start) & (t < chirp_start + chirp_duration)
            chirp_t = t[chirp_mask] - chirp_start
            # Frequency sweep from 2000 to 3500 Hz
            freq = 2000 + 1500 * (chirp_t / chirp_duration)
            signal1[chirp_mask] = 0.3 * np.sin(2 * np.pi * freq * chirp_t)
    # Add background noise
    signal1 += 0.01 * np.random.randn(len(t))
    signals.append(signal1)
    descriptions.append("Bird-like chirps")
    
    # Signal 2: Constant frequency (simulates insect)
    print("  Generating signal 2: Insect-like buzz...")
    signal2 = 0.2 * np.sin(2 * np.pi * 4000 * t)  # 4kHz tone
    # Add amplitude modulation
    signal2 *= (1 + 0.5 * np.sin(2 * np.pi * 10 * t))  # 10Hz modulation
    # Add noise
    signal2 += 0.005 * np.random.randn(len(t))
    signals.append(signal2)
    descriptions.append("Insect buzz at 4kHz")
    
    # Signal 3: Mixed frequencies (simulates natural soundscape)
    print("  Generating signal 3: Mixed soundscape...")
    signal3 = np.zeros_like(t)
    # Low frequency background (100-500 Hz)
    signal3 += 0.1 * np.sin(2 * np.pi * 200 * t)
    signal3 += 0.05 * np.sin(2 * np.pi * 350 * t)
    # Mid frequency sounds (1-3 kHz)
    signal3 += 0.15 * np.sin(2 * np.pi * 1500 * t)
    signal3 += 0.1 * np.sin(2 * np.pi * 2500 * t)
    # Add some random noise
    signal3 += 0.02 * np.random.randn(len(t))
    signals.append(signal3)
    descriptions.append("Mixed frequency soundscape")
    
    # Write WAV files
    for i, (signal, desc) in enumerate(zip(signals, descriptions)):
        # Normalize to prevent clipping
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        # Convert to 16-bit integer
        signal_int16 = np.int16(signal * 32767)
        
        # Create filename with expected format
        filename = f"TestRecording_{timestamps[i]}_001.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(len(signal_int16))
            
            # Convert to bytes
            wav_data = struct.pack('h' * len(signal_int16), *signal_int16)
            wav_file.writeframes(wav_data)
        
        print(f"  ✓ Created: {filename}")
        print(f"    {desc} ({duration}s @ {sample_rate}Hz)")
    
    print(f"\n✓ Test files ready in '{output_dir}' directory")
    print(f"  Files are {duration} seconds each (quick for testing)")
    print(f"  Use these to test the Phase 1 GUI fixes")

if __name__ == "__main__":
    generate_test_wav()