#!/usr/bin/env python3
"""
Create sample WAV files for testing the scikit-maad GUI.
Generates synthetic marine acoustic signals with vessel noise and biological sounds.
"""

import numpy as np
import wave
import struct
import os
import datetime

def create_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    """Create a sine wave at specified frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave_data

def create_vessel_noise(duration, sample_rate=44100):
    """
    Create synthetic vessel noise (low frequency rumble 50-500 Hz).
    Typical of ship engines and propellers.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    noise = np.zeros_like(t)
    
    # Add multiple low frequency components for vessel noise
    frequencies = [60, 120, 180, 250, 350, 450]  # Hz - typical ship frequencies
    for freq in frequencies:
        amplitude = np.random.uniform(0.1, 0.3)
        phase = np.random.uniform(0, 2*np.pi)
        noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add some low-frequency random noise
    noise += 0.05 * np.random.randn(len(t))
    
    # Low-pass filter effect (simple moving average)
    window_size = 100
    noise = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
    
    return noise

def create_dolphin_clicks(duration, sample_rate=44100):
    """
    Create synthetic dolphin echolocation clicks (3-8 kHz).
    Short duration pulses with high frequency content.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)
    
    # Create clicks at random intervals
    num_clicks = int(duration * 10)  # ~10 clicks per second
    for _ in range(num_clicks):
        click_time = np.random.uniform(0, duration)
        click_idx = int(click_time * sample_rate)
        
        if click_idx < len(signal) - 100:
            # Create a short click (1-2 ms duration)
            click_duration = 0.002
            click_samples = int(click_duration * sample_rate)
            click_t = np.linspace(0, click_duration, click_samples)
            
            # Frequency between 3-8 kHz
            freq = np.random.uniform(3000, 8000)
            click = np.sin(2 * np.pi * freq * click_t) * np.exp(-click_t * 1000)
            
            # Add click to signal
            signal[click_idx:click_idx+click_samples] += click[:min(click_samples, len(signal)-click_idx)]
    
    return signal * 0.3

def create_fish_chorus(duration, sample_rate=44100):
    """
    Create synthetic fish chorus sounds (500-2000 Hz).
    Continuous grunts and croaks typical of fish aggregations.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)
    
    # Add multiple frequency components for fish sounds
    base_frequencies = [600, 800, 1200, 1500, 1800]  # Hz
    for base_freq in base_frequencies:
        # Add frequency modulation to make it more natural
        mod_freq = np.random.uniform(1, 5)  # Modulation frequency
        mod_depth = np.random.uniform(50, 150)  # Modulation depth in Hz
        
        frequency = base_freq + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        phase = np.cumsum(2 * np.pi * frequency / sample_rate)
        
        amplitude = np.random.uniform(0.05, 0.15)
        signal += amplitude * np.sin(phase)
    
    # Add amplitude modulation (fish sounds often pulse)
    pulse_freq = np.random.uniform(2, 8)
    amplitude_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * pulse_freq * t)
    signal *= amplitude_envelope
    
    return signal

def create_ambient_ocean(duration, sample_rate=44100):
    """
    Create ambient ocean noise (broad spectrum).
    Wind, waves, and general underwater ambience.
    """
    # Pink noise (1/f noise) - more realistic for ocean
    samples = int(sample_rate * duration)
    white = np.random.randn(samples)
    
    # Simple pink noise approximation
    pink = np.zeros_like(white)
    pink[0] = white[0]
    for i in range(1, len(white)):
        pink[i] = 0.95 * pink[i-1] + 0.05 * white[i]
    
    return pink * 0.1

def save_wav_file(filename, data, sample_rate=44100):
    """Save audio data to WAV file."""
    # Normalize to prevent clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.95
    
    # Convert to 16-bit integers
    data_int = np.array(data * 32767, dtype=np.int16)
    
    # Write WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data_int.tobytes())

def create_test_dataset(output_dir="test_wav_files", num_files=5, duration=30):
    """
    Create a set of test WAV files with different marine acoustic scenarios.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = [
        {
            'name': 'quiet_ocean',
            'description': 'Quiet ocean with minimal activity',
            'vessel': 0.05,
            'biological': 0.1,
            'ambient': 0.3
        },
        {
            'name': 'vessel_passing',
            'description': 'Strong vessel noise with some biological activity',
            'vessel': 0.8,
            'biological': 0.2,
            'ambient': 0.2
        },
        {
            'name': 'dolphin_pod',
            'description': 'Active dolphin pod with clicks',
            'vessel': 0.1,
            'biological': 0.9,
            'ambient': 0.2
        },
        {
            'name': 'fish_spawning',
            'description': 'Fish chorus during spawning aggregation',
            'vessel': 0.05,
            'biological': 0.7,
            'ambient': 0.3
        },
        {
            'name': 'busy_harbor',
            'description': 'Busy harbor with multiple vessels',
            'vessel': 0.95,
            'biological': 0.05,
            'ambient': 0.4
        }
    ]
    
    # Generate files for each scenario
    sample_rate = 44100
    
    print("Creating sample marine acoustic WAV files...")
    print(f"Output directory: {output_dir}\n")
    
    for i, scenario in enumerate(scenarios[:num_files]):
        # Generate filename with timestamp format expected by GUI
        timestamp = datetime.datetime.now() - datetime.timedelta(hours=num_files-i)
        filename = f"marine_{timestamp.strftime('%Y%m%d_%H%M%S')}_{scenario['name']}.wav"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Creating {filename}...")
        print(f"  Scenario: {scenario['description']}")
        
        # Create mixed signal
        signal = np.zeros(int(sample_rate * duration))
        
        # Add vessel noise (anthrophony: 0-1000 Hz)
        if scenario['vessel'] > 0:
            vessel = create_vessel_noise(duration, sample_rate)
            signal += vessel * scenario['vessel']
            print(f"  Vessel noise: {scenario['vessel']*100:.0f}%")
        
        # Add biological sounds (biophony: 1000-8000 Hz)
        if scenario['biological'] > 0:
            # Mix different biological sources
            if 'dolphin' in scenario['name']:
                bio = create_dolphin_clicks(duration, sample_rate)
            elif 'fish' in scenario['name']:
                bio = create_fish_chorus(duration, sample_rate)
            else:
                # Mix both
                bio = 0.5 * create_dolphin_clicks(duration, sample_rate) + \
                      0.5 * create_fish_chorus(duration, sample_rate)
            signal += bio * scenario['biological']
            print(f"  Biological sounds: {scenario['biological']*100:.0f}%")
        
        # Add ambient ocean noise
        ambient = create_ambient_ocean(duration, sample_rate)
        signal += ambient * scenario['ambient']
        print(f"  Ambient noise: {scenario['ambient']*100:.0f}%")
        
        # Save the file
        save_wav_file(filepath, signal, sample_rate)
        print(f"  ✓ Saved: {filepath}\n")
    
    print("="*60)
    print(f"✓ Created {num_files} sample WAV files in '{output_dir}'")
    print("\nFrequency content:")
    print("  • Vessel noise (Anthrophony): 50-500 Hz")
    print("  • Fish chorus (Biophony): 500-2000 Hz")  
    print("  • Dolphin clicks (Biophony): 3000-8000 Hz")
    print("  • Ambient ocean: Broad spectrum")
    print("\nYou can now:")
    print(f"1. Launch the GUI: python main.py")
    print(f"2. Select '{output_dir}' as the input folder")
    print(f"3. Choose an output folder for results")
    print(f"4. Run the analysis to see acoustic indices and visualizations")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample marine acoustic WAV files for testing")
    parser.add_argument("--output", "-o", default="test_wav_files", 
                       help="Output directory for WAV files (default: test_wav_files)")
    parser.add_argument("--num-files", "-n", type=int, default=5,
                       help="Number of files to create (default: 5)")
    parser.add_argument("--duration", "-d", type=int, default=30,
                       help="Duration of each file in seconds (default: 30)")
    
    args = parser.parse_args()
    
    create_test_dataset(args.output, args.num_files, args.duration)