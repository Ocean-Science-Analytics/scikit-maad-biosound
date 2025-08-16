# Scikit-MAAD Acoustic Indices GUI

A graphical tool for analyzing acoustic indices from WAV files using the [scikit-maad](https://scikit-maad.github.io/) library. This tool batch processes passive acoustic data to extract ecoacoustic indices commonly used in soundscape ecology and bioacoustic research.

## What It Does

This application:
- Processes WAV files to compute 60+ acoustic indices
- Generates visualizations of acoustic patterns over time
- Exports results as CSV for further analysis
- Handles batch processing of multiple audio files
- Creates correlation maps and false-color spectrograms

## Quick Start

### For Users

1. **Requirements**
   - Python 3.7+
   - Required packages: `pip install numpy pandas matplotlib scikit-maad`

2. **Launch the GUI**
   ```bash
   python3 SciKit_Maad_File_Processing-GUI_Phase1.py
   ```

3. **Select folders**
   - **Input Folder**: Directory containing your WAV files
   - **Output Folder**: Where results will be saved

4. **Choose time scale**
   - **Hourly**: Averages indices over 1-hour periods
   - **Dataset**: Processes entire files as single units
   - **Manual**: Specify custom time intervals (in seconds)

5. **Click "Run Analysis"** and wait for completion

### Input Format

WAV files must follow this naming convention:
```
[prefix]_YYYYMMDD_HHMMSS_[suffix].wav
```

Examples:
- `Recording_20240515_143022_001.wav`
- `Site1_20240101_060000_A.wav`

### Output Files

The tool generates:
- `Acoustic_Indices.csv` - All computed indices with timestamps
- `output_figures/` folder containing:
  - `correlation_map.png` - Index correlation matrix
  - `individual_features.png` - Time series of 6 key indices
  - `false_color_spectrograms.png` - False-color spectrogram visualization

## Generating Test Data

For testing or development, you can generate sample WAV files:

```bash
python3 test_utils/generate_test_wav.py
```

This creates 3 test files (10 seconds each) with different acoustic characteristics:
- Bird-like chirps
- Insect buzzing
- Mixed soundscape

Test files are saved in `test_data/` with proper naming convention.

## Troubleshooting

### Common Issues

**"Index not available" in plots**
- Some indices (like ROItotal) may not be computed for all audio types
- The tool will show a gray placeholder and continue processing
- All successfully computed indices are still saved to CSV

**"No WAV files found"**
- Check that files have `.wav` extension (lowercase)
- Verify files follow the naming convention
- Ensure input folder path is correct

**Memory errors with large files**
- Try using shorter time intervals with Manual mode
- Process files in smaller batches
- Consider downsampling audio files before processing

## For Developers

### Repository Structure
```
scikit-maad-biosound/
├── SciKit_Maad_File_Processing-GUI_Phase1.py  # Main application (stable)
├── SciKit_Maad_File_Processing-GUI.py         # Original version
├── test_utils/                                 # Test data generation
│   └── generate_test_wav.py
├── archive/                                    # Old versions/variants
├── notes/                                      # Development documentation
└── README.md
```

### Key Functions

- `run_analysis()` - Main processing pipeline
- `safe_plot_index()` - Error-resistant plotting
- `parse_date_and_filename_from_filename()` - Filename parsing

### Acoustic Indices

The tool computes indices in two categories:

**Temporal** (time-domain):
- ZCR, MEANt, VARt, SKEWt, KURTt, LEQt, BGNt, SNRt, MED, Ht, etc.

**Spectral** (frequency-domain):
- MEANf, VARf, ACI, NDSI, BI, ADI, AEI, Hf, TFSD, etc.

See [scikit-maad documentation](https://scikit-maad.github.io/features.html) for detailed descriptions.

## Version History

- **Phase 1** (Current) - Robust error handling, graceful degradation for missing indices
- **Original** - Initial implementation by Jared Stephens (2023-2024)

## Contributing

Issues and improvements can be discussed via GitHub issues or direct collaboration.

## Acknowledgments

- Built on [scikit-maad](https://scikit-maad.github.io/) by Ulloa et al.
- Original GUI by Jared Stephens
- Phase 1 improvements by M. Weirathmueller 
