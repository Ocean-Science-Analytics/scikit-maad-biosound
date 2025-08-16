# Scikit-MAAD Acoustic Indices GUI

Batch process passive acoustic data using this GUI-based wrapper around scikit-maad that allows users to select input/output folders and calculate acoustic indices at custom durations.

Built on the [scikit-maad](https://scikit-maad.github.io/) library, this user-friendly graphical tool processes WAV files to extract acoustic indices commonly used in soundscape ecology, biodiversity monitoring, and bioacoustic research.

## What This Tool Does

**The application:**
- Processes WAV files to compute 60+ acoustic indices (measurements of sound characteristics)
- Generates visualizations showing acoustic patterns over time
- Exports results as CSV files for further analysis in Excel, R, or other tools
- Handles batch processing of multiple audio files automatically
- Creates correlation maps and false-color spectrograms for pattern analysis

**Common use cases:**
- Long-term biodiversity monitoring
- Before/after environmental impact studies  
- Seasonal acoustic pattern analysis
- Habitat quality assessment through soundscape analysis

## Installation & Setup

### Quick Setup

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd scikit-maad-biosound
   ```

2. **Install uv** (if you don't have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or visit [docs.astral.sh/uv](https://docs.astral.sh/uv/) for other installation methods.

3. **Install dependencies:**
   ```bash
   uv add numpy pandas matplotlib scikit-maad
   ```

4. **Run the application:**
   ```bash
   uv run python SciKit_Maad_File_Processing-GUI_Phase1.py
   ```

That's it! The GUI will open and you can start processing audio files.

## Quick Start Guide

### Basic Usage

1. **Launch the application** using one of the installation methods above
2. **Select folders** in the GUI:
   - **Input Folder**: Directory containing your WAV files
   - **Output Folder**: Where results will be saved
3. **Choose time scale**:
   - **Hourly**: Averages indices over 1-hour periods
   - **Dataset**: Processes entire files as single units  
   - **Manual**: Specify custom time intervals (in seconds)
4. **Click "Run Analysis"** and wait for completion
5. **Check your output folder** for results when done

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
uv run python test_utils/generate_test_wav.py
```

This creates 3 test files (10 seconds each).

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

### Running Tests

**Basic tests** (no external dependencies):
```bash
uv run python test_basic.py
```

**Full test suite** (requires all dependencies):
```bash
uv run python test_acoustic_gui.py
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

- **Current** - Robust error handling, graceful degradation for missing indices
- **Original** - Initial implementation by Jared Stephens (2023-2024)

## Contributing

Issues and improvements can be discussed via GitHub issues or direct collaboration.

## Acknowledgments

- Built on [scikit-maad](https://scikit-maad.github.io/) by Ulloa et al.
- Original GUI by Jared Stephens
- Updates and bug fixes by M. Weirathmueller 
