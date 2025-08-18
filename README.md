# Scikit-MAAD Acoustic Indices GUI

## Major Update: Frequency band fixes and GUI updates (August 2025)

This version merges and corrects a few iterations of the marine acoustics GUI that had diverged. The primary fix addresses frequency band calculation errors identified through correspondence with the scikit-maad development team.

### Key Technical Improvements

**1. Corrected Marine Frequency Band Assignments**
   - Anthrophony (vessel noise): 0-1000 Hz band includes the lower frequency human-made sounds, e.g., from vessels.
   - Biophony (biological sounds): 1000-10000 Hz captures cetacean vocalizations and fish choruses
   - **Validation confirmed**: All marine indices (NDSI, BioEnergy, AnthroEnergy, rBA, BI) produce identical results to direct scikit-maad function calls (whew!)

**2. Parallel Processing Implementation**
   - Multiprocessing support reduces computation time by factor of 2-4x
   - Configurable worker processes optimize for available hardware
   - Performance comparison mode available for benchmarking

**3. Run Metadata Tracking**
   - Comprehensive logging of analysis parameters, file manifests, and performance metrics
   - JSON and human-readable summary formats for reproducibility
   - Automatic versioning of sequential runs in the same output directory

**4. Restructured Output Organization**
   - Hierarchical folder structure: `data/`, `figures/`, `metadata/`
   - Metadata includes full parameter sets for method sections
   - Performance reports when comparison mode enabled

**5. Marine Acoustic Test Data Generator**
   - Synthetic WAV files with realistic frequency content
   - Scenarios: vessel transits, dolphin echolocation, fish aggregations
   - Proper timestamp formatting for GUI compatibility

**6. Enhanced Development Infrastructure**
   - Proper Python package structure with `pyproject.toml`
   - Removed brittle `sys.path` manipulation for reliable imports
   - Automated validation system comparing calculations against direct scikit-maad calls
   - Comprehensive test suite covering GUI integration and marine acoustics

### Migration Guide for Existing Projects

#### Previous Version Behavior
The original implementation:
- Saved all outputs directly in the selected output folder (flat structure)
- Calculated frequency bands incorrectly, using only the first two frequency thresholds
- Generated files: `Acoustic_Indices.csv`, `output_figures/` folder with plots
- No run tracking or metadata preservation

#### Current Version Behavior  
The new implementation:
- Creates organized subfolders: `data/`, `figures/`, `metadata/`
- Properly calculates frequency bands across full specified ranges
- Generates same core files but in organized locations
- Adds timestamped metadata files for each run (never overwrites)

#### Critical Considerations

**Frequency Band Corrections Impact These Indices:**
- **NDSI, BI, rBA** - Values will differ substantially from previous calculations
- **AnthroEnergy, BioEnergy** - Now correctly allocated to appropriate frequency ranges
- Other temporal and spectral indices remain unchanged

**File Naming and Overwrite Protection:**
The new **Run Identifier** field (optional) allows you to prefix output files with custom identifiers. The system provides comprehensive overwrite protection:

- Without identifier: `Acoustic_Indices.csv`, `correlation_map.png` (warns if files exist)
- With identifier "StationA_2024": `StationA_2024_Acoustic_Indices.csv`, `StationA_2024_correlation_map.png` (warns if identifier already used)

**Protection scenarios:**
- **No identifier + files exist**: Suggests adding identifier or changing folder
- **Identifier provided + files exist**: Warns about duplicate identifier, suggests modifying it (e.g., adding "_v2")
- **All cases**: User can cancel to make changes, or proceed with explicit overwrite confirmation

Overwrite decisions are recorded in the metadata for full traceability.

This enables flexible naming strategies:
```
outputs/
├── data/
│   ├── StationA_Spring_Acoustic_Indices.csv
│   ├── StationB_Spring_Acoustic_Indices.csv
│   └── StationC_Spring_Acoustic_Indices.csv
└── figures/
    ├── StationA_Spring_correlation_map.png
    └── StationB_Spring_correlation_map.png
```

Alternatively, use separate output folders for complete isolation between stations.

Metadata files are always timestamped and never overwritten, providing a complete history of all processing runs.

#### Recommended Migration Path
1. Create new output folders with "_corrected" suffix to distinguish from previous analyses
2. Process a subset of files to quantify the magnitude of index changes
3. Document the correction in methods sections of ongoing work
4. Consider reprocessing all historical data for consistency 

---

## Overview

A GUI-based tool for batch processing marine passive acoustic data, built on the [scikit-maad](https://scikit-maad.github.io/) library. Designed specifically for underwater recordings, it computes acoustic indices with proper frequency band allocation for distinguishing vessel noise from biological sounds in marine environments.

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

3. **Install the project and dependencies:**
   ```bash
   uv pip install -e .
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```
   
   **Debug and Verbose Modes** (optional):
   ```bash
   python main.py --debug       # Show debug output for troubleshooting
   python main.py --verbose     # Show detailed processing steps
   python main.py --debug --verbose  # Show all diagnostic output
   ```
   
5. **(Optional) Generate test files:**
   ```bash
   python generate_samples.py
   ```

6. **(Optional) Validate calculations:**
   ```bash
   python scripts/validate_calculations.py
   ```
   This runs our implementation against direct scikit-maad calls to ensure calculations are correct. Generates validation reports in JSON and human-readable formats.

That's it! The GUI will open and you can start processing audio files.

## Quick Start Guide

### Basic Usage

1. **Launch the application** using one of the installation methods above
2. **Select folders** in the GUI:
   - **Input Folder**: Directory containing your WAV files
   - **Output Folder**: Where results will be saved
3. **Optional: Add Run Identifier** to prevent filename conflicts
4. **Choose time scale**:
   - **Hourly**: Averages indices over 1-hour periods
   - **Dataset**: Processes entire files as single units  
   - **Manual**: Specify custom time intervals (in seconds)
5. **Optional: Adjust Acoustic Settings** (uses defaults if blank):
   - **Anthrophony Range**: Frequency range for human noise (default: 0-1000 Hz)
   - **Biophony Range**: Frequency range for biological sounds (default: 1000-10000 Hz)
   - **Sensitivity**: Hydrophone sensitivity (default: -35.0 dB)
   - **Gain**: Recording system gain (default: 0.0 dB)
6. **Click "Run Analysis"** and monitor progress bar
7. **Check your output folder** for results when done

### Input Format

WAV files must follow this naming convention:
```
[prefix]_YYYYMMDD_HHMMSS_[suffix].wav
```

Examples:
- `Recording_20240515_143022_001.wav`
- `Site1_20240101_060000_A.wav`

### Output Files

The tool creates an organized folder structure with your results:
```
your_output_folder/
├── data/
│   └── Acoustic_Indices.csv         # All computed indices with timestamps
├── figures/
│   ├── correlation_map.png          # Index correlation matrix
│   ├── individual_features.png      # Time series of 6 key indices
│   └── false_color_spectrograms.png # Visual representation of sound patterns
└── metadata/
    ├── run_metadata_*.json          # Detailed analysis settings and parameters
    └── run_metadata_*_summary.txt   # Human-readable summary of your run
```

**New:** Each analysis run is automatically documented in the metadata folder, making it easy to track what settings were used and reproduce your results.

## Generating Test Data

**New Marine Acoustic Test Files:** Generate realistic sample WAV files with marine sounds:

```bash
python generate_samples.py              # Create 5 sample files (default)
python generate_samples.py -n 10 -d 60  # Create 10 files, 60 seconds each
```

Sample scenarios include:
- **Quiet ocean** - Minimal activity baseline
- **Vessel passing** - Ship engine noise with some biological sounds
- **Dolphin pod** - Echolocation clicks (3-8 kHz)
- **Fish spawning** - Chorus sounds (500-2000 Hz)
- **Busy harbor** - Heavy vessel traffic

Test files are saved in `test_wav_files/` with proper naming convention and realistic frequency content for testing the marine acoustic features.

## Validation and Development Scripts

The `scripts/` folder contains development and validation tools:

- **`validate_calculations.py`** - Validates our implementation against direct scikit-maad calls
- **`compare_versions.py`** - Compares results across different code versions  
- **`test_fixed_marine_indices.py`** - Tests specific marine acoustic calculations
- **`explain_frequency_bands.py`** - Demonstrates frequency band allocation logic

Validation reports are automatically saved to the scripts folder with timestamps.

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

**Debugging multiprocessing issues**
- Run with `python main.py --debug` to see detailed multiprocessing diagnostics
- Shows worker creation, file distribution, and result collection
- Helps identify where processing may be stalling

**Monitoring processing progress**
- Run with `python main.py --verbose` to see detailed processing steps
- Shows spectrogram calculation, index computation, and marine corrections
- Useful for understanding what stage each file is in

## For Developers

### Running Tests

**Basic tests** (no external dependencies):
```bash
uv run python test_basic.py
```

**Full test suite** (requires all dependencies):
```bash
uv run python tests/test_suite.py
```

**Calculation validation** (verify against scikit-maad):
```bash
python scripts/validate_calculations.py
```

### Validation Results

Our implementation has been validated against direct scikit-maad function calls:
- ✅ **12/12 comparisons passed** with perfect numerical agreement
- ✅ **Marine indices verified**: NDSI, BioEnergy, AnthroEnergy, rBA, BI all match exactly (0.00e+00 difference)
- ✅ **Standard indices confirmed**: ZCR, MEANf, VARf, SKEWf, KURTf, Ht, Hf match scikit-maad exactly
- 📄 **Validation reports**: Generated in both JSON (machine-readable) and TXT (human-readable) formats

This validation ensures our marine acoustic processing produces scientifically accurate results. 

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
