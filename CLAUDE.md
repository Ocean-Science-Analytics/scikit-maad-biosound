# Scikit-MAAD Spectral Indices GUI - Development Plan

## Project Overview
Merge the archive GUI improvements with the main GitHub version to create a unified tool that:
- Calculates all 60 acoustic indices correctly
- Allows user-defined frequency bands for marine acoustic analysis
- Implements parallel processing for improved performance
- Maintains backward compatibility

## Key Issues Identified
1. **Frequency Band Limitation**: Original code only uses first two threshold levels, limiting bandwidth calculations
2. **Marine Acoustics Adaptation**: Need to properly assign anthrophony (vessel noise) to 0-1000 Hz and biophony to 1000-8000 Hz
3. **Performance**: Single-threaded processing is slow for large datasets
4. **GUI Feedback**: Missing processing completion notification in some versions

## Implementation Plan

### 1. Add Frequency Band Controls to GUI
**Files to modify**: `SciKit_Maad_File_Processing-GUI.py`

- Add GUI input fields:
  - Anthrophony frequency range (default: 0-1000 Hz)
  - Biophony frequency range (default: 1000-8000 Hz)  
  - Microphone sensitivity S (default: -169.4)
  - Gain G (default: 0)
- Make fields optional with sensible defaults for backward compatibility

### 2. Fix Frequency Band Calculations
**New function to add**: `calculate_biophony_anthrophony_marine()`

```python
def calculate_biophony_anthrophony_marine(Sxx_power, fn, flim_low, flim_mid):
    """
    Calculate anthrophony and biophony for marine environments.
    Anthrophony: flim_low range (vessel noise)
    Biophony: flim_mid range (biological sounds)
    """
    anthrophony_power = Sxx_power[(fn >= flim_low[0]) & (fn < flim_low[1])]
    biophony_power = Sxx_power[(fn >= flim_mid[0]) & (fn < flim_mid[1])]
    return np.sum(anthrophony_power), np.sum(biophony_power)
```

**Indices requiring frequency bands**:
- NDSI, BioEnergy, AnthroEnergy, rBA, BI

**Indices calculated on full bandwidth**:
- ACI, EPS, ROU, ADI, AEI (as confirmed by Sylvain)

### 3. Implement Parallel Processing
**Use**: `multiprocessing.Pool`

```python
from multiprocessing import Pool, cpu_count

def process_files_parallel(file_list, num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count() - 1, 4)  # Leave one core free
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, file_list)
    return results
```

### 4. Testing Strategy

#### Test Files Required
- `test_frequency_bands.py` - Test frequency band calculations
- `test_parallel_processing.py` - Test multiprocessing functionality
- `test_gui_inputs.py` - Test GUI input validation
- `test_indices_calculation.py` - Test all 60 indices computation

#### Test Coverage Areas

##### A. Frequency Band Tests
```python
def test_marine_frequency_bands():
    """Test anthrophony/biophony calculation for marine acoustics"""
    # Test with known signal in 0-1000 Hz range
    # Verify anthrophony energy is captured correctly
    # Test with signal in 1000-8000 Hz range  
    # Verify biophony energy is captured correctly
    
def test_backward_compatibility():
    """Ensure old files work without frequency band inputs"""
    # Process file without specifying frequency bands
    # Verify defaults are applied correctly
```

##### B. Parallel Processing Tests
```python
def test_parallel_vs_sequential():
    """Compare parallel and sequential processing results"""
    # Process same files both ways
    # Verify identical results
    
def test_parallel_performance():
    """Verify parallel processing is faster"""
    # Time both approaches with multiple files
    # Assert parallel is faster for >2 files
```

##### C. GUI Input Validation Tests
```python
def test_frequency_input_validation():
    """Test frequency range input validation"""
    # Test valid ranges (e.g., "0,1000")
    # Test invalid formats
    # Test overlapping ranges
    
def test_sensitivity_gain_validation():
    """Test S and G parameter validation"""
    # Test numeric inputs
    # Test default values
    # Test boundary conditions
```

##### D. Indices Calculation Tests
```python
def test_all_60_indices():
    """Verify all 60 indices are calculated"""
    # Process test file
    # Check output has all expected columns
    # Verify no NaN values for valid input
    
def test_specific_marine_indices():
    """Test marine-specific indices (NDSI, BI, etc.)"""
    # Use known test signals
    # Verify correct calculations with custom frequency bands
```

### 5. File Structure
```
scikit-maad-biosound/
├── SciKit_Maad_File_Processing-GUI.py  # Main unified GUI
├── CLAUDE.md                           # This documentation
├── README_MERGE.md                     # Colleague review document
├── tests/
│   ├── test_frequency_bands.py
│   ├── test_parallel_processing.py
│   ├── test_gui_inputs.py
│   └── test_indices_calculation.py
├── test_data/                          # Test WAV files
└── archive/                            # Original versions for reference
```

### 6. Commands to Run

#### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_frequency_bands.py -v
```

#### Check Code Quality
```bash
# Type checking (if using type hints)
mypy SciKit_Maad_File_Processing-GUI.py

# Linting
pylint SciKit_Maad_File_Processing-GUI.py
```

### 7. Performance Benchmarks
- Current: ~30 seconds per file (single-threaded)
- Target: <10 seconds per file with 4 cores
- Test with dataset of 100+ WAV files

### 8. Known Limitations
- ACI calculation may vary between implementations
- Some indices (e.g., ROItotal) may not always be computed
- Memory usage increases with parallel processing

### 9. Future Enhancements (Not in current scope)
- GPU acceleration for spectrogram computation
- Real-time processing mode
- Web-based interface
- Database storage for results

## Quick Start for Development

1. Review archive versions to understand frequency band modifications
2. Implement GUI changes with backward compatibility
3. Add parallel processing with configurable worker count
4. Write and run tests for each component
5. Benchmark performance improvements
6. Document any breaking changes

## Contact
For questions about the frequency band calculations, refer to the email thread with Sylvain Haupert (CNRS) in notes/OSA_Sylvian_Re_Connection from Carly Batist RE_Scikit-maad.pdf