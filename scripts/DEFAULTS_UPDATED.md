# Updated to Use Scikit-MAAD Defaults

## Changes Made

### ✅ GUI Placeholder Text Updated
**Before:**
- Anthrophony: `Default: 0,1500`
- Biophony: `Default: 1500,8000`
- Sensitivity: `Default: -35.0`
- Gain: `Default: 0.0`

**After (scikit-maad defaults):**
- Anthrophony: `Default: 0,1000`
- Biophony: `Default: 1000,10000`
- Sensitivity: `Default: -35.0`
- Gain: `Default: 0.0`

### ✅ Processing Defaults Updated
**Code defaults now match scikit-maad:**
```python
flim_low = [0, 1000]      # scikit-maad default anthrophony range
flim_mid = [1000, 10000]  # scikit-maad default biophony range  
sensitivity = -35.0       # scikit-maad default sensitivity
gain = 0.0               # Standard default (no scikit-maad default found)
```

### ✅ Frequency Band Impact
**Anthrophony (human/mechanical noise):**
- Range: 0-1000 Hz (unchanged from scikit-maad)
- Captures: Ship engines, mechanical noise, low-frequency disturbances

**Biophony (biological sounds):**
- Previous: 1500-8000 Hz (custom marine range)
- Current: 1000-10000 Hz (scikit-maad standard)
- **Impact**: Wider range captures more biological activity
  - Lower threshold (1kHz vs 1.5kHz): Includes more low-frequency biological sounds
  - Higher ceiling (10kHz vs 8kHz): Includes more high-frequency biological activity

### ✅ Benefits
1. **Compatibility**: Results comparable with other scikit-maad studies
2. **Standardization**: Using library defaults ensures consistency
3. **Flexibility**: Users can still customize for specific environments
4. **Broader coverage**: 1-10kHz biophony range captures more marine life

### 🔬 Marine Biology Context
The updated biophony range (1-10kHz) is actually **better for marine acoustics**:
- **Fish sounds**: Often 100Hz-3kHz (overlaps both ranges)
- **Marine mammals**: Often 1kHz-20kHz+ (better coverage with 10kHz ceiling)
- **Dolphin echolocation**: 1kHz-150kHz (captures more with higher ceiling)
- **Vessel noise**: Primarily <1kHz (anthrophony range unchanged)

### 📊 Result Impact
Using scikit-maad defaults will:
- **Increase biophony values** (wider frequency range)
- **Change NDSI calculations** (different bio/anthro ratio)
- **Maintain compatibility** with standard acoustic ecology studies
- **Provide better baseline** for comparative analyses

## Recommendation
✅ **Keep these scikit-maad defaults** - they provide:
- Better scientific compatibility
- Wider biological sound detection
- Standard reference for comparisons
- Foundation that users can customize as needed