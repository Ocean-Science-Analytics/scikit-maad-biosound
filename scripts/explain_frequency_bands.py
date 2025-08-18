#!/usr/bin/env python3
"""
Explain exactly how flim_low and flim_mid map to anthrophony/biophony
"""

print("🔍 HOW FREQUENCY BANDS WORK IN SCIKIT-MAAD")
print("=" * 60)

print("\nIn the standard scikit-maad library:")
print("  all_spectral_alpha_indices(flim_low, flim_mid, flim_hi)")
print("  ↓")
print("  soundscape_index(flim_antroPh=flim_low, flim_bioPh=flim_mid)")

print("\nSo the mapping is:")
print("  flim_low  → anthropophony (human-made noise)")
print("  flim_mid  → biophony (biological sounds)")
print("  flim_hi   → higher biological sounds")

print("\nDefault values:")
print("  flim_low = [0, 1000]     → anthropophony = 0-1000 Hz")
print("  flim_mid = [1000, 10000] → biophony = 1000-10000 Hz")

print("\nOur current marine settings:")
print("  flim_low = [0, 1500]     → anthropophony = 0-1500 Hz (ship noise)")
print("  flim_mid = [1500, 8000]  → biophony = 1500-8000 Hz (marine life)")

print("\n✅ This is CORRECT for any environment:")
print("  Lower frequencies = human/mechanical noise")
print("  Higher frequencies = biological sounds")

print("\n🎯 The 'marine correction' was just:")
print("  1. Using frequency ranges appropriate for marine acoustics")
print("  2. Making sure the 5 indices (NDSI, rBA, etc.) use these ranges correctly")
print("  3. Allowing users to customize these ranges in the GUI")

print("\n🚫 It was NOT about 'forest vs ocean logic' - that was my mistake!")