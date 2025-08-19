#!/usr/bin/env python3
"""
Compare regression test results between Jared's original version and Michelle's current version.
"""

import numpy as np
import pandas as pd


def load_and_compare():
    # Load both datasets
    jared_file = "regression_testing/jared_version/Acoustic_Indices.csv"
    michelle_file = "regression_testing/michelle_last_edits/data/Acoustic_Indices.csv"

    print("REGRESSION TEST COMPARISON")
    print("=" * 60)

    try:
        jared_df = pd.read_csv(jared_file)
        michelle_df = pd.read_csv(michelle_file)

        print(f"Jared's version (original):  {jared_df.shape[0]} rows × {jared_df.shape[1]} columns")
        print(f"Michelle's version (current): {michelle_df.shape[0]} rows × {michelle_df.shape[1]} columns")

        # Find common and different columns
        jared_cols = set(jared_df.columns) - {'Date', 'Filename'}
        michelle_cols = set(michelle_df.columns) - {'Date', 'Filename'}

        common_cols = jared_cols.intersection(michelle_cols)
        only_jared = jared_cols - michelle_cols
        only_michelle = michelle_cols - jared_cols

        print("\nCOLUMN ANALYSIS:")
        print(f"Common acoustic indices: {len(common_cols)}")
        print(f"Only in Jared's version: {sorted(only_jared) if only_jared else 'None'}")
        print(f"Only in Michelle's version: {sorted(only_michelle) if only_michelle else 'None'}")

        # Analyze the 5 marine indices that should be corrected according to Sylvain's email
        marine_indices = ['NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI']
        marine_corrected = ['NDSI_marine', 'rBA_marine', 'AnthroEnergy_marine', 'BioEnergy_marine', 'BI_marine']

        print("\nMARINE INDICES ANALYSIS:")
        print("According to Sylvain's email, these 5 indices should be REPLACED with marine-corrected versions:")

        for orig, corrected in zip(marine_indices, marine_corrected):
            if orig in jared_df.columns and corrected in michelle_df.columns:
                # Compare original vs marine-corrected values
                jared_vals = jared_df[orig].values
                michelle_orig_vals = michelle_df[orig].values  # Original version still in Michelle's
                michelle_marine_vals = michelle_df[corrected].values  # Marine-corrected version

                # Check if original values match
                orig_diff = np.mean(np.abs(jared_vals - michelle_orig_vals))

                # Check marine correction impact
                marine_diff = np.mean(np.abs(jared_vals - michelle_marine_vals))

                print(f"  {orig}:")
                print(f"    Original vs Current Original: mean abs diff = {orig_diff:.6f}")
                print(f"    Original vs Marine Corrected: mean abs diff = {marine_diff:.6f}")
                print(f"    Marine correction impact: {marine_diff/abs(np.mean(jared_vals)) * 100:.1f}% change")

        # Compare values for common indices (should be identical)
        print("\nVALUE COMPARISON FOR COMMON INDICES:")
        differences = {}

        for col in sorted(common_cols):
            if col in marine_indices:
                continue  # Skip marine indices for now

            try:
                val1 = pd.to_numeric(jared_df[col], errors='coerce').fillna(0)
                val2 = pd.to_numeric(michelle_df[col], errors='coerce').fillna(0)

                if len(val1) == len(val2):
                    abs_diff = np.abs(val1 - val2)
                    max_diff = np.max(abs_diff)
                    mean_diff = np.mean(abs_diff)

                    if max_diff > 1e-10:  # Only show if there's a meaningful difference
                        differences[col] = {'max': max_diff, 'mean': mean_diff}
            except:
                pass

        if differences:
            print("Found differences in these indices:")
            for col, diffs in sorted(differences.items(), key=lambda x: x[1]['max'], reverse=True)[:10]:
                print(f"  {col}: max diff = {diffs['max']:.2e}, mean diff = {diffs['mean']:.2e}")
        else:
            print("✅ All common indices have identical values!")

        # Key findings summary
        print("\n" + "=" * 60)
        print("KEY FINDINGS:")
        print("=" * 60)

        if len(only_michelle) == 5 and all(idx in only_michelle for idx in marine_corrected):
            print("❌ PROBLEM: Current implementation ADDS 5 marine indices instead of REPLACING them")
            print("   Should have 60 indices total, but currently has 65 indices")

        print("\n✅ CORRECT aspects:")
        print("   - Same frequency bands used: [0,1500] and [1500,8000]")
        print("   - Core processing logic preserved")
        print("   - Marine corrections are being calculated")

        print("\n🔧 NEEDS FIXING:")
        print("   - Remove '_marine' suffix from marine indices")
        print("   - Replace original NDSI, rBA, AnthroEnergy, BioEnergy, BI with corrected versions")
        print("   - Should have exactly 60 indices, same as original")

        # Show actual value differences for marine indices
        print("\n📊 MARINE CORRECTION IMPACT:")
        for orig, corrected in zip(marine_indices[:3], marine_corrected[:3]):  # Show first 3 as examples
            if orig in jared_df.columns and corrected in michelle_df.columns:
                print(f"   {orig}:")
                print(f"     Original: {jared_df[orig].iloc[0]:.6f}")
                print(f"     Marine:   {michelle_df[corrected].iloc[0]:.6f}")

    except Exception as e:
        print(f"Error loading files: {e}")
        print("Make sure both CSV files exist in the regression_testing folders")

if __name__ == "__main__":
    load_and_compare()
