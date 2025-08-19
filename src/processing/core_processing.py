#!/usr/bin/env python3
"""
Core processing functions extracted from GUI for standalone testing.
This module contains the processing logic without GUI dependencies.
"""

import os
import sys
from multiprocessing import Pool, cpu_count

import pandas as pd

from processing.standalone_processor import process_single_file_standalone

# Import debug configuration
try:
    from utils.debug_utils import debug_print
except ImportError:
    # Fallback if debug_utils not available (for standalone use)
    def debug_print(message):
        pass


def process_files_sequential(file_paths, params):
    """
    Process files sequentially (for comparison or when parallel is disabled).

    Returns:
        list: Processing results for each file
    """
    results = []
    for filepath in file_paths:
        result = process_single_file_standalone((filepath, params))
        results.append(result)

    return results


def process_files_parallel(file_paths, params):
    """
    Process files in parallel using multiprocessing.

    Returns:
        list: Processing results for each file
    """
    debug_print(f"[DEBUG] Starting parallel processing with {len(file_paths)} files")

    # Check if we're on macOS and running from a GUI
    if sys.platform == "darwin":
        debug_print("[DEBUG] Running on macOS - checking multiprocessing start method")
        try:
            import multiprocessing

            if multiprocessing.get_start_method() != "spawn":
                multiprocessing.set_start_method("spawn", force=True)
                debug_print("[DEBUG] Set multiprocessing method to 'spawn'")
        except RuntimeError:
            debug_print("[DEBUG] Start method already set, continuing...")

    num_workers = min(cpu_count() - 1, 4)  # Leave one core free, max 4 workers
    debug_print(f"[DEBUG] Using {num_workers} worker processes")

    # Prepare arguments for parallel processing
    file_args = [(filepath, params) for filepath in file_paths]
    debug_print(f"[DEBUG] Prepared {len(file_args)} file arguments")

    try:
        debug_print("[DEBUG] Creating process pool...")
        with Pool(num_workers) as pool:
            debug_print("[DEBUG] Pool created, starting map operation...")
            async_result = pool.map_async(process_single_file_standalone, file_args)
            results = async_result.get()
            debug_print(f"[DEBUG] Map operation completed, got {len(results)} results")

        debug_print("[DEBUG] Parallel processing completed successfully")
        return results

    except Exception as e:
        debug_print(f"[DEBUG] ERROR in parallel processing: {e!s}")
        import traceback

        traceback.print_exc()
        return []


def convert_results_to_dataframes(processing_results, filename_list, date_list, mode):
    """
    Convert processing results to pandas DataFrames.

    Returns:
        tuple: (result_df, result_df_per_bin)
    """
    result_df = pd.DataFrame()
    result_df_per_bin = pd.DataFrame()

    # Create filename to date mapping
    dict(zip(filename_list, date_list))

    for i, result in enumerate(processing_results):
        if result is None:
            debug_print(f"[DEBUG] Result {i} is None, skipping")
            continue

        debug_print(
            f"[DEBUG] Processing result {i}: filename={result.get('filename', 'unknown')}"
        )
        filename = os.path.basename(result["filename"])
        parsed_date = result["parsed_date"]

        debug_print(f"[DEBUG] Result has {len(result['results'])} segments")
        for j, segment_result in enumerate(result["results"]):
            if segment_result is None:
                debug_print(f"[DEBUG] Segment {j} is None, skipping")
                continue

            debug_print(
                f"[DEBUG] Processing segment {j}, indices type: {type(segment_result.get('indices'))}"
            )

            # Add main indices with proper column ordering
            indices_row = segment_result["indices"].copy()

            # Create ordered dictionary with Date and Filename first
            ordered_row = {
                "Date": parsed_date,
                "Filename": filename,
                **{
                    k: v for k, v in indices_row.items() if k != "Filename"
                },  # Add all other indices, excluding any existing Filename
            }

            result_df = pd.concat(
                [result_df, pd.DataFrame([ordered_row])], ignore_index=True
            )

            # Add per-bin indices
            if "indices_per_bin" in segment_result:
                indices_per_bin = segment_result["indices_per_bin"].copy()
                indices_per_bin["Filename"] = filename
                indices_per_bin["Date"] = parsed_date
                result_df_per_bin = pd.concat(
                    [result_df_per_bin, indices_per_bin], ignore_index=True
                )

    return result_df, result_df_per_bin


def parse_date_and_filename_from_filename(filename):
    """
    Parse date and filename from WAV file naming convention.

    Expected format: [prefix]_YYYYMMDD_HHMMSS_[suffix].wav

    Args:
        filename: Full path to the WAV file

    Returns:
        tuple: (datetime object, filename with numbering) or (None, None) if parsing fails
    """
    import datetime

    try:
        basename = os.path.basename(filename)
        parts = basename.split("_")

        # Need at least 4 parts: prefix, date, time, suffix
        if len(parts) < 4:
            raise ValueError("Not enough underscore-separated parts")

        # Parse date from second part
        date_str = parts[1]
        if len(date_str) != 8:
            raise ValueError("Date part should be 8 digits (YYYYMMDD)")
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        # Parse time from third part
        time_str = parts[2]
        if len(time_str) != 6:
            raise ValueError("Time part should be 6 digits (HHMMSS)")
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        dt = datetime.datetime(year, month, day, hour, minute, second)
        filename_with_numbering = "_".join(parts[:-1]) + "_" + parts[-1]
        return dt, filename_with_numbering

    except Exception as e:
        print(f"    Could not parse '{os.path.basename(filename)}': {e!s}")
        return None, None
