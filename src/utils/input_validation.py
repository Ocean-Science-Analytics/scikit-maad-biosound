#!/usr/bin/env python3
"""
Input validation utilities for the scikit-maad biosound application.

This module provides comprehensive validation for user inputs to prevent
crashes and provide clear error messages.
"""

import os
from typing import List, Optional, Tuple, Union


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


def validate_directory_exists(
    directory_path: str, directory_name: str = "Directory"
) -> str:
    """
    Validate that a directory exists and is accessible.

    Args:
        directory_path: Path to the directory
        directory_name: Human-readable name for error messages

    Returns:
        str: Absolute path to the directory

    Raises:
        ValidationError: If directory doesn't exist or isn't accessible
    """
    if not directory_path:
        raise ValidationError(f"{directory_name} path cannot be empty")

    directory_path = str(directory_path).strip()
    if not directory_path:
        raise ValidationError(f"{directory_name} path cannot be empty")

    abs_path = os.path.abspath(directory_path)

    if not os.path.exists(abs_path):
        raise ValidationError(f"{directory_name} does not exist:\n{abs_path}")

    if not os.path.isdir(abs_path):
        raise ValidationError(f"{directory_name} is not a directory:\n{abs_path}")

    if not os.access(abs_path, os.R_OK):
        raise ValidationError(f"{directory_name} is not readable:\n{abs_path}")

    return abs_path


def validate_frequency_range(
    freq_range: Union[str, List[float]], range_name: str = "Frequency range"
) -> List[float]:
    """
    Validate frequency range input.

    Args:
        freq_range: Either a string like "0,1500" or a list [0, 1500]
        range_name: Human-readable name for error messages

    Returns:
        List[float]: Validated frequency range [min, max]

    Raises:
        ValidationError: If range is invalid
    """
    # Handle string input
    if isinstance(freq_range, str):
        freq_range = freq_range.strip()
        if not freq_range:
            raise ValidationError(f"{range_name} cannot be empty")

        try:
            parts = freq_range.split(",")
            if len(parts) != 2:
                raise ValidationError(
                    f"{range_name} must have exactly 2 values separated by comma (e.g., '0,1500')"
                )

            freq_range = [float(part.strip()) for part in parts]
        except ValueError:
            raise ValidationError(
                f"{range_name} contains invalid numbers: {freq_range}"
            )

    # Handle list input
    if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
        raise ValidationError(f"{range_name} must contain exactly 2 values")

    try:
        freq_min, freq_max = float(freq_range[0]), float(freq_range[1])
    except (ValueError, TypeError):
        raise ValidationError(f"{range_name} values must be numbers: {freq_range}")

    # Validate range constraints
    if freq_min < 0:
        raise ValidationError(
            f"{range_name} minimum frequency cannot be negative: {freq_min}"
        )

    if freq_max < 0:
        raise ValidationError(
            f"{range_name} maximum frequency cannot be negative: {freq_max}"
        )

    if freq_min >= freq_max:
        raise ValidationError(
            f"{range_name} minimum frequency ({freq_min}) must be less than maximum ({freq_max})"
        )

    if freq_max > 50000:  # Reasonable upper limit for audio
        raise ValidationError(
            f"{range_name} maximum frequency too high ({freq_max} Hz). Consider values under 50,000 Hz"
        )

    return [freq_min, freq_max]


def validate_parameter_value(
    value: Union[str, float],
    param_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """
    Validate a numeric parameter.

    Args:
        value: Parameter value (string or number)
        param_name: Human-readable parameter name
        min_val: Optional minimum allowed value
        max_val: Optional maximum allowed value

    Returns:
        float: Validated parameter value

    Raises:
        ValidationError: If parameter is invalid
    """
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValidationError(f"{param_name} cannot be empty")

    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{param_name} must be a valid number: '{value}'")

    if min_val is not None and num_value < min_val:
        raise ValidationError(
            f"{param_name} must be at least {min_val}, got {num_value}"
        )

    if max_val is not None and num_value > max_val:
        raise ValidationError(
            f"{param_name} must be at most {max_val}, got {num_value}"
        )

    return num_value


def validate_time_interval(interval: Union[str, int]) -> int:
    """
    Validate time interval for manual mode.

    Args:
        interval: Time interval in seconds

    Returns:
        int: Validated time interval

    Raises:
        ValidationError: If interval is invalid
    """
    if isinstance(interval, str):
        interval = interval.strip()
        if not interval:
            raise ValidationError("Time interval cannot be empty")

    try:
        interval_val = int(interval)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Time interval must be a whole number of seconds: '{interval}'"
        )

    if interval_val <= 0:
        raise ValidationError(
            f"Time interval must be positive, got {interval_val} seconds"
        )

    if interval_val > 86400:  # 24 hours
        raise ValidationError(
            f"Time interval too large ({interval_val} seconds). Consider values under 24 hours (86400 seconds)"
        )

    return interval_val


def validate_wav_filename(filename: str) -> bool:
    """
    Validate WAV file naming convention.

    Expected format: [prefix]_YYYYMMDD_HHMMSS_[suffix].wav

    Args:
        filename: Filename to validate

    Returns:
        bool: True if filename follows convention
    """
    if not filename.lower().endswith(".wav"):
        return False

    basename = os.path.basename(filename)
    parts = basename.split("_")

    # Need at least 4 parts: prefix, date, time, suffix.wav
    if len(parts) < 4:
        return False

    # Check date part (second part)
    date_part = parts[1]
    if len(date_part) != 8 or not date_part.isdigit():
        return False

    # Basic date validation
    try:
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])

        if not (1900 <= year <= 2100):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
    except ValueError:
        return False

    # Check time part (third part)
    time_part = parts[2]
    if len(time_part) != 6 or not time_part.isdigit():
        return False

    # Basic time validation
    try:
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])

        if not (0 <= hour <= 23):
            return False
        if not (0 <= minute <= 59):
            return False
        if not (0 <= second <= 59):
            return False
    except ValueError:
        return False

    return True


def find_wav_files(directory: str) -> Tuple[List[str], List[str]]:
    """
    Find WAV files in directory and categorize by naming convention.

    Args:
        directory: Directory to search

    Returns:
        Tuple[List[str], List[str]]: (valid_files, invalid_files)
    """
    valid_files = []
    invalid_files = []

    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                if validate_wav_filename(file):
                    valid_files.append(file_path)
                else:
                    invalid_files.append(file_path)

    return valid_files, invalid_files


def validate_marine_acoustic_parameters(
    flim_low: Union[str, List[float]],
    flim_mid: Union[str, List[float]],
    sensitivity: Union[str, float],
    gain: Union[str, float],
) -> dict:
    """
    Validate complete set of marine acoustic parameters.

    Args:
        flim_low: Anthrophony frequency range
        flim_mid: Biophony frequency range
        sensitivity: Hydrophone sensitivity (dB)
        gain: Recording gain (dB)

    Returns:
        dict: Validated parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    result = {}

    # Validate frequency ranges
    result["flim_low"] = validate_frequency_range(
        flim_low, "Anthrophony frequency range"
    )
    result["flim_mid"] = validate_frequency_range(flim_mid, "Biophony frequency range")

    # Check that frequency ranges don't overlap inappropriately
    if result["flim_low"][1] > result["flim_mid"][0]:
        # Allow some overlap but warn if significant
        overlap = result["flim_low"][1] - result["flim_mid"][0]
        if (
            overlap > (result["flim_mid"][1] - result["flim_mid"][0]) * 0.2
        ):  # More than 20% overlap
            raise ValidationError(
                f"Frequency ranges overlap significantly:\n"
                f"Anthrophony: {result['flim_low'][0]}-{result['flim_low'][1]} Hz\n"
                f"Biophony: {result['flim_mid'][0]}-{result['flim_mid'][1]} Hz\n"
                f"Consider adjusting ranges to minimize overlap"
            )

    # Validate sensitivity and gain
    result["sensitivity"] = validate_parameter_value(
        sensitivity, "Sensitivity", min_val=-200, max_val=0
    )
    result["gain"] = validate_parameter_value(gain, "Gain", min_val=-100, max_val=100)

    return result


def validate_processing_parameters(
    mode: str, time_interval: Optional[Union[str, int]] = None
) -> dict:
    """
    Validate processing mode and related parameters.

    Args:
        mode: Processing mode ('hourly', 'dataset', 'manual')
        time_interval: Time interval for manual mode

    Returns:
        dict: Validated parameters

    Raises:
        ValidationError: If parameters are invalid
    """
    valid_modes = [
        "hourly",
        "dataset",
        "manual",
        "daily",
    ]  # Include legacy 'daily' mode

    if mode not in valid_modes:
        raise ValidationError(
            f"Invalid processing mode: '{mode}'. Must be one of: {valid_modes}"
        )

    result = {"mode": mode}

    if mode == "manual":
        if time_interval is None:
            raise ValidationError("Time interval is required for manual mode")
        result["time_interval"] = validate_time_interval(time_interval)
    elif mode == "hourly":
        result["time_interval"] = 3600  # 1 hour
    else:  # dataset or daily
        result["time_interval"] = 0  # Process entire file

    return result
