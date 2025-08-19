#!/usr/bin/env python3
"""
Generate a side-by-side performance comparison report
"""

import json
from datetime import datetime
from multiprocessing import cpu_count


def generate_performance_report(
    sequential_time,
    parallel_time,
    num_files,
    num_workers,
    output_folder,
    additional_info=None,
):
    """
    Generate a detailed performance comparison report.

    Args:
        sequential_time: Time taken for sequential processing
        parallel_time: Time taken for parallel processing
        num_files: Number of files processed
        num_workers: Number of parallel workers used
        output_folder: Where to save the report
        additional_info: Dict with extra info (file sizes, etc.)
    """

    # Calculate metrics
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = (speedup / num_workers) * 100 if num_workers > 0 else 0
    time_saved = sequential_time - parallel_time

    # Create report data
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {"cpu_cores": cpu_count(), "workers_used": num_workers},
        "processing_info": {
            "num_files": num_files,
            "sequential_time_sec": round(sequential_time, 2),
            "parallel_time_sec": round(parallel_time, 2),
            "time_saved_sec": round(time_saved, 2),
            "time_saved_min": round(time_saved / 60, 2),
        },
        "performance_metrics": {
            "speedup_factor": round(speedup, 2),
            "efficiency_percent": round(efficiency, 1),
            "performance_rating": get_performance_rating(speedup, efficiency),
        },
    }

    if additional_info:
        report_data["additional_info"] = additional_info

    # Generate text report
    report_text = generate_text_report(report_data)

    # Save both JSON and text formats
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_filename = f"performance_report_{timestamp_str}.json"
    txt_filename = f"performance_report_{timestamp_str}.txt"

    import os

    json_path = os.path.join(output_folder, json_filename)
    txt_path = os.path.join(output_folder, txt_filename)

    # Save JSON report
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Save text report
    with open(txt_path, "w") as f:
        f.write(report_text)

    print("\nPerformance reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")

    return report_data


def get_performance_rating(speedup, efficiency):
    """Get a qualitative rating for the performance improvement"""
    if speedup >= 3.0 and efficiency >= 70:
        return "Excellent"
    elif speedup >= 2.0 and efficiency >= 60:
        return "Very Good"
    elif speedup >= 1.5 and efficiency >= 40:
        return "Good"
    elif speedup >= 1.2:
        return "Moderate"
    else:
        return "Poor"


def generate_text_report(data):
    """Generate a human-readable text report"""

    report = f"""
SCIKIT-MAAD PARALLEL PROCESSING PERFORMANCE REPORT
=================================================

Generated: {data["timestamp"]}

SYSTEM CONFIGURATION
-------------------
CPU Cores Available: {data["system_info"]["cpu_cores"]}
Parallel Workers Used: {data["system_info"]["workers_used"]}

PROCESSING SUMMARY
-----------------
Files Processed: {data["processing_info"]["num_files"]}

Sequential Processing: {data["processing_info"]["sequential_time_sec"]} seconds
                      ({data["processing_info"]["sequential_time_sec"] / 60:.1f} minutes)

Parallel Processing:   {data["processing_info"]["parallel_time_sec"]} seconds
                      ({data["processing_info"]["parallel_time_sec"] / 60:.1f} minutes)

Time Saved:           {data["processing_info"]["time_saved_sec"]} seconds
                      ({data["processing_info"]["time_saved_min"]} minutes)

PERFORMANCE METRICS
------------------
Speedup Factor:       {data["performance_metrics"]["speedup_factor"]}x faster
Parallel Efficiency:  {data["performance_metrics"]["efficiency_percent"]}%
Performance Rating:   {data["performance_metrics"]["performance_rating"]}

INTERPRETATION
-------------
"""

    # Add interpretation based on results
    speedup = data["performance_metrics"]["speedup_factor"]
    efficiency = data["performance_metrics"]["efficiency_percent"]

    if speedup >= 3.0:
        report += "🚀 Excellent speedup! Parallel processing is highly effective for your workload.\n"
    elif speedup >= 2.0:
        report += (
            "✅ Very good speedup! Parallel processing provides significant benefits.\n"
        )
    elif speedup >= 1.5:
        report += "👍 Good speedup! Parallel processing is beneficial for larger batch jobs.\n"
    elif speedup >= 1.2:
        report += (
            "📈 Moderate speedup! Consider parallel processing for large datasets.\n"
        )
    else:
        report += "⚠️  Limited speedup. Sequential processing may be sufficient.\n"

    if efficiency >= 70:
        report += "💪 High efficiency - excellent use of available CPU cores.\n"
    elif efficiency >= 50:
        report += "👌 Good efficiency - CPU cores are well utilized.\n"
    elif efficiency >= 30:
        report += "📊 Moderate efficiency - some CPU cores may be underutilized.\n"
    else:
        report += (
            "🔧 Low efficiency - consider workload characteristics or reduce workers.\n"
        )

    report += """
RECOMMENDATIONS
--------------
"""

    num_files = data["processing_info"]["num_files"]

    if speedup >= 2.0:
        report += "• Use parallel processing by default for batch jobs\n"
        report += "• Consider increasing batch sizes to maximize efficiency\n"
    elif speedup >= 1.5:
        report += "• Use parallel processing for 5+ files\n"
        report += "• Sequential processing may be fine for small batches\n"
    else:
        report += "• Parallel processing may not be needed for this workload\n"
        report += "• Consider sequential processing for simplicity\n"

    if efficiency < 50:
        report += "• Consider reducing number of workers for better efficiency\n"

    if num_files < 4:
        report += "• Test with larger batch sizes to see full parallel benefits\n"

    report += """
TECHNICAL NOTES
--------------
• Speedup = Sequential Time ÷ Parallel Time
• Efficiency = (Speedup ÷ Workers) × 100%
• Results may vary with different file sizes and system loads
• Parallel processing overhead is more noticeable with smaller files

For questions about these results, refer to the marine acoustics GUI documentation.
"""

    return report


def create_comparison_summary(old_time, new_time, file_count):
    """Create a quick comparison summary for display"""

    if old_time <= 0:
        return "No comparison data available"

    speedup = old_time / new_time if new_time > 0 else 0
    time_saved = old_time - new_time
    percent_faster = ((old_time - new_time) / old_time) * 100

    summary = f"""
PERFORMANCE IMPROVEMENT SUMMARY
==============================
Files processed: {file_count}
Old processing time: {old_time:.1f} seconds ({old_time / 60:.1f} min)
New processing time: {new_time:.1f} seconds ({new_time / 60:.1f} min)
Time saved: {time_saved:.1f} seconds ({time_saved / 60:.1f} min)
Performance improvement: {speedup:.1f}x faster ({percent_faster:.0f}% reduction)
"""

    return summary


if __name__ == "__main__":
    # Test the report generation
    print("Testing performance report generation...")

    # Mock data
    test_data = generate_performance_report(
        sequential_time=45.2,
        parallel_time=12.8,
        num_files=10,
        num_workers=4,
        output_folder=".",
        additional_info={
            "average_file_size": "5 minutes",
            "frequency_bands": "Marine acoustics (0-1000, 1000-8000 Hz)",
            "gui_version": "Phase 1 with parallel processing",
        },
    )

    print("\nSample report data:")
    print(f"Speedup: {test_data['performance_metrics']['speedup_factor']}x")
    print(f"Efficiency: {test_data['performance_metrics']['efficiency_percent']}%")
    print(f"Rating: {test_data['performance_metrics']['performance_rating']}")

    # Test summary
    summary = create_comparison_summary(45.2, 12.8, 10)
    print(summary)
