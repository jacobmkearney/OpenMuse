"""
Validate Battery subpacket frequency across all test data files.

This script analyzes the timing of Battery subpackets using BOTH device timestamps
(pkt_time) and message timestamps (message_time) to understand the actual
transmission frequency.

The key issue: device timestamps span hours/days (device uptime), but recordings
are only ~60 seconds. We need to use message_time for accurate interval analysis.
"""

import os
import sys

# Add parent directory to path to import MuseLSL3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MuseLSL3.decode_new import parse_message
import numpy as np


def analyze_battery_frequency(filepath):
    """Analyze battery subpacket timing for a single file.

    Returns:
    --------
    dict with keys:
        - file: filename
        - n_battery: number of battery subpackets
        - file_duration: total duration of recording file (first to last message)
        - battery_span: time span between first and last battery packet
        - device_duration: duration based on device timestamps (seconds)
        - intervals_message: list of time intervals between battery packets (message time)
        - mean_interval: mean interval using message timestamps
        - frequency: 1/mean_interval in Hz
        - expected_freq: expected frequency based on FREQ_MAP (0.1 Hz for Battery)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        messages = f.readlines()

    # Get file duration (first to last message)
    first_msg_time = None
    last_msg_time = None

    # Collect battery data with both timestamp types
    battery_data = []
    expected_freq = None

    for message in messages:
        subpackets = parse_message(message)
        if subpackets:
            msg_time = subpackets[0]["message_time"].timestamp()
            if first_msg_time is None:
                first_msg_time = msg_time
            last_msg_time = msg_time

        for sp in subpackets:
            if sp["pkt_type"] == "Battery":
                battery_data.append(
                    {
                        "device_time": sp["pkt_time"],
                        "message_time": sp["message_time"].timestamp(),
                    }
                )
                if expected_freq is None:
                    expected_freq = sp["pkt_freq"]

    file_duration = (
        last_msg_time - first_msg_time if first_msg_time and last_msg_time else None
    )

    if len(battery_data) < 2:
        return {
            "file": os.path.basename(filepath),
            "n_battery": len(battery_data),
            "file_duration": file_duration,
            "battery_span": None,
            "device_duration": None,
            "intervals_message": [],
            "mean_interval": None,
            "frequency": None,
            "expected_freq": expected_freq,
        }

    # Sort by message time (actual recording time)
    battery_data = sorted(battery_data, key=lambda x: x["message_time"])

    # Extract timestamps
    device_times = [d["device_time"] for d in battery_data]
    message_times = [d["message_time"] for d in battery_data]

    # Calculate intervals using MESSAGE time (actual recording time)
    intervals_message = [
        message_times[i] - message_times[i - 1] for i in range(1, len(message_times))
    ]

    battery_span = message_times[-1] - message_times[0]
    device_duration = device_times[-1] - device_times[0]

    mean_interval = np.mean(intervals_message) if intervals_message else None
    frequency = 1.0 / mean_interval if mean_interval and mean_interval > 0 else None

    return {
        "file": os.path.basename(filepath),
        "n_battery": len(battery_data),
        "file_duration": file_duration,
        "battery_span": battery_span,
        "device_duration": device_duration,
        "intervals_message": intervals_message,
        "mean_interval": mean_interval,
        "frequency": frequency,
        "expected_freq": expected_freq,
    }


def main():
    """Analyze all test data files."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data")

    # All test files
    test_files = [
        "data_p20.txt",
        "data_p21.txt",
        "data_p50.txt",
        "data_p51.txt",
        "data_p60.txt",
        "data_p61.txt",
        "data_p1034.txt",
        "data_p1035.txt",
        "data_p1041.txt",
        "data_p1042.txt",
        "data_p1043.txt",
        "data_p1044.txt",
        "data_p1045.txt",
        "data_p1046.txt",
        "data_p4129.txt",
        "test_battery_16_80.txt",
    ]

    print("=" * 80)
    print("Battery Subpacket Frequency Analysis")
    print("=" * 80)
    print()

    results = []

    for filename in test_files:
        filepath = os.path.join(test_data_dir, filename)
        if not os.path.exists(filepath):
            continue

        result = analyze_battery_frequency(filepath)
        results.append(result)

    # Print summary table
    print(
        f"{'File':<25s} {'N':<4s} {'File':>9s} {'BatSpan':>9s} {'Device':>10s} {'Mean Δt':>9s} {'Freq':>9s} {'Exp':>8s}"
    )
    print("-" * 95)

    for r in results:
        if r["n_battery"] >= 2:
            file_dur = f"{r['file_duration']:.1f}s" if r["file_duration"] else "N/A"
            bat_span = f"{r['battery_span']:.1f}s" if r["battery_span"] else "N/A"
            dev_dur = f"{r['device_duration']:.0f}s" if r["device_duration"] else "N/A"
            mean_int = f"{r['mean_interval']:.2f}s" if r["mean_interval"] else "N/A"
            freq_str = f"{r['frequency']:.4f}Hz" if r["frequency"] else "N/A"
            expected_str = (
                f"{r['expected_freq']:.1f}Hz" if r["expected_freq"] else "N/A"
            )
            print(
                f"{r['file']:<25s} {r['n_battery']:<4d} {file_dur:>9s} {bat_span:>9s} {dev_dur:>10s} "
                f"{mean_int:>9s} {freq_str:>9s} {expected_str:>8s}"
            )
        else:
            print(
                f"{r['file']:<25s} {r['n_battery']:<4d} {'N/A':>9s} {'N/A':>9s} {'N/A':>10s} "
                f"{'N/A':>9s} {'N/A':>9s} {'N/A':>8s}"
            )

    print()
    print("=" * 80)
    print("Detailed Statistics")
    print("=" * 80)
    print()

    # Calculate overall statistics
    all_intervals = []
    all_freqs = []

    for r in results:
        if r["intervals_message"]:
            all_intervals.extend(r["intervals_message"])
            if r["frequency"]:
                all_freqs.append(r["frequency"])

    if all_intervals:
        print(f"Overall Statistics (across {len(results)} files):")
        print(f"  Total battery packets: {sum(r['n_battery'] for r in results)}")
        print(f"  Total intervals analyzed: {len(all_intervals)}")
        print(
            f"  Mean interval: {np.mean(all_intervals):.2f} ± {np.std(all_intervals):.2f} seconds"
        )
        print(f"  Median interval: {np.median(all_intervals):.2f} seconds")
        print(f"  Min interval: {np.min(all_intervals):.2f} seconds")
        print(f"  Max interval: {np.max(all_intervals):.2f} seconds")
        print()
        print(
            f"  Mean frequency: {np.mean(all_freqs):.4f} ± {np.std(all_freqs):.4f} Hz"
        )
        print(f"  Expected frequency (from FREQ_MAP): 0.1000 Hz (every 10 seconds)")
        print()

        # Check if it matches expectations
        mean_freq = np.mean(all_freqs)
        if abs(mean_freq - 0.1) < 0.01:
            print("✓ Observed frequency matches expected 0.1 Hz (every 10 seconds)")
        elif abs(mean_freq - 1 / 60) < 0.005:
            print("✓ Observed frequency matches ~60 seconds interval (~0.017 Hz)")
        else:
            print(f"! Observed frequency ({mean_freq:.4f} Hz) analysis:")
            print(f"  - Expected 0.1 Hz = every 10 seconds")
            print(f"  - Expected 0.017 Hz = every 60 seconds")
            print(f"  - Observed: every {1/mean_freq:.1f} seconds")

        print()

        # Interval histogram
        print("Interval Distribution (using message timestamps):")
        bins = [0, 5, 10, 15, 20, 30, 60, 120, np.inf]
        labels = [
            "0-5s",
            "5-10s",
            "10-15s",
            "15-20s",
            "20-30s",
            "30-60s",
            "60-120s",
            ">120s",
        ]

        hist, _ = np.histogram(all_intervals, bins=bins)
        for label, count in zip(labels, hist):
            if count > 0:
                pct = 100 * count / len(all_intervals)
                bar = "#" * int(pct / 2)
                print(f"  {label:>10s}: {count:4d} ({pct:5.1f}%) {bar}")

        print()
        print("Key Finding:")
        print(
            "  The 'File' column shows total recording length (first to last message, ~60s)."
        )
        print(
            "  The 'BatSpan' column shows time between first and last battery packet."
        )
        print("  The 'Device' column shows device uptime span (can be hours/days).")
        print(
            "  Battery intervals should be calculated using MESSAGE timestamps, not device time."
        )
    else:
        print("No battery intervals found in any files.")


if __name__ == "__main__":
    main()

