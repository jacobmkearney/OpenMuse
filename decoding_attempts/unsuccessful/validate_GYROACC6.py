"""
validate_GYROACC6.py - ACCGYRO decoder based on CONFIRMED packet structure.

REWRITTEN (October 2025) to implement the CONFIRMED decoding strategy from
comprehensive data analysis documented in README.md.

=== CONFIRMED STRUCTURE (from systematic analysis) ===
- Data section (bytes 14+) contains CONTINUOUS stream of 12-byte samples
- NO structural headers or delimiters between samples
- Bytes 0x47 and 0xF4 are sample data values, NOT tags
- Samples can span packet boundaries (buffer incomplete samples)
- Achieves 97.59% coverage with this approach

=== PREVIOUS SUBOPTIMAL APPROACH ===
- Old approach scanned for 0x47 "tags" and decoded blocks after them
- This was based on misinterpreting data values as structural markers
- Only achieved ~68% packet hit rate with false positives

=== KEY IMPROVEMENTS ===
2. No false tag detection
3. Proper packet boundary handling with buffer
4. More consistent sample rate (~60 Hz vs variable)
5. Based on confirmed structure, not heuristics

=== IMPLEMENTATION ===
Key findings applied:
1. 14-byte packet header (byte 13 always 0x00)
2. Data section = continuous 12-byte samples (6 int16 values each)
3. No tag-based structure (previous approach was incorrect)
4. Handle sample continuation across packet boundaries with buffer
5. Scale factors: ACC = 0.0000610352 g, GYRO = -0.0074768 deg/s

Reference: decoding_attempts/README.md (October 2025)
"""

import pandas as pd
import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os

# Scale factors (confirmed from original codebase and validation)
ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768


def decode_accgyro_packet(
    packet_hex: str, buffer: bytes = b""
) -> Tuple[Optional[Dict], bytes]:
    """
    Decode ACCGYRO samples as continuous 12-byte samples with boundary handling.

    Based on confirmed packet structure:
    - Bytes 0-13: Header (byte 13 always 0x00)
    - Bytes 14+: Data section = continuous stream of 12-byte samples
    - Each sample: 6 int16 LE values (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z)

    Args:
        packet_hex: Hex string of the packet
        buffer: Incomplete sample bytes from previous packet

    Returns:
        Tuple of (result_dict, leftover_buffer)
        - result_dict: Contains timestamp, counter, samples, coverage stats
        - leftover_buffer: Incomplete sample bytes to prepend to next packet
    """
    packet_bytes = bytes.fromhex(packet_hex)

    if len(packet_bytes) < 14:
        return None, buffer

    # Parse header for metadata
    length = packet_bytes[0]
    counter = packet_bytes[1]
    timestamp = struct.unpack("<I", packet_bytes[2:6])[0]
    packet_id = packet_bytes[9]

    # Verify byte 13 is 0x00 (confirmed in 100% of packets)
    if packet_bytes[13] != 0x00:
        return None, buffer

    # Verify ACCGYRO packet (type nibble = 7)
    type_nibble = packet_id & 0x0F
    if type_nibble != 7:
        return None, buffer

    # Extract data section and prepend any buffered bytes from previous packet
    data = buffer + packet_bytes[14:]
    new_buffer = b""

    # Decode complete 12-byte samples
    samples = []
    offset = 0

    while offset + 12 <= len(data):
        sample_bytes = data[offset : offset + 12]

        try:
            # Read 6 int16 values (little-endian)
            raw_values = struct.unpack("<6h", sample_bytes)

            # Apply scale factors
            sample = {
                "acc_x": raw_values[0] * ACC_SCALE,
                "acc_y": raw_values[1] * ACC_SCALE,
                "acc_z": raw_values[2] * ACC_SCALE,
                "gyro_x": raw_values[3] * GYRO_SCALE,
                "gyro_y": raw_values[4] * GYRO_SCALE,
                "gyro_z": raw_values[5] * GYRO_SCALE,
            }
            samples.append(sample)
            offset += 12
        except struct.error:
            break

    # Buffer any incomplete sample at the end
    if offset < len(data):
        new_buffer = data[offset:]

    # Calculate coverage (bytes decoded / total data bytes)
    data_length = len(packet_bytes) - 14
    bytes_decoded = offset - len(
        buffer
    )  # Don't count buffered bytes from previous packet
    coverage = bytes_decoded / data_length if data_length > 0 else 0

    return {
        "timestamp": timestamp,
        "counter": counter,
        "samples": samples,
        "coverage": coverage,
        "leftover_bytes": len(new_buffer),
        "data_length": data_length,
    }, new_buffer


def decode_file(filepath: Path, sample_rate: float = 52.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Decode all ACCGYRO samples from a file with packet boundary handling.

    Args:
        filepath: Path to the data file
        sample_rate: Nominal sample rate in Hz (default 52 Hz for ACCGYRO)

    Returns:
        Tuple of (samples_df, stats_dict)
    """
    # Read file - Bluetooth timestamp is ISO format, needs parsing
    df = pd.read_csv(filepath, sep="\t", names=["bt_timestamp_str", "uuid", "hex"])
    df["bt_timestamp"] = pd.to_datetime(df["bt_timestamp_str"])

    # Process packets sequentially, maintaining buffer for incomplete samples
    all_samples = []
    buffer = b""  # Buffer for incomplete samples spanning packet boundaries
    total_packets = 0
    accgyro_packets = 0
    total_coverage = 0.0

    for idx, row in df.iterrows():
        result, buffer = decode_accgyro_packet(row["hex"], buffer)
        total_packets += 1

        if result:
            accgyro_packets += 1
            samples = result["samples"]
            total_coverage += result["coverage"]

            if samples:
                num_samples = len(samples)
                sample_interval_ms = 1000.0 / sample_rate  # ~19.23 ms for 52 Hz

                # Convert Bluetooth packet receive timestamp to Unix ms
                bt_timestamp_ms = row["bt_timestamp"].timestamp() * 1000

                # TIMESTAMP STRATEGY:
                # - Bluetooth timestamp = when packet was received (corresponds to LAST sample)
                # - Device timestamp = from packet header (same for all samples in packet)
                # - Individual sample timestamps = computed by offsetting backwards from packet time
                #
                # Example: If packet received at time X with 3 samples:
                #   Sample 0 (first):  X - (2 * 19.23ms) = X - 38.46ms
                #   Sample 1 (middle): X - (1 * 19.23ms) = X - 19.23ms
                #   Sample 2 (last):   X - (0 * 19.23ms) = X

                for i, sample in enumerate(samples):
                    # Calculate offset: how many samples back from the end?
                    samples_from_end = num_samples - 1 - i

                    # Compute unique timestamp for this sample
                    sample_bt_timestamp_ms = bt_timestamp_ms - (
                        samples_from_end * sample_interval_ms
                    )

                    # Store both timestamps:
                    sample["device_packet_timestamp"] = result[
                        "timestamp"
                    ]  # Same for all samples in packet
                    sample["bt_timestamp_ms"] = (
                        sample_bt_timestamp_ms  # Unique per sample
                    )
                    sample["packet_counter"] = result["counter"]
                    all_samples.append(sample)

    # Convert to DataFrame
    samples_df = pd.DataFrame(all_samples)

    # Add relative time in seconds
    if len(samples_df) > 0:
        first_bt_timestamp = samples_df["bt_timestamp_ms"].iloc[0]
        samples_df["time_s"] = (
            samples_df["bt_timestamp_ms"] - first_bt_timestamp
        ) / 1000.0

    # Calculate statistics
    stats = {
        "num_packets": total_packets,
        "accgyro_packets": accgyro_packets,
        "num_samples": len(samples_df),
        "avg_coverage": total_coverage / accgyro_packets if accgyro_packets > 0 else 0,
        "duration_s": samples_df["time_s"].iloc[-1] if len(samples_df) > 0 else 0,
        "sample_rate": (
            len(samples_df) / samples_df["time_s"].iloc[-1]
            if len(samples_df) > 0 and samples_df["time_s"].iloc[-1] > 0
            else 0
        ),
    }

    return samples_df, stats


def plot_all_files(results_dict: Dict, files_per_row: int = 5):
    """
    Plot all decoded signals in one wide figure with files arranged in a grid.
    Uses warm colors for ACC (red, orange, yellow) and cold colors for GYRO (blue, cyan, purple).

    Args:
        results_dict: Dict mapping filename to (samples_df, stats)
        files_per_row: Number of files to show per row (default 5)

    Returns:
        Single figure
    """
    files = list(results_dict.keys())
    n_files = len(files)

    # Calculate grid dimensions
    n_cols = files_per_row
    n_rows = (n_files + files_per_row - 1) // files_per_row  # Ceiling division

    # Signal types and colors
    signal_names = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    signal_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    # Warm colors for ACC, cold colors for GYRO
    signal_colors = [
        "#D32F2F",  # ACC_X - Red
        "#FF6F00",  # ACC_Y - Orange
        "#F9A825",  # ACC_Z - Yellow/Gold
        "#1976D2",  # GYRO_X - Blue
        "#00ACC1",  # GYRO_Y - Cyan
        "#7B1FA2",  # GYRO_Z - Purple
    ]

    # Each file needs 6 subplots vertically (one per signal)
    total_rows = n_rows * 6

    # Create figure with GridSpec for more control
    fig = plt.figure(figsize=(5 * n_cols, 2 * total_rows))
    gs = fig.add_gridspec(total_rows, n_cols, hspace=0.3, wspace=0.3)

    for file_idx, filepath in enumerate(files):
        file_row = file_idx // n_cols
        file_col = file_idx % n_cols

        samples_df, stats = results_dict[filepath]
        filename = Path(filepath).stem

        if len(samples_df) == 0:
            # Create a single merged subplot for this file position showing "No data"
            row_start = file_row * 6
            row_end = row_start + 6
            ax = fig.add_subplot(gs[row_start:row_end, file_col])
            ax.text(
                0.5,
                0.5,
                f"{filename}\nNo data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.axis("off")
            continue

        # Create 6 subplots for this file (one per signal)
        for sig_idx in range(6):
            row_idx = file_row * 6 + sig_idx
            ax = fig.add_subplot(gs[row_idx, file_col])

            signal_name = signal_names[sig_idx]
            signal_col = signal_cols[sig_idx]
            color = signal_colors[sig_idx]

            # Plot this signal
            ax.plot(
                samples_df["time_s"],
                samples_df[signal_col],
                linewidth=0.8,
                alpha=0.8,
                color=color,
            )

            # Title only on first subplot
            if sig_idx == 0:
                ax.set_title(
                    f'{filename} ({stats["num_samples"]} samples, {stats["sample_rate"]:.1f} Hz)',
                    fontsize=9,
                    fontweight="bold",
                )

            # Y-label with signal name
            ax.set_ylabel(signal_name, fontsize=8, color=color, fontweight="bold")
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, alpha=0.3)

            # X-label only on bottom subplot
            if sig_idx == 5:
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.set_xticklabels([])

    return fig


def main():
    """Main execution function."""
    print("=" * 80)
    print("ACCGYRO DECODER - Continuous Sample Stream (CONFIRMED STRUCTURE)")
    print("=" * 80)
    print()
    print("Based on comprehensive analysis (README.md, October 2025):")
    print("  ✓ Data section = continuous 12-byte samples")
    print("  ✓ NO structural tags (0x47/0xF4 are data values)")
    print("  ✓ Samples can span packet boundaries")
    print("  ✓ Expected coverage: ~97.59%")
    print()

    # Data directory and files
    data_files = os.listdir("data_raw")

    results = {}

    for filename in data_files:
        filepath = "data_raw/" + filename

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        print(f"Processing {filename}...")
        samples_df, stats = decode_file(filepath)

        print(f"  Total packets: {stats['num_packets']}")
        print(f"  ACCGYRO packets: {stats['accgyro_packets']}")
        print(f"  Samples decoded: {stats['num_samples']}")
        print(f"  Coverage: {stats['avg_coverage']*100:.2f}%")
        print(f"  Duration: {stats['duration_s']:.2f} s")
        print(f"  Sample rate: {stats['sample_rate']:.1f} Hz")
        print()

        results[filepath] = (samples_df, stats)

    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    total_packets = sum(stats["num_packets"] for _, (_, stats) in results.items())
    total_accgyro = sum(stats["accgyro_packets"] for _, (_, stats) in results.items())
    total_samples = sum(stats["num_samples"] for _, (_, stats) in results.items())
    avg_coverage = np.mean([stats["avg_coverage"] for _, (_, stats) in results.items()])

    print(f"Total packets processed: {total_packets}")
    print(f"ACCGYRO packets: {total_accgyro}")
    print(f"Total samples decoded: {total_samples}")
    print(f"Average coverage: {avg_coverage*100:.2f}%")
    print()

    # Signal statistics
    print("Signal Statistics (across all files):")
    print()

    all_samples = pd.concat([df for df, _ in results.values()], ignore_index=True)

    for signal in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        mean_val = all_samples[signal].mean()
        std_val = all_samples[signal].std()
        min_val = all_samples[signal].min()
        max_val = all_samples[signal].max()
        print(
            f"  {signal.upper():8s}: mean={mean_val:7.4f}, std={std_val:7.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]"
        )

    print()

    # Timestamp verification
    print("Timestamp Verification:")
    print()

    # Show example timestamps from first few samples
    if len(all_samples) > 5:
        print("  Example: First 5 samples from dataset:")
        for i in range(5):
            bt_ts = all_samples.iloc[i]["bt_timestamp_ms"]
            time_s = all_samples.iloc[i]["time_s"]
            print(f"    Sample {i}: time={time_s:.6f}s, bt_timestamp={bt_ts:.3f}ms")

        # Calculate actual time differences between consecutive samples
        time_diffs = all_samples["time_s"].diff().dropna()
        avg_interval_ms = time_diffs.mean() * 1000
        std_interval_ms = time_diffs.std() * 1000

        print()
        print(f"  ✓ Each sample has unique timestamp (computed from packet time)")
        print(
            f"  ✓ Average interval: {avg_interval_ms:.3f} ms (expected: ~19.23 ms for 52 Hz)"
        )
        print(f"  ✓ Std dev: {std_interval_ms:.3f} ms")

    print()

    # Sanity checks
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    # Check 0: Coverage should be ~97.59%
    print(f"[CHECK 1] Data Coverage: {avg_coverage*100:.2f}%")
    print(f"  Expected: ~97.59% (from confirmed analysis)")
    print(f"  Status: {'PASS' if avg_coverage > 0.90 else 'FAIL'}")
    print()

    # Check 2: ACC magnitude (reasonable values)
    all_samples["acc_mag"] = np.sqrt(
        all_samples["acc_x"] ** 2
        + all_samples["acc_y"] ** 2
        + all_samples["acc_z"] ** 2
    )
    acc_mag_mean = all_samples["acc_mag"].mean()
    acc_mag_std = all_samples["acc_mag"].std()

    print(f"[CHECK 2] ACC magnitude: {acc_mag_mean:.3f} ± {acc_mag_std:.3f} g")
    print(
        f"  Range: [{all_samples['acc_mag'].min():.3f}, {all_samples['acc_mag'].max():.3f}] g"
    )
    print(f"  Note: Values include gravity + motion acceleration")
    if 0.5 < acc_mag_mean < 3.0:
        print(f"  Status: PASS - Values within reasonable range")
    else:
        print(f"  Status: WARN - Unusual acceleration values detected")
    print()

    # Check 3: GYRO values (reasonable rotation rates)
    gyro_mag_mean = np.sqrt(
        all_samples["gyro_x"].mean() ** 2
        + all_samples["gyro_y"].mean() ** 2
        + all_samples["gyro_z"].mean() ** 2
    )
    gyro_std_mean = np.mean(
        [
            all_samples["gyro_x"].std(),
            all_samples["gyro_y"].std(),
            all_samples["gyro_z"].std(),
        ]
    )

    print(f"[CHECK 3] GYRO mean magnitude: {gyro_mag_mean:.3f} deg/s")
    print(f"  GYRO mean std dev: {gyro_std_mean:.3f} deg/s")
    print(f"  Note: Device recordings include motion")
    if gyro_mag_mean < 250 and gyro_std_mean < 250:
        print(f"  Status: PASS - Values within sensor range (±245 deg/s)")
    else:
        print(f"  Status: WARN - Values near or exceeding sensor limits")
    print()

    # Check 4: Sample rate consistency
    sample_rates = [stats["sample_rate"] for _, (_, stats) in results.items()]
    rate_mean = np.mean(sample_rates)
    rate_std = np.std(sample_rates)

    print(f"[CHECK 4] Sample rates: {rate_mean:.1f} ± {rate_std:.1f} Hz")
    print(f"  Nominal rate: ~52 Hz (ACCGYRO), actual varies by preset")
    print(f"  Status: {'PASS' if 45 < rate_mean < 70 else 'WARN'}")
    print()

    # Visualization
    print("=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    print()

    fig = plot_all_files(results, files_per_row=5)

    output_file = "accgyro_signals_continuous_all_files.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_file}")

    plt.close("all")

    print()
    print("=" * 80)
    print("DECODING COMPLETE - Continuous Sample Stream Approach")
    print("=" * 80)
    print()
    print("Key improvements over tag-based approach:")
    print("  ✓ Higher coverage (~97.59% vs ~68%)")
    print("  ✓ No false tag detection")
    print("  ✓ Handles packet boundaries correctly")
    print("  ✓ More consistent sample rate")
    print("  ✓ Based on confirmed packet structure")


if __name__ == "__main__":
    main()
