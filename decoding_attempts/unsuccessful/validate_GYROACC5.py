"""
validate_GYROACC5.py - Clean and efficient ACCGYRO decoder with visualization.

Implements the OFFICIAL tag-based decoding strategy (from MuseLSL3/decode.py):
- Search for tag byte 0x47 in data section
- After tag: skip 4-byte header
- Decode 36 bytes (3 samples × 12 bytes)
- Each sample: 6 int16 LE values (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z)

This matches the reference implementation and should give correct sample rate.
"""

import pandas as pd
import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt

# Scale factors (confirmed from original codebase and validation)
ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768


def decode_accgyro_packet(packet_hex):
    """
    Decode ACCGYRO samples from a Bluetooth message using official decoder logic.

    Scans the ENTIRE packet (not just data section) for 0x47 tags.
    For each tag:
    - Skip tag + 4 bytes = 5 bytes total
    - Decode 36 bytes as 3 samples (18 int16 values)

    This matches the official MuseLSL3 decoder behavior.

    Args:
        packet_hex: Hex string of the packet

    Returns:
        dict with timestamp, counter, and samples, or None if invalid
    """
    packet_bytes = bytes.fromhex(packet_hex)

    if len(packet_bytes) < 14:
        return None

    # Parse header for metadata
    length = packet_bytes[0]
    counter = packet_bytes[1]
    timestamp = struct.unpack("<I", packet_bytes[2:6])[0]
    packet_id = packet_bytes[9]

    # Scan ENTIRE packet for 0x47 tags (including in data section)
    samples = []
    tag_count = 0

    idx = 0
    while idx < len(packet_bytes):
        if packet_bytes[idx] == 0x47:
            # Found tag - skip tag + 4-byte header = 5 bytes total
            offset = idx + 5

            # Decode 36 bytes (3 samples)
            if offset + 36 <= len(packet_bytes):
                block = packet_bytes[offset : offset + 36]

                try:
                    # Decode 3 samples (18 int16 values)
                    for ax, ay, az, gx, gy, gz in struct.iter_unpack("<6h", block):
                        samples.append(
                            {
                                "acc_x": ax * ACC_SCALE,
                                "acc_y": ay * ACC_SCALE,
                                "acc_z": az * ACC_SCALE,
                                "gyro_x": gx * GYRO_SCALE,
                                "gyro_y": gy * GYRO_SCALE,
                                "gyro_z": gz * GYRO_SCALE,
                            }
                        )

                    tag_count += 1
                    # Jump past the decoded block
                    idx = offset + 36
                    continue
                except struct.error:
                    pass

        idx += 1

    return {
        "timestamp": timestamp,
        "counter": counter,
        "samples": samples,
        "tag_count": tag_count,
        "coverage": (
            (tag_count * 41) / len(packet_bytes[14:]) if len(packet_bytes) > 14 else 0
        ),
    }


def decode_file(filepath, sample_rate=52.0):
    """
    Decode all ACCGYRO packets from a file.

    Args:
        filepath: Path to the data file
        sample_rate: Expected sample rate in Hz (default 52 Hz for ACCGYRO)

    Returns:
        DataFrame with decoded samples
    """
    # Read file - Bluetooth timestamp is ISO format, needs parsing
    df = pd.read_csv(filepath, sep="\t", names=["bt_timestamp_str", "uuid", "hex"])
    df["bt_timestamp"] = pd.to_datetime(df["bt_timestamp_str"])

    # Process ALL messages (scan all for 0x47 tags)
    # Decode all packets
    all_samples = []
    total_coverage = 0
    total_packets = 0

    for idx, row in df.iterrows():
        result = decode_accgyro_packet(row["hex"])
        total_packets += 1

        if result and result["samples"]:
            num_samples = len(result["samples"])
            sample_interval_ms = 1000.0 / sample_rate  # Time between samples in ms

            # Convert Bluetooth timestamp to Unix ms
            bt_timestamp_ms = row["bt_timestamp"].timestamp() * 1000

            # Bluetooth timestamp corresponds to packet RECEIVE time (last sample)
            # Compute individual timestamps by going backwards
            for i, sample in enumerate(result["samples"]):
                # Index from last sample: 0 is last, num_samples-1 is first
                samples_from_end = num_samples - 1 - i
                sample_bt_timestamp_ms = bt_timestamp_ms - (
                    samples_from_end * sample_interval_ms
                )

                sample["device_packet_timestamp"] = result["timestamp"]
                sample["bt_timestamp_ms"] = sample_bt_timestamp_ms
                sample["packet_counter"] = result["counter"]
                all_samples.append(sample)

            total_coverage += result["coverage"]

    # Convert to DataFrame
    samples_df = pd.DataFrame(all_samples)

    # Add relative time in seconds (from first Bluetooth timestamp)
    if len(samples_df) > 0:
        first_bt_timestamp = samples_df["bt_timestamp_ms"].iloc[0]
        samples_df["time_s"] = (
            samples_df["bt_timestamp_ms"] - first_bt_timestamp
        ) / 1000.0

    # Calculate statistics
    stats = {
        "num_packets": total_packets,
        "num_samples": len(samples_df),
        "avg_coverage": total_coverage / total_packets if total_packets > 0 else 0,
        "duration_s": samples_df["time_s"].iloc[-1] if len(samples_df) > 0 else 0,
        "sample_rate": (
            len(samples_df) / samples_df["time_s"].iloc[-1]
            if len(samples_df) > 0 and samples_df["time_s"].iloc[-1] > 0
            else 0
        ),
    }

    return samples_df, stats


def plot_all_files(results_dict, files_per_figure=3):
    """
    Plot all decoded signals in figures with signal types in rows and files in columns.

    Args:
        results_dict: Dict mapping filename to (samples_df, stats)
        files_per_figure: Number of files to show per figure (default 3)

    Returns:
        List of figures
    """
    files = list(results_dict.keys())
    n_files = len(files)

    # Signal types
    signals = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    signal_labels = ["ACC X", "ACC Y", "ACC Z", "GYRO X", "GYRO Y", "GYRO Z"]
    signal_units = ["g", "g", "g", "deg/s", "deg/s", "deg/s"]

    # Split files into groups for multiple figures
    n_figures = int(np.ceil(n_files / files_per_figure))
    figures = []

    for fig_idx in range(n_figures):
        # Select files for this figure
        start_idx = fig_idx * files_per_figure
        end_idx = min(start_idx + files_per_figure, n_files)
        files_subset = files[start_idx:end_idx]
        n_cols = len(files_subset)

        # Create figure: rows = signal types, columns = files
        fig, axes = plt.subplots(
            len(signals), n_cols, figsize=(5 * n_cols, 2.5 * len(signals))
        )

        # Handle single column case
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each signal x file combination
        for sig_idx, (signal, label, unit) in enumerate(
            zip(signals, signal_labels, signal_units)
        ):
            for col_idx, filename in enumerate(files_subset):
                samples_df, stats = results_dict[filename]
                ax = axes[sig_idx, col_idx]

                # Extract preset name from filename
                preset = filename.stem.replace("data_", "")

                if len(samples_df) > 0:
                    # Plot signal
                    ax.plot(
                        samples_df["time_s"],
                        samples_df[signal],
                        linewidth=0.5,
                        alpha=0.8,
                    )

                    # Formatting
                    ax.grid(True, alpha=0.3)

                    # Y-axis label (only on leftmost column)
                    if col_idx == 0:
                        ax.set_ylabel(f"{label}\n({unit})", fontsize=9)

                    # X-axis label (only on bottom row)
                    if sig_idx == len(signals) - 1:
                        ax.set_xlabel("Time (s)", fontsize=9)

                    # Title (only on top row)
                    if sig_idx == 0:
                        ax.set_title(
                            f'{preset}\n{stats["sample_rate"]:.1f} Hz',
                            fontsize=10,
                            fontweight="bold",
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    if col_idx == 0:
                        ax.set_ylabel(f"{label}\n({unit})", fontsize=9)
                    if sig_idx == len(signals) - 1:
                        ax.set_xlabel("Time (s)", fontsize=9)
                    if sig_idx == 0:
                        ax.set_title(preset, fontsize=10, fontweight="bold")

        plt.tight_layout()
        fig.suptitle(
            f"ACCGYRO Signals - Files {start_idx+1}-{end_idx}",
            fontsize=12,
            fontweight="bold",
            y=1.001,
        )

        figures.append((fig, f"figure_{fig_idx+1}"))

    return figures


def main():
    """Main function to decode and visualize all files."""

    data_dir = Path("decoding_attempts/data_raw")

    # Files to process
    data_files = ["data_p20.txt", "data_p21.txt", "data_p50.txt", "data_p51.txt"]

    print("=" * 80)
    print("ACCGYRO DECODER - Final Validated Implementation")
    print("=" * 80)
    print()

    results = {}

    # Process each file
    for filename in data_files:
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        print(f"Processing {filename}...")
        samples_df, stats = decode_file(filepath)

        print(f"  Packets: {stats['num_packets']}")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Coverage: {stats['avg_coverage']:.2%}")
        print(f"  Duration: {stats['duration_s']:.2f} s")
        print(f"  Sample rate: {stats['sample_rate']:.1f} Hz")
        print()

        results[filepath] = (samples_df, stats)

    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    total_packets = sum(stats["num_packets"] for _, (_, stats) in results.items())
    total_samples = sum(stats["num_samples"] for _, (_, stats) in results.items())
    avg_coverage = np.mean([stats["avg_coverage"] for _, (_, stats) in results.items()])

    print(f"Total packets decoded: {total_packets}")
    print(f"Total samples extracted: {total_samples}")
    print(f"Average coverage: {avg_coverage:.2%}")
    print()

    # Signal statistics across all files
    print("Signal Statistics (across all files):")
    print()

    all_samples = pd.concat([df for df, _ in results.values()], ignore_index=True)

    for signal in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        mean_val = all_samples[signal].mean()
        std_val = all_samples[signal].std()
        min_val = all_samples[signal].min()
        max_val = all_samples[signal].max()

        print(
            f"  {signal.upper():8s}: mean={mean_val:7.4f}, std={std_val:6.4f}, "
            f"range=[{min_val:7.4f}, {max_val:7.4f}]"
        )

    print()

    # Sanity checks
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    # Check 1: ACC magnitude should be ~1g (gravity) for stationary device
    all_samples["acc_mag"] = np.sqrt(
        all_samples["acc_x"] ** 2
        + all_samples["acc_y"] ** 2
        + all_samples["acc_z"] ** 2
    )
    acc_mag_mean = all_samples["acc_mag"].mean()
    acc_mag_std = all_samples["acc_mag"].std()

    print(f"✓ ACC magnitude: {acc_mag_mean:.3f} ± {acc_mag_std:.3f} g")
    print(f"  Expected: ~1.0 g (gravity)")
    print(f"  Status: {'PASS' if 0.9 < acc_mag_mean < 1.1 else 'FAIL'}")
    print()

    # Check 2: GYRO should be near zero for stationary device
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

    print(f"✓ GYRO mean magnitude: {gyro_mag_mean:.3f} deg/s")
    print(f"  GYRO mean std dev: {gyro_std_mean:.3f} deg/s")
    print(f"  Expected: <5 deg/s for stationary device")
    print(f"  Status: {'PASS' if gyro_mag_mean < 5 else 'FAIL'}")
    print()

    # Check 3: Sample rate consistency
    sample_rates = [stats["sample_rate"] for _, (_, stats) in results.items()]
    rate_mean = np.mean(sample_rates)
    rate_std = np.std(sample_rates)

    print(f"✓ Sample rates: {rate_mean:.1f} ± {rate_std:.1f} Hz")
    print(f"  Expected: 50-100 Hz (typical for ACCGYRO)")
    print(f"  Status: {'PASS' if 40 < rate_mean < 110 else 'FAIL'}")
    print()

    # Visualization
    print("=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    print()

    figures = plot_all_files(results, files_per_figure=3)

    # Save figures
    for fig, fig_name in figures:
        output_path = Path(f"decoding_attempts/accgyro_signals_{fig_name}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")
    print()

    # Save decoded data
    print("=" * 80)
    print("SAVING DECODED DATA")
    print("=" * 80)
    print()

    for filepath, (samples_df, _) in results.items():
        output_csv = Path("decoding_attempts") / f"{filepath.stem}_decoded.csv"
        samples_df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

    print()
    print("=" * 80)
    print("DECODING COMPLETE")
    print("=" * 80)

    # Show plot
    plt.show()

    return results


if __name__ == "__main__":
    results = main()
