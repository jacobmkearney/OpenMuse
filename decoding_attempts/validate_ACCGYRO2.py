"""
validate_ACCGYRO.py
===================

Compare two ACCGYRO decoding approaches.

This script processes test_accgyro.txt (90 seconds of recording with specific movements)
and creates a side-by-side comparison plot showing the 6 channels (ACC_X, ACC_Y, ACC_Z,
GYRO_X, GYRO_Y, GYRO_Z) for each approach.
"""

import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import MuseLSL3
sys.path.insert(0, str(Path(__file__).parent.parent))

from MuseLSL3.decode import decode_rawdata


def decode_rawdata2(messages):
    """
    Decode ACC + GYRO from Muse-style messages, skipping all other packet types.

    Each message must be a string: "ISO_TIMESTAMP \\t UUID \\t HEXSTRING".
    This version ensures proper alignment:
      - Scans byte-by-byte until a recognised tag is found
      - Skips over the correct block size for known tags
      - Stops gracefully for incomplete packets

    Returns
    -------
    dict of pandas.DataFrame
        Keys: 'ACC' and 'GYRO'
    """
    import struct
    from datetime import datetime
    import pandas as pd

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    ACC_SCALE = 0.0000610352
    GYRO_SCALE = -0.0074768
    HEADER_LEN = 4

    # Tag → payload length (excluding tag + header)
    PAYLOAD_MAP = {
        0x11: 14,  # EEG4 (4 channels)
        0x12: 28,  # EEG8 (8 channels)
        0x13: 7,  # REF (2 channels)
        0x34: 24,  # Optics4 (4 channels × 3 samples)
        0x35: 48,  # Optics8 (8 channels × 3 samples)
        0x36: 96,  # Optics16 (16 channels × 3 samples)
        0x47: 36,  # ACCGYRO (6 channels × 3 samples)
        0x98: 4,  # Battery
    }

    # ------------------------------------------------------------------
    # Helper: parse ACCGYRO blocks
    # ------------------------------------------------------------------
    def parse_accgyro_block(block: bytes, ts: float):
        """Parse one 36-byte ACC/GYRO block into scaled dicts."""
        acc_rows, gyro_rows = [], []
        try:
            for ax, ay, az, gx, gy, gz in struct.iter_unpack("<6h", block):
                acc_rows.append(
                    {
                        "time": ts,
                        "ACC_X": ax * ACC_SCALE,
                        "ACC_Y": ay * ACC_SCALE,
                        "ACC_Z": az * ACC_SCALE,
                    }
                )
                gyro_rows.append(
                    {
                        "time": ts,
                        "GYRO_X": gx * GYRO_SCALE,
                        "GYRO_Y": gy * GYRO_SCALE,
                        "GYRO_Z": gz * GYRO_SCALE,
                    }
                )
        except struct.error:
            pass
        return acc_rows, gyro_rows

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    acc_rows, gyro_rows = [], []

    for msg in messages:
        parts = msg.split("\t", 2)
        if len(parts) != 3:
            continue

        ts_str, _uuid, hexstr = parts

        # Parse timestamp
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            ts = dt.timestamp()
        except Exception:
            continue

        # Decode hex payload
        try:
            packet = memoryview(bytes.fromhex(hexstr))
        except Exception:
            continue

        i = 0
        n = len(packet)
        while i < n:
            tag = packet[i]

            # Skip unrecognised or filler bytes
            if tag not in PAYLOAD_MAP:
                i += 1
                continue

            payload_len = PAYLOAD_MAP[tag]
            data_start = i + 1 + HEADER_LEN
            data_end = data_start + payload_len

            # Check bounds
            if data_end > n:
                # Incomplete block -> stop scanning this message
                break

            # Handle ACCGYRO
            if tag == 0x47:
                block = bytes(packet[data_start:data_end])
                acc_blk, gyro_blk = parse_accgyro_block(block, ts)
                acc_rows.extend(acc_blk)
                gyro_rows.extend(gyro_blk)

            # Advance index past this full packet
            i = data_end

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out = {}
    if acc_rows:
        out["ACC"] = pd.DataFrame(acc_rows).sort_values("time").reset_index(drop=True)
    if gyro_rows:
        out["GYRO"] = pd.DataFrame(gyro_rows).sort_values("time").reset_index(drop=True)
    return out


def extract_channels_and_stats(data_dict):
    """
    Extract time series and compute sampling statistics.

    Parameters
    ----------
    data_dict : dict
        Dictionary with 'ACC' and 'GYRO' keys

    Returns
    -------
    tuple
        (times, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mean_dt, std_dt)
    """
    if not data_dict["ACC"] or not data_dict["GYRO"]:
        return None, None, None, None, None, None, None, None, None

    # Extract ACC data
    acc_times = np.array([s["time"] for s in data_dict["ACC"]])
    acc_x = np.array([s["ACC_X"] for s in data_dict["ACC"]])
    acc_y = np.array([s["ACC_Y"] for s in data_dict["ACC"]])
    acc_z = np.array([s["ACC_Z"] for s in data_dict["ACC"]])

    # Extract GYRO data
    gyro_times = np.array([s["time"] for s in data_dict["GYRO"]])
    gyro_x = np.array([s["GYRO_X"] for s in data_dict["GYRO"]])
    gyro_y = np.array([s["GYRO_Y"] for s in data_dict["GYRO"]])
    gyro_z = np.array([s["GYRO_Z"] for s in data_dict["GYRO"]])

    # Normalize time to start at 0
    t0 = acc_times[0]
    acc_times = acc_times - t0
    gyro_times = gyro_times - t0

    # Use ACC times as reference (should be same as GYRO)
    times = acc_times

    # Compute time deltas
    if len(times) > 1:
        dts = np.diff(times)
        mean_dt = np.mean(dts)
        std_dt = np.std(dts)
        effective_fs = 1.0 / mean_dt if mean_dt > 0 else 0
    else:
        mean_dt = 0
        std_dt = 0
        effective_fs = 0

    return times, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mean_dt, std_dt


def main():
    """Main function to compare ACCGYRO decoding approaches."""

    # Load test data
    test_file = (
        Path(__file__).parent.parent / "tests" / "test_data" / "test_accgyro.txt"
    )

    print(f"Loading test data from: {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        messages = f.readlines()

    print(f"Loaded {len(messages)} messages")

    # Decode using current method (decode.py)
    print("\n=== Current Method (decode_rawdata) ===")
    current_data = decode_rawdata(messages)

    # Convert pandas DataFrames to dict format for consistency
    current_dict = {
        "ACC": current_data["ACC"].to_dict("records") if "ACC" in current_data else [],
        "GYRO": (
            current_data["GYRO"].to_dict("records") if "GYRO" in current_data else []
        ),
    }

    print(f"Decoded {len(current_dict['ACC'])} ACC samples")
    print(f"Decoded {len(current_dict['GYRO'])} GYRO samples")

    # Decode using new method (parse_message first)
    print("\n=== New Method (parse_message + tag search) ===")
    new_data = decode_rawdata2(messages)

    # Convert pandas DataFrames to dict format for consistency
    new_data = {
        "ACC": new_data["ACC"].to_dict("records") if "ACC" in new_data else [],
        "GYRO": (new_data["GYRO"].to_dict("records") if "GYRO" in new_data else []),
    }

    print(f"Decoded {len(new_data['ACC'])} ACC samples")
    print(f"Decoded {len(new_data['GYRO'])} GYRO samples")

    # Extract channels and compute statistics
    (
        times_curr,
        acc_x_curr,
        acc_y_curr,
        acc_z_curr,
        gyro_x_curr,
        gyro_y_curr,
        gyro_z_curr,
        mean_dt_curr,
        std_dt_curr,
    ) = extract_channels_and_stats(current_dict)

    (
        times_new,
        acc_x_new,
        acc_y_new,
        acc_z_new,
        gyro_x_new,
        gyro_y_new,
        gyro_z_new,
        mean_dt_new,
        std_dt_new,
    ) = extract_channels_and_stats(new_data)

    if times_curr is None or times_new is None:
        print("ERROR: Failed to decode data")
        return

    # Compute effective sampling rates
    fs_curr = 1.0 / mean_dt_curr if mean_dt_curr > 0 else 0
    fs_new = 1.0 / mean_dt_new if mean_dt_new > 0 else 0

    print(
        f"\nCurrent method: Fs = {fs_curr:.2f} Hz (mean_dt = {mean_dt_curr*1000:.2f} ± {std_dt_curr*1000:.2f} ms)"
    )
    print(
        f"New method:     Fs = {fs_new:.2f} Hz (mean_dt = {mean_dt_new*1000:.2f} ± {std_dt_new*1000:.2f} ms)"
    )

    # Create comparison plot
    fig, axes = plt.subplots(6, 2, figsize=(14, 12))
    fig.suptitle("ACCGYRO Decoding Comparison", fontsize=14, fontweight="bold")

    channel_names = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    channel_data_curr = [
        acc_x_curr,
        acc_y_curr,
        acc_z_curr,
        gyro_x_curr,
        gyro_y_curr,
        gyro_z_curr,
    ]
    channel_data_new = [
        acc_x_new,
        acc_y_new,
        acc_z_new,
        gyro_x_new,
        gyro_y_new,
        gyro_z_new,
    ]

    # Left column: Current method
    axes[0, 0].set_title(
        f"Current Method (decode_rawdata)\nFs = {fs_curr:.2f} Hz (dt = {mean_dt_curr*1000:.2f} ± {std_dt_curr*1000:.2f} ms)",
        fontsize=10,
    )

    for i, (name, data) in enumerate(zip(channel_names, channel_data_curr)):
        ax = axes[i, 0]
        ax.plot(times_curr, data, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=9)

    # Right column: New method
    axes[0, 1].set_title(
        f"New Method (parse_message + tag search)\nFs = {fs_new:.2f} Hz (dt = {mean_dt_new*1000:.2f} ± {std_dt_new*1000:.2f} ms)",
        fontsize=10,
    )

    for i, (name, data) in enumerate(zip(channel_names, channel_data_new)):
        ax = axes[i, 1]
        ax.plot(times_new, data, linewidth=0.5, alpha=0.8, color="C1")
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent / "accgyro_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
