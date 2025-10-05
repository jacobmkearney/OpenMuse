"""
validate_ACCGYRO.py
===================

Compare two ACCGYRO decoding approaches:
1. Current approach (decode.py): Searches for 0x47 tag in raw payload
2. New approach: Uses parse_message() to split into packets first, then searches for 0x47 in pkt_data

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
from MuseLSL3.decode_new import parse_message

# Scaling factors from decode.py
ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768


def decode_accgyro_new_method(messages):
    """
    New approach: Use parse_message() to split into packets first,
    then search for 0x47 tag in the pkt_data of all packets.

    Parameters
    ----------
    messages : list of str
        Raw BLE messages

    Returns
    -------
    dict
        Dictionary with 'ACC' and 'GYRO' keys, each containing list of samples
    """
    results = {"ACC": [], "GYRO": []}

    for message in messages:
        # Parse message into packets
        packets = parse_message(message)

        if not packets:
            continue

        # Get message timestamp
        msg_time = packets[0]["message_time"].timestamp()

        # Search for ACCGYRO data in each packet's data section
        # Note: ACCGYRO data (0x47 tags) can appear in ANY packet type!
        # Analysis shows: 52% in Optics16, 36% in EEG8, only 11% in ACCGYRO packets
        for pkt in packets:
            if not pkt["pkt_valid"]:
                continue

            # # Only process Optics and ACCGYRO packets
            # if pkt["pkt_type"] not in [
            #     "Optics4",
            #     "Optics8",
            #     "Optics16",
            #     "ACCGYRO",
            #     "EEG8",
            #     "EEG16",
            # ]:
            #     continue

            # Get the raw data bytes
            pkt_data = pkt["pkt_data"]

            # Handle Battery packets which have decoded data as tuple
            if isinstance(pkt_data, tuple):
                continue

            # Search for 0x47 tag in this packet's data
            # Clean, logical approach: find tags, decode if enough data, skip past decoded blocks
            idx = 0
            while idx < len(pkt_data):
                if pkt_data[idx] == 0x47:
                    # Found ACCGYRO tag
                    # Skip tag byte + 4 byte header = 5 bytes total
                    offset = idx + 5
                    bytes_needed = 36  # 18 int16 values (6 channels x 3 samples)

                    # Check if we have enough data to decode
                    if offset + bytes_needed <= len(pkt_data):
                        # Decode the block
                        block = pkt_data[offset : offset + bytes_needed]

                        # Decode the 18 int16 values
                        # Each sample has 6 channels: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
                        for ax, ay, az, gx, gy, gz in struct.iter_unpack("<6h", block):
                            results["ACC"].append(
                                {
                                    "time": msg_time,
                                    "ACC_X": ax * ACC_SCALE,
                                    "ACC_Y": ay * ACC_SCALE,
                                    "ACC_Z": az * ACC_SCALE,
                                }
                            )
                            results["GYRO"].append(
                                {
                                    "time": msg_time,
                                    "GYRO_X": gx * GYRO_SCALE,
                                    "GYRO_Y": gy * GYRO_SCALE,
                                    "GYRO_Z": gz * GYRO_SCALE,
                                }
                            )

                        # Jump past the entire decoded block (tag + header + data)
                        idx = offset + bytes_needed
                    else:
                        # Not enough data for a full block, skip this tag and continue
                        idx += 1
                else:
                    # Not a tag, move to next byte
                    idx += 1

    return results


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
    new_data = decode_accgyro_new_method(messages)

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
