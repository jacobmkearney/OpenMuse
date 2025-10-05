"""
Muse BLE Message Parser
=======================

Efficiently parses BLE (Bluetooth Low Energy) messages from Muse devices into
structured subpackets for downstream signal processing.

Overview:
---------
Each BLE message contains a hex-encoded payload that may contain multiple concatenated
subpackets. Each subpacket has a 14-byte header followed by variable-length sensor data.

Based on validation in `validate_subpkts.py`:
- 100% validation success across 15 recording presets
- Zero leftover bytes - complete payload coverage
- All packet boundaries correctly identified

Payload Structure:
------------------
Byte Offset | Field Name        | Type      | Description
------------|-------------------|-----------|------------------------------------------
0           | PKT_LEN        | uint8     | Declared packet length (includes header)
1           | PKT_N          | uint8     | Packet counter (0-255, wraps around)
2-5         | PKT_T          | uint32 LE | Timestamp in milliseconds (device time)
6-8         | PKT_UNKNOWN1   | 3 bytes   | Reserved/unknown (consistent per preset)
9           | PKT_ID         | uint8     | Frequency (high nibble) + Type (low nibble)
10-11       | metadata_u16_0    | uint16 LE | High-variance counter (NOT sample count)
12          | metadata_u16_1_lo | uint8     | Low byte, values 0-3
13          | ALWAYS_ZERO       | uint8     | Separator/padding (confirmed 100% = 0x00)
14+         | DATA              | variable  | Sample data (length = PKT_LEN - 14)

Supported Sensor Types:
-----------------------
Code | Type      | Frequencies Observed
-----|-----------|---------------------
1    | EEG4      | 256 Hz
2    | EEG8      | 128 Hz
3    | REF       | 256 Hz
4    | Optics4   | 64 Hz
5    | Optics8   | 64 Hz
6    | Optics16  | 64 Hz
7    | ACCGYRO   | 52 Hz
8    | Battery   | 1 Hz

Parsing Algorithm:
------------------
1. Read declared length from byte 0
2. Extract full packet using the length
3. Decode 14-byte header
4. Extract sensor data (bytes 14+)
5. Validate packet structure
6. Move offset forward and repeat

Performance:
------------
- O(n) complexity - single pass through payload
- Memory efficient
- Handles variable-length packets
- Tested on 20,088+ subpackets across 15 presets

Usage Example:
--------------
    from validate_parse_message import parse_message

    message = "2025-09-25T08:10:06.642793Z\\t273e0013...\\td7..."
    subpackets = parse_message(message)

    for subpkt in subpackets:
        if subpkt['pkt_valid']:
            sensor_type = subpkt['PKT_Type']
            raw_data = subpkt['PKT_DATA']
            # Process sensor-specific data...
"""

import os
import struct
from typing import List, Dict, Optional

import pandas as pd
from datetime import datetime

# Lookup tables for PKT_ID decoding
FREQ_MAP = {
    1: 256.0,
    2: 128.0,
    3: 64.0,
    4: 52.0,
    5: 32.0,
    6: 16.0,
    7: 10.0,
    8: 1.0,
    9: 0.1,
}

TYPE_MAP = {
    1: "EEG4",
    2: "EEG8",
    3: "REF",
    4: "Optics4",
    5: "Optics8",
    6: "Optics16",
    7: "ACCGYRO",
    8: "Battery",
}


def parse_message(message: str) -> List[Dict]:
    """
    Parse a BLE message into a list of subpackets.

    Parameters:
    -----------
    message : str
        Tab-separated string containing:
        - ISO timestamp (BLE message arrival time)
        - UUID (BLE characteristic)
        - Hex-encoded payload

    Returns:
    --------
    List[Dict]
        List of dictionaries, one per subpacket, containing:
        - message_time: Original BLE message timestamp (datetime)
        - message_uuid: BLE characteristic UUID (str)
        - pkt_offset: Byte offset of this packet in the payload (int)
        - PKT_LEN: Declared packet length from byte 0 (int)
        - PKT_N: Packet counter 0-255 from byte 1 (int)
        - PKT_Time: Device timestamp in seconds (float)
        - PKT_UNKNOWN1: Reserved bytes 6-8 (bytes)
        - pkt_freq: Sampling frequency in Hz (float or None)
        - PKT_Type: Sensor type (str or None)
        - pkt_unknown2: Metadata bytes 10-12 (bytes)
        - PKT_DATA: Raw sensor data bytes (bytes)
        - pkt_valid: True if packet passes validation checks (bool)
    """
    # Parse the message line
    ts, uuid, hexstring = message.strip().split("\t", 2)
    message_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    payload = bytes.fromhex(hexstring.strip())

    subpackets = []
    offset = 0
    total_bytes = len(payload)

    # Iterate through payload, extracting subpackets
    while offset < total_bytes:
        # Need at least 14 bytes for header
        if offset + 14 > total_bytes:
            break

        # Read declared length
        declared_len = payload[offset]

        # Check if we have enough bytes for the full packet
        if offset + declared_len > total_bytes:
            break

        # Extract the full packet
        pkt = payload[offset : offset + declared_len]

        # Parse header fields
        PKT_LEN = pkt[0]
        PKT_N = pkt[1]
        PKT_Time_ms = struct.unpack_from("<I", pkt, 2)[0]  # uint32 little-endian
        PKT_Time = PKT_Time_ms * 1e-3  # Convert to seconds
        PKT_UNKNOWN1 = pkt[6:9]  # Reserved bytes 6-8
        PKT_ID = pkt[9]
        pkt_unknown2 = pkt[10:13]  # Metadata bytes 10-12
        byte_13 = pkt[13]

        # Decode ID byte
        freq_code = (PKT_ID >> 4) & 0x0F
        type_code = PKT_ID & 0x0F
        pkt_freq = FREQ_MAP.get(freq_code)
        PKT_Type = TYPE_MAP.get(type_code)

        # Extract data section
        PKT_DATA = pkt[14:] if len(pkt) > 14 else b""

        # Validate packet
        pkt_valid = (
            pkt_freq is not None  # Frequency code is recognized
            and PKT_Type is not None  # Type code is recognized
            and byte_13 == 0  # Byte 13 must be 0x00
            and PKT_LEN == len(pkt)  # Declared length matches actual length
            and PKT_LEN >= 14  # Minimum header size
        )

        # Build subpacket dictionary with only requested fields
        pkt_dict = {
            "message_time": message_time,
            "message_uuid": uuid,
            "pkt_offset": offset,
            "PKT_LEN": PKT_LEN,
            "PKT_N": PKT_N,
            "PKT_Time": PKT_Time,
            "PKT_UNKNOWN1": PKT_UNKNOWN1,
            "pkt_freq": pkt_freq,
            "PKT_Type": PKT_Type,
            "pkt_unknown2": pkt_unknown2,
            "PKT_DATA": PKT_DATA,
            "pkt_valid": pkt_valid,
        }

        subpackets.append(pkt_dict)

        # Move to next packet
        offset += declared_len

    return subpackets


def test_parse_message(data_dir: str = "./data_raw") -> Dict:
    """
    Comprehensive test of parse_message() function.

    Tests:
    1. 100% payload coverage (no leftover bytes)
    2. 100% subpacket validity
    3. Performance metrics

    Returns:
    --------
    Dict containing test results and statistics
    """
    import time

    files = sorted(os.listdir(data_dir))

    total_messages = 0
    total_payload_bytes = 0
    total_decoded_bytes = 0
    total_subpackets = 0
    valid_subpackets = 0
    total_parse_time = 0.0

    file_results = []

    for filename in files:
        if not filename.endswith(".txt"):
            continue

        preset = filename.replace("data_", "").replace(".txt", "")

        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()

        file_messages = 0
        file_payload_bytes = 0
        file_decoded_bytes = 0
        file_subpackets = 0
        file_valid = 0

        for message in lines:
            if not message.strip():
                continue

            # Extract payload size before parsing
            parts = message.strip().split("\t", 2)
            if len(parts) < 3:
                continue

            payload = bytes.fromhex(parts[2].strip())
            payload_size = len(payload)

            # Time the parsing
            start_time = time.perf_counter()
            subpackets = parse_message(message)
            parse_time = time.perf_counter() - start_time

            total_parse_time += parse_time

            # Calculate decoded bytes
            decoded_bytes = sum(s["PKT_LEN"] for s in subpackets)

            # Count valid subpackets
            valid_count = sum(1 for s in subpackets if s["pkt_valid"])

            file_messages += 1
            file_payload_bytes += payload_size
            file_decoded_bytes += decoded_bytes
            file_subpackets += len(subpackets)
            file_valid += valid_count

        total_messages += file_messages
        total_payload_bytes += file_payload_bytes
        total_decoded_bytes += file_decoded_bytes
        total_subpackets += file_subpackets
        valid_subpackets += file_valid

        # Calculate file-level metrics
        coverage = (
            100.0 * file_decoded_bytes / file_payload_bytes
            if file_payload_bytes > 0
            else 0
        )
        validity = 100.0 * file_valid / file_subpackets if file_subpackets > 0 else 0

        file_results.append(
            {
                "preset": preset,
                "messages": file_messages,
                "payload_bytes": file_payload_bytes,
                "decoded_bytes": file_decoded_bytes,
                "coverage": coverage,
                "subpackets": file_subpackets,
                "valid": file_valid,
                "validity": validity,
            }
        )

    # Overall metrics
    overall_coverage = (
        100.0 * total_decoded_bytes / total_payload_bytes
        if total_payload_bytes > 0
        else 0
    )
    overall_validity = (
        100.0 * valid_subpackets / total_subpackets if total_subpackets > 0 else 0
    )
    avg_time_per_message = (
        total_parse_time / total_messages if total_messages > 0 else 0
    )
    throughput_mbps = (
        (total_payload_bytes / total_parse_time) / (1024 * 1024)
        if total_parse_time > 0
        else 0
    )

    return {
        "summary": {
            "total_messages": total_messages,
            "total_payload_bytes": total_payload_bytes,
            "total_decoded_bytes": total_decoded_bytes,
            "total_subpackets": total_subpackets,
            "valid_subpackets": valid_subpackets,
            "coverage_pct": overall_coverage,
            "validity_pct": overall_validity,
            "total_parse_time_sec": total_parse_time,
            "avg_time_per_message_ms": avg_time_per_message * 1000,
            "throughput_mbps": throughput_mbps,
        },
        "files": file_results,
        "passed": overall_coverage == 100.0 and overall_validity == 100.0,
    }


# -------------------------------------------------------------------------
# Main Script - Run tests and display results
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("MUSE BLE MESSAGE PARSER - VALIDATION TEST")
    print("=" * 80)
    print("\nRunning comprehensive tests...")

    # Run the test suite
    test_results = test_parse_message(data_dir="./data_raw")

    summary = test_results["summary"]

    # Display test results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    # Coverage and Validity Tests
    print("\n✓ COVERAGE TEST:")
    print(f"  Total payload bytes: {summary['total_payload_bytes']:,}")
    print(f"  Decoded bytes:       {summary['total_decoded_bytes']:,}")
    print(f"  Coverage:            {summary['coverage_pct']:.2f}%")
    if summary["coverage_pct"] == 100.0:
        print("  Status:              ✓ PASSED - 100% payload decoded")
    else:
        print(
            f"  Status:              ✗ FAILED - {100-summary['coverage_pct']:.2f}% leftover"
        )

    print("\n✓ VALIDITY TEST:")
    print(f"  Total subpackets:    {summary['total_subpackets']:,}")
    print(f"  Valid subpackets:    {summary['valid_subpackets']:,}")
    print(f"  Validity:            {summary['validity_pct']:.2f}%")
    if summary["validity_pct"] == 100.0:
        print("  Status:              ✓ PASSED - 100% valid subpackets")
    else:
        invalid = summary["total_subpackets"] - summary["valid_subpackets"]
        print(f"  Status:              ✗ FAILED - {invalid} invalid subpackets")

    # Performance Metrics
    print("\n✓ PERFORMANCE METRICS:")
    print(f"  Total messages:      {summary['total_messages']:,}")
    print(f"  Total parse time:    {summary['total_parse_time_sec']:.3f} seconds")
    print(f"  Avg time/message:    {summary['avg_time_per_message_ms']:.3f} ms")
    print(f"  Throughput:          {summary['throughput_mbps']:.2f} MB/s")

    # Check if suitable for real-time
    if summary["avg_time_per_message_ms"] < 1.0:
        print(f"  Status:              ✓ SUITABLE for real-time (< 1 ms per message)")
    elif summary["avg_time_per_message_ms"] < 10.0:
        print(f"  Status:              ⚠ MARGINAL for real-time (< 10 ms per message)")
    else:
        print(f"  Status:              ✗ TOO SLOW for real-time (> 10 ms per message)")

    # Overall test result
    print("\n" + "=" * 80)
    if test_results["passed"]:
        print("OVERALL: ✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("OVERALL: ✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 80)

    # Per-file breakdown
    print("\n" + "=" * 80)
    print("PER-FILE BREAKDOWN")
    print("=" * 80)
    df_files = pd.DataFrame(test_results["files"])
    print(df_files.to_markdown(index=False, floatfmt=".2f"))

    # Parse all messages and collect sample data for display
    print("\n" + "=" * 80)
    print("SAMPLE SUBPACKETS")
    print("=" * 80)

    data_dir = "./data_raw"
    files = sorted(os.listdir(data_dir))
    all_results = []

    for filename in files[:3]:  # Just first 3 files for sample
        if not filename.endswith(".txt"):
            continue

        preset = filename.replace("data_", "").replace(".txt", "")

        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()

        for message in lines[:2]:  # Just 2 messages per file
            if not message.strip():
                continue

            subpackets = parse_message(message)

            for subpkt in subpackets[:1]:  # Just 1 subpacket per message
                result = {
                    "preset": preset,
                    "PKT_N": subpkt["PKT_N"],
                    "PKT_Type": subpkt["PKT_Type"],
                    "pkt_freq": subpkt["pkt_freq"],
                    "PKT_LEN": subpkt["PKT_LEN"],
                    "data_bytes": len(subpkt["PKT_DATA"]),
                    "unknown1": subpkt["PKT_UNKNOWN1"].hex(),
                    "unknown2": subpkt["pkt_unknown2"].hex(),
                    "valid": subpkt["pkt_valid"],
                }
                all_results.append(result)

    df_sample = pd.DataFrame(all_results)
    print(df_sample.to_markdown(index=False))

    print("\n✓ Script completed successfully!")

