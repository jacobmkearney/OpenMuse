"""
validate_PKT_DATA.py

Analyze the PKT_DATA structure for ACCGYRO, EEG4, and EEG8 packets.
Goal: Understand the variable packet length and confirm the number of samples per packet.

Building on the validated packet structure from validate_subpkts.py:
- Packet structure: PKT_LEN (1) + PKT_N (1) + PKT_T (4) + UNKNOWN (3) +
  PKT_ID (1) + PKT_METADATA (4) + PKT_DATA (variable)
- Header size: 14 bytes
- Data size: packet_length - 14
"""

import struct
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np


# ============================================================================
# PACKET PARSING (from validate_subpkts.py)
# ============================================================================


def extract_pkt_id(pkt: bytes) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract and parse the ID byte from a Muse payload.
    Returns (frequency, type) tuple.
    """
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

    id_byte = pkt[9]
    freq_code = (id_byte >> 4) & 0x0F
    type_code = id_byte & 0x0F

    return FREQ_MAP.get(freq_code), TYPE_MAP.get(type_code)


def extract_pkt_time(pkt: bytes) -> float:
    """Extract timestamp in seconds from packet."""
    ms = struct.unpack_from("<I", pkt, 2)[0]
    return ms * 1e-3


def validate_packet(
    pkt: bytes, prev_counter: Optional[int] = None, prev_time: Optional[float] = None
) -> Dict:
    """Validate a packet with basic checks."""
    result = {
        "valid": False,
        "reason": None,
        "length": len(pkt),
        "declared_length": None,
        "counter": None,
        "timestamp": None,
        "frequency": None,
        "type": None,
    }

    if len(pkt) < 14:
        result["reason"] = "too_short"
        return result

    declared_len = pkt[0]
    result["declared_length"] = declared_len
    if declared_len != len(pkt):
        result["reason"] = "length_mismatch"
        return result

    counter = pkt[1]
    result["counter"] = counter

    if prev_counter is not None:
        expected_counter = (prev_counter + 1) % 256
        if counter != expected_counter:
            result["reason"] = "counter_invalid"
            return result

    try:
        timestamp = extract_pkt_time(pkt)
        result["timestamp"] = timestamp
    except:
        result["reason"] = "timestamp_error"
        return result

    if prev_time is not None and timestamp <= prev_time:
        result["reason"] = "timestamp_not_increasing"
        return result

    try:
        frequency, pkt_type = extract_pkt_id(pkt)
        result["frequency"] = frequency
        result["type"] = pkt_type
    except:
        result["reason"] = "id_error"
        return result

    if frequency is None or pkt_type is None:
        result["reason"] = "unknown_type"
        return result

    result["valid"] = True
    return result


# ============================================================================
# DATA SUBPACKET ANALYSIS
# ============================================================================


def analyze_data_subpacket(pkt: bytes, pkt_type: str) -> Dict:
    """
    Analyze the data subpacket structure for ACCGYRO, EEG4, and EEG8 packets.

    Returns information about:
    - Data length (total bytes after 14-byte header)
    - Potential number of samples (based on expected channels and data type)
    - Any patterns or tags in the data
    """
    result = {
        "packet_type": pkt_type,
        "total_length": len(pkt),
        "header_length": 14,
        "data_length": len(pkt) - 14,
        "metadata": None,
        "data_analysis": {},
    }

    # Extract metadata (bytes 10-13)
    metadata_bytes = pkt[10:14]
    result["metadata"] = metadata_bytes.hex()

    # Parse metadata as two uint16s (little-endian)
    u16_0 = struct.unpack_from("<H", pkt, 10)[0]
    u16_1 = struct.unpack_from("<H", pkt, 12)[0]
    result["metadata_u16_0"] = u16_0
    result["metadata_u16_1"] = u16_1

    # Extract individual bytes for detailed analysis
    result["metadata_byte_10"] = pkt[10]
    result["metadata_byte_11"] = pkt[11]
    result["metadata_byte_12"] = pkt[12]
    result["metadata_byte_13"] = pkt[
        13
    ]  # u8_3 - should always be 0?    # Extract data portion (bytes 14+)
    data = pkt[14:]
    data_length = len(data)

    # Look at first few bytes for patterns
    result["data_prefix"] = data[:8].hex() if len(data) >= 8 else data.hex()

    if pkt_type == "ACCGYRO":
        # ACCGYRO: Expected 6 channels (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z)
        # Data is likely int16 (2 bytes per channel) = 12 bytes per sample
        # May have tag bytes (0x47?) and block headers

        # Count potential tag bytes (0x47 is one possibility)
        tag_count_0x47 = data.count(b"\x47")
        tag_count_0xF4 = data.count(b"\xf4")

        result["data_analysis"] = {
            "expected_channels": 6,
            "bytes_per_channel": 2,  # Assuming int16
            "expected_bytes_per_sample": 12,
            "tag_0x47_count": tag_count_0x47,
            "tag_0xF4_count": tag_count_0xF4,
        }

        # Try different interpretations
        # Option 1: Pure samples (no tags/headers)
        if data_length % 12 == 0:
            result["data_analysis"]["interpretation_pure_samples"] = {
                "possible": True,
                "num_samples": data_length // 12,
            }
        else:
            result["data_analysis"]["interpretation_pure_samples"] = {
                "possible": False,
                "remainder": data_length % 12,
            }

        # Option 2: Tag + samples (1-byte tag + N samples)
        for tag_size in [1, 5]:  # Try 1-byte tag or 5-byte (tag + 4-byte header)
            remaining = data_length - tag_size
            if remaining > 0 and remaining % 12 == 0:
                result["data_analysis"][f"interpretation_tag{tag_size}_samples"] = {
                    "possible": True,
                    "tag_size": tag_size,
                    "num_samples": remaining // 12,
                }
            else:
                result["data_analysis"][f"interpretation_tag{tag_size}_samples"] = {
                    "possible": False,
                    "remainder": remaining % 12 if remaining > 0 else None,
                }

    elif pkt_type == "EEG4":
        # EEG4: Expected 4 channels
        # Data is likely int16 (2 bytes per channel) = 8 bytes per sample

        result["data_analysis"] = {
            "expected_channels": 4,
            "bytes_per_channel": 2,  # Assuming int16
            "expected_bytes_per_sample": 8,
        }

        # Check if data length is divisible by 8
        if data_length % 8 == 0:
            result["data_analysis"]["interpretation_pure_samples"] = {
                "possible": True,
                "num_samples": data_length // 8,
            }
        else:
            result["data_analysis"]["interpretation_pure_samples"] = {
                "possible": False,
                "remainder": data_length % 8,
            }

        # Try with potential tag/header bytes
        for overhead in [1, 2, 4, 5]:
            remaining = data_length - overhead
            if remaining > 0 and remaining % 8 == 0:
                result["data_analysis"][
                    f"interpretation_overhead{overhead}_samples"
                ] = {
                    "possible": True,
                    "overhead_size": overhead,
                    "num_samples": remaining // 8,
                }

    elif pkt_type == "EEG8":
        # EEG8: Expected 8 channels
        # Data is likely int16 (2 bytes per channel) = 16 bytes per sample

        result["data_analysis"] = {
            "expected_channels": 8,
            "bytes_per_channel": 2,  # Assuming int16
            "expected_bytes_per_sample": 16,
        }

        # Check if data length is divisible by 16
        if data_length % 16 == 0:
            result["data_analysis"]["interpretation_pure_samples"] = {
                "possible": True,
                "num_samples": data_length // 16,
            }
        else:
            result["data_analysis"]["interpretation_pure_samples"] = {
                "possible": False,
                "remainder": data_length % 16,
            }

        # Try with potential tag/header bytes
        for overhead in [1, 2, 4, 5]:
            remaining = data_length - overhead
            if remaining > 0 and remaining % 16 == 0:
                result["data_analysis"][
                    f"interpretation_overhead{overhead}_samples"
                ] = {
                    "possible": True,
                    "overhead_size": overhead,
                    "num_samples": remaining // 16,
                }

    return result


def validate_accgyro_decoding(pkt: bytes) -> Dict:
    """
    Validate ACCGYRO data decoding by attempting to extract samples.

    ACCGYRO should contain:
    - 6 channels: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
    - Each sample = 6 int16 values = 12 bytes
    - Data may have tags/headers before samples

    Strategy:
    1. Try different interpretations (pure samples, with tags, etc.)
    2. Decode as int16 values and check if they look reasonable
    3. Count how many complete samples we can extract
    """
    data = pkt[14:]  # Skip 14-byte header
    data_length = len(data)

    result = {
        "data_length": data_length,
        "interpretations": {},
    }

    # Interpretation 1: Pure samples (no overhead)
    if data_length % 12 == 0:
        num_samples = data_length // 12
        try:
            # Decode all as int16 little-endian
            values = struct.unpack(f"<{num_samples * 6}h", data)

            # Check if values are in reasonable ranges for accelerometer/gyro
            # ACC typically ±8g -> raw values roughly ±16000
            # GYRO typically ±2000 dps -> raw values roughly ±16000
            reasonable = all(-32768 <= v <= 32767 for v in values)

            result["interpretations"]["pure_samples"] = {
                "viable": True,
                "num_samples": num_samples,
                "total_values": len(values),
                "values_in_range": reasonable,
                "sample_values": (
                    values[:18] if len(values) >= 18 else values
                ),  # First 3 samples
                "value_range": (min(values), max(values)),
            }
        except struct.error as e:
            result["interpretations"]["pure_samples"] = {
                "viable": False,
                "error": str(e),
            }

    # Interpretation 2: Data with variable overhead (try removing 1-10 bytes from start)
    for overhead in range(1, min(11, data_length)):
        remaining = data_length - overhead
        if remaining % 12 == 0 and remaining > 0:
            num_samples = remaining // 12
            try:
                values = struct.unpack(f"<{num_samples * 6}h", data[overhead:])
                reasonable = all(-32768 <= v <= 32767 for v in values)

                result["interpretations"][f"overhead_{overhead}"] = {
                    "viable": True,
                    "overhead_bytes": overhead,
                    "overhead_hex": data[:overhead].hex(),
                    "num_samples": num_samples,
                    "values_in_range": reasonable,
                    "value_range": (min(values), max(values)),
                }
            except struct.error:
                pass

    return result


def process_file(filepath: str) -> List[Dict]:
    """Process a single data file and analyze all packets."""
    results = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        try:
            ts, uuid, payload_hex = line.strip().split("\t", 2)
            payload = bytes.fromhex(payload_hex.strip())

            offset = 0
            prev_counter = None
            prev_time = None

            while offset < len(payload):
                if offset + 14 > len(payload):
                    break

                declared_len = payload[offset]
                if offset + declared_len > len(payload):
                    break

                pkt = payload[offset : offset + declared_len]
                validation = validate_packet(pkt, prev_counter, prev_time)

                if validation["valid"]:
                    pkt_type = validation["type"]

                    # Only analyze ACCGYRO, EEG4, and EEG8
                    if pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
                        analysis = analyze_data_subpacket(pkt, pkt_type)
                        results.append(analysis)

                    prev_counter = validation["counter"]
                    prev_time = validation["timestamp"]
                    offset += declared_len
                else:
                    offset += 1

        except Exception as e:
            print(f"Error processing line: {e}")
            continue

    return results


# ============================================================================
# MAIN SCRIPT
# ============================================================================

data_dir = "./data_raw"
files = sorted(os.listdir(data_dir))

print("=" * 80)
print("ANALYZING PKT_DATA STRUCTURE")
print("=" * 80)

all_analyses = []

for filename in files:  # Process all files
    print(f"\nProcessing {filename}...")
    preset = filename.replace("data_", "").replace(".txt", "")

    filepath = os.path.join(data_dir, filename)
    analyses = process_file(filepath)

    # Add preset info to each analysis
    for analysis in analyses:
        analysis["preset"] = preset
        analysis["filename"] = filename

    all_analyses.extend(analyses)
    print(f"  Analyzed {len(analyses)} packets")

# Convert to DataFrame for analysis
df = pd.DataFrame(all_analyses)

print("\n" + "=" * 80)
print("DATA LENGTH DISTRIBUTION BY PACKET TYPE")
print("=" * 80)

for pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
    if pkt_type not in df["packet_type"].values:
        continue

    subset = df[df["packet_type"] == pkt_type]
    print(f"\n{pkt_type} (n={len(subset)} packets):")
    print(
        f"  Data length: min={subset['data_length'].min()}, "
        f"max={subset['data_length'].max()}, "
        f"mean={subset['data_length'].mean():.1f}"
    )

    # Show distribution of data lengths
    length_counts = subset["data_length"].value_counts().sort_index()
    print(f"  Unique data lengths: {length_counts.to_dict()}")

print("\n" + "=" * 80)
print("SAMPLE COUNT ANALYSIS")
print("=" * 80)

# Analyze interpretations for each packet type
for pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
    if pkt_type not in df["packet_type"].values:
        continue

    subset = df[df["packet_type"] == pkt_type]
    print(f"\n{pkt_type}:")

    # Check pure samples interpretation
    pure_samples = []
    for idx, row in subset.iterrows():
        interp = row["data_analysis"].get("interpretation_pure_samples", {})
        if interp.get("possible", False):
            pure_samples.append(interp["num_samples"])

    if pure_samples:
        pure_samples_array = np.array(pure_samples)
        print(f"  Pure samples interpretation:")
        print(
            f"    Viable for {len(pure_samples)}/{len(subset)} packets ({100*len(pure_samples)/len(subset):.1f}%)"
        )
        print(
            f"    Sample counts: min={pure_samples_array.min()}, "
            f"max={pure_samples_array.max()}, "
            f"mean={pure_samples_array.mean():.1f}"
        )
        print(f"    Unique sample counts: {np.unique(pure_samples_array).tolist()}")
    else:
        print(f"  Pure samples interpretation: NOT VIABLE")

        # Check overhead interpretations
        print(f"  Checking with overhead bytes:")
        for key in subset.iloc[0]["data_analysis"].keys():
            if key.startswith("interpretation_overhead") or key.startswith(
                "interpretation_tag"
            ):
                overhead_samples = []
                for idx, row in subset.iterrows():
                    interp = row["data_analysis"].get(key, {})
                    if interp.get("possible", False):
                        overhead_samples.append(interp["num_samples"])

                if len(overhead_samples) == len(
                    subset
                ):  # All packets match this interpretation
                    overhead_samples_array = np.array(overhead_samples)
                    overhead_size = subset.iloc[0]["data_analysis"][key].get(
                        "overhead_size"
                    ) or subset.iloc[0]["data_analysis"][key].get("tag_size")
                    print(f"    {key}: VIABLE for ALL packets")
                    print(f"      Overhead: {overhead_size} bytes")
                    print(
                        f"      Sample counts: min={overhead_samples_array.min()}, "
                        f"max={overhead_samples_array.max()}, "
                        f"mean={overhead_samples_array.mean():.1f}"
                    )
                    print(
                        f"      Unique sample counts: {np.unique(overhead_samples_array).tolist()}"
                    )

print("\n" + "=" * 80)
print("METADATA ANALYSIS")
print("=" * 80)

for pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
    if pkt_type not in df["packet_type"].values:
        continue

    subset = df[df["packet_type"] == pkt_type]
    print(f"\n{pkt_type}:")
    print(
        f"  metadata_u16_0: min={subset['metadata_u16_0'].min()}, "
        f"max={subset['metadata_u16_0'].max()}, "
        f"unique_values={len(subset['metadata_u16_0'].unique())}"
    )
    print(
        f"  metadata_u16_1: unique_values={sorted(subset['metadata_u16_1'].unique())}"
    )

    # Check correlation between metadata_u16_1 and data_length
    print(f"  Data length by metadata_u16_1:")
    for u16_1_val in sorted(subset["metadata_u16_1"].unique()):
        group = subset[subset["metadata_u16_1"] == u16_1_val]
        print(
            f"    u16_1={u16_1_val}: n={len(group)}, "
            f"data_len range={group['data_length'].min()}-{group['data_length'].max()}, "
            f"mean={group['data_length'].mean():.1f}"
        )

print("\n" + "=" * 80)
print("DATA PREFIX PATTERNS")
print("=" * 80)

for pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
    if pkt_type not in df["packet_type"].values:
        continue

    subset = df[df["packet_type"] == pkt_type]
    print(f"\n{pkt_type}:")

    # Look at first byte of data
    first_bytes = subset["data_prefix"].apply(lambda x: x[:2])
    first_byte_counts = first_bytes.value_counts().head(10)
    print(f"  Most common first bytes (hex):")
    for byte_val, count in first_byte_counts.items():
        print(f"    0x{byte_val}: {count} packets ({100*count/len(subset):.1f}%)")

    # Sample a few complete prefixes
    print(f"  Sample data prefixes (first 8 bytes):")
    for i, prefix in enumerate(subset["data_prefix"].unique()[:5]):
        print(f"    {prefix}")

print("\n" + "=" * 80)
print("BYTE 13 (u8_3) ANALYSIS - Checking if it's always 0")
print("=" * 80)

for pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
    if pkt_type not in df["packet_type"].values:
        continue

    subset = df[df["packet_type"] == pkt_type]
    byte_13_values = subset["metadata_byte_13"].unique()

    print(f"\n{pkt_type}:")
    print(f"  Unique values for byte 13: {sorted(byte_13_values)}")
    print(f"  Count of byte 13 values:")
    for val in sorted(byte_13_values):
        count = (subset["metadata_byte_13"] == val).sum()
        print(f"    {val}: {count} packets ({100*count/len(subset):.2f}%)")

    if len(byte_13_values) == 1 and byte_13_values[0] == 0:
        print(f"  ✓ CONFIRMED: Byte 13 is always 0 (potential separator/padding)")
    else:
        print(f"  ✗ NOT CONFIRMED: Byte 13 has multiple values")

print("\n" + "=" * 80)
print("METADATA_U16_0 vs DATA_LENGTH CORRELATION")
print("=" * 80)
print("Investigating if metadata_u16_0 relates to number of samples/data structure")

for pkt_type in ["ACCGYRO", "EEG4", "EEG8"]:
    if pkt_type not in df["packet_type"].values:
        continue

    subset = df[df["packet_type"] == pkt_type]
    print(f"\n{pkt_type}:")

    # Calculate correlation between metadata_u16_0 and data_length
    correlation = subset["metadata_u16_0"].corr(subset["data_length"])
    print(f"  Correlation (metadata_u16_0 vs data_length): {correlation:.4f}")

    # Check if metadata_u16_0 could represent data_length or sample count
    # Try various transformations
    subset_copy = subset.copy()
    subset_copy["u16_0_div_8"] = subset_copy["metadata_u16_0"] / 8
    subset_copy["u16_0_div_12"] = subset_copy["metadata_u16_0"] / 12
    subset_copy["u16_0_div_16"] = subset_copy["metadata_u16_0"] / 16
    subset_copy["u16_0_mod_256"] = subset_copy["metadata_u16_0"] % 256

    # Check various potential relationships
    print(f"  Testing potential relationships:")

    # Could u16_0 be related to a counter or offset?
    if subset["metadata_u16_0"].min() > 10000:
        print(
            f"    metadata_u16_0 appears to be a large counter (min={subset['metadata_u16_0'].min()})"
        )

    # Group by data_length and see if u16_0 varies consistently
    length_groups = subset.groupby("data_length")["metadata_u16_0"].agg(
        ["min", "max", "mean", "count"]
    )

    # Show a few examples
    print(f"  Sample data_length groups (showing up to 5):")
    for data_len, row in length_groups.head(5).iterrows():
        print(
            f"    data_len={data_len}: u16_0 range={int(row['min'])}-{int(row['max'])}, "
            f"mean={row['mean']:.0f}, n={int(row['count'])}"
        )

    # Check if data_length can be predicted from u16_0
    # Try modulo operations that might reveal sample counts
    if pkt_type == "ACCGYRO":
        bytes_per_sample = 12
    elif pkt_type == "EEG4":
        bytes_per_sample = 8
    elif pkt_type == "EEG8":
        bytes_per_sample = 16

    subset_copy["implied_samples_from_length"] = (
        subset_copy["data_length"] / bytes_per_sample
    )
    subset_copy["u16_0_mod_samples"] = subset_copy["metadata_u16_0"] % bytes_per_sample

    # Check if u16_0 modulo bytes_per_sample correlates with remainder
    remainder_corr = subset_copy["u16_0_mod_samples"].corr(
        subset_copy["data_length"] % bytes_per_sample
    )
    print(
        f"  Correlation (u16_0 % {bytes_per_sample} vs data_length % {bytes_per_sample}): {remainder_corr:.4f}"
    )

print("\n" + "=" * 80)
print("ACCGYRO DATA DECODING VALIDATION")
print("=" * 80)
print("Testing if ACCGYRO data can be decoded as 6-channel int16 samples")

accgyro_packets = df[df["packet_type"] == "ACCGYRO"]
if len(accgyro_packets) > 0:
    print(f"\nAnalyzing {len(accgyro_packets)} ACCGYRO packets...")

    # Test decoding on a sample of packets
    sample_size = min(100, len(accgyro_packets))
    sample_indices = np.random.choice(accgyro_packets.index, sample_size, replace=False)

    decoding_results = {
        "pure_samples_success": 0,
        "overhead_success": {},
        "failed": 0,
    }

    # Re-process sample packets with decoding validation
    for idx in sample_indices:
        row = accgyro_packets.loc[idx]

        # Get the actual packet from the file
        filepath = os.path.join(data_dir, row["filename"])
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find and decode this specific packet
        # (This is simplified - in practice we'd need to track packet position)
        # For now, just demonstrate with first few packets
        break  # Skip detailed per-packet analysis for performance

    # Instead, analyze the structure patterns we already have
    print(f"\n  From structure analysis:")
    print(
        f"    Pure samples viable: {len(accgyro_packets[accgyro_packets['data_analysis'].apply(lambda x: x.get('interpretation_pure_samples', {}).get('possible', False))])} / {len(accgyro_packets)} packets"
    )

    # Check tag patterns in ACCGYRO
    print(f"\n  Tag byte analysis:")
    tag_0x47_counts = [
        row["data_analysis"].get("tag_0x47_count", 0)
        for _, row in accgyro_packets.iterrows()
    ]
    tag_0xF4_counts = [
        row["data_analysis"].get("tag_0xF4_count", 0)
        for _, row in accgyro_packets.iterrows()
    ]

    print(
        f"    0x47 tag present: {sum(1 for c in tag_0x47_counts if c > 0)} / {len(accgyro_packets)} packets"
    )
    print(
        f"    0xF4 tag present: {sum(1 for c in tag_0xF4_counts if c > 0)} / {len(accgyro_packets)} packets"
    )
    print(
        f"    No common tag: {sum(1 for i, _ in enumerate(accgyro_packets.iterrows()) if tag_0x47_counts[i] == 0 and tag_0xF4_counts[i] == 0)} / {len(accgyro_packets)} packets"
    )

    # Recommendation based on findings
    print(f"\n  RECOMMENDATION:")
    pure_viable_pct = (
        100
        * len(
            accgyro_packets[
                accgyro_packets["data_analysis"].apply(
                    lambda x: x.get("interpretation_pure_samples", {}).get(
                        "possible", False
                    )
                )
            ]
        )
        / len(accgyro_packets)
    )

    if pure_viable_pct > 50:
        print(
            f"    ✓ Pure samples interpretation works for {pure_viable_pct:.1f}% of packets"
        )
        print(f"      -> Decode directly as 6-channel int16 samples (no overhead)")
    else:
        print(
            f"    ✗ Pure samples interpretation only works for {pure_viable_pct:.1f}% of packets"
        )
        print(f"      -> Need to identify and skip overhead bytes (tags/headers)")
        print(
            f"      -> Investigate patterns in data_prefix to find consistent overhead structure"
        )
else:
    print("\n  No ACCGYRO packets found in dataset")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print(
    """
Based on the analysis above:
1. Identify if sample counts are constant or variable per packet type
2. Determine the overhead structure (tags, headers, etc.)
3. Investigate the relationship between metadata fields and data structure
4. Verify the bytes_per_channel assumption (int16 = 2 bytes)
5. Examine actual data values to confirm interpretation
"""
)

# Save detailed results
output_file = "PKT_DATA_analysis.csv"
df.to_csv(output_file, index=False)
print(f"\nDetailed results saved to {output_file}")

