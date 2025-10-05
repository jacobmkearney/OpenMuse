"""
validate_GYROACC3.py

Comprehensive ACCGYRO decoder validation.

Goal: Test different decoding strategies for ACCGYRO packets and determine which
produces reasonable accelerometer and gyroscope values.

Expected ACCGYRO structure:
- 6 channels: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
- Each sample = 6 int16 values = 12 bytes
- Data section (bytes 14+) may have:
  - Tag bytes (0x47, 0xF4, or others)
  - Headers/metadata
  - Variable number of samples per packet

Validation criteria:
- Decoded values should be int16 (-32768 to 32767)
- Accelerometer raw values typically: ±16000 (for ±8g range)
- Gyroscope raw values typically: ±16000 (for ±2000 dps range)
- Values should show some variation (not all zeros)
- Should decode to complete samples (no partial samples)
"""

import struct
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

ACC_SCALE = 0.0000610352  # Scale factor for accelerometer
GYRO_SCALE = -0.0074768  # Scale factor for gyroscope

# Expected ranges for raw int16 values (before scaling)
ACC_REASONABLE_RANGE = (-20000, 20000)  # ±8g range
GYRO_REASONABLE_RANGE = (-20000, 20000)  # ±2000 dps range


# ============================================================================
# PACKET PARSING UTILITIES
# ============================================================================


def extract_pkt_id(pkt: bytes) -> Tuple[Optional[float], Optional[str]]:
    """Extract and parse the ID byte from a Muse packet."""
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


def validate_packet_basic(pkt: bytes) -> bool:
    """Basic validation: length, type, byte 13."""
    if len(pkt) < 14:
        return False
    if pkt[0] != len(pkt):  # Declared length matches
        return False
    if pkt[13] != 0:  # Byte 13 always 0
        return False

    _, pkt_type = extract_pkt_id(pkt)
    return pkt_type == "ACCGYRO"


# ============================================================================
# DECODING STRATEGIES
# ============================================================================


def decode_strategy_pure_samples(data: bytes) -> Optional[Dict]:
    """
    Strategy 1: Data is pure samples, no overhead.
    data_length must be divisible by 12.
    """
    if len(data) % 12 != 0:
        return None

    num_samples = len(data) // 12
    try:
        values = struct.unpack(f"<{num_samples * 6}h", data)

        samples = []
        for i in range(num_samples):
            idx = i * 6
            samples.append(
                {
                    "ACC_X": values[idx + 0],
                    "ACC_Y": values[idx + 1],
                    "ACC_Z": values[idx + 2],
                    "GYRO_X": values[idx + 3],
                    "GYRO_Y": values[idx + 4],
                    "GYRO_Z": values[idx + 5],
                }
            )

        return {
            "strategy": "pure_samples",
            "num_samples": num_samples,
            "samples": samples,
            "overhead_bytes": 0,
            "overhead_hex": "",
        }
    except struct.error:
        return None


def decode_strategy_skip_n_bytes(data: bytes, n: int) -> Optional[Dict]:
    """
    Strategy 2: Skip first N bytes, then decode as samples.
    """
    if n >= len(data):
        return None

    remaining = data[n:]
    if len(remaining) % 12 != 0:
        return None

    num_samples = len(remaining) // 12
    if num_samples == 0:
        return None

    try:
        values = struct.unpack(f"<{num_samples * 6}h", remaining)

        samples = []
        for i in range(num_samples):
            idx = i * 6
            samples.append(
                {
                    "ACC_X": values[idx + 0],
                    "ACC_Y": values[idx + 1],
                    "ACC_Z": values[idx + 2],
                    "GYRO_X": values[idx + 3],
                    "GYRO_Y": values[idx + 4],
                    "GYRO_Z": values[idx + 5],
                }
            )

        return {
            "strategy": f"skip_{n}_bytes",
            "num_samples": num_samples,
            "samples": samples,
            "overhead_bytes": n,
            "overhead_hex": data[:n].hex(),
        }
    except struct.error:
        return None


def decode_strategy_find_tag(data: bytes, tag: int) -> Optional[Dict]:
    """
    Strategy 3: Find tag byte (0x47 or 0xF4), skip tag + 4 bytes header, decode samples.
    """
    tag_byte = tag.to_bytes(1, "big")
    tag_idx = data.find(tag_byte)

    if tag_idx == -1:
        return None

    # Skip tag (1 byte) + header (4 bytes)
    start = tag_idx + 5
    if start >= len(data):
        return None

    remaining = data[start:]
    if len(remaining) % 12 != 0:
        return None

    num_samples = len(remaining) // 12
    if num_samples == 0:
        return None

    try:
        values = struct.unpack(f"<{num_samples * 6}h", remaining)

        samples = []
        for i in range(num_samples):
            idx = i * 6
            samples.append(
                {
                    "ACC_X": values[idx + 0],
                    "ACC_Y": values[idx + 1],
                    "ACC_Z": values[idx + 2],
                    "GYRO_X": values[idx + 3],
                    "GYRO_Y": values[idx + 4],
                    "GYRO_Z": values[idx + 5],
                }
            )

        return {
            "strategy": f"tag_0x{tag:02x}_offset_{tag_idx}",
            "num_samples": num_samples,
            "samples": samples,
            "overhead_bytes": start,
            "overhead_hex": data[:start].hex(),
            "tag_found_at": tag_idx,
        }
    except struct.error:
        return None


# ============================================================================
# VALIDATION
# ============================================================================


def validate_samples(samples: List[Dict]) -> Dict:
    """
    Validate that decoded samples contain reasonable accelerometer/gyro values.
    """
    if not samples:
        return {
            "valid": False,
            "reason": "no_samples",
        }

    # Extract all values
    acc_x = [s["ACC_X"] for s in samples]
    acc_y = [s["ACC_Y"] for s in samples]
    acc_z = [s["ACC_Z"] for s in samples]
    gyro_x = [s["GYRO_X"] for s in samples]
    gyro_y = [s["GYRO_Y"] for s in samples]
    gyro_z = [s["GYRO_Z"] for s in samples]

    all_acc = acc_x + acc_y + acc_z
    all_gyro = gyro_x + gyro_y + gyro_z
    all_values = all_acc + all_gyro

    # Check 1: All values in int16 range
    if not all(-32768 <= v <= 32767 for v in all_values):
        return {
            "valid": False,
            "reason": "values_out_of_range",
        }

    # Check 2: Not all zeros
    if all(v == 0 for v in all_values):
        return {
            "valid": False,
            "reason": "all_zeros",
        }

    # Check 3: Accelerometer values in reasonable range
    acc_in_range = sum(
        1 for v in all_acc if ACC_REASONABLE_RANGE[0] <= v <= ACC_REASONABLE_RANGE[1]
    )
    acc_in_range_pct = 100 * acc_in_range / len(all_acc)

    # Check 4: Gyro values in reasonable range
    gyro_in_range = sum(
        1 for v in all_gyro if GYRO_REASONABLE_RANGE[0] <= v <= GYRO_REASONABLE_RANGE[1]
    )
    gyro_in_range_pct = 100 * gyro_in_range / len(all_gyro)

    # Check 5: Some variation (std dev > 0)
    acc_has_variation = np.std(all_acc) > 1.0
    gyro_has_variation = np.std(all_gyro) > 1.0

    # Check 6: Accelerometer Z should be around ±1g when device is still (raw ~±16384)
    # This is a sanity check - at least one axis should show gravity
    acc_magnitude = np.sqrt(
        np.mean(acc_x) ** 2 + np.mean(acc_y) ** 2 + np.mean(acc_z) ** 2
    )
    gravity_present = 8000 < acc_magnitude < 24000  # Rough range for 1g

    # Overall validation
    valid = (
        acc_in_range_pct > 80
        and gyro_in_range_pct > 80
        and (acc_has_variation or gyro_has_variation)
    )

    return {
        "valid": valid,
        "acc_in_range_pct": acc_in_range_pct,
        "gyro_in_range_pct": gyro_in_range_pct,
        "acc_has_variation": acc_has_variation,
        "gyro_has_variation": gyro_has_variation,
        "gravity_present": gravity_present,
        "acc_magnitude": acc_magnitude,
        "acc_stats": {
            "mean": [np.mean(acc_x), np.mean(acc_y), np.mean(acc_z)],
            "std": [np.std(acc_x), np.std(acc_y), np.std(acc_z)],
            "range": [
                (min(acc_x), max(acc_x)),
                (min(acc_y), max(acc_y)),
                (min(acc_z), max(acc_z)),
            ],
        },
        "gyro_stats": {
            "mean": [np.mean(gyro_x), np.mean(gyro_y), np.mean(gyro_z)],
            "std": [np.std(gyro_x), np.std(gyro_y), np.std(gyro_z)],
            "range": [
                (min(gyro_x), max(gyro_x)),
                (min(gyro_y), max(gyro_y)),
                (min(gyro_z), max(gyro_z)),
            ],
        },
    }


def try_all_strategies(pkt: bytes) -> List[Dict]:
    """
    Try all decoding strategies on a packet and return successful ones with validation.
    """
    data = pkt[14:]  # Skip 14-byte header
    results = []

    # Strategy 1: Pure samples
    decoded = decode_strategy_pure_samples(data)
    if decoded:
        validation = validate_samples(decoded["samples"])
        decoded["validation"] = validation
        results.append(decoded)

    # Strategy 2: Skip N bytes (try 1-10)
    for n in range(1, min(11, len(data))):
        decoded = decode_strategy_skip_n_bytes(data, n)
        if decoded:
            validation = validate_samples(decoded["samples"])
            decoded["validation"] = validation
            results.append(decoded)

    # Strategy 3: Find tag 0x47
    decoded = decode_strategy_find_tag(data, 0x47)
    if decoded:
        validation = validate_samples(decoded["samples"])
        decoded["validation"] = validation
        results.append(decoded)

    # Strategy 4: Find tag 0xF4
    decoded = decode_strategy_find_tag(data, 0xF4)
    if decoded:
        validation = validate_samples(decoded["samples"])
        decoded["validation"] = validation
        results.append(decoded)

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def analyze_file(filepath: str, max_packets: int = 100) -> List[Dict]:
    """Analyze ACCGYRO packets from a file."""
    results = []
    packet_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        try:
            ts, uuid, payload_hex = line.strip().split("\t", 2)
            payload = bytes.fromhex(payload_hex.strip())

            offset = 0
            while offset < len(payload) and packet_count < max_packets:
                if offset + 14 > len(payload):
                    break

                declared_len = payload[offset]
                if offset + declared_len > len(payload):
                    break

                pkt = payload[offset : offset + declared_len]

                if validate_packet_basic(pkt):
                    # Try all decoding strategies
                    strategy_results = try_all_strategies(pkt)

                    if strategy_results:
                        results.append(
                            {
                                "packet_num": packet_count,
                                "timestamp": ts,
                                "packet_length": len(pkt),
                                "data_length": len(pkt) - 14,
                                "strategies": strategy_results,
                            }
                        )
                        packet_count += 1

                offset += declared_len

        except Exception as e:
            print(f"Error processing line: {e}")
            continue

    return results


# ============================================================================
# SCRIPT
# ============================================================================

data_dir = "./data_raw"
files = sorted(os.listdir(data_dir))

print("=" * 80)
print("ACCGYRO DECODER VALIDATION")
print("=" * 80)

# Analyze multiple files with ACCGYRO packets
test_files = ["data_p20.txt", "data_p1034.txt", "data_p1041.txt"]  # Different presets
all_results = []

for test_file in test_files:
    print(f"\nAnalyzing {test_file}...")
    filepath = os.path.join(data_dir, test_file)
    file_results = analyze_file(filepath, max_packets=20)
    all_results.extend(file_results)
    print(f"  Found {len(file_results)} ACCGYRO packets")

results = all_results
print(f"\nTotal analyzed: {len(results)} ACCGYRO packets")

# ============================================================================
# STRATEGY SUCCESS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DECODING STRATEGY SUCCESS RATES")
print("=" * 80)

strategy_stats = {}

for result in results:
    for strategy_result in result["strategies"]:
        strategy_name = strategy_result["strategy"]

        if strategy_name not in strategy_stats:
            strategy_stats[strategy_name] = {
                "total": 0,
                "valid": 0,
                "acc_in_range_sum": 0,
                "gyro_in_range_sum": 0,
                "num_samples_list": [],
            }

        strategy_stats[strategy_name]["total"] += 1

        validation = strategy_result["validation"]
        if validation["valid"]:
            strategy_stats[strategy_name]["valid"] += 1

        strategy_stats[strategy_name]["acc_in_range_sum"] += validation.get(
            "acc_in_range_pct", 0
        )
        strategy_stats[strategy_name]["gyro_in_range_sum"] += validation.get(
            "gyro_in_range_pct", 0
        )
        strategy_stats[strategy_name]["num_samples_list"].append(
            strategy_result["num_samples"]
        )

# Print summary
print(
    f"\nFound {len(strategy_stats)} different strategies that produced decodable output:"
)

# Sort by success rate
sorted_strategies = sorted(
    strategy_stats.items(),
    key=lambda x: (x[1]["valid"] / x[1]["total"], x[1]["total"]),
    reverse=True,
)

for strategy_name, stats in sorted_strategies[:10]:  # Top 10
    success_rate = 100 * stats["valid"] / stats["total"]
    avg_acc_in_range = stats["acc_in_range_sum"] / stats["total"]
    avg_gyro_in_range = stats["gyro_in_range_sum"] / stats["total"]
    avg_samples = np.mean(stats["num_samples_list"])

    print(f"\n{strategy_name}:")
    print(f"  Applied to: {stats['total']} packets")
    print(f"  Valid: {stats['valid']} ({success_rate:.1f}%)")
    print(f"  Avg ACC in range: {avg_acc_in_range:.1f}%")
    print(f"  Avg GYRO in range: {avg_gyro_in_range:.1f}%")
    print(f"  Avg samples/packet: {avg_samples:.1f}")
    print(f"  Sample counts: {sorted(set(stats['num_samples_list']))}")

# ============================================================================
# DETAILED EXAMPLES
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED EXAMPLES OF SUCCESSFUL DECODING")
print("=" * 80)

# Find packets where strategies succeeded
for i, result in enumerate(results[:5]):  # First 5 packets
    print(
        f"\n--- Packet {result['packet_num']} (length={result['packet_length']}, data={result['data_length']}) ---"
    )

    valid_strategies = [s for s in result["strategies"] if s["validation"]["valid"]]

    if valid_strategies:
        best = valid_strategies[0]
        print(f"Best strategy: {best['strategy']}")
        print(f"  Overhead: {best['overhead_bytes']} bytes ({best['overhead_hex']})")
        print(f"  Samples: {best['num_samples']}")
        print(
            f"  Validation: ACC {best['validation']['acc_in_range_pct']:.1f}% in range, "
            f"GYRO {best['validation']['gyro_in_range_pct']:.1f}% in range"
        )

        # Show first sample
        if best["samples"]:
            sample = best["samples"][0]
            print(f"  First sample:")
            print(
                f"    ACC:  X={sample['ACC_X']:6d}, Y={sample['ACC_Y']:6d}, Z={sample['ACC_Z']:6d}"
            )
            print(
                f"    GYRO: X={sample['GYRO_X']:6d}, Y={sample['GYRO_Y']:6d}, Z={sample['GYRO_Z']:6d}"
            )

            # Show scaled values
            print(f"  Scaled (physical units):")
            print(
                f"    ACC:  X={sample['ACC_X']*ACC_SCALE:8.4f}, Y={sample['ACC_Y']*ACC_SCALE:8.4f}, Z={sample['ACC_Z']*ACC_SCALE:8.4f} g"
            )
            print(
                f"    GYRO: X={sample['GYRO_X']*GYRO_SCALE:8.4f}, Y={sample['GYRO_Y']*GYRO_SCALE:8.4f}, Z={sample['GYRO_Z']*GYRO_SCALE:8.4f} dps"
            )
    else:
        print(f"  No valid decoding strategies found")
        print(f"  Tried {len(result['strategies'])} strategies")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if sorted_strategies:
    best_strategy = sorted_strategies[0]
    best_name = best_strategy[0]
    best_stats = best_strategy[1]
    success_rate = 100 * best_stats["valid"] / best_stats["total"]

    print(f"\n✓ BEST STRATEGY: {best_name}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(
        f"  Average samples per packet: {np.mean(best_stats['num_samples_list']):.1f}"
    )

    if "skip_" in best_name:
        n_bytes = int(best_name.split("_")[1])
        print(f"\n  IMPLEMENTATION:")
        print(f"    1. Skip first {n_bytes} bytes of data section")
        print(f"    2. Decode remaining bytes as int16 little-endian")
        print(
            f"    3. Group into samples of 6 values: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z"
        )
        print(f"    4. Scale: ACC *= {ACC_SCALE}, GYRO *= {GYRO_SCALE}")

    elif "tag_" in best_name:
        tag = best_name.split("_")[1]
        print(f"\n  IMPLEMENTATION:")
        print(f"    1. Find {tag} tag byte in data section")
        print(f"    2. Skip tag + 4 bytes (header)")
        print(f"    3. Decode remaining bytes as int16 little-endian")
        print(
            f"    4. Group into samples of 6 values: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z"
        )
        print(f"    5. Scale: ACC *= {ACC_SCALE}, GYRO *= {GYRO_SCALE}")

    elif best_name == "pure_samples":
        print(f"\n  IMPLEMENTATION:")
        print(f"    1. Data section contains only samples (no overhead)")
        print(f"    2. Decode all bytes as int16 little-endian")
        print(
            f"    3. Group into samples of 6 values: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z"
        )
        print(f"    4. Scale: ACC *= {ACC_SCALE}, GYRO *= {GYRO_SCALE}")

print("\n")
