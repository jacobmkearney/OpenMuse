"""
validate_GYROACC4.py

Find the CONSISTENT ACCGYRO decoding structure across all packets.

Key insight: The manufacturer uses ONE consistent encoding scheme, not random variations.
We should find a pattern that works for ALL packets, not different strategies for different packets.

Validation criteria (no range checking):
- Structure must be consistent across packets
- Data must decode to complete samples (no partial samples)
- Overhead structure (if any) should be identifiable and consistent
- Mathematical alignment should be perfect

Strategy:
1. Analyze the data section structure systematically
2. Look for consistent patterns in overhead bytes
3. Find the ONE decoding rule that works for all packets
"""

import struct
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter


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


def validate_packet_basic(pkt: bytes) -> bool:
    """Basic validation: length, type, byte 13."""
    if len(pkt) < 14:
        return False
    if pkt[0] != len(pkt):
        return False
    if pkt[13] != 0:
        return False

    _, pkt_type = extract_pkt_id(pkt)
    return pkt_type == "ACCGYRO"


# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================


def analyze_data_structure(data: bytes) -> Dict:
    """
    Analyze the structure of the data section without making assumptions.
    Focus on:
    - Length and divisibility
    - Byte patterns
    - Potential alignment markers
    """
    result = {
        "length": len(data),
        "divisible_by_12": len(data) % 12 == 0,
        "first_8_bytes": data[:8].hex() if len(data) >= 8 else data.hex(),
        "last_8_bytes": data[-8:].hex() if len(data) >= 8 else data.hex(),
    }

    # Check divisibility by 12 with different overhead assumptions
    for overhead in range(20):  # Check first 20 bytes as potential overhead
        remaining = len(data) - overhead
        if remaining > 0 and remaining % 12 == 0:
            result[f"overhead_{overhead}_viable"] = True
            result[f"overhead_{overhead}_samples"] = remaining // 12

    # Look for common tag bytes
    result["tag_0x47_positions"] = [i for i, b in enumerate(data) if b == 0x47]
    result["tag_0xF4_positions"] = [i for i, b in enumerate(data) if b == 0xF4]

    # Byte frequency analysis (first 20 bytes)
    prefix = data[: min(20, len(data))]
    result["prefix_bytes"] = [b for b in prefix]

    return result


def find_consistent_pattern(all_analyses: List[Dict]) -> Dict:
    """
    Find the pattern that is CONSISTENT across all packets.
    """
    print("\n" + "=" * 80)
    print("SEARCHING FOR CONSISTENT PATTERN")
    print("=" * 80)

    total_packets = len(all_analyses)
    print(f"\nAnalyzing {total_packets} packets for consistency...\n")

    # Check 1: Pure samples (no overhead)
    pure_viable = sum(1 for a in all_analyses if a["divisible_by_12"])
    pure_pct = 100 * pure_viable / total_packets
    print(
        f"Pure samples (data % 12 == 0): {pure_viable}/{total_packets} ({pure_pct:.1f}%)"
    )

    # Check 2: Consistent overhead
    overhead_consistency = {}
    for overhead in range(20):
        key = f"overhead_{overhead}_viable"
        count = sum(1 for a in all_analyses if a.get(key, False))
        if count > 0:
            pct = 100 * count / total_packets
            overhead_consistency[overhead] = {"count": count, "pct": pct}
            if pct >= 90:  # If works for 90%+ of packets
                print(
                    f"Overhead {overhead} bytes: {count}/{total_packets} ({pct:.1f}%) ✓ HIGHLY CONSISTENT"
                )
            elif pct >= 50:
                print(
                    f"Overhead {overhead} bytes: {count}/{total_packets} ({pct:.1f}%)"
                )

    # Check 3: Tag presence consistency
    has_0x47 = sum(1 for a in all_analyses if len(a["tag_0x47_positions"]) > 0)
    has_0xF4 = sum(1 for a in all_analyses if len(a["tag_0xF4_positions"]) > 0)
    has_both = sum(
        1
        for a in all_analyses
        if len(a["tag_0x47_positions"]) > 0 and len(a["tag_0xF4_positions"]) > 0
    )
    has_neither = sum(
        1
        for a in all_analyses
        if len(a["tag_0x47_positions"]) == 0 and len(a["tag_0xF4_positions"]) == 0
    )

    print(f"\nTag presence:")
    print(
        f"  0x47 present: {has_0x47}/{total_packets} ({100*has_0x47/total_packets:.1f}%)"
    )
    print(
        f"  0xF4 present: {has_0xF4}/{total_packets} ({100*has_0xF4/total_packets:.1f}%)"
    )
    print(
        f"  Both tags: {has_both}/{total_packets} ({100*has_both/total_packets:.1f}%)"
    )
    print(
        f"  Neither tag: {has_neither}/{total_packets} ({100*has_neither/total_packets:.1f}%)"
    )

    # Check 4: First tag position consistency
    if has_0x47 > total_packets * 0.5:
        first_0x47_positions = [
            a["tag_0x47_positions"][0]
            for a in all_analyses
            if len(a["tag_0x47_positions"]) > 0
        ]
        position_counts = Counter(first_0x47_positions)
        print(f"\n  0x47 first occurrence positions (top 5):")
        for pos, count in position_counts.most_common(5):
            print(
                f"    Position {pos}: {count} packets ({100*count/has_0x47:.1f}% of packets with tag)"
            )

    if has_0xF4 > total_packets * 0.5:
        first_0xF4_positions = [
            a["tag_0xF4_positions"][0]
            for a in all_analyses
            if len(a["tag_0xF4_positions"]) > 0
        ]
        position_counts = Counter(first_0xF4_positions)
        print(f"\n  0xF4 first occurrence positions (top 5):")
        for pos, count in position_counts.most_common(5):
            print(
                f"    Position {pos}: {count} packets ({100*count/has_0xF4:.1f}% of packets with tag)"
            )

    # Check 5: Prefix pattern analysis
    print(f"\nFirst byte analysis:")
    first_bytes = [
        a["prefix_bytes"][0] if len(a["prefix_bytes"]) > 0 else None
        for a in all_analyses
    ]
    first_byte_counts = Counter(first_bytes)
    for byte_val, count in first_byte_counts.most_common(10):
        if byte_val is not None:
            print(
                f"  0x{byte_val:02X}: {count} packets ({100*count/total_packets:.1f}%)"
            )

    # Check 6: Length patterns
    lengths = [a["length"] for a in all_analyses]
    length_counts = Counter(lengths)
    print(f"\nData length distribution (top 10):")
    for length, count in length_counts.most_common(10):
        print(f"  {length} bytes: {count} packets ({100*count/total_packets:.1f}%)")

    # New analysis: Check if 0xF4 at position 5 is a reliable indicator
    print(f"\n0xF4 at position 5 analysis:")
    f4_at_5 = sum(1 for a in all_analyses if 5 in a["tag_0xF4_positions"])
    print(
        f"  0xF4 at position 5: {f4_at_5}/{total_packets} ({100*f4_at_5/total_packets:.1f}%)"
    )

    if f4_at_5 > 0:
        # For packets with 0xF4 at position 5, check if data after position 10 is divisible by 12
        f4_at_5_packets = [a for a in all_analyses if 5 in a["tag_0xF4_positions"]]
        # Assuming tag (1 byte) + 4 byte header = 5 bytes, start at position 10
        after_tag_divisible = sum(
            1 for a in f4_at_5_packets if (a["length"] - 10) % 12 == 0
        )
        print(
            f"  After skipping 10 bytes (tag+header), divisible by 12: {after_tag_divisible}/{f4_at_5} ({100*after_tag_divisible/f4_at_5:.1f}%)"
        )

        # Try different skip amounts
        for skip in [9, 10, 11, 12, 13, 14]:
            viable = sum(1 for a in f4_at_5_packets if (a["length"] - skip) % 12 == 0)
            if viable > f4_at_5 * 0.8:  # If works for 80%+
                print(
                    f"  After skipping {skip} bytes, divisible by 12: {viable}/{f4_at_5} ({100*viable/f4_at_5:.1f}%) ✓"
                )

    # Check 7: Relationship between length and overhead
    print(f"\nLength modulo analysis:")
    for mod in [12, 8, 16]:
        remainders = [a["length"] % mod for a in all_analyses]
        remainder_counts = Counter(remainders)
        print(f"  Length % {mod}: {dict(remainder_counts)}")

    return {
        "overhead_consistency": overhead_consistency,
        "tag_presence": {
            "0x47": has_0x47,
            "0xF4": has_0xF4,
            "both": has_both,
            "neither": has_neither,
        },
    }


def analyze_remainder_patterns(all_analyses: List[Dict]) -> None:
    """
    For packets that don't divide evenly by 12, analyze what the remainder consists of.
    This might reveal the overhead structure.
    """
    print("\n" + "=" * 80)
    print("ANALYZING REMAINDER PATTERNS")
    print("=" * 80)

    # Group packets by remainder when divided by 12
    by_remainder = {}
    for a in all_analyses:
        remainder = a["length"] % 12
        if remainder not in by_remainder:
            by_remainder[remainder] = []
        by_remainder[remainder].append(a)

    print(f"\nPackets grouped by (data_length % 12):")
    for remainder in sorted(by_remainder.keys()):
        count = len(by_remainder[remainder])
        pct = 100 * count / len(all_analyses)
        print(f"\n  Remainder {remainder}: {count} packets ({pct:.1f}%)")

        if count > 0 and remainder > 0:
            # Look at first N bytes where N = remainder
            prefix_patterns = Counter(
                [a["first_8_bytes"][: remainder * 2] for a in by_remainder[remainder]]
            )
            print(f"    Common prefix patterns (first {remainder} bytes):")
            for pattern, pcount in prefix_patterns.most_common(3):
                print(f"      {pattern}: {pcount} packets")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def collect_packets(filepath: str, max_packets: int = 200) -> List[bytes]:
    """Collect ACCGYRO packets from a file."""
    packets = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        try:
            ts, uuid, payload_hex = line.strip().split("\t", 2)
            payload = bytes.fromhex(payload_hex.strip())

            offset = 0
            while offset < len(payload) and len(packets) < max_packets:
                if offset + 14 > len(payload):
                    break

                declared_len = payload[offset]
                if offset + declared_len > len(payload):
                    break

                pkt = payload[offset : offset + declared_len]

                if validate_packet_basic(pkt):
                    packets.append(pkt)

                offset += declared_len

        except Exception as e:
            continue

    return packets


# ============================================================================
# SCRIPT
# ============================================================================

data_dir = "./data_raw"
files = sorted(os.listdir(data_dir))

print("=" * 80)
print("SYSTEMATIC ACCGYRO STRUCTURE ANALYSIS")
print("=" * 80)
print("\nGoal: Find the ONE consistent decoding pattern used by the device")

# Collect packets from multiple files to ensure we capture all variations
test_files = ["data_p20.txt", "data_p1034.txt", "data_p1041.txt", "data_p1043.txt"]
all_packets = []

for test_file in test_files:
    print(f"\nCollecting from {test_file}...")
    filepath = os.path.join(data_dir, test_file)
    packets = collect_packets(filepath, max_packets=50)
    all_packets.extend(packets)
    print(f"  Collected {len(packets)} ACCGYRO packets")

print(f"\nTotal packets collected: {len(all_packets)}")

# Analyze structure of all packets
all_analyses = []
for pkt in all_packets:
    data = pkt[14:]  # Skip header
    analysis = analyze_data_structure(data)
    all_analyses.append(analysis)

# Find consistent pattern
consistency_results = find_consistent_pattern(all_analyses)

# Analyze remainder patterns
analyze_remainder_patterns(all_analyses)

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find the most consistent overhead value
best_overhead = None
best_consistency = 0

for overhead, stats in consistency_results["overhead_consistency"].items():
    if stats["pct"] > best_consistency:
        best_consistency = stats["pct"]
        best_overhead = overhead

if best_consistency >= 90:
    print(f"\n✓ CONSISTENT PATTERN FOUND:")
    print(f"  Skip first {best_overhead} bytes, then decode as int16 samples")
    print(f"  Consistency: {best_consistency:.1f}% of packets")
    print(f"\n  DECODER IMPLEMENTATION:")
    print(f"    1. Extract data section (packet[14:])")
    print(f"    2. Skip first {best_overhead} bytes")
    print(f"    3. Decode remaining bytes as int16 little-endian")
    print(
        f"    4. Group into 6-value samples: ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z"
    )

    # Show what the overhead bytes might represent
    if best_overhead > 0:
        print(f"\n  The {best_overhead}-byte overhead might contain:")
        overhead_examples = [
            a["first_8_bytes"][: best_overhead * 2] for a in all_analyses[:5]
        ]
        print(f"    Examples: {overhead_examples[:3]}")

        # Check if overhead has patterns
        all_overhead = [
            a["first_8_bytes"][: best_overhead * 2]
            for a in all_analyses
            if len(a["first_8_bytes"]) >= best_overhead * 2
        ]
        unique_overheads = len(set(all_overhead))
        print(f"    Unique patterns: {unique_overheads}/{len(all_overhead)}")

        if unique_overheads == 1:
            print(f"    ✓ Overhead is CONSTANT: {all_overhead[0]}")
        elif unique_overheads < len(all_overhead) * 0.1:
            print(f"    ✓ Overhead has few patterns (likely metadata)")
        else:
            print(f"    ! Overhead is highly variable (might contain data)")
else:
    print(f"\n⚠ NO HIGHLY CONSISTENT PATTERN FOUND")
    print(
        f"  Best candidate: {best_overhead} bytes overhead ({best_consistency:.1f}% consistency)"
    )
    print(f"\n  This suggests:")
    print(
        f"    - The data structure might be more complex than simple overhead + samples"
    )
    print(f"    - There might be multiple encoding formats used")
    print(f"    - Further investigation needed into tag-based structure")

print("\n")
