"""
Verify the pattern: look at ALL 0x47 occurrences and the 4 bytes before them
"""

from MuseLSL3.decode import parse_message
import struct
from collections import Counter

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()

print("=" * 80)
print("PATTERN ANALYSIS: What comes BEFORE 0x47 tags?")
print("=" * 80)

patterns_before = []
all_positions = []

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        leftover = sp["leftover"]

        # Find all 0x47 positions
        for pos in range(len(leftover)):
            if leftover[pos] == 0x47:
                all_positions.append(pos)

                # Get 4 bytes before (if available)
                if pos >= 4:
                    pattern = leftover[pos - 4 : pos]
                    patterns_before.append(
                        {
                            "pattern": pattern.hex(),
                            "bytes": tuple(pattern),
                            "msg_idx": msg_idx,
                            "pos": pos,
                            "primary_type": sp["pkt_type"],
                            "leftover_size": len(leftover),
                        }
                    )

print(f"\nFound {len(all_positions)} occurrences of 0x47 in leftovers")
print(
    f"Positions: min={min(all_positions)}, max={max(all_positions)}, common={Counter(all_positions).most_common(5)}"
)

print(f"\n{len(patterns_before)} have at least 4 bytes before them")

# Analyze the 4-byte patterns
print("\n" + "-" * 80)
print("4-BYTE PATTERNS BEFORE 0x47:")
print("-" * 80)

# Count unique patterns
pattern_counts = Counter(p["pattern"] for p in patterns_before)
print(f"\nUnique patterns: {len(pattern_counts)}")
print("\nMost common patterns:")
for pattern, count in pattern_counts.most_common(10):
    print(f"  {pattern}: {count} times")

# Analyze byte-by-byte
print("\n" + "-" * 80)
print("BYTE-BY-BYTE ANALYSIS:")
print("-" * 80)

for byte_offset in range(4):
    byte_values = [p["bytes"][byte_offset] for p in patterns_before]
    byte_counts = Counter(byte_values)
    print(f"\nByte at position [-{4-byte_offset}] before 0x47:")
    print(f"  Unique values: {len(byte_counts)}")
    print(f"  Range: 0x{min(byte_values):02x} - 0x{max(byte_values):02x}")
    if len(byte_counts) <= 10:
        print(f"  Values: {sorted(byte_counts.keys())}")
    print(f"  Most common: {byte_counts.most_common(5)}")

    # Check if it's constant
    if len(byte_counts) == 1:
        print(f"  ✓ CONSTANT: always 0x{min(byte_values):02x}")

# Try to interpret the 4 bytes as structured data
print("\n" + "-" * 80)
print("INTERPRETATION ATTEMPTS:")
print("-" * 80)

print("\nHypothesis: 4 bytes are [?, ?, ?, 0x01] where last byte is constant")
last_byte_counts = Counter(p["bytes"][3] for p in patterns_before)
print(f"Last byte values: {last_byte_counts}")

if len(last_byte_counts) == 1 and list(last_byte_counts.keys())[0] == 0x01:
    print("✓ Last byte is ALWAYS 0x01")

    # Look at first 2 bytes as potential uint16
    print("\nFirst 2 bytes as uint16 (little endian):")
    first_2_as_uint16 = [
        struct.unpack("<H", bytes(p["bytes"][:2]))[0] for p in patterns_before
    ]
    print(f"  Range: {min(first_2_as_uint16)} - {max(first_2_as_uint16)}")
    print(
        f"  Values: {sorted(set(first_2_as_uint16))[:20]} {'...' if len(set(first_2_as_uint16)) > 20 else ''}"
    )

    # Look at third byte
    third_bytes = [p["bytes"][2] for p in patterns_before]
    third_counts = Counter(third_bytes)
    print(f"\nThird byte:")
    print(f"  Range: 0x{min(third_bytes):02x} - 0x{max(third_bytes):02x}")
    print(f"  Most common: {third_counts.most_common(10)}")

# Check if position 135 is special
print("\n" + "-" * 80)
print("POSITION ANALYSIS:")
print("-" * 80)
pos_135_count = sum(1 for pos in all_positions if pos == 135)
print(
    f"Position 135: {pos_135_count} / {len(all_positions)} ({100*pos_135_count/len(all_positions):.1f}%)"
)

# Show a few full examples for reference
print("\n" + "-" * 80)
print("FULL EXAMPLES (first 3):")
print("-" * 80)
for i, p in enumerate(patterns_before[:3]):
    print(f"\nExample {i+1}:")
    print(f"  Message {p['msg_idx']}, Position {p['pos']}/{p['leftover_size']}")
    print(f"  Primary packet: {p['primary_type']}")
    print(f"  4 bytes before 0x47: {p['pattern']} = {p['bytes']}")
