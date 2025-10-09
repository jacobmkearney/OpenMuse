"""Analyze leftover bytes to understand structure and find ACCGYRO data (0x47)"""

import struct
from collections import defaultdict
from MuseLSL3.decode import parse_message

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:500]  # Analyze first 500 messages

print("=" * 80)
print("ANALYZING LEFTOVER DATA STRUCTURE")
print("=" * 80)

# Collect statistics
leftover_stats = defaultdict(lambda: {"count": 0, "sizes": [], "examples": []})
tag_positions = defaultdict(list)

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp_idx, sp in enumerate(subpackets):
        if not sp["pkt_valid"]:
            continue

        leftover = sp["leftover"]
        if leftover and len(leftover) > 0:
            pkt_type = sp["pkt_type"]
            leftover_stats[pkt_type]["count"] += 1
            leftover_stats[pkt_type]["sizes"].append(len(leftover))

            # Store first few examples
            if len(leftover_stats[pkt_type]["examples"]) < 3:
                leftover_stats[pkt_type]["examples"].append(
                    {
                        "msg_idx": msg_idx,
                        "sp_idx": sp_idx,
                        "size": len(leftover),
                        "hex": leftover[:60].hex(),
                        "full_leftover": leftover,
                    }
                )

            # Search for 0x47 tag (ACCGYRO) in leftover
            for i in range(len(leftover)):
                if leftover[i] == 0x47:
                    tag_positions[pkt_type].append(
                        {
                            "msg_idx": msg_idx,
                            "position": i,
                            "leftover_size": len(leftover),
                            "context": leftover[
                                max(0, i - 4) : min(len(leftover), i + 40)
                            ].hex(),
                        }
                    )

# Print statistics
print("\n1. LEFTOVER STATISTICS BY PACKET TYPE")
print("-" * 80)
for pkt_type in sorted(leftover_stats.keys()):
    stats = leftover_stats[pkt_type]
    sizes = stats["sizes"]
    print(f"\n{pkt_type}:")
    print(f"  - Count: {stats['count']} packets with leftover data")
    print(f"  - Size range: {min(sizes)} - {max(sizes)} bytes")
    print(f"  - Average size: {sum(sizes)/len(sizes):.1f} bytes")
    print(f"  - Common sizes: {set(sizes) if len(set(sizes)) < 10 else 'varies'}")

# Analyze 0x47 tag positions
print("\n\n2. ACCGYRO TAG (0x47) OCCURRENCES IN LEFTOVERS")
print("-" * 80)
if not tag_positions:
    print("No 0x47 tags found in leftover data")
else:
    for pkt_type in sorted(tag_positions.keys()):
        occurrences = tag_positions[pkt_type]
        print(f"\n{pkt_type}: Found {len(occurrences)} occurrences of 0x47")

        # Analyze positions
        positions = [occ["position"] for occ in occurrences]
        print(f"  - Position range: {min(positions)} - {max(positions)}")
        print(f"  - Common positions: {sorted(set(positions))[:10]}")

        # Show first few examples with context
        print(f"\n  First examples with context:")
        for i, occ in enumerate(occurrences[:5]):
            print(f"\n  Example {i+1}:")
            print(f"    Position: {occ['position']} / {occ['leftover_size']} bytes")
            print(f"    Context (hex): {occ['context']}")

            # Try to decode what comes after 0x47
            context_bytes = bytes.fromhex(occ["context"])
            tag_offset = min(4, occ["position"])  # Where 0x47 is in context

            if tag_offset + 36 < len(context_bytes):
                # Try to interpret as ACCGYRO data (36 bytes = 18 int16 values)
                data_section = context_bytes[tag_offset + 1 : tag_offset + 1 + 36]
                print(f"    Next 36 bytes: {data_section.hex()}")

                # Decode as int16 values
                try:
                    values = struct.unpack("<18h", data_section)
                    print(f"    As int16 values: {values[:6]}... (showing first 6)")
                except:
                    print(f"    Could not decode as int16")

# Detailed analysis of leftover structure
print("\n\n3. DETAILED LEFTOVER STRUCTURE ANALYSIS")
print("-" * 80)

for pkt_type in ["Battery", "EEG4", "REF"]:
    if pkt_type not in leftover_stats:
        continue

    print(f"\n{pkt_type} - First 3 examples:")
    print("-" * 80)

    for ex_idx, ex in enumerate(leftover_stats[pkt_type]["examples"]):
        leftover = ex["full_leftover"]
        print(f"\nExample {ex_idx+1} (msg {ex['msg_idx']}, {len(leftover)} bytes):")
        print(f"  Full hex: {leftover.hex()}")

        # Look for patterns: does it start with a length byte?
        print(f"\n  First byte: 0x{leftover[0]:02x} ({leftover[0]})")
        print(f"  Could be length?: {leftover[0]} vs actual size {len(leftover)}")

        # Look for tag bytes
        print(f"\n  Scanning for known tags:")
        tags = [0x11, 0x12, 0x13, 0x34, 0x35, 0x36, 0x47, 0x98]
        for i in range(min(20, len(leftover))):
            if leftover[i] in tags:
                tag_name = {
                    0x11: "EEG4",
                    0x12: "EEG8",
                    0x13: "REF",
                    0x34: "Optics4",
                    0x35: "Optics8",
                    0x36: "Optics16",
                    0x47: "ACCGYRO",
                    0x98: "Battery",
                }
                print(
                    f"    Position {i}: 0x{leftover[i]:02x} = {tag_name.get(leftover[i], 'Unknown')}"
                )

print("\n\n4. HYPOTHESIS TESTING: IS THERE A HEADER BEFORE DATA?")
print("-" * 80)

# Test hypothesis: leftover might have structure like [TAG, data...] or [LEN, TAG, data...]
for pkt_type in ["Battery", "ACCGYRO", "EEG4"]:
    if pkt_type not in leftover_stats:
        continue

    examples = leftover_stats[pkt_type]["examples"]
    if not examples:
        continue

    print(f"\n{pkt_type}:")
    for ex in examples[:2]:
        leftover = ex["full_leftover"]
        if len(leftover) < 40:
            continue

        print(f"\n  Example (msg {ex['msg_idx']}, {len(leftover)} bytes):")
        print(f"    First 20 bytes: {leftover[:20].hex()}")

        # Test if byte 0 could be a length
        if leftover[0] <= len(leftover) and leftover[0] >= 36:
            print(f"    Byte[0]={leftover[0]} could be total length")

        # Test if there's a pattern matching main packet structure
        # Main packet: [len, counter, time(4), unk1(3), tag, unk2(3), 0x00, data...]
        if len(leftover) >= 14:
            potential_tag = leftover[9] if leftover[0] >= 14 else leftover[0]
            print(f"    If structured like main packet:")
            print(f"      Byte[9] = 0x{leftover[9]:02x} (tag position in main packet)")
            print(
                f"      Byte[13] = 0x{leftover[13]:02x} (should be 0x00 if same structure)"
            )

print("\n\nANALYSIS COMPLETE")
