"""
Detailed analysis: Do leftovers contain additional packets with full headers?
"""

import struct
from MuseLSL3.decode import parse_message, TAGS

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:100]

print("=" * 80)
print("TESTING HYPOTHESIS: Leftovers contain additional packets with 14-byte headers")
print("=" * 80)


def try_parse_leftover_as_packets(leftover_data):
    """Try to parse leftover data as if it contains full packet structures"""
    packets = []
    offset = 0

    while offset < len(leftover_data):
        if offset + 14 > len(leftover_data):
            break

        # Try to read as packet
        declared_len = leftover_data[offset]

        # Check if declared_len is reasonable
        if declared_len < 14 or offset + declared_len > len(leftover_data):
            break

        pkt = leftover_data[offset : offset + declared_len]

        # Ensure we have full packet
        if len(pkt) < 14:
            break

        # Parse header
        pkt_len = pkt[0]
        pkt_n = pkt[1]
        pkt_time_ms = struct.unpack_from("<I", pkt, 2)[0]
        pkt_unknown1 = pkt[6:9]
        pkt_id = pkt[9]
        pkt_unknown2 = pkt[10:13]
        byte_13 = pkt[13]

        # Decode ID
        freq_code = (pkt_id >> 4) & 0x0F
        type_code = pkt_id & 0x0F

        # Check if it looks valid
        is_valid = (
            pkt_len == len(pkt)
            and pkt_len >= 14
            and pkt_id in TAGS
            and byte_13 == 0x00  # Critical: byte 13 should be 0x00
        )

        packets.append(
            {
                "offset": offset,
                "len": pkt_len,
                "counter": pkt_n,
                "time_ms": pkt_time_ms,
                "id": pkt_id,
                "tag": TAGS.get(pkt_id, "Unknown"),
                "byte_13": byte_13,
                "valid": is_valid,
                "data_len": pkt_len - 14 if pkt_len >= 14 else 0,
            }
        )

        offset += declared_len

    return packets, offset


# Analyze leftovers from different packet types
test_cases = []

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp_idx, sp in enumerate(subpackets):
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        leftover = sp["leftover"]

        # Try to parse leftover as packets
        parsed, bytes_consumed = try_parse_leftover_as_packets(leftover)

        if parsed:
            test_cases.append(
                {
                    "msg_idx": msg_idx,
                    "sp_idx": sp_idx,
                    "primary_type": sp["pkt_type"],
                    "primary_counter": sp["pkt_n"],
                    "leftover_size": len(leftover),
                    "parsed_packets": parsed,
                    "bytes_consumed": bytes_consumed,
                    "parse_complete": bytes_consumed == len(leftover),
                }
            )

# Report findings
print(f"\nAnalyzed {len(test_cases)} packets with leftover data\n")

# Count successful parses
successful_parses = [
    tc
    for tc in test_cases
    if tc["parse_complete"] and all(p["valid"] for p in tc["parsed_packets"])
]
print(
    f"Successfully parsed as additional packets: {len(successful_parses)} / {len(test_cases)}"
)

if successful_parses:
    print("\n" + "=" * 80)
    print("SUCCESS! Leftovers DO contain additional packets with full 14-byte headers")
    print("=" * 80)

    # Show examples
    print("\nFirst 5 examples:")
    for i, tc in enumerate(successful_parses[:5]):
        print(f"\n{i+1}. Message {tc['msg_idx']}, Primary packet: {tc['primary_type']}")
        print(f"   Leftover size: {tc['leftover_size']} bytes")
        print(f"   Found {len(tc['parsed_packets'])} additional packet(s):")
        for p in tc["parsed_packets"]:
            print(
                f"     - {p['tag']} (0x{p['id']:02x}): {p['len']} bytes, {p['data_len']} data bytes"
            )
            print(
                f"       Counter: {p['counter']}, Time: {p['time_ms']}ms, Byte[13]=0x{p['byte_13']:02x}"
            )

    # Statistics
    print("\n" + "-" * 80)
    print("Statistics on additional packets:")
    from collections import Counter

    all_additional_packets = []
    for tc in successful_parses:
        all_additional_packets.extend(tc["parsed_packets"])

    tag_counts = Counter(p["tag"] for p in all_additional_packets)
    print(f"\nPacket types found in leftovers:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    # Check if ACCGYRO (0x47) follows specific packet types
    print("\n" + "-" * 80)
    print("Which primary packet types have ACCGYRO in their leftovers?")
    accgyro_after = Counter()
    for tc in successful_parses:
        for p in tc["parsed_packets"]:
            if p["tag"] == "ACCGYRO":
                accgyro_after[tc["primary_type"]] += 1

    if accgyro_after:
        print("ACCGYRO found after:")
        for ptype, count in sorted(accgyro_after.items(), key=lambda x: -x[1]):
            print(f"  {ptype}: {count} times")

else:
    print("\nLeftovers do NOT appear to have full packet headers")
    print("\nTesting alternative: TAG byte directly followed by data (no header)")
    print("-" * 80)

    # Count how many times we find valid tags at expected positions
    tag_found = 0
    for tc in test_cases[:10]:
        leftover = test_cases[0]["leftover"] if tc["msg_idx"] == 0 else None
        # Would need leftover bytes here - showing structure only

print("\n\nCONCLUSION")
print("=" * 80)
if successful_parses:
    print("✓ Leftovers contain ADDITIONAL FULL PACKETS with 14-byte headers")
    print("✓ Structure: [len, counter, time(4), unk1(3), TAG, unk2(3), 0x00, data...]")
    print("✓ These should be parsed recursively just like the main payload")
else:
    print("✗ Leftovers do NOT follow the full packet structure")
    print("  Need alternative parsing strategy")
