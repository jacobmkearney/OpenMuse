"""
CRITICAL FINDING: Byte[0] of leftover can be a TAG (0x12 = EEG8)
Let's systematically check if leftovers start with TAG bytes
"""

from MuseLSL3.decode import parse_message, TAGS
import struct

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:200]

print("=" * 80)
print("CHECKING: Does leftover start with a TAG byte?")
print("=" * 80)

tag_at_start = 0
total_leftovers = 0

leftover_examples = []

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        leftover = sp["leftover"]
        total_leftovers += 1

        byte0 = leftover[0]
        if byte0 in TAGS:
            tag_at_start += 1
            if len(leftover_examples) < 20:
                leftover_examples.append(
                    {
                        "msg_idx": msg_idx,
                        "primary_type": sp["pkt_type"],
                        "primary_counter": sp["pkt_n"],
                        "leftover_size": len(leftover),
                        "first_tag": TAGS[byte0],
                        "first_tag_byte": byte0,
                        "leftover": leftover,
                    }
                )

print(
    f"\nLeftovers starting with valid TAG: {tag_at_start} / {total_leftovers} ({100*tag_at_start/total_leftovers:.1f}%)"
)

if tag_at_start > 0:
    print(f"\n✓✓✓ Most leftovers START with a TAG byte! ✓✓✓")
    print("\nTrying to parse these as packets...")
    print("=" * 80)

    # Now try to parse these leftovers as packets
    for i, ex in enumerate(leftover_examples[:10]):
        print(f"\n{'='*60}")
        print(
            f"Example {i+1}: Message {ex['msg_idx']}, Primary={ex['primary_type']}, counter={ex['primary_counter']}"
        )
        print(f"Leftover: {ex['leftover_size']} bytes, First tag: {ex['first_tag']}")
        print(f"Hex (first 40): {ex['leftover'][:40].hex()}")

        leftover = ex["leftover"]

        # Test different header structures:
        print(f"\nTesting packet structures:")

        # Structure 1: [TAG, data...] - no header
        print(f"  1. [TAG, data...] (no header)")
        print(
            f"     TAG at [0]: 0x{leftover[0]:02x} = {TAGS.get(leftover[0], 'unknown')}"
        )
        if leftover[0] == 0x12:  # EEG8 needs specific data length
            print(f"     EEG8 typically has data, can't verify without knowing length")

        # Structure 2: [LEN, TAG, data...]
        print(f"  2. [LEN, TAG, data...]")
        pkt_len = leftover[0]
        if len(leftover) >= 2:
            potential_tag = leftover[1]
            print(
                f"     Len: {pkt_len}, TAG at [1]: 0x{potential_tag:02x} = {TAGS.get(potential_tag, 'unknown')}"
            )
            if potential_tag in TAGS and 2 <= pkt_len <= len(leftover):
                print(
                    f"     ✓ Could be: packet of {pkt_len} bytes with {TAGS[potential_tag]}"
                )

        # Structure 3: [TAG, LEN, data...]
        print(f"  3. [TAG, LEN, data...]")
        if len(leftover) >= 2:
            tag = leftover[0]
            pkt_len = leftover[1]
            if tag in TAGS:
                print(f"     TAG: {TAGS[tag]}, Len: {pkt_len}")
                if 2 <= pkt_len <= len(leftover) - 2:
                    print(
                        f"     ✓ Could be: {TAGS[tag]} packet with {pkt_len} bytes of data"
                    )

        # Structure 4: Standard 14-byte header starting with LEN
        print(
            f"  4. Full 14-byte header [LEN, counter, time(4), unk(3), TAG, unk(3), 0x00, data...]"
        )
        if len(leftover) >= 14:
            pkt_len = leftover[0]
            pkt_n = leftover[1]
            pkt_time_ms = struct.unpack_from("<I", leftover, 2)[0]
            pkt_id = leftover[9]
            byte_13 = leftover[13]

            print(f"     Len: {pkt_len}, counter: {pkt_n}, time: {pkt_time_ms}ms")
            print(f"     TAG at [9]: 0x{pkt_id:02x} = {TAGS.get(pkt_id, 'unknown')}")
            print(f"     Byte[13]: 0x{byte_13:02x} (need 0x00)")

            if (
                pkt_id in TAGS
                and byte_13 == 0x00
                and pkt_len >= 14
                and pkt_len <= len(leftover)
            ):
                print(f"     ✓✓✓ VALID FULL PACKET HEADER! ✓✓✓")
                print(f"     Type: {TAGS[pkt_id]}, Data bytes: {pkt_len - 14}")

                # Try to parse rest of leftover
                if pkt_len < len(leftover):
                    remaining = leftover[pkt_len:]
                    print(f"     Remaining: {len(remaining)} bytes")
                    if len(remaining) >= 14 and remaining[0] >= 14:
                        print(f"     Next packet might start at offset {pkt_len}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
