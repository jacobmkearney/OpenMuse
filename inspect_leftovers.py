"""
Simple inspection of leftover data to understand its structure
"""

from MuseLSL3.decode import parse_message, TAGS
import struct

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:100]

print("=" * 80)
print("EXAMINING LEFTOVER STRUCTURE")
print("=" * 80)

# Get first few examples with different primary types
examples_by_type = {}

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        pkt_type = sp["pkt_type"]
        if pkt_type not in examples_by_type:
            examples_by_type[pkt_type] = []

        if len(examples_by_type[pkt_type]) < 2:
            examples_by_type[pkt_type].append(
                {"msg_idx": msg_idx, "primary_pkt": sp, "leftover": sp["leftover"]}
            )

# Analyze each example in detail
for pkt_type, examples in examples_by_type.items():
    print(f"\n{'='*80}")
    print(f"PRIMARY PACKET TYPE: {pkt_type}")
    print(f"{'='*80}")

    for ex_idx, ex in enumerate(examples):
        leftover = ex["leftover"]
        primary = ex["primary_pkt"]

        print(f"\nExample {ex_idx+1} (Message {ex['msg_idx']}):")
        print(
            f"  Primary packet: counter={primary['pkt_n']}, time={primary['pkt_time']:.3f}s"
        )
        print(f"  Leftover size: {len(leftover)} bytes")
        print(f"  Leftover hex (first 80 bytes): {leftover[:80].hex()}")

        # Try interpretation 1: Starts with packet length
        if len(leftover) > 0:
            byte_0 = leftover[0]
            print(f"\n  Byte[0] = {byte_0} (0x{byte_0:02x})")
            print(f"    Could be length? {byte_0} vs actual {len(leftover)}")

            # If byte_0 looks like a length, try parsing as packet
            if 14 <= byte_0 <= len(leftover):
                print(
                    f"    ✓ Byte[0] could be packet length (need >= 14, have {len(leftover)})"
                )

                # Try to parse first potential packet
                if len(leftover) >= 14:
                    print(f"\n  Attempting to parse as packet with header:")
                    pkt_len = leftover[0]
                    pkt_n = leftover[1]
                    pkt_time_ms = struct.unpack_from("<I", leftover, 2)[0]
                    byte_6_8 = leftover[6:9]
                    pkt_id = leftover[9]
                    byte_10_12 = leftover[10:13]
                    byte_13 = leftover[13]

                    print(f"    pkt_len: {pkt_len}")
                    print(f"    pkt_n (counter): {pkt_n}")
                    print(f"    pkt_time: {pkt_time_ms}ms = {pkt_time_ms/1000:.3f}s")
                    print(f"    bytes[6-8]: {byte_6_8.hex()}")
                    print(
                        f"    pkt_id (byte[9]): 0x{pkt_id:02x} -> {TAGS.get(pkt_id, 'UNKNOWN TAG')}"
                    )
                    print(f"    bytes[10-12]: {byte_10_12.hex()}")
                    print(
                        f"    byte[13]: 0x{byte_13:02x} (should be 0x00 for valid packet)"
                    )

                    # Check validity
                    is_valid_tag = pkt_id in TAGS
                    is_byte_13_zero = byte_13 == 0x00
                    is_length_consistent = pkt_len == len(leftover[:pkt_len])

                    print(f"\n    Validity check:")
                    print(f"      TAG recognized: {is_valid_tag}")
                    print(f"      Byte[13] == 0x00: {is_byte_13_zero}")
                    print(f"      Length consistent: {is_length_consistent}")

                    if is_valid_tag and is_byte_13_zero:
                        print(f"      ✓✓✓ LOOKS LIKE A VALID PACKET! ✓✓✓")
                        print(
                            f"      Data section would be bytes[14:{pkt_len}] = {pkt_len-14} bytes"
                        )

                        # Try to parse remaining data
                        if pkt_len < len(leftover):
                            remaining = leftover[pkt_len:]
                            print(
                                f"\n      Remaining after first packet: {len(remaining)} bytes"
                            )
                            print(
                                f"      Next byte: 0x{remaining[0]:02x} ({remaining[0]})"
                            )

                            # Try to parse second packet
                            if len(remaining) >= 14 and 14 <= remaining[0] <= len(
                                remaining
                            ):
                                pkt2_len = remaining[0]
                                pkt2_id = remaining[9]
                                pkt2_byte13 = remaining[13]
                                print(
                                    f"      Could have 2nd packet: len={pkt2_len}, tag=0x{pkt2_id:02x} ({TAGS.get(pkt2_id, 'UNKNOWN')}), byte[13]=0x{pkt2_byte13:02x}"
                                )
                    else:
                        print(f"      ✗ NOT a valid packet structure")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
