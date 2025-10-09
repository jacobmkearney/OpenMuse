"""
Final systematic analysis: Parse leftover by searching for 0x47 tags
and checking if there's a valid 14-byte header starting BEFORE it
"""

from MuseLSL3.decode import parse_message, TAGS
import struct

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:100]

print("=" * 80)
print("FINDING ACCGYRO PACKETS (0x47) IN LEFTOVERS")
print("Testing if they have 14-byte headers at position [offset-13:offset+1]")
print("=" * 80)

valid_headers_found = 0
total_0x47_found = 0

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        leftover = sp["leftover"]

        # Find all 0x47 positions
        for pos in range(len(leftover)):
            if leftover[pos] == 0x47:
                total_0x47_found += 1

                # Check if there's room for a 14-byte header ending at pos
                header_start = pos - 13  # Header would be at [pos-13:pos+1]

                if header_start >= 0 and pos + 1 + 36 <= len(leftover):
                    # Try to parse header
                    potential_header = leftover[header_start : pos + 1]

                    if len(potential_header) == 14:
                        pkt_len = potential_header[0]
                        pkt_n = potential_header[1]
                        pkt_time_ms = struct.unpack_from("<I", potential_header, 2)[0]
                        byte_6_8 = potential_header[6:9]
                        pkt_id = potential_header[9]
                        byte_10_12 = potential_header[10:13]
                        byte_13 = potential_header[13]

                        # Check validity
                        is_valid = (
                            pkt_id == 0x47  # TAG is ACCGYRO
                            and byte_13 == 0x00  # Byte 13 is 0x00
                            and pkt_len >= 14  # Length is reasonable
                            and pkt_len <= len(leftover) - header_start  # Packet fits
                        )

                        if is_valid:
                            valid_headers_found += 1

                            if valid_headers_found <= 5:
                                print(f"\n{'='*60}")
                                print(f"Found valid ACCGYRO packet in leftover!")
                                print(
                                    f"Message {msg_idx}, Primary: {sp['pkt_type']}, counter={sp['pkt_n']}"
                                )
                                print(
                                    f"  Packet starts at offset {header_start} in leftover ({len(leftover)} bytes total)"
                                )
                                print(f"  Header: {potential_header.hex()}")
                                print(f"    pkt_len: {pkt_len}")
                                print(f"    pkt_n (counter): {pkt_n}")
                                print(
                                    f"    pkt_time: {pkt_time_ms}ms = {pkt_time_ms/1000:.3f}s"
                                )
                                print(f"    bytes[6-8]: {byte_6_8.hex()}")
                                print(f"    pkt_id: 0x{pkt_id:02x} = {TAGS[pkt_id]}")
                                print(f"    bytes[10-12]: {byte_10_12.hex()}")
                                print(f"    byte[13]: 0x{byte_13:02x} ✓")
                                print(f"  Data length: {pkt_len - 14} bytes")

                                # Show data
                                data_section = leftover[
                                    pos + 1 : pos + 1 + min(36, pkt_len - 14)
                                ]
                                print(
                                    f"  Data preview (first 36 bytes): {data_section.hex()}"
                                )

print(f"\n\n{'='*80}")
print("RESULTS")
print("=" * 80)
print(f"Total 0x47 tags found: {total_0x47_found}")
print(f"Valid 14-byte headers found: {valid_headers_found}")
print(
    f"Success rate: {100*valid_headers_found/total_0x47_found if total_0x47_found > 0 else 0:.1f}%"
)

if valid_headers_found > 0:
    print("\n✓✓✓ CONFIRMED! ACCGYRO data in leftovers HAS full 14-byte headers")
    print("✓ Structure: [len, counter, time(4), unk(3), 0x47, unk(3), 0x00, data...]")
    print("\nThis means leftovers contain ADDITIONAL FULL PACKETS, not just raw data!")
else:
    print("\n✗ No valid headers found. Need to investigate alternative structures.")
