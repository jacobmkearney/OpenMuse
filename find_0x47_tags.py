"""
Find where 0x47 (ACCGYRO tag) appears and examine context around it
"""

from MuseLSL3.decode import parse_message, TAGS, decode_accgyro
import struct

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:100]

print("=" * 80)
print("FINDING 0x47 TAGS IN LEFTOVERS AND EXAMINING CONTEXT")
print("=" * 80)

count_found = 0

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp_idx, sp in enumerate(subpackets):
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        leftover = sp["leftover"]

        # Find all 0x47 positions
        for pos in range(len(leftover)):
            if leftover[pos] == 0x47:
                count_found += 1

                if count_found <= 5:  # Show first 5 examples
                    print(f"\n{'='*80}")
                    print(
                        f"Found 0x47 at position {pos} in leftover (total {len(leftover)} bytes)"
                    )
                    print(
                        f"Message {msg_idx}, Primary packet: {sp['pkt_type']}, counter={sp['pkt_n']}"
                    )
                    print(f"{'='*80}")

                    # Show context before 0x47
                    start = max(0, pos - 20)
                    print(f"\nContext ({20 if pos >= 20 else pos} bytes BEFORE 0x47):")
                    print(f"  Hex: {leftover[start:pos].hex()}")

                    # Show the 0x47 and what follows
                    end = min(len(leftover), pos + 50)
                    print(f"\n0x47 and next {end-pos-1} bytes:")
                    print(f"  Hex: {leftover[pos:end].hex()}")

                    # Try to parse what comes IMMEDIATELY after 0x47 as ACCGYRO data
                    if pos + 1 + 36 <= len(leftover):
                        print(
                            f"\n  Hypothesis 1: 0x47 directly followed by 36 bytes of ACCGYRO data"
                        )
                        data_section = leftover[pos + 1 : pos + 1 + 36]
                        print(f"    Data: {data_section.hex()}")

                        # Decode as 18 int16 values (3 samples x 6 channels)
                        try:
                            values = struct.unpack("<18h", data_section)
                            print(f"    As int16: {values}")
                            print(
                                f"    Sample 1 (ACC): ({values[0]}, {values[1]}, {values[2]})"
                            )
                            print(
                                f"    Sample 1 (GYRO): ({values[3]}, {values[4]}, {values[5]})"
                            )

                            # Check if values are reasonable for ACCGYRO
                            # Typical ACC range: -32768 to 32767 (raw)
                            # Typical GYRO range: similar
                            reasonable = all(-32768 <= v <= 32767 for v in values)
                            if reasonable:
                                print(f"    ✓ Values look reasonable for sensor data")
                        except:
                            print(f"    ✗ Could not decode")

                    # Check if 0x47 has a header BEFORE it
                    if pos >= 14:
                        print(f"\n  Hypothesis 2: 14-byte header BEFORE 0x47")
                        header_start = pos - 14
                        potential_header = leftover[header_start:pos]
                        print(f"    Header bytes: {potential_header.hex()}")

                        pkt_len = potential_header[0]
                        pkt_n = potential_header[1]
                        pkt_time_ms = struct.unpack_from("<I", potential_header, 2)[0]
                        byte_6_8 = potential_header[6:9]
                        pkt_id = potential_header[9]
                        byte_10_12 = potential_header[10:13]
                        byte_13 = potential_header[13]

                        print(
                            f"    Would indicate: len={pkt_len}, counter={pkt_n}, time={pkt_time_ms}ms"
                        )
                        print(
                            f"    tag at byte[9]: 0x{pkt_id:02x} -> {TAGS.get(pkt_id, 'UNKNOWN')}"
                        )
                        print(f"    byte[13]: 0x{byte_13:02x} (need 0x00)")

                        if pkt_id == 0x47 and byte_13 == 0x00:
                            print(f"    ✓✓✓ VALID HEADER FOR ACCGYRO!")
                            # Try to find where this packet should start
                            # It should start at header_start and be pkt_len bytes long
                            if pkt_len >= 14:
                                packet_start = header_start
                                packet_end = header_start + pkt_len
                                print(
                                    f"    Packet should span bytes[{packet_start}:{packet_end}]"
                                )

                                if packet_end <= len(leftover):
                                    print(f"    ✓ Packet fits in leftover")
                                    data_len = pkt_len - 14
                                    print(
                                        f"    Data section: {data_len} bytes (need 36 for 3 ACC GYRO samples)"
                                    )
                        else:
                            print(f"    ✗ Not a valid ACCGYRO header")

                    # Check bytes immediately BEFORE 0x47 for patterns
                    if pos >= 4:
                        print(f"\n  Bytes immediately before 0x47:")
                        for i in range(min(4, pos), 0, -1):
                            byte_val = leftover[pos - i]
                            print(f"    [-{i}]: 0x{byte_val:02x} ({byte_val})")

print(f"\n\nTotal 0x47 tags found in leftovers: {count_found}")
