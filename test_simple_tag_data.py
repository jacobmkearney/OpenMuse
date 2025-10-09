"""
Test hypothesis: Leftover contains mini-packets with structure [TAG, data...]
or maybe [LEN, ..., TAG, data...]
"""

from MuseLSL3.decode import parse_message, TAGS, ACC_SCALE, GYRO_SCALE
import struct
import numpy as np

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:50]

print("=" * 80)
print("HYPOTHESIS: Leftover = sequence of [TAG, 36_bytes_data] chunks")
print("=" * 80)

successful_decodes = 0
total_attempts = 0

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"] or not sp["leftover"] or sp["pkt_type"] != "ACCGYRO":
            continue

        leftover = sp["leftover"]
        total_attempts += 1

        # Try to parse as sequence of [TAG, 36 bytes] chunks
        offset = 0
        chunks_found = []

        while offset < len(leftover):
            # Look for a TAG byte
            if leftover[offset] in TAGS:
                tag = leftover[offset]
                tag_name = TAGS[tag]

                # If it's ACCGYRO, expect 36 bytes after
                if tag == 0x47 and offset + 1 + 36 <= len(leftover):
                    data_bytes = leftover[offset + 1 : offset + 1 + 36]

                    # Try to decode
                    try:
                        values = np.frombuffer(data_bytes, dtype="<i2", count=18)
                        # Apply scaling
                        acc_data = values.reshape(-1, 6)[:, :3] * ACC_SCALE
                        gyro_data = values.reshape(-1, 6)[:, 3:] * GYRO_SCALE

                        # Check if reasonable (ACC in m/s^2 should be roughly -20 to 20, GYRO in rad/s roughly -10 to 10)
                        acc_reasonable = np.all((acc_data > -50) & (acc_data < 50))
                        gyro_reasonable = np.all((gyro_data > -50) & (gyro_data < 50))

                        if acc_reasonable and gyro_reasonable:
                            chunks_found.append(
                                {
                                    "offset": offset,
                                    "tag": tag_name,
                                    "data_len": 36,
                                    "valid": True,
                                }
                            )
                            offset += 1 + 36  # Move past TAG + data
                        else:
                            offset += 1  # Not valid data, try next byte
                    except:
                        offset += 1
                else:
                    # Not ACCGYRO or not enough bytes
                    offset += 1
            else:
                # Not a TAG
                offset += 1

        if chunks_found:
            successful_decodes += 1
            if msg_idx < 5:  # Show first few
                print(f"\n{'='*60}")
                print(
                    f"Message {msg_idx}, Primary ACCGYRO packet, counter={sp['pkt_n']}"
                )
                print(f"Leftover: {len(leftover)} bytes")
                print(f"Found {len(chunks_found)} ACCGYRO chunk(s) in leftover:")
                for chunk in chunks_found:
                    print(
                        f"  - At offset {chunk['offset']}: {chunk['tag']}, {chunk['data_len']} bytes"
                    )

print(f"\n\nRESULTS:")
print(
    f"Successfully decoded ACCGYRO from leftovers: {successful_decodes} / {total_attempts}"
)

# Now try a more sophisticated approach: maybe there's a sub-header
print("\n" + "=" * 80)
print("ALTERNATIVE: Look for ANY valid packet structure in leftovers")
print("=" * 80)

# Let me check if at position 0 there's often a specific pattern
first_bytes_accgyro = []

for message in messages:
    subpackets = parse_message(message)
    for sp in subpackets:
        if sp["pkt_valid"] and sp["leftover"] and sp["pkt_type"] == "ACCGYRO":
            leftover = sp["leftover"]
            if len(leftover) >= 20:
                first_bytes_accgyro.append(leftover[:20].hex())

print(f"\nFirst 20 bytes of ACCGYRO leftovers (first 10 examples):")
for i, fb in enumerate(first_bytes_accgyro[:10]):
    print(f"  {i+1}. {fb}")

# Check if there's a pattern - maybe position 0 has a secondary packet
print("\n\nChecking if leftover starts with a mini-header...")
for message in messages[:10]:
    subpackets = parse_message(message)
    for sp in subpackets:
        if sp["pkt_valid"] and sp["leftover"] and sp["pkt_type"] == "ACCGYRO":
            leftover = sp["leftover"]
            if len(leftover) >= 10:
                # Maybe structure is: [pkt_id, ...] at start?
                # Or maybe: [len, pkt_id, data...]
                byte0 = leftover[0]
                byte1 = leftover[1]

                # Check if byte 0 or 1 is a valid TAG
                if byte0 in TAGS:
                    print(f"\nByte[0] = 0x{byte0:02x} = {TAGS[byte0]}")
                if byte1 in TAGS:
                    print(f"Byte[1] = 0x{byte1:02x} = {TAGS[byte1]}")

                # Check other patterns
                print(f"  First 10 bytes: {leftover[:10].hex()}")
                break
    break

print("\n\nCONCLUSION:")
print("=" * 80)
print("Based on analysis, leftover structure is NOT simply [TAG, data, TAG, data...]")
print("Need to understand the actual structure of secondary packets")
