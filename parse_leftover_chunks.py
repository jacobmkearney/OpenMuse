"""
Parse leftovers with structure: [TAG, LEN, data...] repeated
"""

from MuseLSL3.decode import parse_message, TAGS
import struct

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:50]

print("=" * 80)
print("PARSING LEFTOVERS AS: [TAG, LEN, data...] chunks")
print("=" * 80)


def parse_leftover_chunks(leftover_data):
    """Parse leftover as sequence of [TAG, LEN, data...] chunks"""
    chunks = []
    offset = 0

    while offset < len(leftover_data):
        if offset + 2 > len(leftover_data):
            break

        tag_byte = leftover_data[offset]

        # Check if it's a valid tag
        if tag_byte not in TAGS:
            # Not a tag, might be leftover noise or different structure
            break

        data_len = leftover_data[offset + 1]

        # Check if we have enough bytes
        if offset + 2 + data_len > len(leftover_data):
            break

        # Extract chunk
        data = leftover_data[offset + 2 : offset + 2 + data_len]

        chunks.append(
            {
                "offset": offset,
                "tag": TAGS[tag_byte],
                "tag_byte": tag_byte,
                "data_len": data_len,
                "data": data,
                "total_chunk_size": 2 + data_len,
            }
        )

        offset += 2 + data_len

    return chunks, offset


# Parse all leftovers
successful_parses = 0
total_attempts = 0
all_chunks = []

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"] or not sp["leftover"]:
            continue

        leftover = sp["leftover"]
        total_attempts += 1

        chunks, bytes_consumed = parse_leftover_chunks(leftover)

        if chunks and bytes_consumed == len(leftover):
            successful_parses += 1
            all_chunks.extend(chunks)

            if msg_idx < 10:  # Show first 10
                print(f"\n{'='*60}")
                print(
                    f"Message {msg_idx}, Primary: {sp['pkt_type']}, counter={sp['pkt_n']}"
                )
                print(f"Leftover: {len(leftover)} bytes")
                print(
                    f"Parsed {len(chunks)} chunk(s), {bytes_consumed} bytes consumed:"
                )
                for i, chunk in enumerate(chunks):
                    print(
                        f"  {i+1}. {chunk['tag']} (0x{chunk['tag_byte']:02x}): {chunk['data_len']} bytes"
                    )
                    if chunk["tag"] == "ACCGYRO":
                        print(f"     ✓ FOUND ACCGYRO DATA!")
                    # Show first 20 bytes of data
                    print(f"     Data preview: {chunk['data'][:20].hex()}...")

print(f"\n\n{'='*80}")
print("RESULTS")
print("=" * 80)
print(
    f"Successfully parsed: {successful_parses} / {total_attempts} ({100*successful_parses/total_attempts:.1f}%)"
)

# Count chunk types
from collections import Counter

chunk_types = Counter(c["tag"] for c in all_chunks)
print(f"\nChunk types found:")
for tag, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
    print(f"  {tag}: {count}")

# Check ACCGYRO data lengths
accgyro_chunks = [c for c in all_chunks if c["tag"] == "ACCGYRO"]
if accgyro_chunks:
    accgyro_lens = Counter(c["data_len"] for c in accgyro_chunks)
    print(f"\nACCGYRO data lengths:")
    for length, count in sorted(accgyro_lens.items()):
        print(f"  {length} bytes: {count} times")
        if length == 36:
            print(f"    ✓ This is 3 samples × 6 channels × 2 bytes = 36 bytes!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if successful_parses > total_attempts * 0.8:
    print("✓✓✓ SUCCESS! Leftover structure is: [TAG, LEN, data...] repeated")
    print("✓ TAG: 1 byte identifying the data type (0x36=Optics16, 0x47=ACCGYRO, etc.)")
    print("✓ LEN: 1 byte specifying data length")
    print("✓ DATA: LEN bytes of sensor data")
else:
    print("✗ This structure doesn't work for all cases")
    print(f"  Only {successful_parses}/{total_attempts} parsed successfully")
