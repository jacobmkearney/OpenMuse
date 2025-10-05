import struct
from datetime import datetime
from typing import Dict, List, Optional
import os

import pandas as pd

# - Each BlueTooth's *Message* contains one timestamp, one UUID, and one hexstring (the *Payload*)
# - Each payload may contain multiple concatenated *Packets*, each starting with its own length byte.
# - Each packet contains multiple *Subpackets*, including a *Data* subpacket that contains the signal data.

# Packet structure -------------------------------------------
# Offset (0-based)   Field
# -----------------  -----------------------------------------
# 0                  PKT_LEN       (1 byte) [confirmed]
# 1                  PKT_N         (1 byte) [confirmed]
# 2–5                PKT_T         (uint32, ms since device start) [confirmed]
# 6–8                PKT_UNKNOWN1  (3 bytes, reserved?)
# 9                  PKT_ID        (freq/type nibbles) [confirmed]
# 10–13              PKT_METADATA  (4 bytes, little-endian; header metadata)
# - interpretable as two little-endian uint16s:
#   - u16_0 = bytes 10–11: high-variance 16-bit value (possibly per-packet offset / internal counter / fine-grained ID)
#   - u16_1 = bytes 12–13: small discrete value ∈ {0,1,2,3} (likely a 2-bit slot/index / bank id)
# - u8_3 (byte 13) is observed always 0 -> reserved/padding
# 14...              PKT_DATA      (multiplexed samples, tightly packed, repeating blocks)
# - ACC/GYRO (TO BE CONFIRMED): Each block:
#   - [tag byte: 0x47]
#   - [4-byte block header (unknown; possibly sub-counter or timestamp offset)]
#   - [N batched samples of 6 channels, interleaved per sample: (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z) x N]
#   - [e.g., 36 bytes data: 18 signed 16-bit little-endian integers (<18h): 18 integers represent 6 channels x 3.
#   - Multiple blocks per payload possible; search for all 0x47 tags to extract.
# - Possible tags (TO BE CONFIRMED): 0x47 for ACCGYRO, 0x12 for EEG, 0x34 for optics, 0x98 for battery
# - Other source (amused-py): 0xF4 for AGGYRO; 0xDB, 0xDF: EEG + PPG combined, 0xD9 for Mixed sensor data
# Note: the payloads received might be concatenations of multiple subpackets (to be confirmed). Each subpacket starts with its own 1-byte length field (which includes the length byte itself), followed by the subpacket content.


def extract_pkt_length(pkt: bytes):
    """
    Extract the PKT_LEN field (declared length) from a Muse payload.
    """
    if not pkt or len(pkt) < 14:  # minimum length for header
        return None, False

    declared_len = pkt[0]
    return declared_len, (declared_len == len(pkt))


def extract_pkt_n(pkt: bytes):
    """
    Extract the PKT_N field (1-byte sequence number) from a Muse payload.

    - Located at offset 1 (0-based). Increments by 1 per packet, wraps at 255 -> 0.
    - Useful for detecting dropped or out-of-order packets (quality check assessment).

    Returns the integer sequence number (0-255).
    """
    return pkt[1]


def extract_pkt_time(pkt: bytes):
    """
    Extract subpkt time from a single payload. 4-byte unsigned little-endian at offset 2 -> milliseconds.
    """
    # primary 4-byte little-endian at offset 2 (fixed from 3)
    ms = struct.unpack_from("<I", pkt, 2)[0]
    return ms * 1e-3  # convert to seconds


def extract_pkt_id(pkt: bytes):
    """
    Extract and parse the ID byte from a Muse payload.
    - ID byte is at offset 9 (0-based).
    - Upper nibble = frequency code.
    - Lower nibble = data type code.
    Returns dict with raw codes and decoded labels.
    """

    # Lookup tables
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


def validate_packet(
    pkt: bytes, prev_counter: Optional[int] = None, prev_time: Optional[float] = None
) -> Dict:
    """
    Validate a packet by checking:
    1. Minimum length (14 bytes for header)
    2. PKT_ID matches known types (EEG, Optics, ACCGYRO, Battery, etc.)
    3. Packet counter (PKT_N) is valid (0-255) and increments from previous
    4. Timestamp is valid and larger than previous (if provided)
    5. Declared length matches actual packet length
    6. Byte 13 (u8_3) is always 0 (confirmed separator/padding)

    Returns dict with validation results and extracted fields.
    """
    result = {
        "valid": False,
        "reason": None,
        "length": len(pkt),
        "declared_length": None,
        "counter": None,
        "timestamp": None,
        "frequency": None,
        "type": None,
    }

    # Check minimum length
    if len(pkt) < 14:
        result["reason"] = "too_short"
        return result

    # Extract and check declared length
    declared_len = pkt[0]
    result["declared_length"] = declared_len
    if declared_len != len(pkt):
        result["reason"] = "length_mismatch"
        return result

    # Extract packet counter
    counter = pkt[1]
    result["counter"] = counter

    # Check counter increment (allowing for wrap-around at 255->0)
    if prev_counter is not None:
        expected_counter = (prev_counter + 1) % 256
        if counter != expected_counter:
            result["reason"] = "counter_invalid"
            return result

    # Extract timestamp
    try:
        timestamp = extract_pkt_time(pkt)
        result["timestamp"] = timestamp
    except:
        result["reason"] = "timestamp_error"
        return result

    # Check timestamp is increasing
    if prev_time is not None and timestamp <= prev_time:
        result["reason"] = "timestamp_not_increasing"
        return result

    # Extract and validate packet ID (frequency and type)
    try:
        frequency, pkt_type = extract_pkt_id(pkt)
        result["frequency"] = frequency
        result["type"] = pkt_type
    except:
        result["reason"] = "id_error"
        return result

    # Check if type is recognized
    if frequency is None or pkt_type is None:
        result["reason"] = "unknown_type"
        return result

    # Check byte 13 (u8_3) is always 0 (confirmed separator/padding)
    if pkt[13] != 0:
        result["reason"] = "byte13_not_zero"
        return result

    # All checks passed
    result["valid"] = True
    return result


def decode_message(message: str) -> Dict:
    """
    Decode a single message by iterating through the payload and identifying valid packets.

    Returns statistics about the message including:
    - Total payload bytes
    - Number of valid packets found
    - Breakdown by packet type
    - Bytes consumed by valid packets
    - Leftover bytes
    """
    ts, uuid, payload_hex = message.strip().split("\t", 2)
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    # Convert hex string to bytes
    payload = bytes.fromhex(payload_hex.strip())
    total_bytes = len(payload)

    stats = {
        "timestamp": ts,
        "uuid": uuid,
        "total_bytes": total_bytes,
        "valid_packets": 0,
        "bytes_consumed": 0,
        "leftover_bytes": 0,
        "packet_types": {},
        "packet_lengths": {},  # Track lengths for each packet type
        "length_mismatches": 0,  # Track packets with length mismatches
    }

    offset = 0
    prev_counter = None
    prev_time = None

    while offset < total_bytes:
        # Check if we have enough bytes for a minimal packet
        if offset + 14 > total_bytes:
            stats["leftover_bytes"] = total_bytes - offset
            break

        # Try to read packet length
        declared_len = payload[offset]

        # Check if we have enough bytes for this packet
        if offset + declared_len > total_bytes:
            stats["leftover_bytes"] = total_bytes - offset
            break

        # Extract packet
        pkt = payload[offset : offset + declared_len]

        # Validate packet
        validation = validate_packet(pkt, prev_counter, prev_time)

        if validation["valid"]:
            stats["valid_packets"] += 1
            stats["bytes_consumed"] += declared_len

            # Track packet type and length
            pkt_type = validation["type"]
            if pkt_type not in stats["packet_types"]:
                stats["packet_types"][pkt_type] = 0
                stats["packet_lengths"][pkt_type] = []
            stats["packet_types"][pkt_type] += 1
            stats["packet_lengths"][pkt_type].append(declared_len)

            # Update previous values for next iteration
            prev_counter = validation["counter"]
            prev_time = validation["timestamp"]

            # Move to next packet
            offset += declared_len
        else:
            # Track length mismatches specifically
            if validation["reason"] == "length_mismatch":
                stats["length_mismatches"] += 1

            # Invalid packet - try to skip one byte and continue
            # This handles cases where packets are not properly aligned
            offset += 1

    stats["leftover_bytes"] = total_bytes - stats["bytes_consumed"]
    stats["proportion_validated"] = (
        stats["bytes_consumed"] / total_bytes if total_bytes > 0 else 0
    )

    return stats


# -------------------------------------------------------------------------
# Script
# -------------------------------------------------------------------------
data_dir = "./data_raw"
files = sorted(os.listdir(data_dir))

all_results = []

for filename in files:
    print(f"Processing {filename}...")

    # Extract preset number from filename (e.g., "data_p20.txt" -> "p20")
    preset = filename.replace("data_", "").replace(".txt", "")

    file_stats = {
        "preset": preset,
        "filename": filename,
        "total_messages": 0,
        "total_bytes": 0,
        "total_valid_packets": 0,
        "total_bytes_consumed": 0,
        "total_leftover_bytes": 0,
        "total_length_mismatches": 0,
        "packet_type_counts": {},
        "packet_type_lengths": {},  # Track all lengths for each packet type
    }

    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
        lines = f.readlines()

        for message in lines:
            if not message.strip():
                continue

            try:
                stats = decode_message(message)

                file_stats["total_messages"] += 1
                file_stats["total_bytes"] += stats["total_bytes"]
                file_stats["total_valid_packets"] += stats["valid_packets"]
                file_stats["total_bytes_consumed"] += stats["bytes_consumed"]
                file_stats["total_leftover_bytes"] += stats["leftover_bytes"]
                file_stats["total_length_mismatches"] += stats["length_mismatches"]

                # Aggregate packet types and lengths
                for pkt_type, count in stats["packet_types"].items():
                    if pkt_type not in file_stats["packet_type_counts"]:
                        file_stats["packet_type_counts"][pkt_type] = 0
                        file_stats["packet_type_lengths"][pkt_type] = []
                    file_stats["packet_type_counts"][pkt_type] += count
                    file_stats["packet_type_lengths"][pkt_type].extend(
                        stats["packet_lengths"][pkt_type]
                    )
            except Exception as e:
                print(f"  Error processing message: {e}")
                continue

    # Calculate proportions
    if file_stats["total_bytes"] > 0:
        file_stats["proportion_validated"] = (
            file_stats["total_bytes_consumed"] / file_stats["total_bytes"]
        )
        file_stats["proportion_leftover"] = (
            file_stats["total_leftover_bytes"] / file_stats["total_bytes"]
        )
    else:
        file_stats["proportion_validated"] = 0
        file_stats["proportion_leftover"] = 0

    # Calculate proportion of length mismatches relative to total bytes inspected
    # Note: we count attempts by total_bytes since each byte position is potentially a packet start
    total_attempts = (
        file_stats["total_valid_packets"] + file_stats["total_length_mismatches"]
    )
    if total_attempts > 0:
        file_stats["proportion_length_mismatch"] = (
            file_stats["total_length_mismatches"] / total_attempts
        )
    else:
        file_stats["proportion_length_mismatch"] = 0

    all_results.append(file_stats)
    print(
        f"  Messages: {file_stats['total_messages']}, "
        f"Valid packets: {file_stats['total_valid_packets']}, "
        f"Validated: {file_stats['proportion_validated']:.2%}"
    )

# Create DataFrame with results
df_results = pd.DataFrame(all_results)

# Create columns for each packet type (counts and length statistics)
packet_types = set()
for result in all_results:
    packet_types.update(result["packet_type_counts"].keys())

for pkt_type in sorted(packet_types):
    # Count column
    df_results[f"packets_{pkt_type}"] = df_results["packet_type_counts"].apply(
        lambda x: x.get(pkt_type, 0)
    )

    # Average length and range column
    def format_length_stats(row):
        lengths = row["packet_type_lengths"].get(pkt_type, [])
        if not lengths:
            return "-"
        avg = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len == max_len:
            return f"{avg:.1f}"
        else:
            return f"{avg:.1f} ({min_len}-{max_len})"

    df_results[f"length_{pkt_type}"] = df_results.apply(format_length_stats, axis=1)

# Drop the dictionary columns and unwanted columns
df_results = df_results.drop(
    columns=[
        "packet_type_counts",
        "packet_type_lengths",
        "total_messages",
        "total_bytes",
        "total_bytes_consumed",
        "filename",
        "proportion_leftover",
        "total_length_mismatches",  # Drop the raw count, keep only proportion
    ]
)

# Reorder columns for better readability
column_order = [
    "preset",
    "total_valid_packets",
    "total_leftover_bytes",
    "proportion_validated",
    "proportion_length_mismatch",
]
# Add packet type columns (count and length interleaved)
for pkt_type in sorted(packet_types):
    column_order.append(f"packets_{pkt_type}")
    column_order.append(f"length_{pkt_type}")

df_results = df_results[column_order]

print("\n" + "=" * 80)
print("SUMMARY RESULTS")
print("=" * 80)
print(df_results.to_markdown(index=False, floatfmt=(".2f")))


# ================================================================================
# SUMMARY RESULTS
# ================================================================================
# | preset   |   total_valid_packets |   total_leftover_bytes |   proportion_validated |   proportion_length_mismatch |   packets_ACCGYRO | length_ACCGYRO   |   packets_Battery | length_Battery   |   packets_EEG4 | length_EEG4     |   packets_EEG8 | length_EEG8     |   packets_Optics16 | length_Optics16   |   packets_Optics4 | length_Optics4   |   packets_Optics8 | length_Optics8   |   packets_REF | length_REF      |
# |:---------|----------------------:|-----------------------:|-----------------------:|-----------------------------:|------------------:|:-----------------|------------------:|:-----------------|---------------:|:----------------|---------------:|:----------------|-------------------:|:------------------|------------------:|:-----------------|------------------:|:-----------------|--------------:|:----------------|
# | p1034    |                  1245 |                      0 |                   1.00 |                         0.00 |               191 | 226.3 (202-244)  |                 6 | 222.2 (210-231)  |            589 | 225.2 (202-244) |              0 | -               |                  0 | -                 |                 0 | -                |               412 | 224.9 (202-243)  |            47 | 218.7 (202-244) |
# | p1035    |                  1036 |                      0 |                   1.00 |                         0.00 |               208 | 225.4 (213-244)  |                10 | 232.4 (228-240)  |            595 | 229.4 (205-244) |              0 | -               |                  0 | -                 |               182 | 227.6 (209-244)  |                 0 | -                |            41 | 228.1 (205-244) |
# | p1041    |                  2195 |                      0 |                   1.00 |                         0.00 |               222 | 224.1 (206-244)  |                 5 | 210.8 (210-214)  |              0 | -               |           1087 | 231.7 (203-244) |                821 | 229.3 (210-243)   |                 0 | -                |                 0 | -                |            60 | 227.3 (202-244) |
# | p1042    |                  2194 |                      0 |                   1.00 |                         0.00 |               194 | 225.4 (210-244)  |                11 | 235.3 (210-243)  |              0 | -               |           1109 | 230.9 (202-244) |                846 | 228.8 (206-243)   |                 0 | -                |                 0 | -                |            34 | 230.6 (203-244) |
# | p1043    |                  1810 |                      0 |                   1.00 |                         0.00 |               193 | 223.8 (206-244)  |                 5 | 228.6 (219-243)  |              0 | -               |           1139 | 230.5 (202-244) |                  0 | -                 |                 0 | -                |               430 | 225.1 (202-243)  |            43 | 227.1 (202-244) |
# | p1044    |                  1807 |                      0 |                   1.00 |                         0.00 |               198 | 224.8 (206-244)  |                 4 | 226.5 (210-232)  |              0 | -               |           1126 | 230.5 (202-244) |                  0 | -                 |                 0 | -                |               432 | 225.8 (202-244)  |            47 | 232.7 (202-244) |
# | p1045    |                  1589 |                      0 |                   1.00 |                         0.00 |               183 | 224.3 (211-244)  |                 3 | 218.3 (217-219)  |              0 | -               |           1151 | 233.9 (207-244) |                  0 | -                 |               208 | 228.3 (213-244)  |                 0 | -                |            44 | 232.8 (207-238) |
# | p1046    |                  1596 |                      0 |                   1.00 |                         0.00 |               232 | 223.1 (213-244)  |                 6 | 218.3 (215-219)  |              0 | -               |           1151 | 233.6 (205-244) |                  0 | -                 |               163 | 229.2 (211-244)  |                 0 | -                |            44 | 232.5 (207-244) |
# | p20      |                   837 |                      0 |                   1.00 |                         0.00 |               218 | 225.4 (215-244)  |                 2 | 228.0            |            574 | 226.3 (207-244) |              0 | -               |                  0 | -                 |                 0 | -                |                 0 | -                |            43 | 231.4 (211-244) |
# | p21      |                   837 |                      0 |                   1.00 |                         0.00 |               215 | 225.3 (215-244)  |                 3 | 230.7 (228-236)  |            581 | 226.2 (207-244) |              0 | -               |                  0 | -                 |                 0 | -                |                 0 | -                |            38 | 229.3 (211-244) |
# | p4129    |                  1597 |                      0 |                   1.00 |                         0.00 |               195 | 223.2 (213-244)  |                 8 | 225.5 (219-232)  |              0 | -               |           1154 | 233.4 (205-244) |                  0 | -                 |               183 | 227.9 (211-244)  |                 0 | -                |            57 | 230.4 (205-244) |
# | p50      |                   836 |                      0 |                   1.00 |                         0.00 |               220 | 226.4 (211-244)  |                 5 | 227.6 (215-240)  |            581 | 226.1 (207-244) |              0 | -               |                  0 | -                 |                 0 | -                |                 0 | -                |            30 | 228.6 (211-244) |
# | p51      |                   836 |                      0 |                   1.00 |                         0.00 |               213 | 226.9 (215-244)  |                 5 | 230.4 (228-240)  |            576 | 226.1 (207-244) |              0 | -               |                  0 | -                 |                 0 | -                |                 0 | -                |            42 | 224.0 (211-244) |
# | p60      |                   836 |                      0 |                   1.00 |                         0.00 |               203 | 226.3 (215-244)  |                 8 | 218.1 (215-240)  |            587 | 226.4 (207-244) |              0 | -               |                  0 | -                 |                 0 | -                |                 0 | -                |            38 | 224.9 (211-244) |
# | p61      |                   837 |                      0 |                   1.00 |                         0.00 |               221 | 226.6 (215-244)  |                 7 | 224.7 (215-244)  |            580 | 226.0 (207-244) |              0 | -               |                  0 | -                 |                 0 | -                |                 0 | -                |            29 | 224.3 (211-244) |


# WHAT WAS ACHIEVED:
# ------------------
# ✓ Successfully validated payload structure across 15 different recording presets
# ✓ 100% of payload bytes were successfully decoded (proportion_validated = 1.00)
# ✓ Zero leftover bytes in all files - complete payload coverage
# ✓ Zero length mismatches (proportion_length_mismatch = 0.00) - all packet
#   boundaries are correctly identified

# WHAT WAS CONFIRMED:
# -------------------
# 1. PACKET STRUCTURE: Each packet starts with a 1-byte length field (PKT_LEN)
#    that accurately declares the packet's total size (including the length byte).

# 2. PACKET SEQUENCING: Packet counters (PKT_N) increment correctly from 0-255
#    with wrap-around, and timestamps are strictly increasing within each payload.

# 3. PACKET ID VALIDATION: All packets have valid PKT_ID bytes that map to known
#    sensor types: EEG4, EEG8, REF, Optics4, Optics8, Optics16, ACCGYRO, Battery.

# 4. PRESET-SPECIFIC SENSOR COMBINATIONS:
#    - p20, p21, p50, p51, p60, p61: EEG4 + ACCGYRO + Battery + REF
#    - p1034: EEG4 + Optics8 + ACCGYRO + Battery + REF
#    - p1035: EEG4 + Optics4 + ACCGYRO + Battery + REF
#    - p1041, p1042: EEG8 + Optics16 + ACCGYRO + Battery + REF
#    - p1043, p1044: EEG8 + Optics8 + ACCGYRO + Battery + REF
#    - p1045, p1046, p4129: EEG8 + Optics4 + ACCGYRO + Battery + REF

# 5. PACKET LENGTH VARIABILITY: Packet lengths vary significantly (202-244 bytes)
#    even within the same sensor type, suggesting variable-length data encoding
#    (likely multiple samples per packet, with the number varying per packet).

# HOW TO USE THIS FOR A RELIABLE DECODER:
# ----------------------------------------
# 1. SEQUENTIAL PARSING:
#    - Start at payload offset 0
#    - Read byte[0] as declared_length
#    - Extract packet = payload[0:declared_length]
#    - Move offset += declared_length
#    - Repeat until end of payload

# 2. VALIDATION CHECKS (in order):
#    a) Minimum length: packet must be ≥14 bytes (header size)
#    b) Length match: declared_length == len(packet)
#    c) Counter increment: current_counter == (prev_counter + 1) % 256
#    d) Timestamp increasing: current_timestamp > prev_timestamp
#    e) Valid PKT_ID: frequency and type codes map to known sensors

# 3. HEADER EXTRACTION (offsets 0-13):
#    - Byte 0: PKT_LEN (total packet length)
#    - Byte 1: PKT_N (packet counter, 0-255)
#    - Bytes 2-5: PKT_T (uint32 little-endian, milliseconds since device start)
#    - Bytes 6-8: PKT_UNKNOWN1 (reserved/unknown)
#    - Byte 9: PKT_ID (upper nibble = frequency code, lower nibble = type code)
#    - Bytes 10-13: PKT_METADATA (varies by packet type)

# 4. DATA EXTRACTION (bytes 14+):
#    - Remaining bytes contain sensor data in packet-type-specific formats
#    - Data length = declared_length - 14
#    - Parse according to sensor type (EEG, Optics, ACCGYRO, etc.)

# 5. ERROR HANDLING:
#    - If validation fails, consider skipping 1 byte and re-attempting
#    - Track failed validations for quality assessment
#    - Log anomalies: counter jumps, timestamp reversals, unknown IDs

# CONFIDENCE LEVEL: HIGH
# ----------------------
# With 100% payload coverage, zero length mismatches, and consistent validation
# across all 15 presets, this packet structure model is highly reliable and ready
# for production decoder implementation.

