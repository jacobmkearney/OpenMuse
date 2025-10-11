"""
Muse BLE Message Parser
=======================

Efficiently parses BLE (Bluetooth Low Energy) messages from Muse devices into
structured packets for downstream signal processing.

Message Structure:
------------------
Each BLE MESSAGE (timestamped transmission) contains one or more PACKETS.
Each PACKET contains one or more DATA SUBPACKETS with sensor data.

MESSAGE (BLE transmission with message timestamp)
  └─ PACKET (14-byte header + variable data)
       ├─ Packet Header (14 bytes)
       └─ DATA SUBPACKETS (variable, multiple per packet)
            ├─ TAG (1 byte): Frequency (high nibble) + Type (low nibble)
            ├─ Subpacket counter (1 byte): Per-sensor-type counter (0-255, wraps)
            ├─ Unknown metadata (3 bytes): Function unknown
            └─ Sensor data (variable): Format depends on sensor type


Packet Header Structure (14 bytes):
------------------------------------
Byte Offset | Field Name     | Type      | Description
------------|----------------|-----------|------------------------------------------
0           | pkt_len        | uint8     | Declared packet length (includes header)
1           | pkt_n          | uint8     | Global packet counter (0-255, wraps)
2-5         | pkt_time       | uint32 LE | Machine timestamp (256 kHz ticks, device uptime)
6-8         | pkt_unknown1   | 3 bytes   | Reserved/unknown (consistent per preset)
9           | pkt_id         | uint8     | Primary sensor: Freq (high nibble) + Type (low)
10-12       | pkt_unknown2   | 3 bytes   | Metadata bytes (function unknown)
13          | always_zero    | uint8     | Separator (confirmed 100% = 0x00)

IMPORTANT: Packet timestamps (pkt_time, bytes 2-5) are in 256 kHz clock ticks,
NOT milliseconds. Divide by 256000 to convert to seconds.


Data Subpacket Structure (variable length):
--------------------------------------------
Byte Offset | Field Name     | Type      | Description
------------|----------------|-----------|------------------------------------------
0           | TAG            | uint8     | Frequency (high nibble) + Type (low nibble)
1           | subpkt_counter | uint8     | Per-sensor-type counter (0-255, wraps)
2-4         | subpkt_unknown | 3 bytes   | Metadata bytes (function unknown)
5+          | sensor_data    | variable  | Encoded sensor samples (length varies by type)

Multiple DATA SUBPACKETS can appear in sequence within a single packet. Each
subpacket has its own TAG identifying the sensor type. The subpkt_counter is
INDEPENDENT for each sensor type (EEG has its own counter, ACCGYRO has its own, etc.).


Counter Systems:
----------------
1. pkt_n (byte 1 of packet header): GLOBAL counter across all sensors
   - Increments for each packet regardless of sensor type
   - Shared across EEG, ACCGYRO, Optics, Battery, etc.
   - Wraps at 255 back to 0

2. subpkt_counter (byte 1 after TAG): PER-SENSOR-TYPE counter
   - Independent counter for each sensor type
   - EEG has its own sequence: 0, 1, 2, 3...
   - ACCGYRO has its own sequence: 0, 1, 2, 3...
   - Tracks data continuity within each sensor stream
   - Wraps at 255 back to 0
   - Validated alignment with expected timing (ratio ~1.003-1.005)


Notes & Known Issues:
---------------------
1. TIMESTAMPS NOT STRICTLY MONOTONIC: Packet timestamps (pkt_time) are mostly
   increasing but can occasionally arrive out-of-order (6.5% ACCGYRO, 2.1% EEG).
   Current backfilling assumes chronological order but data chunks are interleaved.

2. UNKNOWN FIELDS:
   - Packet header bytes 6-8 (pkt_unknown1) and 10-12 (pkt_unknown2)
   - Subpacket bytes 2-4 (subpkt_unknown)
   These have consistent patterns but their exact meaning is not yet decoded.


Usage Example:
--------------
Example of raw BLE message (2 messages):

2025-10-05T19:06:25.799053+00:00	273e0013-4c4d-454d-96be-f03bac821358	d700005528d18993014700fe02006ec088fea4039700fb00940076c09bfebe031bfcc901a4fc38c073fe8203e5feb200effe1200733801ff1f0008000280ffdffff7ff0180fcdf0028000e804a2013a8050a801201b0380160e1be77ed097ab09eab27eaed7b77dc6c482e928faa216c681a42871202ce3801fbac6f4545d54817dbc8c6c5b953ff3f000000c8030000001008000012030c3901a956e7e2190ab8d943ad30237c0a00407ffeffffffe57109ecfbc2b5120449390101d1ffcf6663a9ffffffffffffff0c3db0c8988400baac596bedeebc
2025-10-05T19:06:25.829038+00:00	273e0013-4c4d-454d-96be-f03bac821358	f00100a628d189930112051c0000ff3f0000000000ad005e7027940c6f1a1582d915a3bd00000000000012067a000000805ffdffffff0dedd0eaade2a31bccffefafe7bcffffffffffffff1207980000faf823dae7100eb9f189ac38bfcfff3f0000000000dc0460016ca81d1208d500000da0fab07bed8a00000000000000000044fcffffffbd27789957b68d120912010030c7ff2feb3bcfffffffffffffffebf391fb32f91c4376b23d7f97e0120a4f0100ff7f4d000000007b898a92b8dc2fe6a609e01d756f00000000000000120b6e010000c039dbdfffff392218e8ff117771c3ffffff07dcffffffffffffff
"""

import struct
from typing import List, Dict, Union
from datetime import datetime
import numpy as np

# Lookup tables for PKT_ID decoding
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

# Based on the possible FREQ/TYPE combinations, possible ID TAG bytes are:
TAGS = {
    0x11: "EEG4",
    0x12: "EEG8",
    # 0x13: "REF",  # Not observed in any test data files
    0x34: "Optics4",
    0x35: "Optics8",
    0x36: "Optics16",
    0x47: "ACCGYRO",
    0x53: "Unknown",  # 24-byte structure at 32 Hz (freq_code=5, type_code=3)
    0x98: "Battery",
}

ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768
OPTICS_SCALE = 1.0 / 32768.0


def parse_message(message: str, parse_leftovers=True) -> List[Dict]:
    """
    Parse a BLE message into a list of packets.

    Parameters:
    -----------
    message : str
        Tab-separated string containing:
        - ISO timestamp (BLE message arrival time)
        - UUID (BLE characteristic)
        - Hex-encoded payload
    """
    # Parse the message line
    try:
        ts, uuid, hexstring = message.strip().split("\t", 2)
        message_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        payload = bytes.fromhex(hexstring.strip())
    except (ValueError, AttributeError) as e:
        # Malformed message - return empty list
        return []

    subpackets = []
    offset = 0
    total_bytes = len(payload)

    # Iterate through payload, extracting subpackets
    while offset < total_bytes:
        # Need at least 14 bytes for header
        if offset + 14 > total_bytes:
            break

        # Read declared length
        declared_len = payload[offset]

        # Check if we have enough bytes for the full packet
        if offset + declared_len > total_bytes:
            break

        # Extract the full packet
        pkt = payload[offset : offset + declared_len]

        # Parse header fields
        pkt_len = pkt[0]
        pkt_n = pkt[1]
        pkt_time_ticks = struct.unpack_from("<I", pkt, 2)[0]  # uint32 little-endian
        # Device timestamp is in clock ticks at 256 kHz (2^18 Hz)
        pkt_time = pkt_time_ticks / 256000.0  # Convert ticks to seconds
        pkt_unknown1 = pkt[6:9]  # Reserved bytes 6-8
        pkt_id = pkt[9]
        pkt_unknown2 = pkt[10:13]  # Metadata bytes 10-12
        byte_13 = pkt[13]

        # Decode ID byte
        freq_code = (pkt_id >> 4) & 0x0F
        type_code = pkt_id & 0x0F
        pkt_freq = FREQ_MAP.get(freq_code)
        pkt_type = TYPE_MAP.get(type_code)

        # Validate packet
        pkt_valid = (
            pkt_freq is not None  # Frequency code is recognized
            and pkt_type is not None  # Type code is recognized
            and byte_13 == 0  # Byte 13 must be 0x00
            and pkt_len == len(pkt)  # Declared length matches actual length
            and pkt_len >= 14  # Minimum header size
        )

        # Extract data section
        pkt_data = pkt[14:] if len(pkt) > 14 else b""

        leftover = None

        data_accgyro = []
        data_battery = []
        data_eeg = []
        data_optics = []

        # Parse primary data
        if pkt_valid and pkt_type == "Battery":
            d, leftover = decode_battery(pkt_data, pkt_time)
            data_battery.append(d)
        if pkt_valid and pkt_type == "ACCGYRO":
            d, leftover = decode_accgyro(pkt_data, pkt_time)
            data_accgyro.append(d)
        if pkt_valid and pkt_type in ["EEG4", "EEG8"]:
            n_channels = 4 if pkt_type == "EEG4" else 8
            d, leftover = decode_eeg(pkt_data, pkt_time, n_channels)
            data_eeg.append(d)
        if pkt_valid and pkt_type in ["Optics4", "Optics8", "Optics16"]:
            n_channels = (
                4 if pkt_type == "Optics4" else 8 if pkt_type == "Optics8" else 16
            )
            d, leftover = decode_optics(pkt_data, pkt_time, n_channels)
            data_optics.append(d)

        # Parse leftovers for additional sensor data
        # Empirical structure: [TAG byte][4 bytes header][data bytes...]
        # The 4-byte header appears consistently across packet types (contains metadata/timing)
        if parse_leftovers and leftover is not None and len(leftover) > 0:
            # Collect leftover data arrays by type (will be inserted before primary data)
            leftover_data_accgyro = []
            leftover_data_eeg = []
            leftover_data_optics = []
            leftover_data_battery = []

            # Track counts per sensor type for time backfilling
            leftover_counts = {
                "ACCGYRO": 0,
                "EEG4": 0,
                "EEG8": 0,
                "Optics4": 0,
                "Optics8": 0,
                "Optics16": 0,
                "Battery": 0,
            }

            # Iteratively parse all sensor types from leftover
            # Continue as long as we find valid TAGs with sufficient data
            while leftover is not None and len(leftover) > 0:
                # Check if first byte is a valid TAG
                tag_byte = leftover[0]

                if tag_byte not in TAGS:
                    break  # Invalid TAG, stop parsing

                tag_type = TAGS[tag_byte]

                # Determine required bytes for this packet type
                # Structure: [TAG (1 byte)][header (4 bytes)][data (variable bytes)]
                bytes_for_type = {
                    "ACCGYRO": 1 + 4 + 36,  # 3 samples × 6 channels × 2 bytes
                    "Battery": 1 + 4 + 20,  # 2 bytes SOC + 18 bytes metadata
                    "EEG4": 1 + 4 + 28,  # 4 samples × 4 channels × 14 bits
                    "EEG8": 1 + 4 + 28,  # 2 samples × 8 channels × 14 bits
                    "Optics4": 1 + 4 + 30,  # 3 samples × 4 channels × 20 bits
                    "Optics8": 1 + 4 + 40,  # 2 samples × 8 channels × 20 bits
                    "Optics16": 1 + 4 + 40,  # 1 sample × 16 channels × 20 bits
                    "Unknown": 1 + 4 + 24,  # 24-byte structure (validated 100%)
                }

                bytes_needed = bytes_for_type.get(tag_type)
                if bytes_needed is None or len(leftover) < bytes_needed:
                    break  # Not enough data remaining

                # Skip Unknown packets (no decoder implemented yet)
                # But consume the bytes to continue parsing other sensors
                if tag_type == "Unknown":
                    leftover = leftover[bytes_needed:]
                    continue  # Skip to next leftover packet

                # Parse based on sensor type
                leftover_counts[tag_type] += 1

                # Skip TAG (1 byte) + header (4 bytes) = 5 bytes total
                # Note: The 4-byte header likely contains timing information,
                # but for now we use theoretical backfilling based on sampling rates
                leftover_pkt_data = leftover[5:]

                if tag_type == "ACCGYRO":
                    # Each ACCGYRO packet contains 3 samples at 52 Hz
                    samples_per_packet = 3
                    leftover_pkt_time = pkt_time - (
                        leftover_counts["ACCGYRO"] * samples_per_packet / 52.0
                    )
                    d, new_leftover = decode_accgyro(
                        leftover_pkt_data, leftover_pkt_time
                    )
                    if d.size > 0:
                        leftover_data_accgyro.append(d)
                        leftover = new_leftover
                    else:
                        break

                elif tag_type == "Battery":
                    # Battery packets at ~0.1 Hz (every 10 seconds)
                    leftover_pkt_time = pkt_time - (leftover_counts["Battery"] * 10.0)
                    d, new_leftover = decode_battery(
                        leftover_pkt_data, leftover_pkt_time
                    )
                    if d.size > 0:
                        leftover_data_battery.append(d)
                        leftover = new_leftover
                    else:
                        break

                elif tag_type in ["EEG4", "EEG8"]:
                    # EEG at 256 Hz
                    n_channels = 4 if tag_type == "EEG4" else 8
                    n_samples = 4 if tag_type == "EEG4" else 2
                    leftover_pkt_time = pkt_time - (
                        leftover_counts[tag_type] * n_samples / 256.0
                    )
                    d, new_leftover = decode_eeg(
                        leftover_pkt_data, leftover_pkt_time, n_channels
                    )
                    if d.size > 0:
                        leftover_data_eeg.append(d)
                        leftover = new_leftover
                    else:
                        break

                elif tag_type in ["Optics4", "Optics8", "Optics16"]:
                    # Optics at ~64 Hz (nominal)
                    n_channels = (
                        4
                        if tag_type == "Optics4"
                        else 8 if tag_type == "Optics8" else 16
                    )
                    n_samples = (
                        3
                        if tag_type == "Optics4"
                        else 2 if tag_type == "Optics8" else 1
                    )
                    leftover_pkt_time = pkt_time - (
                        leftover_counts[tag_type] * n_samples / 64.0
                    )
                    d, new_leftover = decode_optics(
                        leftover_pkt_data, leftover_pkt_time, n_channels
                    )
                    if d.size > 0:
                        leftover_data_optics.append(d)
                        leftover = new_leftover
                    else:
                        break
                else:
                    break  # Unknown type, stop parsing

            # Insert leftover data BEFORE primary data (oldest to newest)
            # Leftover packets are older than the primary packet
            # Reverse the lists since we parsed them newest-to-oldest
            leftover_data_accgyro.reverse()
            leftover_data_eeg.reverse()
            leftover_data_optics.reverse()
            leftover_data_battery.reverse()

            data_accgyro = leftover_data_accgyro + data_accgyro
            data_eeg = leftover_data_eeg + data_eeg
            data_optics = leftover_data_optics + data_optics
            data_battery = leftover_data_battery + data_battery

        # Build subpacket dictionary with only requested fields
        subpkt_dict = {
            "message_time": message_time,
            "message_uuid": uuid,
            "pkt_offset": offset,
            "pkt_len": pkt_len,
            "pkt_n": pkt_n,
            "pkt_time": pkt_time,
            "pkt_unknown1": pkt_unknown1,
            "pkt_freq": pkt_freq,
            "pkt_type": pkt_type,
            "pkt_unknown2": pkt_unknown2,
            "pkt_valid": pkt_valid,
            "leftover": leftover,
            "data_accgyro": data_accgyro,
            "data_battery": data_battery,
            "data_eeg": data_eeg,
            "data_optics": data_optics,
        }

        subpackets.append(subpkt_dict)

        # Move to next packet
        offset += declared_len

    return subpackets


def decode_battery(pkt_data: bytes, pkt_time: float):
    """
    Decode battery data (battery percentage, 0-100%) from Battery packet.

    Battery packets are transmitted at approximately 0.1 Hz (every ~10 seconds).

    Notes:
    ----------------
    - SINGLE SAMPLE PER PACKET: Despite variable packet lengths (196-230 bytes),
       each Battery packet seems to contain only ONE battery reading.
    - PACKET STRUCTURE: Battery packets have consistent structure in first 14 bytes,
       followed by variable-length diagnostic data (bytes 14+).

    Battery Data Format (first 14 bytes, empirically determined):
    --------------------------------------------------------------
    Bytes 0-1:   raw_soc (uint16 LE) - State of charge (SOC), divide by 256 for percentage
    Bytes 2-3:   voltage_raw (uint16 LE) - Voltage reading (scaling unknown)
    Bytes 4-5:   temp_raw (uint16 LE) - Temperature
    Bytes 6-7:   Variable field (420-6102) - Possibly timestamp offset or sample counter
    Bytes 8-9:   Constant 0xFFFF (65535) - Likely "not used" or invalid marker
    Bytes 10-11: Nearly constant (130-136) - Metadata field
    Bytes 12-13: Constant 0x4272 (29250) - Fixed identifier or packet type marker
    Bytes 14+:   Variable-length data

    Leftover Structure (validated across 119 Battery packets in test data):
    -----------------------------------------------------------------------
    Battery packets bundle additional sensor data in their leftovers with a highly
    structured format. The leftover bytes contain multiple sensor packets chained
    together, starting from pkt_data[20:].

    VALIDATED FINDINGS (in original pkt_data):
    - Offset 20: Valid TAG byte present in 99.2% of packets (118/119 packets)
    - After offset 20, additional sensor packets follow in sequence
    - Each sensor packet has structure: [TAG_BYTE (1 byte)][DATA (variable bytes)]
    - EEG8 packets show consistent 33-byte spacing (1 TAG + 32 data bytes)
      appearing at offsets: 20, 53, 86, 119, 152, 185... (+33 byte intervals)
    """
    # First 2 bytes are state-of-charge (SOC) as uint16 little-endian
    raw_soc = struct.unpack("<H", pkt_data[0:2])[0]
    battery_percent = raw_soc / 256.0

    # Return as (1 sample of) times, data array (float64 for timestamp precision)
    data = np.array([pkt_time, battery_percent], dtype=np.float64)

    # Leftover starts at offset 20 (skip 2 bytes battery data + 18 bytes header/metadata)
    # This makes Battery leftovers consistent with ACCGYRO: leftover[0] is a TAG byte
    leftover = pkt_data[20:] if len(pkt_data) > 20 else b""

    return data, leftover


def decode_eeg(pkt_data: bytes, pkt_time: float, n_channels: int):
    """
    Decode EEG data from EEG4 or EEG8 packet.

    EEG packets contain a variable number of samples of n-channel EEG data,
    where each value is encoded as a 14-bit unsigned integer.

    Data Format:
    ------------
    - Each sample contains n_channels × 14-bit values
    - Bits are packed in little-endian bit order (LSB-first)
    - Sampling rate: 256 Hz

    Supported Configurations:
    -------------------------
    - EEG4: 4 channels, 4 samples/packet, 28 bytes (224 bits)
    - EEG8: 8 channels, 2 samples/packet, 28 bytes (224 bits)

    Note: Different EEG modes bundle different numbers of samples per packet.
    More channels → fewer samples per packet (to keep packet size constant at 28 bytes).

    Scaling:
    --------
    - Raw 14-bit unsigned integers (range 0-16383) are scaled by (1450 / 16383)
    - This converts to microvolts (µV)

    Timing:
    -------
    - Samples are back-filled from pkt_time at 256 Hz sampling rate
    - Sample times calculated based on n_samples per packet

    Parameters:
    -----------
    pkt_data : bytes
        Raw packet data containing EEG sensor values
    pkt_time : float
        Packet timestamp (seconds)
    n_channels : int
        Number of EEG channels (4 or 8)

    Returns:
    --------
    data : np.ndarray, shape (n_samples, n_channels + 1)
        Array with columns: [time, EEG_01, EEG_02, ..., EEG_N]
        Values are in microvolts (µV)
    leftover : bytes
        Remaining unparsed bytes after primary EEG data
    """
    # Infer number of samples per packet based on channel count
    # Validated via leftover TAG analysis (100% validation across all test files)
    if n_channels == 4:
        n_samples = 4  # EEG4: 4 samples × 4 channels × 14 bits = 28 bytes
    elif n_channels == 8:
        n_samples = 2  # EEG8: 2 samples × 8 channels × 14 bits = 28 bytes
    else:
        raise ValueError(f"Unsupported n_channels: {n_channels}. Expected 4 or 8.")

    # Calculate required bytes: n_samples × n_channels × 14 bits / 8 bits per byte
    bits_needed = n_samples * n_channels * 14
    bytes_needed = bits_needed // 8

    if bytes_needed > len(pkt_data):
        return np.empty((0, n_channels + 1), dtype=np.float32), pkt_data

    block = pkt_data[0:bytes_needed]

    # Convert bytes to bit array (LSB first within each byte)
    bits = []
    for byte in block:
        for bit_pos in range(8):
            bits.append((byte >> bit_pos) & 1)

    # Parse n_samples samples, each with n_channels × 14-bit values
    data = np.zeros((n_samples, n_channels), dtype=np.float32)

    for sample_idx in range(n_samples):
        for channel_idx in range(n_channels):
            # Calculate bit position for this value
            bit_start = (sample_idx * n_channels + channel_idx) * 14
            bit_end = bit_start + 14

            # Extract 14 bits and convert to integer (little-endian bit order)
            int_value = 0
            for bit_idx in range(14):
                if bits[bit_start + bit_idx]:
                    int_value |= 1 << bit_idx

            # Scale and store (convert to microvolts)
            data[sample_idx, channel_idx] = int_value * (1450.0 / 16383.0)

    # Add back-filled time vector at 256 Hz sampling rate
    sampling_rate = 256.0
    times = (
        pkt_time
        - (n_samples - 1) * (1 / sampling_rate)
        + np.arange(n_samples) * (1 / sampling_rate)
    )

    # Add times as the first column (keep as float64 for precision with large timestamps)
    times_col = times.reshape(-1, 1).astype(np.float64)
    data = np.hstack((times_col, data))

    leftover = pkt_data[bytes_needed:]

    return data, leftover


def decode_accgyro(pkt_data: bytes, pkt_time: float):
    """
    Decode accelerometer and gyroscope data from ACCGYRO packet.

    ACCGYRO packets are transmitted at 52 Hz, with each packet containing 3 samples
    of 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope).

    Data Format:
    ------------
    - 36 bytes total (18 × int16 little-endian values)
    - Arranged as 3 samples × 6 channels
    - Channels: [ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]

    Scaling:
    --------
    - Accelerometer: ACC_SCALE = 0.0000610352 (units: m/s²)
    - Gyroscope: GYRO_SCALE = -0.0074768 (units: rad/s)

    Timing:
    -------
    - Samples are back-filled from pkt_time at 52 Hz intervals
    - Sample times: [pkt_time - 2/52, pkt_time - 1/52, pkt_time]

    Returns:
    --------
    data : np.ndarray, shape (3, 7)
        Array with columns: [time, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
    leftover : bytes
        Remaining unparsed bytes after primary ACCGYRO data
    """

    bytes_needed = 36  # 18 int16 values
    if bytes_needed > len(pkt_data):
        return np.empty((0, 7), dtype=np.float32), pkt_data

    block = pkt_data[0:bytes_needed]

    # interpret 18 int16 values → reshape to 3 samples × 6 channels
    data = np.frombuffer(block, dtype="<i2", count=18).reshape(-1, 6)
    data = data.astype(np.float32)

    # apply scaling: first 3 cols (ACC), next 3 cols (GYRO)
    data[:, 0:3] *= ACC_SCALE
    data[:, 3:6] *= GYRO_SCALE

    # Add a back-filled time vector according to 52 Hz sampling rate
    num_samples = data.shape[0]
    times = pkt_time - (num_samples - 1) * (1 / 52) + np.arange(num_samples) * (1 / 52)

    # Add times as the first column (keep as float64 for precision with large timestamps)
    data = np.hstack((times.reshape(-1, 1), data))

    leftover = pkt_data[bytes_needed:]  # leftover bytes undecoded

    return data, leftover


def decode_optics(pkt_data: bytes, pkt_time: float, n_channels: int):
    """
    Decode optical sensor data from OPTICS packet.

    OPTICS packets contain a variable number of samples of n-channel optical data,
    where each value is encoded as a 20-bit unsigned integer.

    Data Format:
    ------------
    - Each sample contains n_channels × 20-bit values
    - Bits are packed in little-endian byte order
    - Each 20-bit value spans across bytes with LSB first

    Supported Configurations:
    -------------------------
    - Optics4:  4 channels, 3 samples/packet, 30 bytes (240 bits)
    - Optics8:  8 channels, 2 samples/packet, 40 bytes (320 bits)
    - Optics16: 16 channels, 1 sample/packet, 40 bytes (320 bits)

    Note: Different optics modes bundle different numbers of samples per packet.
    More channels → fewer samples per packet (to keep packet size manageable).

    Scaling:
    --------
    - Raw 20-bit integers are divided by 32768 to normalize

    Timing:
    -------
    - Samples are back-filled from pkt_time at the inferred sampling rate
    - Sample times calculated based on n_samples per packet

    Parameters:
    -----------
    pkt_data : bytes
        Raw packet data containing optical sensor values
    pkt_time : float
        Packet timestamp (seconds)
    n_channels : int
        Number of optical channels (4, 8, or 16)

    Returns:
    --------
    data : np.ndarray, shape (n_samples, n_channels + 1)
        Array with columns: [time, OPTICS_01, OPTICS_02, ..., OPTICS_N]
    leftover : bytes
        Remaining unparsed bytes after primary OPTICS data
    """
    # Infer number of samples per packet based on channel count
    # Validated via leftover TAG analysis (100% validation across all test files)
    if n_channels == 4:
        n_samples = 3  # Optics4: 3 samples × 4 channels × 20 bits = 30 bytes
    elif n_channels == 8:
        n_samples = 2  # Optics8: 2 samples × 8 channels × 20 bits = 40 bytes
    elif n_channels == 16:
        n_samples = 1  # Optics16: 1 sample × 16 channels × 20 bits = 40 bytes
    else:
        raise ValueError(f"Unsupported n_channels: {n_channels}. Expected 4, 8, or 16.")

    # Calculate required bytes: n_samples × n_channels × 20 bits / 8 bits per byte
    bits_needed = n_samples * n_channels * 20
    bytes_needed = bits_needed // 8

    if bytes_needed > len(pkt_data):
        return np.empty((0, n_channels + 1), dtype=np.float32), pkt_data

    block = pkt_data[0:bytes_needed]

    # Convert bytes to bit array (LSB first within each byte)
    bits = []
    for byte in block:
        for bit_pos in range(8):
            bits.append((byte >> bit_pos) & 1)

    # Parse n_samples samples, each with n_channels × 20-bit values
    data = np.zeros((n_samples, n_channels), dtype=np.float32)

    for sample_idx in range(n_samples):
        for channel_idx in range(n_channels):
            # Calculate bit position for this value
            bit_start = (sample_idx * n_channels + channel_idx) * 20
            bit_end = bit_start + 20

            # Extract 20 bits and convert to integer (little-endian bit order)
            int_value = 0
            for bit_idx in range(20):
                if bits[bit_start + bit_idx]:
                    int_value |= 1 << bit_idx

            # Scale and store
            data[sample_idx, channel_idx] = int_value * OPTICS_SCALE

    # Add back-filled time vector
    # Use a nominal sampling rate estimate based on observed packet rates
    # (Actual rates vary by optics mode but this provides reasonable timestamps)
    sampling_rate = 64.0  # Nominal rate for time interpolation
    times = (
        pkt_time
        - (n_samples - 1) * (1 / sampling_rate)
        + np.arange(n_samples) * (1 / sampling_rate)
    )

    # Add times as the first column (keep as float64 for precision with large timestamps)
    times_col = times.reshape(-1, 1).astype(np.float64)
    data = np.hstack((times_col, data))

    leftover = pkt_data[bytes_needed:]

    return data, leftover


# ============================================================================
# Wrapper Functions
# ============================================================================


def decode_message(message: str, parse_leftovers: bool = True) -> dict:
    """
    Converts new parse_message() output to old format expected by stream.py:
    {
        "ACC": [{"time": float, "ACC_X": float, "ACC_Y": float, "ACC_Z": float}, ...],
        "GYRO": [{"time": float, "GYRO_X": float, "GYRO_Y": float, "GYRO_Z": float}, ...]
    }

    Parameters:
    -----------
    message : str
        Tab-separated BLE message string
    parse_leftovers : bool
        Whether to parse leftover data (default: True)

    Returns:
    --------
    dict : Dictionary with "ACC" and "GYRO" keys containing lists of sample dicts
    """
    subpackets = parse_message(message, parse_leftovers=parse_leftovers)

    result = {
        "ACC": [],
        "GYRO": [],
    }

    for subpkt in subpackets:
        # Extract ACCGYRO data
        if "data_accgyro" in subpkt and len(subpkt["data_accgyro"]) > 0:
            for acc_array in subpkt["data_accgyro"]:
                # acc_array shape: (3, 7) with columns [time, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
                for i in range(acc_array.shape[0]):
                    row = acc_array[i, :]

                    # Split into ACC and GYRO
                    result["ACC"].append(
                        {
                            "time": float(row[0]),
                            "ACC_X": float(row[1]),
                            "ACC_Y": float(row[2]),
                            "ACC_Z": float(row[3]),
                        }
                    )

                    result["GYRO"].append(
                        {
                            "time": float(row[0]),
                            "GYRO_X": float(row[4]),
                            "GYRO_Y": float(row[5]),
                            "GYRO_Z": float(row[6]),
                        }
                    )

    return result


def decode_rawdata(messages: list, parse_leftovers: bool = True) -> dict:
    """
    Decode multiple raw BLE messages and return organized data in Pandas DataFrames.

    Parameters:
    -----------
    messages : list
        List of BLE message strings (one per line)
    parse_leftovers : bool
        Whether to parse leftover data (default: True)

    Returns:
    --------
    dict : Dictionary with pandas DataFrames for each sensor type:
        {
            "ACC": DataFrame with columns [time, ACC_X, ACC_Y, ACC_Z],
            "GYRO": DataFrame with columns [time, GYRO_X, GYRO_Y, GYRO_Z]
        }
    """
    import pandas as pd

    all_acc = []
    all_gyro = []

    for message in messages:
        decoded = decode_message(message, parse_leftovers=parse_leftovers)
        all_acc.extend(decoded.get("ACC", []))
        all_gyro.extend(decoded.get("GYRO", []))

    return {
        "ACC": pd.DataFrame(all_acc),
        "GYRO": pd.DataFrame(all_gyro),
    }
