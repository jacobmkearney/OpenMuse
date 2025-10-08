"""
Muse BLE Message Parser
=======================

Efficiently parses BLE (Bluetooth Low Energy) messages from Muse devices into
structured packets for downstream signal processing.

Functions:
----------
- parse_message(message: str) -> List[Dict]: Parse BLE message into packets
- decode_battery(pkt_data: bytes, pkt_time: float) -> tuple[np.ndarray, np.ndarray]:
    Extract battery percentage from packet data, returns (times, data) arrays


Packet Structure:
------------------
Each packet contains multiple subpackets encoding metadata and sensor data.

Byte Offset | Field Name        | Type      | Description
------------|-------------------|-----------|------------------------------------------
0           | pkt_len        | uint8     | Declared packet length (includes header)
1           | pkt_n          | uint8     | Packet counter (0-255, wraps around)
2-5         | pkt_time       | uint32 LE | Timestamp in milliseconds (device time)
6-8         | pkt_unknown1   | 3 bytes   | Reserved/unknown (consistent per preset)
9           | pkt_id         | uint8     | Frequency (high nibble) + Type (low nibble)
10-12       | pkt_unknown2   | 3 bytes   | Metadata bytes (function unknown)
13          | always_zero       | uint8     | Separator/padding (confirmed 100% = 0x00)
14+         | pkt_data       | variable  | Sample data (length = pkt_len - 14)


Notes & Known Issues:
---------------------
1. TIMESTAMPS NOT STRICTLY MONOTONIC: Device timestamps (pkt_time) are mostly
   increasing but can occasionally arrive out-of-order (to be confirmed).

2. UNKNOWN FIELDS: Bytes 6-8 (pkt_unknown1) and 10-12 (pkt_unknown2) have
   consistent patterns per preset but their exact meaning is not yet decoded. They
   may contain:
   - Fine-grained timing information
   - Sample indexing within packet
   - Device state flags
   - Quality indicators

   These fields are preserved in output for future analysis.

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

# Based on the FREQ/TYPE nibbles, possible ID bytes are: 0x11, 0x12, 0x13, 0x34, 0x35, 0x36, 0x47, 0x98

ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768


def parse_message(message: str) -> List[Dict]:
    """
    Parse a BLE message into a list of packets.

    Parameters:
    -----------
    message : str
        Tab-separated string containing:
        - ISO timestamp (BLE message arrival time)
        - UUID (BLE characteristic)
        - Hex-encoded payload

    Returns:
    --------
    List[Dict]
        List of dictionaries, one per packet, containing:
        - message_time: Original BLE message timestamp (datetime)
        - message_uuid: BLE characteristic UUID (str)
        - pkt_offset: Byte offset of this packet in the payload (int)
        - pkt_len: Declared packet length from byte 0 (int)
        - pkt_n: Packet counter 0-255 from byte 1 (int)
        - pkt_time: Device timestamp in seconds (float)
        - pkt_unknown1: Reserved bytes 6-8 (bytes)
        - pkt_freq: Sampling frequency in Hz (float or None)
        - pkt_type: Sensor type (str or None)
        - pkt_unknown2: Metadata bytes 10-12 (bytes)
        - pkt_data: Decoded data tuple (times, data) for Battery subpackets,
                       raw bytes for other types (tuple[np.ndarray, np.ndarray] or bytes)
        - pkt_valid: True if packet passes validation checks (bool)
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
        pkt_time_ms = struct.unpack_from("<I", pkt, 2)[0]  # uint32 little-endian
        pkt_time = pkt_time_ms * 1e-3  # Convert to seconds
        pkt_unknown1 = pkt[6:9]  # Reserved bytes 6-8
        pkt_id = pkt[9]
        pkt_unknown2 = pkt[10:13]  # Metadata bytes 10-12
        byte_13 = pkt[13]

        # Decode ID byte
        freq_code = (pkt_id >> 4) & 0x0F
        type_code = pkt_id & 0x0F
        pkt_freq = FREQ_MAP.get(freq_code)
        pkt_type = TYPE_MAP.get(type_code)

        # Extract data section
        pkt_data = pkt[14:] if len(pkt) > 14 else b""

        # Validate packet
        pkt_valid = (
            pkt_freq is not None  # Frequency code is recognized
            and pkt_type is not None  # Type code is recognized
            and byte_13 == 0  # Byte 13 must be 0x00
            and pkt_len == len(pkt)  # Declared length matches actual length
            and pkt_len >= 14  # Minimum header size
        )

        leftover = None
        # Decode battery percentage if this is a Battery subpacket
        if pkt_valid and pkt_type == "Battery":
            pkt_data, leftover = decode_battery(pkt_data, pkt_time)
        if pkt_valid and pkt_type == "ACCGYRO":
            pkt_data, leftover = decode_accgyro(pkt_data, pkt_time)

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
            "pkt_data": pkt_data,
            "pkt_valid": pkt_valid,
            "leftover": leftover,
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
    Bytes 14+:   Variable-length diagnostic data (182-216 bytes)
                 Contains ~2.7 "0x11XX" patterns per packet (structured event/log data)
                 ~9% 0xFF bytes, ~9% 0x00 bytes - diagnostic logs or system events

    Example:
    --------
        >>> from MuseLSL3.decode_new import parse_message
        >>> message = "2025-10-05T14:06:55.620681+00:00\\t...\\td700000166..."
        >>> subpackets = parse_message(message)
        >>> for sp in subpackets:
        ...     if sp['subpkt_type'] == 'Battery':
        ...         times, data = sp['subpkt_data']
        ...         print(f"Battery at {times[0]:.3f}s: {data[0]:.2f}%")
        Battery at 1234.567s: 16.53%
    """
    # First 2 bytes are state-of-charge (SOC) as uint16 little-endian
    raw_soc = struct.unpack("<H", pkt_data[0:2])[0]
    battery_percent = raw_soc / 256.0

    # Return as (1 sample of) times, data array
    data = np.array([pkt_time, battery_percent], dtype=np.float32)

    leftover = pkt_data[2:]  # leftover bytes undecoded

    return data, leftover


def decode_accgyro(pkt_data: bytes, pkt_time: float):

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

    # Add times as the first column
    data = np.hstack((times.astype(np.float32).reshape(-1, 1), data))

    leftover = pkt_data[bytes_needed:]  # leftover bytes undecoded

    return data, leftover


# with open("../tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
#     messages = f.readlines()
# pkts = [parse_message(m) for m in messages]
# print(f"Total packets: {len(pkts)}")
# accgyro = [p for sublist in pkts for p in sublist if p["pkt_type"] == "ACCGYRO"]
# data = np.vstack([a["pkt_data"] for a in accgyro if a["pkt_valid"]])
# data = pd.DataFrame(
#     data, columns=["time", "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
# )

# data.plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True)
# leftovers = [a["leftover"] for a in accgyro if a["pkt_valid"] and a["leftover"]]
