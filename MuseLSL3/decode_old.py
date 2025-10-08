import struct
from datetime import datetime
from typing import Dict, List, Optional

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

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
EEG_SCALE = 1450.0 / 16383.0
ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768


def parse_uint14_le_values(buffer: bytes) -> List[int]:
    """Decode a buffer of little-endian 14-bit integers with minimal allocations."""
    n_vals = (len(buffer) * 8) // 14
    out = [0] * n_vals
    bit_buffer = 0
    bits_in_buffer = 0
    out_index = 0

    for byte in buffer:
        bit_buffer |= byte << bits_in_buffer
        bits_in_buffer += 8

        while bits_in_buffer >= 14:
            out[out_index] = bit_buffer & 0x3FFF
            out_index += 1
            bit_buffer >>= 14
            bits_in_buffer -= 14

    return out


# -------------------------------------------------------------------------
# Packet parser
# -------------------------------------------------------------------------


def decode_pkt_eeg(packet: bytes, offset: int):
    payload_len = 28
    end_index = offset + payload_len
    if end_index > len(packet):
        return end_index, None, []

    values = parse_uint14_le_values(packet[offset:end_index])
    entry = {f"EEG_{i}": value * EEG_SCALE for i, value in enumerate(values)}
    return end_index, "EEG", [entry]


def decode_pkt_accgyro(packet: bytes, offset: int):
    bytes_needed = 36  # 18 int16 values
    end_index = offset + bytes_needed
    if end_index > len(packet):
        return end_index, None, []

    block = packet[offset:end_index]
    out = []
    for ax, ay, az, gx, gy, gz in struct.iter_unpack("<6h", block):
        out.append(
            {
                "ACC_X": ax * ACC_SCALE,
                "ACC_Y": ay * ACC_SCALE,
                "ACC_Z": az * ACC_SCALE,
                "GYRO_X": gx * GYRO_SCALE,
                "GYRO_Y": gy * GYRO_SCALE,
                "GYRO_Z": gz * GYRO_SCALE,
            }
        )

    return end_index, "ACC_GYRO", out


PACKET_DECODERS = {
    0x12: decode_pkt_eeg,  # EEG
    0x47: decode_pkt_accgyro,  # ACC + GYRO
}


def decode_packet(packet: bytes, tag: int, tag_index: int):
    offset = tag_index + 5  # tag byte + 4 byte header
    decoder = PACKET_DECODERS.get(tag)
    if decoder is None:
        return tag_index + 1, None, []
    return decoder(packet, offset)


# -------------------------------------------------------------------------
# Wrapper
# -------------------------------------------------------------------------


def decode_message(message: str) -> Optional[Dict[str, List[dict]]]:
    """Decode a single Muse log message into modality-specific samples."""
    parts = message.split("\t", 2)
    if len(parts) != 3:
        return None

    ts_str, _uuid, hexstr = parts

    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    ts = dt.timestamp()  # Convert to float seconds

    try:
        packet = memoryview(bytes.fromhex(hexstr))
    except ValueError:
        return None

    decoded: Dict[str, List[dict]] = {"EEG": [], "ACC": [], "GYRO": []}
    append_eeg = decoded["EEG"].append
    append_acc = decoded["ACC"].append
    append_gyro = decoded["GYRO"].append

    idx = 0
    packet_len = len(packet)
    while idx < packet_len:
        tag = packet[idx]
        next_idx, name, entries = decode_packet(packet, tag, idx)
        if name == "EEG":
            for entry in entries:
                entry["time"] = ts
                append_eeg(entry)
        elif name == "ACC_GYRO":
            for entry in entries:
                entry["time"] = ts
                append_acc(
                    {
                        "time": ts,
                        "ACC_X": entry["ACC_X"],
                        "ACC_Y": entry["ACC_Y"],
                        "ACC_Z": entry["ACC_Z"],
                    }
                )
                append_gyro(
                    {
                        "time": ts,
                        "GYRO_X": entry["GYRO_X"],
                        "GYRO_Y": entry["GYRO_Y"],
                        "GYRO_Z": entry["GYRO_Z"],
                    }
                )
        idx = next_idx if next_idx > idx else idx + 1

    return decoded


# -------------------------------------------------------------------------
# Static decoder
# -------------------------------------------------------------------------


def decode_rawdata(messages: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Decode raw Muse log messages into DataFrames for each modality.
    Each message is 'timestamp \t uuid \t hexstring'.
    """
    dfs: Dict[str, List[dict]] = {"EEG": [], "ACC": [], "GYRO": []}

    for message in messages:
        decoded_message = decode_message(message)
        if not decoded_message:
            continue

        dfs["EEG"].extend(decoded_message["EEG"])
        dfs["ACC"].extend(decoded_message["ACC"])
        dfs["GYRO"].extend(decoded_message["GYRO"])

    out = {}
    for modality, rows in dfs.items():
        if rows:
            out[modality] = pd.DataFrame(rows).sort_values("time")
    return out


# ==============================================================================
#  Test
# ==============================================================================
# import urllib.request

# import MuseLSL3
# import matplotlib.pyplot as plt

# url = f"https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_p1045.txt"
# lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# with open("../tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
# data = MuseLSL3.decode_rawdata(lines)
# data["ACC"]["time"] = data["ACC"]["time"] - data["ACC"]["time"].iloc[0]
# data["ACC"].plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)

