import struct
from datetime import datetime, timezone
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# Packet structure -------------------
# Offset (0-based)   Field
# -----------------  -----------------------------------------
# 0                  SUBPKT_LEN       (1 byte) [confirmed]
# 1                  SUBPKT_N         (1 byte) [confirmed]
# 2–5                SUBPKT_T         (uint32, ms since device start) [confirmed]
# 6–8                SUBPKT_UNKNOWN1  (3 bytes, reserved?)
# 9                  SUBPKT_ID        (freq/type nibbles) [confirmed]
# 10–13              SUBPKT_UNKNOWN2  (unknown, first one could be a counter)
# 14...              SUBPKT_DATA      (samples, tightly packed)


def extract_pkt_length(payload: bytes):
    """
    Extract the SUBPKT_LEN field (declared length) from a Muse payload.
    """
    if not payload or len(payload) < 14:  # minimum length for header
        return None, False

    declared_len = payload[0]
    return declared_len, (declared_len == len(payload))


def extract_pkt_n(payload: bytes):
    """
    Extract the SUBPKT_N field (1-byte sequence number) from a Muse payload.

    - Located at offset 1 (0-based). Increments by 1 per packet, wraps at 255 -> 0.
    - Useful for detecting dropped or out-of-order packets (quality check assessment).

    Returns the integer sequence number (0-255), or None if payload too short.
    """
    if len(payload) <= 1:
        return None
    return payload[1]


def extract_pkt_time(payload: bytes):
    """
    Extract subpkt time from a single payload. 4-byte unsigned little-endian at offset 3 -> milliseconds.
    """
    # primary 4-byte little-endian at offset 3
    if len(payload) >= 3 + 4:
        ms = struct.unpack_from("<I", payload, 3)[0]
        return ms * 1e-3  # convert to seconds

    return None


def extract_pkt_id(payload: bytes):
    """
    Extract and parse the ID byte from a Muse payload.
    - ID byte is at offset 9 (0-based).
    - Upper nibble = frequency code.
    - Lower nibble = data type code.
    Returns dict with raw codes and decoded labels.
    """

    if len(payload) <= 9:
        return None  # not enough data

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

    id_byte = payload[9]
    freq_code = (id_byte >> 4) & 0x0F
    type_code = id_byte & 0x0F

    return FREQ_MAP.get(freq_code), TYPE_MAP.get(type_code)


def extract_pkt_info(payload: bytes):

    pkt_len, valid = extract_pkt_length(payload)
    if not valid or pkt_len is None:
        return None
    pkt_n = extract_pkt_n(payload)
    pkt_time = extract_pkt_time(payload)
    pkt_freq, pkt_type = extract_pkt_id(payload)

    return {
        "length": pkt_len,
        "sequence": pkt_n,
        "time": pkt_time,
        "frequency": pkt_freq,
        "type": pkt_type,
    }


def extract_pkt_accgyro(
    payload: bytes,
    time_current=None,
    time_prev=None,
):
    """
    Decode one ACC/GYRO subpacket into timestamped samples.
    """
    # ---- Constants
    BYTES_PER_SAMPLE = 12  # 6 channels × 2 bytes
    GYRO_DT = 1.0 / 52.0  # 52  Hz for ACC/GYRO

    # ---- Parse data block
    data_bytes = payload[14:]
    n_samples = len(data_bytes) // BYTES_PER_SAMPLE
    leftover = len(data_bytes) % BYTES_PER_SAMPLE

    if n_samples == 0:
        return None

    block = data_bytes[: n_samples * BYTES_PER_SAMPLE]
    raw_i16 = np.frombuffer(block, dtype="<i2")
    raw = raw_i16.astype(np.int32).reshape(n_samples, 6)

    # ---- Convert to physical units
    acc = raw[:, 0:3].astype(np.float32) / 16384.0  # ±2 g
    gyr = raw[:, 3:6].astype(np.float32) / 131.0  # ±250 dps

    # ---- Timestamps
    if time_prev is None:
        time_prev = time_current - n_samples * GYRO_DT
    ts = time_prev + GYRO_DT * np.arange(n_samples, dtype=np.float64)

    return {
        "type": "ACCGYRO",
        "time": ts,
        "leftover": leftover,
        "ACC": acc,
        "GYRO": gyr,
    }


# Constants ----------------------------
# GYRO/ACC settings

GYRO_FS = 52.0
GYRO_DT = 1.0 / GYRO_FS
GYRO_TAG = 0x47
GYRO_HEADER_LEN = 1 + 4
GYRO_BLOCK_BYTES = 18 * 2
GYRO_MIN_BLOCK_LEN = GYRO_HEADER_LEN + GYRO_BLOCK_BYTES
# -------------------------------------

# def extract_pkt_accgyro(payload: bytes, time_prev: Optional[float], time_current: float):
#     """
#     Yield per-sample tuples:
#       (time_s, acc_x_g, acc_y_g, acc_z_g, gyro_x_dps, gyro_y_dps, gyro_z_dps)

#     time_prev and time_current are seconds; last sample equals time_current.
#     If time_prev is None the function uses GYRO_DT_S spacing to align the block.
#     Minimal allocations: unpacks each block once and yields floats per sample.
#     """
#     plen = len(payload)
#     start = 0

#     blocks = []
#     while start + GYRO_MIN_BLOCK_LEN <= plen:
#         idx = payload.find(bytes([GYRO_TAG]), start)
#         if idx == -1:
#             break
#         if idx + GYRO_MIN_BLOCK_LEN > plen:
#             break
#         block_start = idx + GYRO_HEADER_LEN
#         try:
#             raw = struct.unpack_from("<18h", payload, block_start)
#         except struct.error:
#             start = idx + 1
#             continue
#         blocks.append(raw)
#         start = block_start + GYRO_BLOCK_BYTES

#     if not blocks:
#         return

#     total_samples = len(blocks) * 3

#     if time_prev is None:
#         time_prev = time_current - total_samples * GYRO_DT

#     dt_s_uniform = (time_current - time_prev) / float(total_samples)

#     sample_index = 0
#     for raw_block in blocks:
#         for s in range(3):
#             base = s * 6
#             acc_x = raw_block[base + 0] / 16384.0
#             acc_y = raw_block[base + 1] / 16384.0
#             acc_z = raw_block[base + 2] / 16384.0
#             gyro_x = raw_block[base + 3] / 131.0
#             gyro_y = raw_block[base + 4] / 131.0
#             gyro_z = raw_block[base + 5] / 131.0

#             t_s = time_prev + (sample_index + 1) * dt_s_uniform
#             yield (t_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
#             sample_index += 1


# MASTER FUNCTIONS --------------------
def decode_payload(
    payload: bytes,
    time_prev: Optional[float] = None,
):

    # --- Header fields ---
    pkt_len, length_ok = extract_pkt_length(payload)
    pkt_n = extract_pkt_n(payload)
    pkt_time = extract_pkt_time(payload)
    pkt_freq, pkt_type = extract_pkt_id(payload)

    if not length_ok:
        return None  # incomplete packet

    # # --- Decode ACC/GYRO samples if applicable ---
    # if pkt_type and pkt_type == "ACCGYRO" and pkt_time is not None:
    #     samples = list(extract_pkt_accgyro(payload, time_prev, pkt_time))
    #     result["samples"] = samples
    # TODO


# Convenience: collect into NumPy arrays in one allocation
def stream_to_arrays(it):
    """
    Consume the stream and return (times_s, acc_with_t, gyro_with_t)

    - acc: shape (N,4) float32 columns = [time_s, acc_x, acc_y, acc_z]
    - gyro: shape (N,4) float32 columns = [time_s, gyro_x, gyro_y, gyro_z]
    """
    rows = list(it)
    if not rows:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
    arr = np.array(
        rows, dtype=np.float64
    )  # shape (N,7): time, acc_x..acc_z, gyro_x..gyro_z
    times = arr[:, 0].astype(np.float64)  # (N,)

    acc = arr[:, 1:4].astype(np.float32)  # (N,3)
    gyro = arr[:, 4:7].astype(np.float32)  # (N,3)

    # prepend time column (cast times to float32 to keep arrays compact)
    ts = times.astype(np.float32).reshape(-1, 1)  # (N,1)
    acc = np.hstack((ts, acc))  # (N,4)
    gyro = np.hstack((ts, gyro))  # (N,4)

    return acc, gyro


def decode_rawdata(lines: list[str]):
    """
    Decode raw lines into a list of dicts, one per line.
    Each dict contains metadata and decoded samples (if any).
    """
    # Decode each package -------------------------------------------------
    decoded = []
    t_prev = None

    for ln in lines:
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            decoded.append(None)
            continue

        try:
            ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
            uuid = parts[1]
            payload = bytes.fromhex(parts[2])
        except Exception:
            decoded.append(None)
            continue

        pkt_time = extract_pkt_time(payload)
        pkt_type = extract_pkt_id(payload)[1]  # (freq, type)

        pkt_decoded = None
        if pkt_type == "ACCGYRO":
            pkt_decoded = extract_pkt_accgyro(
                payload, time_current=pkt_time, time_prev=t_prev
            )

        # Always append a dict (even if None for data)
        decoded.append(
            {
                "uuid": uuid,
                "line_time": ts,
                "pkt_time": pkt_time,
                "pkt_type": pkt_type,
                "data": pkt_decoded,
            }
        )

        t_prev = pkt_time

    # Postprocess by collecting data --------------------------------
    acc_chunks, gyro_chunks = [], []

    for pkt in decoded:
        if not pkt or not pkt["data"]:
            continue
        if pkt["pkt_type"] == "ACCGYRO":
            ts = pkt["data"]["time"][:, None]
            acc = np.hstack([ts, pkt["data"]["ACC"]])
            gyr = np.hstack([ts, pkt["data"]["GYRO"]])
            acc_chunks.append(acc)
            gyro_chunks.append(gyr)

    df_acc = pd.DataFrame(
        np.vstack(acc_chunks), columns=["time", "ACC_X", "ACC_Y", "ACC_Z"]
    ) if acc_chunks else pd.DataFrame(columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])

    df_gyro = pd.DataFrame(
        np.vstack(gyro_chunks), columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    ) if gyro_chunks else pd.DataFrame(columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])

    return {"ACC": df_acc, "GYRO": df_gyro}


# import urllib.request
# import matplotlib.pyplot as plt
# import MuseLSL3

# url = "https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_p1034.txt"

# lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# times, t = MuseLSL3.decode_rawdata(lines)
# df_acc.plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"])
# df_gyro.plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"])
