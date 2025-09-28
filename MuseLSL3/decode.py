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
# 10–13              SUBPKT_METADATA  (4 bytes, little-endian; header metadata)
# - interpretable as two little-endian uint16s:
#   - u16_0 = bytes 10–11: high-variance 16-bit value (possibly per-packet offset / internal counter / fine-grained ID)
#   - u16_1 = bytes 12–13: small discrete value ∈ {0,1,2,3} (likely a 2-bit slot/index / bank id)
# - u8_3 (byte 13) is observed always 0 -> reserved/padding
# 14...              SUBPKT_DATA      (multiplexed samples, tightly packed, repeating blocks)
# - ACC/GYRO (TO BE CONFIRMED): Each block:
#   - [tag byte: 0x47]
#   - [4-byte block header (unknown; possibly sub-counter or timestamp offset)]
#   - [N batched samples of 6 channels, interleaved per sample: (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z) x N]
#   - [e.g., 36 bytes data: 18 signed 16-bit little-endian integers (<18h): 18 integers represent 6 channels x 3.
#   - Multiple blocks per payload possible; search for all 0x47 tags to extract.
# Note: the payloads received are actually concatenations of multiple subpackets. Each subpacket starts with its own 1-byte length field (which includes the length byte itself), followed by the subpacket content.


# ==============================================================================
# Packet Info
# ==============================================================================
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

    Returns the integer sequence number (0-255).
    """
    return payload[1]


def extract_pkt_time(payload: bytes):
    """
    Extract subpkt time from a single payload. 4-byte unsigned little-endian at offset 2 -> milliseconds.
    """
    # primary 4-byte little-endian at offset 2 (fixed from 3)
    ms = struct.unpack_from("<I", payload, 2)[0]
    return ms * 1e-3  # convert to seconds


def extract_pkt_id(payload: bytes):
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

    id_byte = payload[9]
    freq_code = (id_byte >> 4) & 0x0F
    type_code = id_byte & 0x0F

    return FREQ_MAP.get(freq_code), TYPE_MAP.get(type_code)


def extract_pkt_metadata(payload: bytes):
    """
    Extract SUBPKT_METADATA from a Muse payload.

    Structure (offsets are 0-based):
      - bytes 10-11 : u16_0 (uint16 little-endian) - high-variance field (candidate: per-packet offset / internal counter)
      - bytes 12-13 : u16_1 (uint16 little-endian) - low-variance small set {0,1,2,3} (candidate: slot/index/bank id)
      - bytes 10..13 also viewed as u8_0..u8_3 for fine-grained inspection

    It could be a slot_hint if equals raw_u16_1 if its value is in {0,1,2,3} else None.
    """

    # raw views
    u16_0 = struct.unpack_from("<H", payload, 10)[0]
    u16_1 = struct.unpack_from("<H", payload, 12)[0]
    u8_0 = payload[10]
    u8_1 = payload[11]
    u8_2 = payload[12]
    u8_3 = payload[13]

    return {
        "metadata_0": int(u16_0),
        "metadata_1": int(u16_1),
        # "u8_0": int(u8_0),
        # "u8_1": int(u8_1),
        # "u8_2": int(u8_2),
        # "u8_3": int(u8_3),
    }


def extract_pkt_info(payload: bytes) -> Optional[Dict]:
    pkt_len, valid = extract_pkt_length(payload)
    if not valid or pkt_len is None:
        return None
    pkt_n = extract_pkt_n(payload)
    pkt_time = extract_pkt_time(payload)
    pkt_freq, pkt_type = extract_pkt_id(payload)
    pkt_metadata = extract_pkt_metadata(payload)

    return {
        "length": pkt_len,
        "sequence": pkt_n,
        "time": pkt_time,
        "frequency": pkt_freq,
        "type": pkt_type,
        "metadata_0": pkt_metadata["metadata_0"],
        "metadata_1": pkt_metadata["metadata_1"],
    }


# ==============================================================================
# Packet Data
# ==============================================================================
def extract_pkt_battery(payload: bytes, time):
    """
    Decode a Muse S (Athena) Battery packet.

    Subpacket layout: battery payload (7 x uint16, little-endian).

    Fields (based on MindMonitor / TI BQ fuel-gauge registers):
      0: raw_state_of_charge (uint16): Reported as a fixed-point register (divide by 256 to get %)
      1: raw_voltage (uint16): Fuel-gauge voltage reading (needs scaling by x16 to get millivolts)
      2: raw_temperature (uint16): Temperature register. Unit unknown.
      3-6: diagnostic / reserved registers (uint16 each): Exact purpose unclear (may be current, status flags, etc).

    """
    HEADER_LEN = 14
    if len(payload) < HEADER_LEN + 14:
        return None
    data = payload[HEADER_LEN : HEADER_LEN + 14]
    raw_soc, raw_mv, raw_temp, r1, r2, r3, r4 = struct.unpack("<7H", data)
    return {
        "battery": raw_soc / 256.0,
        "voltage": raw_mv * 16,
        "temperature": raw_temp,
        "leftover": str(r1) + "," + str(r2) + "," + str(r3) + "," + str(r4),
        "time": time,
    }


def extract_pkt_accgyro(payload: bytes, time: float) -> Optional[dict]:
    """
    Decode a Muse S ACC/GYRO packet (type 'ACCGYRO').

    - Looks for 0x47-tagged blocks after the 14-byte header.
    - Each block: tag (1) + 4-byte subheader + variable bytes data (multiple of 12 bytes / 6 int16 per sample).
    - Number of samples per block is variable: (data_len // 2) // 6.
    - ACC scaling: raw / 8192.0 -> g
    - GYRO scaling: raw * 0.06103515625 -> dps
    - Timestamps: base time + (sample_index × 1/52).

    Returns:
        dict with keys:
            "time"     : base packet time,
            "acc"      : list of (t, [x, y, z]) tuples,
            "gyro"     : list of (t, [x, y, z]) tuples,
            "leftover" : debugging info (unparsed bytes, subheaders).
    """
    HEADER_LEN = 14
    TAG = 0x47
    SUBHEADER_LEN = 4
    SAMPLE_SIZE_BYTES = 12  # 6 int16s for one ACC + GYRO sample
    PERIOD = 1.0 / 52.0

    if len(payload) < HEADER_LEN + 1 + SUBHEADER_LEN + SAMPLE_SIZE_BYTES:
        return None

    data = payload[HEADER_LEN:]
    acc_samples, gyro_samples = [], []
    leftovers = []
    total_sample_idx = 0
    i = 0

    while i < len(data):
        # Find the next 0x47 tag
        tag_idx = data.find(TAG, i)
        if tag_idx == -1:
            if i < len(data):
                leftovers.append(f"unparsed_data: {data[i:].hex()}")
            break

        # We have unparsed bytes before the tag
        if tag_idx > i:
            leftovers.append(
                f"unparsed_bytes_before_tag_at_{tag_idx}: {data[i:tag_idx].hex()}"
            )

        block_start_idx = tag_idx + 1 + SUBHEADER_LEN
        if block_start_idx > len(data):
            leftovers.append(f"incomplete_block_header_at_{tag_idx}")
            break

        # Extract subheader
        sub_header = struct.unpack_from("<I", data, tag_idx + 1)[0]
        leftovers.append(f"sub_header_at_{tag_idx}: {sub_header}")

        # Find the end of the current block of samples
        next_tag_idx = data.find(TAG, block_start_idx)
        if next_tag_idx == -1:
            block_end_idx = len(data)
        else:
            block_end_idx = next_tag_idx

        block_data_len = block_end_idx - block_start_idx
        num_samples = block_data_len // SAMPLE_SIZE_BYTES

        if num_samples > 0:
            num_int16 = num_samples * 6
            raw_vals = struct.unpack_from(f"<{num_int16}h", data, block_start_idx)

            for k in range(num_samples):
                j = k * 6
                acc_raw = raw_vals[j : j + 3]
                gyro_raw = raw_vals[j + 3 : j + 6]

                t = time + total_sample_idx * PERIOD

                acc_samples.append((t, [v / 8192.0 for v in acc_raw]))
                gyro_samples.append((t, [v * 0.06103515625 for v in gyro_raw]))

                total_sample_idx += 1

        # Move to the end of the processed block
        i = block_end_idx

    if not acc_samples:
        return None

    return {
        "time": time,
        "acc": acc_samples,
        "gyro": gyro_samples,
        "leftover": ", ".join(leftovers) if leftovers else "",
    }


# ==============================================================================
# ACC/GYRO
# ==============================================================================
def extract_pkt_accgyro2(
    payload: bytes,
    time_current: float,
) -> Optional[Dict]:
    """
    Decode one ACC/GYRO subpacket into timestamped samples.
    """
    # ---- Constants
    BYTES_PER_SAMPLE = 12  # 6 channels × 2 bytes
    GYRO_DT = 1.0 / 52.0  # 52 Hz for ACC/GYRO

    # ---- Parse data block
    data_bytes = payload[14:]
    n_samples = len(data_bytes) // BYTES_PER_SAMPLE
    leftover = len(data_bytes) % BYTES_PER_SAMPLE

    if n_samples == 0:
        return None

    # Debug: log if leftover (possible truncated packet)
    if leftover > 0:
        print(
            f"Warning: {leftover} leftover bytes in ACCGYRO packet (n_samples={n_samples}, data_len={len(data_bytes)}). Possible truncation or format mismatch."
        )

    block = data_bytes[: n_samples * BYTES_PER_SAMPLE]
    raw_i16 = np.frombuffer(block, dtype="<i2")
    raw = raw_i16.astype(np.int32).reshape(n_samples, 6)

    # ---- Convert to physical units
    acc = raw[:, 0:3].astype(np.float32) / 16384.0  # ±2 g
    gyr = raw[:, 3:6].astype(np.float32) / 131.0  # ±250 dps

    # ---- Timestamps (assume time_current is for last sample; interpolate backwards)
    # This ensures continuity across packets without gaps/overlaps.
    ts = (
        time_current
        - (n_samples - 1) * GYRO_DT
        + GYRO_DT * np.arange(n_samples, dtype=np.float64)
    )

    return {
        "type": "ACCGYRO",
        "time": ts,
        "leftover": leftover,
        "ACC": acc,
        "GYRO": gyr,
    }


def extract_pkt_accgyro1(
    payload: bytes,
    time_current: float,
    time_prev: Optional[float] = None,
) -> Optional[Dict]:
    """
    Decode one ACC/GYRO subpacket into timestamped samples.

    Muse ACC/GYRO packets are organised as blocks:
      [0x47 tag][4-byte header][18x int16 = 3 samples × (ACCx3 + GYROx3)]
    """
    GYRO_TAG = 0x47
    GYRO_HEADER_LEN = 1 + 4
    GYRO_BLOCK_BYTES = 18 * 2  # 18 shorts = 36 bytes
    GYRO_MIN_BLOCK_LEN = GYRO_HEADER_LEN + GYRO_BLOCK_BYTES
    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS

    plen = len(payload)
    if plen <= 14:  # no room for header + data
        return None

    data_bytes = payload[14:]
    blocks = []
    start = 0

    while start + GYRO_MIN_BLOCK_LEN <= len(data_bytes):
        # search for block tag
        idx = data_bytes.find(bytes([GYRO_TAG]), start)
        if idx == -1:
            break
        if idx + GYRO_MIN_BLOCK_LEN > len(data_bytes):
            break
        block_start = idx + GYRO_HEADER_LEN
        try:
            raw = struct.unpack_from("<18h", data_bytes, block_start)
        except struct.error:
            start = idx + 1
            continue
        blocks.append(raw)
        start = block_start + GYRO_BLOCK_BYTES

    if not blocks:
        return None

    # convert to numpy array
    raw_all = np.array(blocks, dtype=np.int16).reshape(-1, 6)  # (N,6)
    acc = raw_all[:, 0:3].astype(np.float32) / 16384.0
    gyr = raw_all[:, 3:6].astype(np.float32) / 131.0

    n_samples = raw_all.shape[0]

    # timestamps: spread uniformly between time_prev and time_current
    if time_prev is None:
        # fallback: backfill assuming fixed 52 Hz
        ts = time_current - (n_samples - 1) * GYRO_DT + GYRO_DT * np.arange(n_samples)
    else:
        dt_uniform = (time_current - time_prev) / float(n_samples)
        ts = time_prev + dt_uniform * (np.arange(n_samples) + 1)

    return {
        "type": "ACCGYRO",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "blocks": len(blocks),
        "payload_len": plen,
    }


def _timestamps_for(
    n_samples: int, time_current: float, time_prev: Optional[float]
) -> np.ndarray:
    if time_prev is not None and np.isfinite(time_prev):
        if n_samples == 1:
            return np.array([time_current], dtype=np.float64)
        return np.linspace(time_prev, time_current, n_samples, dtype=np.float64)
    # fallback: assume last sample at time_current and fixed sampling rate
    return (
        time_current
        - (n_samples - 1) * GYRO_DT
        + GYRO_DT * np.arange(n_samples, dtype=np.float64)
    )


def extract_pkt_accgyro_tagged(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    """
    Expect tagged blocks: [0x47][4-byte hdr][18 int16] repeated.
    Each block -> 3 samples (each sample: ACCx3, GYROx3) stored as 18 int16.
    Return None if no valid blocks are found.
    """

    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14
    GYRO_TAG = 0x47  # used by tagged-block format

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    nbytes = len(data)
    blocks: List[List[int]] = []
    pos = 0
    while True:
        idx = data.find(bytes([GYRO_TAG]), pos)
        if idx == -1:
            break
        block_start = idx + 1 + 4  # skip tag + 4-byte header
        if block_start + 18 * 2 <= nbytes:
            try:
                raw18 = struct.unpack_from("<18h", data, block_start)
            except struct.error:
                pos = idx + 1
                continue
            blocks.append(raw18)
            pos = block_start + 18 * 2
        else:
            break

    if not blocks:
        return None

    raw_all = np.array(blocks, dtype=np.int16).reshape(-1, 6)  # shape (N,6)
    acc_raw = raw_all[:, :3].astype(np.int32)
    gyr_raw = raw_all[:, 3:6].astype(np.int32)
    n_samples = acc_raw.shape[0]

    acc = acc_raw.astype(np.float32) / 16384.0
    gyr = gyr_raw.astype(np.float32) / 131.0
    ts = _timestamps_for(n_samples, time_current, time_prev)

    diagnostics = {
        "payload_len": len(payload),
        "data_len": nbytes,
        "blocks_found": len(blocks),
        "n_samples": int(n_samples),
    }

    return {
        "type": "ACCGYRO",
        "method": "tagged_blocks",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def extract_pkt_accgyro_split(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    """
    Expect contiguous split: first 3*N int16 are ACC (n x 3), next 3*N int16 are GYRO (n x 3).
    Return None if total shorts not divisible by 6.
    """

    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14
    GYRO_TAG = 0x47  # used by tagged-block format

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    nbytes = len(data)
    if nbytes % 2 != 0:
        # must be even number of bytes for int16
        return None

    raw_i16 = np.frombuffer(data, dtype="<i2").astype(np.int32)
    total_shorts = raw_i16.size
    if total_shorts % 6 != 0:
        return None

    n_samples = total_shorts // 6
    acc_raw = raw_i16[: 3 * n_samples].reshape(n_samples, 3)
    gyr_raw = raw_i16[3 * n_samples : 6 * n_samples].reshape(n_samples, 3)

    acc = acc_raw.astype(np.float32) / 16384.0
    gyr = gyr_raw.astype(np.float32) / 131.0
    ts = _timestamps_for(n_samples, time_current, time_prev)

    diagnostics = {
        "payload_len": len(payload),
        "data_len": nbytes,
        "n_samples": int(n_samples),
        "layout": "split_first_acc_then_gyro",
    }

    return {
        "type": "ACCGYRO",
        "method": "split_acc_then_gyro",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def extract_pkt_accgyro_interleaved(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    """
    Expect interleaved samples: accx accy accz gyrx gyry gyrz repeated (6 int16 per sample).
    Return None if total shorts not divisible by 6.
    """

    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14
    GYRO_TAG = 0x47  # used by tagged-block format

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    nbytes = len(data)
    if nbytes % 2 != 0:
        return None

    raw_i16 = np.frombuffer(data, dtype="<i2").astype(np.int32)
    total_shorts = raw_i16.size
    if total_shorts % 6 != 0:
        return None

    n_samples = total_shorts // 6
    arr = raw_i16.reshape(n_samples, 6)
    acc_raw = arr[:, :3]
    gyr_raw = arr[:, 3:6]

    acc = acc_raw.astype(np.float32) / 16384.0
    gyr = gyr_raw.astype(np.float32) / 131.0
    ts = _timestamps_for(n_samples, time_current, time_prev)

    diagnostics = {
        "payload_len": len(payload),
        "data_len": nbytes,
        "n_samples": int(n_samples),
        "layout": "interleaved_per_sample",
    }

    return {
        "type": "ACCGYRO",
        "method": "interleaved",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def extract_pkt_accgyro_separate(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    """
    Decodes a Muse packet that can contain either accelerometer or gyroscope data.

    Args:
        payload: The raw byte payload of the packet.
        time_current: The timestamp of the current packet.
        time_prev: The timestamp of the previous packet.

    Returns:
        A dictionary containing the decoded data, or None if the packet is invalid.
    """

    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14
    GYRO_TAG = 0x47  # used by tagged-block format

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    # Each sample is composed of 3 signed 16-bit integers (6 bytes)
    if len(data) % 6 != 0:
        return None

    # Unpack the data into a numpy array of signed 16-bit integers
    raw = np.frombuffer(data, dtype="<i2").astype(np.int32).reshape(-1, 3)
    n_samples = raw.shape[0]
    ts = _timestamps_for(n_samples, time_current, time_prev)

    # Scale the raw data to physical units (g for accelerometer, dps for gyroscope)
    acc = raw.astype(np.float32) / 16384.0
    gyr = raw.astype(np.float32) / 131.0

    # Determine whether the packet contains accelerometer or gyroscope data
    # based on the plausible range of values.
    if np.all(np.abs(acc) < 8):  # Plausible range for accelerometer data in g
        # This is an accelerometer packet, so fill the gyroscope data with NaNs
        gyr = np.full_like(acc, np.nan, dtype=np.float32)
        sensor_type = "ACC"
    else:
        # This is a gyroscope packet, so fill the accelerometer data with NaNs
        acc = np.full_like(gyr, np.nan, dtype=np.float32)
        sensor_type = "GYRO"

    return {
        "type": "ACCGYRO",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "sensor_type": sensor_type,
        "diagnostics": {
            "payload_len": len(payload),
            "n_samples": n_samples,
        },
    }


def extract_pkt_accgyro_corrected(
    payload: bytes, time_current: float
) -> Optional[Dict]:
    """
    Decode one ACC/GYRO subpacket using tagged/block-based layout (repeating [0x47 tag][4-byte hdr][18 int16 for 3 interleaved samples]).
    Assumes Muse S format: searches for 0x47 tags, extracts blocks, handles partial data.
    """
    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14
    GYRO_TAG = 0x47

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    nbytes = len(data)
    blocks = []
    pos = 0
    while True:
        idx = data.find(bytes([GYRO_TAG]), pos)
        if idx == -1:
            break
        block_start = idx + 1 + 4  # skip tag + 4-byte header
        if block_start + 36 > nbytes:
            break
        try:
            raw18 = struct.unpack_from("<18h", data, block_start)
            blocks.append(raw18)
        except struct.error:
            pos = idx + 1
            continue
        pos = block_start + 36

    if not blocks:
        return None

    raw_all = np.array(blocks, dtype=np.int16).astype(np.int32).reshape(-1, 6)
    acc = raw_all[:, :3].astype(np.float32) / 16384.0
    gyr = raw_all[:, 3:6].astype(np.float32) / 131.0
    n_samples = raw_all.shape[0]
    ts = (
        time_current
        - (n_samples - 1) * GYRO_DT
        + GYRO_DT * np.arange(n_samples, dtype=np.float64)
    )

    diagnostics = {
        "payload_len": len(payload),
        "data_len": nbytes,
        "n_samples": int(n_samples),
        "layout": "tagged_blocks",
        "leftover_shorts": (nbytes - pos) // 2,
    }

    return {
        "type": "ACCGYRO",
        "method": "tagged",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


# ==============================================================================
# Main functions
# ==============================================================================


def decode_rawdata(lines: list[str]) -> Dict[str, pd.DataFrame]:
    """
    Decode raw lines into structured DataFrames.

    Returns:
        dict with keys:
            "ACC"     -> DataFrame(time, ACC_X, ACC_Y, ACC_Z)
            "GYRO"    -> DataFrame(time, GYRO_X, GYRO_Y, GYRO_Z)
            "Battery" -> DataFrame(time, battery, temperature)
            "Leftover"-> list of leftover strings (per-packet diagnostics)
    """
    decoded = []
    leftovers_all = []

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
            pkt_decoded = extract_pkt_accgyro(payload, time=pkt_time)
        elif pkt_type == "Battery":
            pkt_decoded = extract_pkt_battery(payload, time=pkt_time)

        decoded.append(
            {
                "uuid": uuid,
                "line_time": ts,
                "pkt_time": pkt_time,
                "pkt_type": pkt_type,
                "data": pkt_decoded,
            }
        )

    # Collect data -----------------------------------------------------
    acc, gyro, batt = [], [], []

    for pkt in decoded:
        if not pkt or not pkt["data"]:
            continue

        data = pkt["data"]
        if pkt["pkt_type"] == "ACCGYRO":
            # expand acc + gyro
            for t, acc_vals in data["acc"]:
                acc.append([t, *acc_vals])
            for t, gyro_vals in data["gyro"]:
                gyro.append([t, *gyro_vals])
            if data.get("leftover"):
                leftovers_all.append(data["leftover"])

        elif pkt["pkt_type"] == "Battery":
            batt.append([data["time"], data["battery"], data["temperature"]])
            if data.get("leftover"):
                leftovers_all.append(data["leftover"])

    # Convert to DataFrames --------------------------------------------
    df_acc = (
        pd.DataFrame(acc, columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
        if acc
        else pd.DataFrame(columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
    )

    df_gyro = (
        pd.DataFrame(gyro, columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
        if gyro
        else pd.DataFrame(columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
    )

    df_batt = (
        pd.DataFrame(batt, columns=["time", "battery", "temperature"])
        if batt
        else pd.DataFrame(columns=["time", "battery", "temperature"])
    )

    return {
        "ACC": df_acc,
        "GYRO": df_gyro,
        "Battery": df_batt,
        "Leftover": leftovers_all,
        "Raw": decoded,
    }


def decode_rawdata2(lines: list[str]) -> Dict[str, pd.DataFrame]:
    """
    Decode raw lines into structured DataFrames.

    Returns:
        dict with keys:
            "ACC"     -> DataFrame(time, ACC_X, ACC_Y, ACC_Z)
            "GYRO"    -> DataFrame(time, GYRO_X, GYRO_Y, GYRO_Z)
            "Battery" -> DataFrame(time, battery, temperature)
            "Leftover"-> list of leftover strings (per-packet diagnostics)
    """
    decoded = []
    leftovers_all = []

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

        # Parse concatenated subpackets from the payload
        pos = 0
        sub_decoded = []
        while pos < len(payload):
            if pos + 1 > len(payload):
                break
            sub_len = payload[pos]
            if pos + sub_len > len(payload):
                break  # Incomplete subpacket—drop
            sub_payload = payload[pos : pos + sub_len]
            pos += sub_len

            pkt_info = extract_pkt_info(sub_payload)
            if not pkt_info:
                continue

            pkt_type = pkt_info["type"]
            pkt_time = pkt_info["time"]

            pkt_decoded = None
            if pkt_type == "ACCGYRO":
                pkt_decoded = extract_pkt_accgyro(sub_payload, time=pkt_time)
            elif pkt_type == "Battery":
                pkt_decoded = extract_pkt_battery(sub_payload, time=pkt_time)

            if pkt_decoded:
                sub_decoded.append(pkt_decoded)

        decoded.append(
            {
                "uuid": uuid,
                "line_time": ts,
                "data": sub_decoded,  # List of decoded subpackets
            }
        )

    # Collect data -----------------------------------------------------
    acc, gyro, batt = [], [], []

    for pkt in decoded:
        if not pkt or not pkt["data"]:
            continue

        for data in pkt["data"]:
            if "acc" in data and "gyro" in data:  # ACCGYRO
                for t, acc_vals in data["acc"]:
                    acc.append([t, *acc_vals])
                for t, gyro_vals in data["gyro"]:
                    gyro.append([t, *gyro_vals])
                if data.get("leftover"):
                    leftovers_all.append(data["leftover"])

            elif "battery" in data:  # Battery
                batt.append([data["time"], data["battery"], data["temperature"]])
                if data.get("leftover"):
                    leftovers_all.append(data["leftover"])

    # Convert to DataFrames --------------------------------------------
    df_acc = (
        (
            pd.DataFrame(acc, columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
            if acc
            else pd.DataFrame(columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    df_gyro = (
        (
            pd.DataFrame(gyro, columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
            if gyro
            else pd.DataFrame(columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    df_batt = (
        (
            pd.DataFrame(batt, columns=["time", "battery", "temperature"])
            if batt
            else pd.DataFrame(columns=["time", "battery", "temperature"])
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return {
        "ACC": df_acc,
        "GYRO": df_gyro,
        "Battery": df_batt,
        "Leftover": leftovers_all,
        "Raw": decoded,
    }


def decode_rawdata3(lines: list[str]) -> Dict[str, pd.DataFrame]:
    """
    Decode raw lines into structured DataFrames.

    Returns:
        dict with keys:
            "ACC"      -> DataFrame(time, ACC_X, ACC_Y, ACC_Z)
            "GYRO"     -> DataFrame(time, GYRO_X, GYRO_Y, GYRO_Z)
            "Battery"  -> DataFrame(time, battery, temperature)
            "Leftover" -> list of leftover strings (per-packet diagnostics)
    """
    decoded = []
    leftovers_all = []

    for ln in lines:
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            continue

        try:
            ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
            uuid = parts[1]
            payload = bytes.fromhex(parts[2])
        except Exception:
            continue

        # --- Loop through concatenated subpackets ---
        offset = 0
        while offset < len(payload):
            subpacket_len = payload[offset]
            if subpacket_len == 0:
                break

            subpacket = payload[offset : offset + subpacket_len]
            if len(subpacket) < subpacket_len:
                break  # Incomplete subpacket

            pkt_time = extract_pkt_time(subpacket)
            pkt_freq, pkt_type = extract_pkt_id(subpacket)

            pkt_decoded = None
            if pkt_type == "ACCGYRO":
                pkt_decoded = extract_pkt_accgyro(subpacket, time=pkt_time)
            elif pkt_type == "Battery":
                pkt_decoded = extract_pkt_battery(subpacket, time=pkt_time)

            if pkt_decoded:
                decoded.append(
                    {
                        "uuid": uuid,
                        "line_time": ts,
                        "pkt_time": pkt_time,
                        "pkt_type": pkt_type,
                        "data": pkt_decoded,
                    }
                )

            offset += subpacket_len

    # Collect data -----------------------------------------------------
    acc, gyro, batt = [], [], []

    for pkt in decoded:
        if not pkt or not pkt["data"]:
            continue

        data = pkt["data"]
        if pkt["pkt_type"] == "ACCGYRO":
            # expand acc + gyro
            for t, acc_vals in data["acc"]:
                acc.append([t, *acc_vals])
            for t, gyro_vals in data["gyro"]:
                gyro.append([t, *gyro_vals])
            if data.get("leftover"):
                leftovers_all.append(data["leftover"])

        elif pkt["pkt_type"] == "Battery":
            batt.append([data["time"], data["battery"], data["temperature"]])
            if data.get("leftover"):
                leftovers_all.append(data["leftover"])

    # Convert to DataFrames --------------------------------------------
    df_acc = (
        pd.DataFrame(acc, columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
        if acc
        else pd.DataFrame(columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
    )

    df_gyro = (
        pd.DataFrame(gyro, columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
        if gyro
        else pd.DataFrame(columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
    )

    df_batt = (
        pd.DataFrame(batt, columns=["time", "battery", "temperature"])
        if batt
        else pd.DataFrame(columns=["time", "battery", "temperature"])
    )

    return {
        "ACC": df_acc,
        "GYRO": df_gyro,
        "Battery": df_batt,
        "Leftover": leftovers_all,
        "Raw": decoded,
    }


import urllib.request
import matplotlib.pyplot as plt
import MuseLSL3

for preset in ["p21", "p1034", "p1045"]:
    print(f"Decoding preset {preset}")
    url = f"https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_{preset}.txt"

    lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

    data = decode_rawdata3(lines)

    # # Battery
    # print(f"Battery data: {len(data['Battery'])} samples")
    # if len(data["Battery"]) > 0:
    #     print(data["Battery"].describe())
    #     data["Battery"].plot(x="time", y="battery")
    #     data["Battery"].plot(x="time", y="temperature")

    # ACC + GYRO
    # Print proportion of empty data
    accgyro_pkts = []
    for pkt in data["Raw"]:
        if pkt and pkt["pkt_type"] == "ACCGYRO":
            accgyro_pkts.append(pkt)
    n_accgyro = len(accgyro_pkts)
    n_accgyro / 60
    prop_none = sum(1 for pkt in accgyro_pkts if pkt["data"] is None) / len(
        accgyro_pkts
    )

    if len(data["ACC"]) == 0 or len(data["GYRO"]) == 0:
        print("No data decoded")
        continue
    len(data["ACC"]) / 60
    data["ACC"].plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)
    data["GYRO"].plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True)
    data["ACC"].plot(y="time")  # Ok, monotonic
    data["Leftover"]
    # print(
    #     f"Same length: {len(data['ACC']) == len(data['GYRO'])}"
    # )  # Both have same length, good

# 1. Using these decoding functions on a file, I get only 9 ACC and GYRO samples per s (len(data["ACC"]) == 540 for a 1min recording), which does not match the expected 52 Hz. The data content is None for ~1/4 of the accgyro packets. The signals also do not look like smooth ACC/GYRO traces. This suggests that something is wrong with the decoding.
# 2. I am attaching an Android debug log from a custom decompiled MindMonitor apk and a bluetooth HCI snoop (analyzed with Wireshark). Can we use this information to find what is wrong?
# 3. Compare  that against solutions in https://github.com/AbosaSzakal/MuseAthenaDataformatParser and https://github.com/Amused-EEG/amused-py

# The ACC/GYRO signals look bad however, with both are likely contaminated and not pure and correctly decoded accgyro data.
# 1. Compare the ACCGYRO decoding functions and outline the differences and similarities.

# 4. Write me a drop-in replacement function that implements the correct decoding method.
