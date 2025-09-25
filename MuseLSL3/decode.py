from datetime import datetime, timezone
import struct
from typing import Optional
import numpy as np
import pandas as pd

# Constants ----------------------------
# GYRO/ACC settings

GYRO_FS = 52.0
GYRO_DT = 1.0 / GYRO_FS
GYRO_TAG = 0x47
GYRO_HEADER_LEN = 1 + 4
GYRO_BLOCK_BYTES = 18 * 2
GYRO_MIN_BLOCK_LEN = GYRO_HEADER_LEN + GYRO_BLOCK_BYTES
# -------------------------------------


def extract_time(payload: bytes):
    """
    Extract packet time from a single payload. 4-byte unsigned little-endian at offset 3 -> milliseconds.
    """
    # primary 4-byte little-endian at offset 3
    if len(payload) >= 3 + 4:
        ms = struct.unpack_from("<I", payload, 3)[0]
        return ms * 1e-3  # convert to seconds

    return None


def extract_gyroacc(payload: bytes, time_prev: Optional[float], time_current: float):
    """
    Yield per-sample tuples:
      (time_s, acc_x_g, acc_y_g, acc_z_g, gyro_x_dps, gyro_y_dps, gyro_z_dps)

    time_prev and time_current are seconds; last sample equals time_current.
    If time_prev is None the function uses GYRO_DT_S spacing to align the block.
    Minimal allocations: unpacks each block once and yields floats per sample.
    """
    plen = len(payload)
    start = 0

    blocks = []
    while start + GYRO_MIN_BLOCK_LEN <= plen:
        idx = payload.find(bytes([GYRO_TAG]), start)
        if idx == -1:
            break
        if idx + GYRO_MIN_BLOCK_LEN > plen:
            break
        block_start = idx + GYRO_HEADER_LEN
        try:
            raw = struct.unpack_from("<18h", payload, block_start)
        except struct.error:
            start = idx + 1
            continue
        blocks.append(raw)
        start = block_start + GYRO_BLOCK_BYTES

    if not blocks:
        return

    total_samples = len(blocks) * 3

    if time_prev is None:
        time_prev = time_current - total_samples * GYRO_DT

    dt_s_uniform = (time_current - time_prev) / float(total_samples)

    sample_index = 0
    for raw_block in blocks:
        for s in range(3):
            base = s * 6
            acc_x = raw_block[base + 0] / 16384.0
            acc_y = raw_block[base + 1] / 16384.0
            acc_z = raw_block[base + 2] / 16384.0
            gyro_x = raw_block[base + 3] / 131.0
            gyro_y = raw_block[base + 4] / 131.0
            gyro_z = raw_block[base + 5] / 131.0

            t_s = time_prev + (sample_index + 1) * dt_s_uniform
            yield (t_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
            sample_index += 1


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
    Each line is expected to be tab-separated with 3 parts:
    ISO timestamp, UUID, and hex payload.
    """
    times, uuids, data = [], [], []
    for ln in lines:
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
            times.append(ts)
            uuids.append(parts[1])
            data.append(bytes.fromhex(parts[2]))
        except Exception:
            continue

    acc_chunks = []
    gyro_chunks = []
    t_prev = None

    for pkt in data:
        t_current = extract_time(pkt)

        it = extract_gyroacc(pkt, t_prev, t_current)
        acc, gyro = stream_to_arrays(it)
        # acc, gyro = extract_gyroacc(pkt, t_prev, t_current)
        if acc.size:
            acc_chunks.append(acc)
            gyro_chunks.append(gyro)
        t_prev = t_current

    df_acc = pd.DataFrame(
        np.vstack(acc_chunks), columns=["time", "ACC_X", "ACC_Y", "ACC_Z"]
    )
    df_gyro = pd.DataFrame(
        np.vstack(gyro_chunks), columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    )

    return times, df_acc, df_gyro


# df_acc.plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"])
# df_gyro.plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"])
