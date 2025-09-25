from datetime import datetime, timezone
import struct
from typing import Optional
import numpy as np
import pandas as pd


def extract_time(payload: bytes) -> Optional[datetime]:
    """
    Extract packet time from a single payload and return a UTC datetime. 4-byte unsigned little-endian at offset 3 -> milliseconds.
    """
    # primary 4-byte little-endian at offset 3
    if len(payload) >= 3 + 4:
        ms = struct.unpack_from("<I", payload, 3)[0]
        return ms * 1e-3

    return None


def extract_gyroacc(payload: bytes, time_prev, time_current):
    """
    Parse all 0x47 ACC/GYRO blocks in payload and return two arrays:
      - acc: shape (N, 3) floats in g (columns acc_x, acc_y, acc_z)
      - gyro: shape (N, 3) floats in dps (columns gyro_x, gyro_y, gyro_z)

    Each 0x47 block contains 18 int16 values = 3 samples * 6 channels.
    Samples are returned in chronological order (within-block order preserved).
    If no blocks found, returns two empty arrays with shape (0, 3).
    """
    TAG = 0x47
    HEADER_LEN = 1 + 4
    BLOCK_BYTES = 18 * 2
    MIN_BLOCK_LEN = HEADER_LEN + BLOCK_BYTES
    SAMPLE_RATE = 52.0

    acc_blocks = []
    gyro_blocks = []
    start = 0
    plen = len(payload)

    while start + MIN_BLOCK_LEN <= plen:
        idx = payload.find(bytes([TAG]), start)
        if idx == -1:
            break
        if idx + MIN_BLOCK_LEN > plen:
            break
        block_start = idx + HEADER_LEN
        try:
            raw = struct.unpack_from("<18h", payload, block_start)
        except struct.error:
            start = idx + 1
            continue
        vals = np.array(raw, dtype=np.int16).reshape((3, 6)).astype(np.float32)
        acc_block = vals[:, 0:3] / 16384.0
        gyro_block = vals[:, 3:6] / 131.0
        acc_blocks.append(acc_block)
        gyro_blocks.append(gyro_block)
        start = block_start + BLOCK_BYTES

    if not acc_blocks:
        return (np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32))

    acc = np.vstack(acc_blocks)  # (total_samples, 3)
    gyro = np.vstack(gyro_blocks)  # (total_samples, 3)
    total_samples = acc.shape[0]

    # compute uniform per-sample spacing (ms)
    if time_prev is not None:
        dt_ms = (time_current - time_prev) / float(total_samples)
        # t = time_prev + dt_ms * [1, 2, ..., total_samples] so last == time_current
        t = time_prev + dt_ms * np.arange(1, total_samples + 1)
    else:
        dt_ms = 1000.0 / SAMPLE_RATE
        # align last sample to time_current
        t = time_current - dt_ms * (np.arange(total_samples - 1, -1, -1))
        t = t[::-1]  # ascending order

    # prepend time column
    acc = np.hstack((t.reshape(-1, 1), acc.astype(np.float32)))
    gyro = np.hstack((t.reshape(-1, 1), gyro.astype(np.float32)))

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

    d_gyro = []
    d_acc = []
    t_current, t_prev = None, None
    for pkt in data:
        t_current = extract_time(pkt)
        acc, gyro = extract_gyroacc(pkt, t_prev, t_current)
        d_acc.append(acc)
        d_gyro.append(gyro)
        t_prev = t_current

    df_acc = pd.DataFrame(np.vstack(d_acc), columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])

    return times
