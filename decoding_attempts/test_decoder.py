import numpy as np
import pandas as pd
import os
import datetime as dt


def parse_lines(lines):
    """
    Each line has three tab-separated fields:
    - Timestamp (ISO 8601 with microseconds and timezone)
    - UUID / session ID
    - Hex payload (the actual Muse packet)
    """
    fromiso = dt.datetime.fromisoformat
    tobytes = bytes.fromhex

    times, uuids, data = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        times.append(fromiso(parts[0].replace("Z", "+00:00")).timestamp())
        uuids.append(parts[1])
        data.append(tobytes(parts[2]))

    # If more than one UUID, warn
    if len(set(uuids)) > 1:
        print(f"Warning: Multiple UUIDs found: {set(uuids)}")

    return times, uuids, data


def _decode_ch52(raw_bytes):

    CH52_HEADERS = {
        0xCD,
        0xCF,
        0xD3,
        0xD5,
        0xD7,
        0xD9,
        0xDB,
        0xDD,
        0xDF,
        0xE1,
        0xE3,
        0xE4,
        0xE7,
        0xEA,
        0xEB,
        0xF2,
    }
    try:
        header = raw_bytes[0]
        if header not in CH52_HEADERS:
            # Not a valid CH52 packet
            return np.empty((0, 6), dtype=np.int16)

        payload = raw_bytes[1:]
        n = len(payload) // 12  # 6 channels × 2 bytes
        if n == 0:
            return np.empty((0, 6), dtype=np.int16)

        data = np.frombuffer(payload[: n * 12], dtype="<i2")
        samples = data.reshape(n, 6)
        return samples
    except Exception:
        return np.empty((0, 6), dtype=np.int16)


def decode_channels(lines):
    times, uuids, data = parse_lines(lines)

    all_ch52 = []
    all_times = []

    for t, raw in zip(times, data):
        ch52 = _decode_ch52(raw)
        if ch52.size == 0:
            continue

        # If each packet contains multiple samples, spread timestamps
        # Here: assume uniform spacing within the packet
        n = ch52.shape[0]
        if n > 1:
            # distribute timestamps linearly across the packet duration
            # (you may want to refine this if you know the exact sample rate)
            ts = np.linspace(t, t, n)  # placeholder: all same timestamp
        else:
            ts = [t]

        all_ch52.append(ch52)
        all_times.extend(ts)

    df = pd.DataFrame(
        np.vstack(all_ch52), columns=["ACCx", "ACCy", "ACCz", "GYRx", "GYRy", "GYRz"]
    )
    df.insert(0, "Time", all_times)

    return df


if __name__ == "__main__":
    all_results = []
    for fname in os.listdir("data_raw"):
        # fname = "data_p1034.txt"

        with open(os.path.join("data_raw", fname), "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = decode_channels(lines)
        df.plot(
            x="Time", y=["ACCx", "ACCy", "ACCz", "GYRx", "GYRy", "GYRz"], subplots=True
        )
