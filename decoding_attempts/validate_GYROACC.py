import numpy as np
import pandas as pd
import datetime as dt
import struct
import os


def _parse_lines(lines):
    fromiso = dt.datetime.fromisoformat
    tobytes = bytes.fromhex
    times, uuids, data = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        times.append(fromiso(parts[0].replace("Z", "+00:00")).timestamp())
        uuids.append(parts[1])
        data.append(tobytes(parts[2]))
    return times, uuids, data


def parse_acc_gyro(lines: list[str], debug=True) -> pd.DataFrame:
    """
    Parse Muse ACC/GYRO packets.
    - Packets tagged with 0x47.
    - After tag byte, skip 4 metadata bytes.
    - Then read 18 int16 little-endian values = 3 samples * 6 channels.
    - ACC scaling: ±2g -> 1 g = 16384 counts
    - GYRO scaling: ±250 dps -> 1 dps ≈ 131 counts
    - Timestamps spaced at 52 Hz, aligned so payload time = last sample.
    """
    times, uuids, data = _parse_lines(lines)

    all_values, all_sample_times = [], []
    sample_rate = 52.0
    dt_sample = 1.0 / sample_rate

    dumped = False

    for payload_time, payload in zip(times, data):
        pos = 0
        plen = len(payload)
        while pos < plen:
            tag = payload[pos]
            # if not target tag, skip forward
            if tag != 0x47:
                pos += 1
                continue

            # ensure enough bytes for tag + 4 metadata + 36 data
            min_needed = pos + 1 + 4 + 36
            if min_needed > plen:
                # advance past tag to avoid stalling and continue scanning
                pos += 1
                continue

            payload_start = pos + 1 + 4
            end_index = payload_start + 36
            block = payload[payload_start:end_index]
            try:
                vals = list(struct.unpack("<18h", block))
            except struct.error:
                pos += 1
                continue

            values = np.array(vals, dtype=np.int16).reshape((3, 6))
            if debug and not dumped:
                print("First ACC/GYRO packet dump (18 raw int16):", vals)
                dumped = True

            acc = values[:, 0:3].astype(np.float32) / 16384.0
            gyro = values[:, 3:6].astype(np.float32) / 131.0
            scaled_values = np.hstack((acc, gyro))

            this_times = np.array(
                [payload_time - (2 - i) * dt_sample for i in range(3)], dtype=np.float64
            )

            all_values.append(scaled_values)
            all_sample_times.append(this_times)

            pos = end_index

    if not all_values:
        print("No valid ACC/GYRO data parsed.")
        return pd.DataFrame()

    all_values = np.vstack(all_values)
    all_sample_times = np.concatenate(all_sample_times)

    df = pd.DataFrame(
        all_values, columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    )
    df["time"] = all_sample_times

    # sanity check
    observed_dur = all_sample_times[-1] - all_sample_times[0]
    expected_dur = (len(all_sample_times) - 1) * dt_sample
    print(
        f"Observed duration: {observed_dur:.2f} s, expected {expected_dur:.2f} s from {len(all_sample_times)} samples"
    )

    return df


if __name__ == "__main__":
    data_dir = "./data_raw/"
    files = sorted(os.listdir(data_dir))

    all_dfs = {}

    for filename in files:
        print(f"Processing {filename}...")
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = parse_acc_gyro(lines, debug=False)

        # Compute Observed Frequency
        time_diffs = np.diff(df["time"].values)
        observed_freq = 1.0 / np.median(time_diffs)
        print(f"Observed sampling frequency: {observed_freq:.2f} Hz")

        all_dfs[filename] = df

    all_dfs["data_p50.txt"].plot(
        x="time",
        y=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        subplots=True,
    )
