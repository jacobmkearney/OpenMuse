import os
import glob
import struct
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the extract functions from previous code


def extract_pkt_accgyro_split(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    if len(data) % 2 != 0:
        data = data[:-1]

    raw_i16 = np.frombuffer(data, dtype="<i2").astype(np.int32)
    total_shorts = raw_i16.size
    if total_shorts % 6 != 0:
        leftover = total_shorts % 6
        total_shorts -= leftover
    else:
        leftover = 0

    if total_shorts < 6:
        return None

    n_samples = total_shorts // 6
    acc_raw = raw_i16[: 3 * n_samples].reshape(n_samples, 3)
    gyr_raw = raw_i16[3 * n_samples : 3 * n_samples + 3 * n_samples].reshape(
        n_samples, 3
    )

    acc = acc_raw.astype(np.float32) / 16384.0
    gyr = gyr_raw.astype(np.float32) / 131.0

    ts = _timestamps_for(n_samples, time_current, time_prev)

    diagnostics = {
        "n_samples": n_samples,
        "leftover_shorts": leftover,
        "layout": "split_acc_then_gyro",
    }

    return {
        "type": "ACCGYRO",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def extract_pkt_accgyro_interleaved(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    if len(data) % 2 != 0:
        data = data[:-1]

    raw_i16 = np.frombuffer(data, dtype="<i2").astype(np.int32)
    total_shorts = raw_i16.size
    if total_shorts % 6 != 0:
        leftover = total_shorts % 6
        total_shorts -= leftover
    else:
        leftover = 0

    if total_shorts < 6:
        return None

    n_samples = total_shorts // 6
    arr = raw_i16[:total_shorts].reshape(n_samples, 6)
    acc_raw = arr[:, :3]
    gyr_raw = arr[:, 3:]

    acc = acc_raw.astype(np.float32) / 16384.0
    gyr = gyr_raw.astype(np.float32) / 131.0
    ts = _timestamps_for(n_samples, time_current, time_prev)

    diagnostics = {
        "n_samples": n_samples,
        "leftover_shorts": leftover,
        "layout": "interleaved_per_sample",
    }

    return {
        "type": "ACCGYRO",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def extract_pkt_accgyro_separate(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
    GYRO_FS = 52.0
    GYRO_DT = 1.0 / GYRO_FS
    HEADER_OFFSET = 14

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    if len(data) % 2 != 0:
        data = data[:-1]

    raw_i16 = np.frombuffer(data, dtype="<i2").astype(np.int32)
    total_shorts = raw_i16.size
    if total_shorts % 3 != 0:
        leftover = total_shorts % 3
        total_shorts -= leftover
    else:
        leftover = 0

    if total_shorts < 3:
        return None

    n_samples = total_shorts // 3
    raw = raw_i16[:total_shorts].reshape(n_samples, 3)
    ts = _timestamps_for(n_samples, time_current, time_prev)

    # Scale
    acc_trial = raw.astype(np.float32) / 16384.0
    gyr_trial = raw.astype(np.float32) / 131.0

    # Detect type by plausible range (acc norm ~1, gyro ~0 when still)
    acc_norm = np.mean(np.linalg.norm(acc_trial, axis=1))
    gyr_norm = np.mean(np.linalg.norm(gyr_trial, axis=1))

    if acc_norm > 0.5 and acc_norm < 1.5:  # Plausible for acc
        acc = acc_trial
        gyr = np.full_like(acc, np.nan)
        sensor_type = "ACC"
    else:
        acc = np.full_like(gyr_trial, np.nan)
        gyr = gyr_trial
        sensor_type = "GYRO"

    diagnostics = {
        "n_samples": n_samples,
        "leftover_shorts": leftover,
        "sensor_type": sensor_type,
        "layout": "separate_3ch",
    }

    return {
        "type": "ACCGYRO",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def extract_pkt_accgyro_tagged(
    payload: bytes, time_current: float, time_prev: Optional[float] = None
) -> Optional[Dict]:
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
    found = 0
    while pos < nbytes:
        idx = data.find(bytes([GYRO_TAG]), pos)
        if idx == -1:
            break
        block_start = idx + 1 + 4  # tag + 4 hdr
        if block_start + 36 > nbytes:
            break
        try:
            raw18 = struct.unpack_from("<18h", data, block_start)
            blocks.append(raw18)
            found += 1
        except struct.error:
            pos = idx + 1
            continue
        pos = block_start + 36

    if not blocks:
        return None

    raw_all = np.array(blocks, dtype=np.int16).reshape(-1, 6)
    acc = raw_all[:, :3].astype(np.float32) / 16384.0
    gyr = raw_all[:, 3:6].astype(np.float32) / 131.0
    n_samples = raw_all.shape[0]
    ts = _timestamps_for(n_samples, time_current, time_prev)

    diagnostics = {
        "n_samples": n_samples,
        "blocks_found": len(blocks),
        "layout": "tagged_blocks",
        "leftover_bytes": nbytes - pos if pos < nbytes else 0,
    }

    return {
        "type": "ACCGYRO",
        "time": ts,
        "ACC": acc,
        "GYRO": gyr,
        "diagnostics": diagnostics,
    }


def _timestamps_for(
    n_samples: int, time_current: float, time_prev: Optional[float]
) -> np.ndarray:
    GYRO_DT = 1.0 / 52.0
    if time_prev is not None:
        if n_samples == 0:
            return np.array([])
        return (
            np.linspace(
                time_prev, time_current, n_samples, endpoint=False, dtype=np.float64
            )
            + (time_current - time_prev) / n_samples
        )
    else:
        return (
            time_current
            - (n_samples - 1) * GYRO_DT
            + GYRO_DT * np.arange(n_samples, dtype=np.float64)
        )


def decode_rawdata(lines: List[str], extract_fn) -> Dict[str, pd.DataFrame]:
    decoded = []
    prev_imu_time = None

    for ln in lines:
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            decoded.append(None)
            continue

        try:
            ts = pd.to_datetime(parts[0]).timestamp()
            uuid = parts[1]
            payload = bytes.fromhex(parts[2])
        except Exception:
            decoded.append(None)
            continue

        pkt_time = extract_pkt_time(payload)
        pkt_freq, pkt_type = extract_pkt_id(payload)

        pkt_decoded = None
        if pkt_type == "ACCGYRO":
            pkt_decoded = extract_fn(payload, pkt_time, prev_imu_time)
            if pkt_decoded and len(pkt_decoded["time"]) > 0:
                prev_imu_time = pkt_decoded["time"][-1]

        decoded.append(
            {
                "uuid": uuid,
                "line_time": ts,
                "pkt_time": pkt_time,
                "pkt_type": pkt_type,
                "data": pkt_decoded,
            }
        )

    acc_chunks, gyro_chunks = [], []
    diagnostics_list = []

    for pkt in decoded:
        if not pkt or not pkt["data"]:
            continue
        if pkt["pkt_type"] == "ACCGYRO":
            ts = pkt["data"]["time"][:, None]
            acc = pkt["data"]["ACC"]
            gyr = pkt["data"]["GYRO"]
            acc_chunks.append(np.hstack([ts, acc]))
            gyro_chunks.append(np.hstack([ts, gyr]))
            diagnostics_list.append(pkt["data"]["diagnostics"])

    df_acc = pd.DataFrame(
        np.vstack(acc_chunks) if acc_chunks else [],
        columns=["time", "ACC_X", "ACC_Y", "ACC_Z"],
    )
    df_gyro = pd.DataFrame(
        np.vstack(gyro_chunks) if gyro_chunks else [],
        columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    )

    return {"ACC": df_acc, "GYRO": df_gyro, "diagnostics": diagnostics_list}


# Header extract functions (from original code)


def extract_pkt_time(payload: bytes) -> Optional[float]:
    if len(payload) >= 6:
        ms = struct.unpack_from("<I", payload, 2)[0]
        return ms * 1e-3
    return None


def extract_pkt_id(payload: bytes) -> Tuple[Optional[float], Optional[str]]:
    if len(payload) <= 9:
        return None, None

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


# Main script

methods = {
    "split": extract_pkt_accgyro_split,
    "interleaved": extract_pkt_accgyro_interleaved,
    "separate": extract_pkt_accgyro_separate,
    "tagged": extract_pkt_accgyro_tagged,
}

folder = "./data_raw/"
files: List[str] = glob.glob(os.path.join(folder, "*.txt"))

for file in files:
    # file = "./data_raw\\data_p1034.txt"

    with open(file, "r") as f:
        lines = f.readlines()

    print(f"Processing file: {file}")

    # Create one figure per file
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), sharex=True)
    fig.suptitle(f"Decoding Methods for {os.path.basename(file)}")

    row = 0
    for method_name, extract_fn in methods.items():
        data = decode_rawdata(lines, extract_fn)
        diagnostics = data["diagnostics"]
        avg_leftover = (
            np.mean(
                [
                    d.get("leftover_shorts", 0) or d.get("leftover_bytes", 0)
                    for d in diagnostics
                ]
            )
            if diagnostics
            else 0
        )
        n_pkts = len(diagnostics)

        print(f"  Method: {method_name}")
        print(f"    Diagnostics: n_pkts={n_pkts}, avg_leftover={avg_leftover}")

        df_acc = data["ACC"]
        df_gyro = data["GYRO"]

        # Plot ACC on left, GYRO on right
        ax_acc = axs[row, 0]
        ax_gyro = axs[row, 1]

        if not df_acc.empty:
            t_rel = df_acc["time"] - df_acc["time"].min()
            for col in ["ACC_X", "ACC_Y", "ACC_Z"]:
                ax_acc.plot(t_rel, df_acc[col], label=col)
            acc_mean = df_acc[["ACC_X", "ACC_Y", "ACC_Z"]].mean().values
            acc_std = df_acc[["ACC_X", "ACC_Y", "ACC_Z"]].std().values
            acc_norm = np.linalg.norm(acc_mean)
            plaus_acc = abs(acc_norm - 1) + acc_std.mean()
            ax_acc.set_title(
                f"{method_name} ACC (n={len(df_acc)}, norm={acc_norm:.2f}, plaus={plaus_acc:.2f})"
            )
            ax_acc.legend()
            ax_acc.grid(True)
            print(
                f"    ACC: n={len(df_acc)}, mean={acc_mean}, std={acc_std}, norm={acc_norm:.2f}"
            )

        else:
            ax_acc.set_title(f"{method_name} ACC: empty")
            print("    ACC: empty")

        if not df_gyro.empty:
            t_rel = df_gyro["time"] - df_gyro["time"].min()
            for col in ["GYRO_X", "GYRO_Y", "GYRO_Z"]:
                ax_gyro.plot(t_rel, df_gyro[col], label=col)
            gyro_mean = df_gyro[["GYRO_X", "GYRO_Y", "GYRO_Z"]].mean().values
            gyro_std = df_gyro[["GYRO_X", "GYRO_Y", "GYRO_Z"]].std().values
            plaus_gyro = np.abs(gyro_mean).mean() + gyro_std.mean()
            ax_gyro.set_title(
                f"{method_name} GYRO (n={len(df_gyro)}, plaus={plaus_gyro:.2f})"
            )
            ax_gyro.legend()
            ax_gyro.grid(True)
            print(f"    GYRO: n={len(df_gyro)}, mean={gyro_mean}, std={gyro_std}")

        else:
            ax_gyro.set_title(f"{method_name} GYRO: empty")
            print("    GYRO: empty")

        row += 1

    axs[-1, 0].set_xlabel("Relative Time (s)")
    axs[-1, 1].set_xlabel("Relative Time (s)")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# ISOLATION
# --------------------------------------------------------------------

import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Core decoder ---


def extract_pkt_accgyro(payload: bytes, time: float):
    HEADER_OFFSET = 14
    GYRO_TAG = 0x47
    GYRO_DT = 1.0 / 52.0

    if len(payload) <= HEADER_OFFSET:
        return None

    data = payload[HEADER_OFFSET:]
    nbytes = len(data)
    blocks, pos = [], 0

    while pos < nbytes:
        idx = data.find(bytes([GYRO_TAG]), pos)
        if idx == -1:
            break
        block_start = idx + 1 + 4  # tag + 4 hdr
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

    raw_all = np.array(blocks, dtype=np.int16).reshape(-1, 6)
    acc = raw_all[:, :3].astype(np.float32) / 16384.0
    gyr = raw_all[:, 3:6].astype(np.float32) / 131.0
    n_samples = raw_all.shape[0]
    ts = time - (n_samples - 1) * GYRO_DT + GYRO_DT * np.arange(n_samples)

    return {"time": ts, "ACC": acc, "GYRO": gyr}


def extract_pkt_time(payload: bytes):
    if len(payload) >= 6:
        ms = struct.unpack_from("<I", payload, 2)[0]
        return ms * 1e-3
    return None


def extract_pkt_id(payload: bytes):
    if len(payload) <= 9:
        return None, None
    TYPE_MAP = {7: "ACCGYRO"}
    id_byte = payload[9]
    type_code = id_byte & 0x0F
    return None, TYPE_MAP.get(type_code)


def decode_rawdata(lines):
    decoded = []
    prev_imu_time = None

    for ln in lines:
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            ts = pd.to_datetime(parts[0]).timestamp()
            payload = bytes.fromhex(parts[2])
        except Exception:
            continue

        pkt_time = extract_pkt_time(payload)
        _, pkt_type = extract_pkt_id(payload)
        if pkt_type == "ACCGYRO":
            pkt = extract_pkt_accgyro(payload, pkt_time)
            if pkt:
                decoded.append(pkt)

    if not decoded:
        return pd.DataFrame(), pd.DataFrame()

    acc_chunks = [np.hstack([d["time"][:, None], d["ACC"]]) for d in decoded]
    gyr_chunks = [np.hstack([d["time"][:, None], d["GYRO"]]) for d in decoded]

    df_acc = pd.DataFrame(
        np.vstack(acc_chunks), columns=["time", "ACC_X", "ACC_Y", "ACC_Z"]
    )
    df_gyr = pd.DataFrame(
        np.vstack(gyr_chunks), columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    )
    return df_acc, df_gyr


# --- Main ---

file = "./data_raw/data_p1045.txt"  # change to your file
with open(file, "r") as f:
    lines = f.readlines()

df_acc, df_gyr = decode_rawdata(lines)

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

if not df_acc.empty:
    t_rel = df_acc["time"] - df_acc["time"].min()
    for col in ["ACC_X", "ACC_Y", "ACC_Z"]:
        axs[0].plot(t_rel, df_acc[col], label=col)
    axs[0].set_title("Accelerometer")
    axs[0].legend()
    axs[0].grid(True)

if not df_gyr.empty:
    t_rel = df_gyr["time"] - df_gyr["time"].min()
    for col in ["GYRO_X", "GYRO_Y", "GYRO_Z"]:
        axs[1].plot(t_rel, df_gyr[col], label=col)
    axs[1].set_title("Gyroscope")
    axs[1].legend()
    axs[1].grid(True)

axs[1].set_xlabel("Relative Time (s)")
plt.tight_layout()
plt.show()
