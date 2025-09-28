import os
import struct
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

DATA_DIR = "./data_raw/"
FILES = sorted(p for p in os.listdir(DATA_DIR) if p.endswith(".txt"))

TYPE_MAP = {
    1: "EEG4",
    2: "EEG8",
    3: "DRL_REF",
    4: "Optics4",
    5: "Optics8",
    6: "Optics16",
    7: "ACC_GYRO",
    8: "Battery",
}


# --------------------------------------------------------------------
def parse_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None
    ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
    payload = bytes.fromhex(parts[2])
    return ts, payload


def extract_header(payload: bytes):
    if len(payload) < 14:
        return None
    subpkt_len = payload[0]
    id_byte = payload[9]
    type_code = id_byte & 0x0F
    return {
        "declared_len": subpkt_len,
        "actual_len": len(payload),
        "type": TYPE_MAP.get(type_code),
    }


def extract_battery(payload: bytes):
    HEADER_LEN = 14
    if len(payload) < HEADER_LEN + 14:
        return None
    data = payload[HEADER_LEN : HEADER_LEN + 14]
    raw_soc, raw_mv, raw_temp, r1, r2, r3, r4 = struct.unpack("<7H", data)
    return {
        "battery_percent": raw_soc / 256.0,  # map to 0–100 %
        "voltage_mV": raw_mv * 16,  # adjust scale
        "temperature_C": raw_temp / 100.0,  # may still need correction
        "diag1": r1,
        "diag2": r2,
        "diag3": r3,
        "diag4": r4,
    }


# --------------------------------------------------------------------
def validate_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue
            ts, payload = parsed
            hdr = extract_header(payload)
            if not hdr or hdr["type"] != "Battery":
                continue
            values = extract_battery(payload)
            if not values:
                continue
            values["ts"] = ts
            rows.append(values)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------
if __name__ == "__main__":
    all_summaries = []
    for fn in FILES:
        print(f"Processing {fn}...")
        df = validate_file(os.path.join(DATA_DIR, fn))
        if df.empty:
            continue

        summary = {
            "file": fn,
            "n": len(df),
            "battery_mean": df["battery_percent"].mean(),
            "battery_min": df["battery_percent"].min(),
            "battery_max": df["battery_percent"].max(),
            "voltage_mean_mV": df["voltage_mV"].mean(),
            "temp_mean_C": df["temperature_C"].mean() / 100.0,  # convert centi-°C
        }
        all_summaries.append(summary)

        # Plot battery percent and voltage for sanity check
        plt.figure(figsize=(10, 4))
        plt.plot(df["ts"], df["battery_percent"], label="Battery %")
        plt.plot(df["ts"], df["voltage_mV"] / 50, label="Voltage (scaled)")
        plt.title(f"Battery trace: {fn}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\nSummary:")
    print(pd.DataFrame(all_summaries).to_markdown(index=False, floatfmt=".2f"))
