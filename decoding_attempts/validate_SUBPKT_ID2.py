import os
import datetime as dt
import numpy as np
import pandas as pd

# ------------------------
# Known type mapping
# ------------------------
DATA_TYPE_MAP = {
    1: "EEG4",
    2: "EEG8",
    3: "DRL_REF",
    4: "Optics4",
    5: "Optics8",
    6: "Optics16",
    7: "ACC_GYRO",
    8: "Battery",
}

# ------------------------
# Helpers
# ------------------------

def parse_lines_fast(lines):
    """Parse lines into timestamps, uuids, and payloads."""
    tobytes = bytes.fromhex
    times = []
    uuids = []
    data = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        ts = parts[0]
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            times.append(dt.datetime.fromisoformat(ts).timestamp())
            uuids.append(parts[1])
            data.append(tobytes(parts[2]))
        except Exception:
            continue
    return np.array(times, dtype=np.float64), uuids, data

def extract_type_ids(data, location=10):
    """Extract type nibble from ID byte at given location (1-based)."""
    type_ids = []
    for payload in data:
        if len(payload) < location:
            continue
        id_byte = payload[location - 1]
        type_ids.append(id_byte & 0x0F)
    return np.array(type_ids, dtype=np.uint8)

# ------------------------
# Main analysis
# ------------------------

def compute_packet_frequencies(filename, data_dir="./data_raw", id_loc=10):
    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
        lines = f.readlines()
    times, uuids, data = parse_lines_fast(lines)
    if len(times) < 2:
        return {}

    type_ids = extract_type_ids(data, location=id_loc)
    n = min(len(times), len(type_ids))
    times = times[:n]
    type_ids = type_ids[:n]

    duration = times[-1] - times[0]
    if duration <= 0:
        return {}

    freqs = {}
    for t_id in np.unique(type_ids):
        label = DATA_TYPE_MAP.get(int(t_id), f"Unknown{t_id}")
        count = np.sum(type_ids == t_id)
        freqs[label] = count / duration
    return freqs

if __name__ == "__main__":
    data_dir = "./data_raw"
    files = sorted(os.listdir(data_dir))

    all_results = []
    for filename in files:
        print(f"Processing {filename}...")
        freqs = compute_packet_frequencies(filename, data_dir=data_dir, id_loc=10)
        freqs["File"] = filename
        all_results.append(freqs)

    results_df = pd.DataFrame(all_results).fillna(0).set_index("File")
    print("\nObserved packet frequencies (packets/s):")
    print(results_df.to_markdown(floatfmt=".2f"))


# Observed packet frequencies (packets/s):
# | File           |   EEG4 |   DRL_REF |   Optics8 |   ACC_GYRO |   Battery |   Optics4 |   EEG8 |   Optics16 |
# |:---------------|-------:|----------:|----------:|-----------:|----------:|----------:|-------:|-----------:|
# | data_p1034.txt |   9.73 |      0.78 |      6.80 |       3.15 |      0.10 |      0.00 |   0.00 |       0.00 |
# | data_p1035.txt |   9.82 |      0.68 |      0.00 |       3.43 |      0.17 |      3.00 |   0.00 |       0.00 |
# | data_p1041.txt |   0.00 |      0.99 |      0.00 |       3.66 |      0.08 |      0.00 |  17.91 |      13.53 |
# | data_p1042.txt |   0.00 |      0.56 |      0.00 |       3.20 |      0.18 |      0.00 |  18.31 |      13.97 |
# | data_p1043.txt |   0.00 |      0.71 |      7.10 |       3.19 |      0.08 |      0.00 |  18.80 |       0.00 |
# | data_p1044.txt |   0.00 |      0.78 |      7.13 |       3.27 |      0.07 |      0.00 |  18.59 |       0.00 |
# | data_p1045.txt |   0.00 |      0.73 |      0.00 |       3.03 |      0.05 |      3.44 |  19.03 |       0.00 |
# | data_p1046.txt |   0.00 |      0.73 |      0.00 |       3.83 |      0.10 |      2.69 |  19.00 |       0.00 |
# | data_p20.txt   |   9.48 |      0.71 |      0.00 |       3.60 |      0.03 |      0.00 |   0.00 |       0.00 |
# | data_p21.txt   |   9.59 |      0.63 |      0.00 |       3.55 |      0.05 |      0.00 |   0.00 |       0.00 |
# | data_p4129.txt |   0.00 |      0.94 |      0.00 |       3.22 |      0.13 |      3.02 |  19.05 |       0.00 |
# | data_p50.txt   |   9.59 |      0.50 |      0.00 |       3.63 |      0.08 |      0.00 |   0.00 |       0.00 |
# | data_p51.txt   |   9.51 |      0.69 |      0.00 |       3.52 |      0.08 |      0.00 |   0.00 |       0.00 |
# | data_p60.txt   |   9.71 |      0.63 |      0.00 |       3.36 |      0.13 |      0.00 |   0.00 |       0.00 |
# | data_p61.txt   |   9.57 |      0.48 |      0.00 |       3.65 |      0.12 |      0.00 |   0.00 |       0.00 |
