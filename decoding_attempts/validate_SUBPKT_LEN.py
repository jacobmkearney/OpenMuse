import os
import struct
import pandas as pd
from datetime import datetime

DATA_DIR = "./data_raw/"
FILES = sorted(p for p in os.listdir(DATA_DIR) if p.endswith(".txt"))

# --------------------------------------------------------------------
# Maps for ID-byte decoding
# --------------------------------------------------------------------
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
    3: "DRL_REF",
    4: "Optics4",
    5: "Optics8",
    6: "Optics16",
    7: "ACC_GYRO",
    8: "Battery",
}

# Expected payload lengths (fill in once you’ve measured them empirically)
EXPECTED_LEN_BY_TYPE = {
    "EEG4": None,
    "EEG8": None,
    "ACC_GYRO": None,
    "Optics4": None,
    "Optics8": None,
    "Optics16": None,
    "Battery": None,
    "DRL_REF": None,
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
    freq_code = (id_byte >> 4) & 0x0F
    type_code = id_byte & 0x0F
    return {
        "declared_len": subpkt_len,
        "actual_len": len(payload),
        "freq_code": freq_code,
        "freq_hz": FREQ_MAP.get(freq_code),
        "type_code": type_code,
        "type": TYPE_MAP.get(type_code),
    }


# --------------------------------------------------------------------
def validate_file(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue
            _, payload = parsed
            hdr = extract_header(payload)
            if not hdr:
                continue
            exp_len = EXPECTED_LEN_BY_TYPE.get(hdr["type"])
            hdr["expected_len"] = exp_len
            hdr["len_match"] = (
                None if exp_len is None else (hdr["actual_len"] == exp_len)
            )
            hdr["declared_match"] = hdr["declared_len"] == hdr["actual_len"]
            results.append(hdr)
    return pd.DataFrame(results)


# --------------------------------------------------------------------
if __name__ == "__main__":
    all_results = []
    for fn in FILES:
        print(f"Processing {fn}...")
        df = validate_file(os.path.join(DATA_DIR, fn))
        if df.empty:
            continue
        n = len(df)
        summary = {
            "file": fn,
            "n_packets": n,
            "declared_vs_actual_mismatch_prop": (~df["declared_match"]).mean(),
            "expected_len_mismatch_prop": (
                df["len_match"].eq(False).mean() if "len_match" in df else None
            ),
            "types_seen": df["type"].dropna().unique().tolist(),
        }
        all_results.append(summary)

    print("\nValidation summary (proportions):")
    print(pd.DataFrame(all_results).to_markdown(index=False, floatfmt=".3f"))

# Validation summary (proportions):
# | file           |   n_packets |   declared_vs_actual_mismatch_prop |   expected_len_mismatch_prop | types_seen                                             |
# |:---------------|------------:|-----------------------------------:|-----------------------------:|:-------------------------------------------------------|
# | data_p1034.txt |        1245 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG4', 'Optics8', 'DRL_REF', 'Battery']  |
# | data_p1035.txt |        1036 |                              0.000 |                        0.000 | ['Optics4', 'EEG4', 'ACC_GYRO', 'DRL_REF', 'Battery']  |
# | data_p1041.txt |        2195 |                              0.000 |                        0.000 | ['EEG8', 'ACC_GYRO', 'Optics16', 'DRL_REF', 'Battery'] |
# | data_p1042.txt |        2194 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG8', 'Optics16', 'DRL_REF', 'Battery'] |
# | data_p1043.txt |        1810 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG8', 'Optics8', 'DRL_REF', 'Battery']  |
# | data_p1044.txt |        1807 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG8', 'Optics8', 'DRL_REF', 'Battery']  |
# | data_p1045.txt |        1589 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG8', 'DRL_REF', 'Optics4', 'Battery']  |
# | data_p1046.txt |        1596 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG8', 'DRL_REF', 'Optics4', 'Battery']  |
# | data_p20.txt   |         837 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG4', 'DRL_REF', 'Battery']             |
# | data_p21.txt   |         837 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG4', 'DRL_REF', 'Battery']             |
# | data_p4129.txt |        1597 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG8', 'Optics4', 'DRL_REF', 'Battery']  |
# | data_p50.txt   |         836 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG4', 'DRL_REF', 'Battery']             |
# | data_p51.txt   |         836 |                              0.000 |                        0.000 | ['ACC_GYRO', 'EEG4', 'DRL_REF', 'Battery']             |
# | data_p60.txt   |         836 |                              0.000 |                        0.000 | ['EEG4', 'DRL_REF', 'ACC_GYRO', 'Battery']             |
# | data_p61.txt   |         837 |                              0.000 |                        0.000 | ['EEG4', 'DRL_REF', 'ACC_GYRO', 'Battery']             |
