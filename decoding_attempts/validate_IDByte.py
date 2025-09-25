import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import datetime as dt
import os
import scipy

# --- Prior knowledge ---

### Constructor Information

# Muse S Athena specs (From the [Muse website](https://eu.choosemuse.com/products/muse-s-athena) - note that these info might not be up to date or fully accurate):
# - Wireless Connection: BLE 5.3, 2.4 GHz
# - EEG Channels: 4 EEG channels (TP9, AF7, AF8, TP10) + 1 (or 4?) amplified Aux channels
#   - Sample Rate: 256 Hz
#   - Sample Resolution: 14 bits / sample
# - Accelerometer: Three-axis at 52Hz, 16-bit resolution, range +/- 2G
# - Gyroscope: Three-axis at 52Hz, 16-bit resolution, range +/- 250dps
# - PPG Sensor: Triple wavelength: IR (850nm), Near-IR (730nm), Red (660nm), 64 Hz sample rate, 20-bit resolution
# - fNIRS Sensor: 5-optode bilateral frontal cortex hemodynamics, 64 Hz sample rate, 20-bit resolution
#   - Might result in 1, 4, 5, 8, 16 OPTICS channels

### Presets

# Different presets enable/disable some channels, but the exact combinations are not fully documented.
# - p20-p61: Red LED in the centre is off
# - p1034, p1041, p1042, p1043: red LED in the centre is brightly on (suggesting the activation of OPTICS or PPG channels)
# - p1035, p1044, p1045, p4129: red LED in the centre is dimmer

# Based on these specs, we can derive the following plausible expectations regarding each channel type:


KNOWN_RATES = {
    "EEG": 256.0,
    "AUX": 256.0,
    "ACC": 52.0,
    "GYRO": 52.0,
    "PPG": 64.0,
    "OPTICS": 64.0,
}

KNOWN_CHANNELS = {
    "EEG": [0, 4],  #  256 Hz, 14 bits
    "AUX": [0, 1, 4],  # 256 Hz, 14 bits
    "ACC": [0, 3],  #  52 Hz, 16 bits
    "GYRO": [0, 3],  #  52 Hz, 16 bits
    "PPG": [0, 3],  # 64 Hz, 20 bits
    "OPTICS": [0, 1, 4, 5, 8, 16],  # 64 Hz, 20 bits
}

# Each data file (e.g., ./data_raw/data_p1045.txt) should contain some combination of these channels.
# The exact combination depends on the preset used during recording.
# Importantly, these channels types are likely indistinguishable from the data alone, so it is best to group them according to their data characteristics in GROUPS, namely CH256, CH52, CH64.

# --- Expectations ---

# These are different possible channel numbers for each group that might show up in various presets.
EXPECTED_GROUPS = {
    "CH256": list(
        set(
            KNOWN_CHANNELS["EEG"]
            + KNOWN_CHANNELS["AUX"]
            + [
                i + j
                for i in KNOWN_CHANNELS["EEG"]
                for j in KNOWN_CHANNELS["AUX"]
                if j > 0
            ]
        )
    ),
    "CH52": list(
        set(
            KNOWN_CHANNELS["ACC"]
            + KNOWN_CHANNELS["GYRO"]
            + [
                i + j
                for i in KNOWN_CHANNELS["ACC"]
                for j in KNOWN_CHANNELS["GYRO"]
                if j > 0
            ]
        )
    ),
    "CH64": list(
        set(
            KNOWN_CHANNELS["PPG"]
            + KNOWN_CHANNELS["OPTICS"]
            + [
                i + j
                for i in KNOWN_CHANNELS["PPG"]
                for j in KNOWN_CHANNELS["OPTICS"]
                if j > 0
            ]
        )
    ),
}

# Expected sampling rates (Hz) and bits per sample for each candidate channel type
EXPECTED_RATES = {
    "CH256": 256.0,
    "CH52": 52.0,
    "CH64": 64.0,
}
BITS_PER_SAMPLE = {
    "CH256": 14,  # treat bit-width as a constraint, not packing
    "CH52": 16,
    "CH64": 20,
}


# --- Assumptions ---

# Prior work (https://github.com/AbosaSzakal/MuseAthenaDataformatParser) suggested that the data packages follows this format:

# <length in bytes>
# <packet counter>
# <unknown 7 bytes>
# <packet id byte - see below>
# <unknown 4 bytes, first one likely a counter>
# <samples tightly packed>
# <other subpackets in the same format>

# Packet ID Byte

# First 4 bits (Frequency):
# - 0 = frequency not valid
# - 1 = 256 Hz
# - 2 = 128 Hz
# - 3 = 64 Hz
# - 4 = 52 Hz
# - 5 = 32 Hz
# - 6 = 16 Hz
# - 7 = 10 Hz
# - 8 = 1 Hz
# - 9 = 0.1 Hz
#
# Second 4 bits (Data Type):
# - 0 = not valid
# - 1 = EEG, 4 channels
# - 2 = EEG, 8 channels
# - 3 = DRL/REF
# - 4 = Optics, 4 channels
# - 5 = Optics, 8 channels
# - 6 = Optics, 16 channels
# - 7 = Accelerometer + Gyroscope
# - 8 = Battery
# Note: This leaves quite a few types of data unaccounted for


# --- GOAL ---

# Validate the structure of the data for each file. Namely:
# 1. Each line has three tab-separated fields:
# - Timestamp (ISO 8601 with microseconds and timezone)
# - UUID
# - Hex payload (the actual Muse packet)
# STATUS: DONE

# 2. Payload structure: id byte for FREQ
# Extract the the number and rate of the FREQ identifiers in the id bytes
# Test: identifiers of higher frequencies should not appear more frequently than those of lower frequencies.
# STATUS:
# - Initial information suggest that the frequency bit is at byte 12 (1-based indexing) of the payload
# - However, empirical results suggested that the location is most likely 10th

# 3. Payload structure: id byte for DATA TYPE
# Extract the the number and rate of the DATA TYPE identifiers in the id bytes
# Test: The rate of channel types should be consistent with the expected rates of the channel types and the possible number of channels
# STATUS: Results confirmed previous findings: 10th byte consistent with type info


# ======================================================================
# Functions ============================================================
# ======================================================================
def _parse_lines(lines):
    fromiso = dt.datetime.fromisoformat
    tobytes = bytes.fromhex

    errors = 0

    times, uuids, data = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print(f"Skipping malformed line: {line.strip()}")
            errors += 1
            continue
        times.append(fromiso(parts[0].replace("Z", "+00:00")).timestamp())
        uuids.append(parts[1])
        data.append(tobytes(parts[2]))

    # If more than one UUID, warn
    if len(set(uuids)) > 1:
        print(f"WARNING: Multiple UUIDs found: {set(uuids)}")

    return times, uuids, data, errors / len(lines)


def assess_id_bytes(data, location=12):
    """Analyse ID byte (at given location) containing freq (upper nibble) and type (lower nibble)."""

    # Maps ---------------------------------------------------------------
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

    EXPECTED_FREQ_BY_TYPE = {
        "EEG4": 256.0,
        "EEG8": 256.0,
        "ACC_GYRO": 52.0,
        "Optics4": 64.0,
        "Optics8": 64.0,
        "Optics16": 64.0,
        "Battery": None,
        "DRL_REF": None,
    }

    # Extract nibbles ----------------------------------------------------
    freq_ids, type_ids = [], []
    errors = 0
    for payload in data:
        if len(payload) < location:
            errors += 1
            continue
        id_byte = payload[location - 1]
        freq_ids.append((id_byte >> 4) & 0x0F)
        type_ids.append(id_byte & 0x0F)

    n = len(data)
    parse_err_rate = errors / n if n > 0 else 1.0

    rez = {"IDLoc": location, "IDParseErrors": parse_err_rate}

    # Frequency summary --------------------------------------------------
    freq_counts = pd.Series(freq_ids).value_counts().sort_index()
    freq_df = pd.DataFrame({"count": freq_counts})
    freq_df["freq_hz"] = freq_df.index.map(FREQ_MAP)
    rez["FreqUnknown"] = (
        freq_df[freq_df["freq_hz"].isna()]["count"].sum() / freq_df["count"].sum()
    )

    corr = freq_df["count"].corr(freq_df["freq_hz"], method="spearman")
    rez["FreqPattern"] = 1 - np.abs(corr) if len(freq_df) > 2 else 1.0

    # Type summary -------------------------------------------------------
    type_counts = pd.Series(type_ids).value_counts().sort_index()
    type_df = pd.DataFrame({"count": type_counts})
    type_df["type"] = type_df.index.map(DATA_TYPE_MAP)
    rez["TypeUnknown"] = (
        type_df[type_df["type"].isna()]["count"].sum() / type_df["count"].sum()
    )
    rez["TypeDiversity"] = (8 - len(type_df["type"].dropna())) / 8
    rez["TypeUseful"] = int(not any(type_df["type"].isin(["EEG4", "EEG8", "ACC_GYRO"])))

    # Consistency check --------------------------------------------------
    type_df["freq_hz2"] = type_df["type"].map(EXPECTED_FREQ_BY_TYPE)
    merged = freq_df.merge(type_df, on="count", how="inner")
    merged["freq_diff"] = np.abs(merged["freq_hz"] - merged["freq_hz2"])

    rez["TypeFreqMismatch"] = merged["freq_diff"].mean()

    return rez


# Assessment -------------------------------------------
def assess_structure(filename, loc_id=12):
    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
        lines = f.readlines()
    times, uuids, data, err_rate = _parse_lines(lines)

    result = {
        "Preset": filename.replace("data_", "").replace(".txt", ""),
        "LineStructureErrors": err_rate,
    }

    result.update(assess_id_bytes(data, location=loc_id))

    return pd.DataFrame(result, index=[0])


# ------------------------
# Script entrypoint
# ------------------------
if __name__ == "__main__":
    data_dir = "./data_raw"
    files = sorted(os.listdir(data_dir))

    results = []

    for filename in files:
        print(f"Processing {filename}...")

        # Grid search for best ID byte location (joint freq+type check)
        grid = [assess_structure(filename, loc_id=loc_id) for loc_id in range(8, 20)]
        grid = pd.concat([d.dropna(how="all", axis=1) for d in grid], ignore_index=True)

        # Score: combine how well freq + type match expectations
        score = (
            grid["FreqUnknown"]
            + grid["FreqPattern"]
            + grid["TypeUnknown"]
            + grid["TypeUseful"]
        ) / 4

        # Pick best location
        best_idx = score.idxmin()
        results.append(grid.loc[[best_idx], :])

    results = pd.concat(results, ignore_index=True)

    print("\nID byte results:")
    print(results.to_markdown(floatfmt=(".2f")))


# ID byte results:
# |    | Preset   |   LineStructureErrors |   IDLoc |   IDParseErrors |   FreqUnknown |   FreqPattern |   TypeUnknown |   TypeDiversity |   TypeUseful |   TypeFreqMismatch |
# |---:|:---------|----------------------:|--------:|----------------:|--------------:|--------------:|--------------:|----------------:|-------------:|-------------------:|
# |  0 | p1034    |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.38 |            0 |               0.00 |
# |  1 | p1035    |                  0.00 |      10 |            0.00 |          0.00 |          0.10 |          0.00 |            0.38 |            0 |               0.00 |
# |  2 | p1041    |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.38 |            0 |               0.00 |
# |  3 | p1042    |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.38 |            0 |               0.00 |
# |  4 | p1043    |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.38 |            0 |               0.00 |
# |  5 | p1044    |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.38 |            0 |               0.00 |
# |  6 | p1045    |                  0.00 |      10 |            0.00 |          0.00 |          0.10 |          0.00 |            0.38 |            0 |               0.00 |
# |  7 | p1046    |                  0.00 |      10 |            0.00 |          0.00 |          0.10 |          0.00 |            0.38 |            0 |               0.00 |
# |  8 | p20      |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.50 |            0 |               0.00 |
# |  9 | p21      |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.50 |            0 |               0.00 |
# | 10 | p4129    |                  0.00 |      10 |            0.00 |          0.00 |          0.10 |          0.00 |            0.38 |            0 |               0.00 |
# | 11 | p50      |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.50 |            0 |               0.00 |
# | 12 | p51      |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.50 |            0 |               0.00 |
# | 13 | p60      |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.50 |            0 |               0.00 |
# | 14 | p61      |                  0.00 |      10 |            0.00 |          0.00 |          0.00 |          0.00 |            0.50 |            0 |               0.00 |
