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
# STATUS: NOT DONE YET


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


# Analyze the packet id bytes -------------------------------------------
def _extract_freq_ids(data, location=12):
    """Extract frequency identifiers (upper 4 bits of the id byte) from each payload."""
    freq_ids = []
    errors = 0
    for payload in data:
        if (
            len(payload) < location
        ):  # need at least 'location' bytes to access the id byte
            errors += 1
            continue  # malformed or too short
        id_byte = payload[location - 1]  # 'location'th byte (0-indexed)
        freq_code = (id_byte >> 4) & 0x0F
        freq_ids.append(freq_code)
    return freq_ids, errors / len(data)


def assess_freq_ids(data, location=12):
    freq_ids, err_rate = _extract_freq_ids(data, location=location)
    counts = pd.Series(freq_ids).value_counts().sort_index()

    # Map frequency codes to Hz (from spec)
    freq_map = {
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

    counts = pd.DataFrame(counts)
    counts["freq"] = counts.index.map(freq_map)

    # How many codes detected are unexpected (i.e., not in freq_map)
    unexpected = counts[counts["freq"].isna()]["count"].sum() / counts["count"].sum()

    # Spearman correlation between count and freq
    corr = counts["count"].corr(counts["freq"], method="spearman")

    result = {
        "FreqLoc": location,
        "FreqParseErrors": err_rate,
        "FreqUnnexpected": unexpected,
        "FreqExpectedPattern": 1 - np.abs(corr),
    }

    if len(counts) <= 2:
        result["FreqExpectedPattern"] = 1.0  # not enough data to assess pattern
    return result


def _extract_type_ids(data, location=12):
    data_ids = []
    errors = 0
    for payload in data:
        if len(payload) < location:
            errors += 1
            continue
        id_byte = payload[location - 1]
        dtype_code = id_byte & 0x0F
        data_ids.append(dtype_code)
    return data_ids, errors / len(data)


def assess_types_ids(data, location=12):
    DATA_TYPE_MAP = {
        # 0: "Invalid",
        1: "EEG4",
        2: "EEG8",
        3: "DRL_REF",
        4: "Optics4",
        5: "Optics8",
        6: "Optics16",
        7: "ACC_GYRO",
        8: "Battery",
    }

    dtype_ids, err_rate = _extract_type_ids(data, location=location)
    counts = pd.Series(dtype_ids).value_counts().sort_index()
    counts = pd.DataFrame(counts)
    counts["type"] = counts.index.map(DATA_TYPE_MAP)
    unexpected = counts[counts["type"].isna()]["count"].sum() / counts["count"].sum()

    # Check if contains at least EEG or ACC_GYRO
    contains = 0 if any(counts["type"].isin(["EEG4", "EEG8", "ACC_GYRO"])) else 1

    return {
        "TypeLoc": location,
        "TypeParseErrors": err_rate,
        "TypeUnexpected": unexpected,
        "TypeCounts": (8 - len(counts["type"].dropna())) / 8,
        "TypeRelevant": contains,
    }


# Assessment -------------------------------------------
def assess_structure(filename, loc_freq=10, loc_type=12):
    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
        lines = f.readlines()
    times, uuids, data, err_rate = _parse_lines(lines)

    result = {
        "Preset": filename.replace("data_", "").replace(".txt", ""),
        "LineStructureErrors": err_rate,
    }

    result.update(assess_freq_ids(data, location=loc_freq))
    result.update(assess_types_ids(data, location=loc_type))

    return pd.DataFrame(result, index=[0])


# ------------------------
# Script entrypoint
# ------------------------
if __name__ == "__main__":
    data_dir = "./data_raw"
    files = sorted(os.listdir(data_dir))

    results = []
    optimized_freq = []
    optimized_type = []

    for filename in files:
        print(f"Processing {filename}...")

        result = assess_structure(filename, loc_freq=12)
        results.append(result)

        # Grid search for different freq id locations (result: suggests 10 is best )
        best_loc_freq = []
        for loc_freq in range(8, 20):
            best_loc_freq.append(assess_structure(filename, loc_freq=loc_freq))
        best_loc_freq = pd.concat(best_loc_freq, ignore_index=True)
        score = (
            best_loc_freq["FreqUnnexpected"] + best_loc_freq["FreqExpectedPattern"]
        ) / 2
        optimized_freq.append(
            assess_structure(
                filename, loc_freq=best_loc_freq["FreqLoc"][score.idxmin()]
            )[
                [
                    "Preset",
                    "FreqLoc",
                    "FreqParseErrors",
                    "FreqUnnexpected",
                    "FreqExpectedPattern",
                ]
            ]
        )

        # Grid search for different type id locations (result: suggests 12 is best )
        best_loc_type = []
        for loc_type in range(10, 20):
            best_loc_type.append(
                assess_structure(filename, loc_freq=10, loc_type=loc_type)
            )
        best_loc_type = pd.concat(best_loc_type, ignore_index=True)
        score = (best_loc_type["TypeUnexpected"] + best_loc_type["TypeRelevant"]) / 2
        optimized_type.append(
            assess_structure(
                filename, loc_freq=10, loc_type=best_loc_type["TypeLoc"][score.idxmin()]
            )[
                [
                    "Preset",
                    "TypeLoc",
                    "TypeParseErrors",
                    "TypeUnexpected",
                    "TypeCounts",
                    "TypeRelevant",
                ]
            ]
        )

    results = pd.concat(results, ignore_index=True)
    optimized_freq = pd.concat(optimized_freq, ignore_index=True)
    optimized_type = pd.concat(optimized_type, ignore_index=True)

    # print(results.to_markdown(floatfmt=(".2f")))
    print("\nFreq bytes results:")
    print(optimized_freq.to_markdown(floatfmt=(".2f")))
    print("\nType bytes results:")
    print(optimized_type.to_markdown(floatfmt=(".2f")))
