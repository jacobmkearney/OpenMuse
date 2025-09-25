import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import datetime as dt
import os

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

# Each data file (./data_raw/) should contain some combination of these channels.
# The exact combination depends on the preset used during recording.
# Importantly, these channels types are likely indistinguishable from the data alone, so it is best to group them according to their data characteristics in GROUPS, namely CH256, CH52, CH64.

# --- Expectations ---


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

# Goal for this script is to try various decoding approaches and validate
# See: https://github.com/AbosaSzakal/MuseAthenaDataformatParser

# -----------------------
# PID -> semantics maps
# -----------------------
FREQ_NIBBLE_TO_HZ = {
    0: None,
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

# dtype nibble -> logical expected channels per sample (set of possibilities)
EXPECTED_CHANNELS_BY_DTYPE = {
    0: set(),
    1: {4, 8},  # EEG, 4 or sometimes packed as 8 ints (keep flexible)
    2: {8},  # EEG 8 channels
    3: set(),  # DRL/REF - not used for main signals
    4: {4},  # Optics 4
    5: {8},  # Optics 8
    6: {16},  # Optics 16
    7: {6},  # IMU: 3 gyro + 3 accel -> 6 values per logical sample
    8: {1},  # Battery: 1 small value
}


# ======================================================================
# Functions ============================================================
# ======================================================================
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


# ---------------------------
# Generic bit-unpack helper
# ---------------------------
def unpack_nbit_le_signed(data: bytes, nbits: int) -> List[int]:
    """
    Unpack little-endian n-bit signed integers from `data`.
    Returns list of Python ints. Works for arbitrary nbits (<=32).
    Interprets packed stream as successive little-endian n-bit words.
    """
    if nbits <= 0 or nbits > 32:
        raise ValueError("nbits must be 1..32")
    out = []
    bit_mask = (1 << nbits) - 1
    sign_bit = 1 << (nbits - 1)

    buffer = 0
    bits_in_buf = 0
    for b in data:
        buffer |= b << bits_in_buf
        bits_in_buf += 8
        while bits_in_buf >= nbits:
            val = buffer & bit_mask
            # sign-extend if necessary
            if val & sign_bit:
                val = val - (1 << nbits)
            out.append(int(val))
            buffer >>= nbits
            bits_in_buf -= nbits
    return out


# ---------------------------
# bytes-per-block mapping
# ---------------------------
def bytes_per_block_for_pid(pid_byte: int) -> Optional[int]:
    """
    Using README conventions - compute bytes for one logical block for the pid type.
    Returns bytes per block, or None if unknown (e.g. battery).
    """
    dtype = pid_byte & 0x0F
    # IMU: 3 samples, 6x12-bit => 3 * (6*12) bits = 216 bits = 27 bytes
    if dtype == 7:
        return 3 * (6 * 12) // 8  # 27
    # Optics:
    if dtype == 4:  # 4 channels
        return 3 * (4 * 20) // 8  # 30
    if dtype == 5:  # 8 channels
        return 3 * (8 * 20) // 8  # 60
    if dtype == 6:  # 16 channels
        return 3 * (16 * 20) // 8  # 120
    # EEG (README indicates 2 samples with 8x14 ints -> 16 ints * 14 bits = 224 bits = 28 bytes)
    if dtype in (1, 2):
        return 2 * (8 * 14) // 8  # 28
    # Battery and unknown - variable / small: return None
    return None


# -------------------------------------------------
# Decoders (raw ints -> arrays, not rescaled)
# -------------------------------------------------
def decode_imu_region(region: bytes) -> Dict[str, Any]:
    """Decode IMU 12-bit little-endian. Return raw rows and per-sample raw arrays."""
    ints = unpack_nbit_le_signed(region, 12)
    n_rows = len(ints) // 6
    if n_rows == 0:
        empty_rows = np.empty((0, 6), dtype=np.int32)
        return {
            "raw": empty_rows,
            "gyro": empty_rows[:, 0:3].astype(float),
            "acc": empty_rows[:, 3:6].astype(float),
            "acc_mag": np.empty((0,)),
        }
    trimmed = ints[: n_rows * 6]
    rows = np.array(trimmed, dtype=np.int32).reshape(n_rows, 6)
    gyr = rows[:, 0:3].astype(float)
    acc = rows[:, 3:6].astype(float)
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    return {"raw": rows, "gyro": gyr, "acc": acc, "acc_mag": acc_mag}


def decode_eeg_region(region: bytes) -> Dict[str, Any]:
    """Decode EEG 14-bit little-endian. Return raw rows (rows x 16 typically)."""
    ints = unpack_nbit_le_signed(region, 14)
    n_rows = len(ints) // 16
    if n_rows == 0:
        return {"raw": np.empty((0, 16), dtype=np.int32)}
    trimmed = ints[: n_rows * 16]
    rows = np.array(trimmed, dtype=np.int32).reshape(n_rows, 16)
    return {"raw": rows}


def decode_optical_region(region: bytes, channels: int = 4) -> Dict[str, Any]:
    """Decode optics 20-bit little-endian. Return raw rows (samples x channels)."""
    ints = unpack_nbit_le_signed(region, 20)
    n_samples = len(ints) // channels  # store as sequential samples x channels
    if n_samples == 0:
        return {"raw": np.empty((0, channels), dtype=np.int32)}
    trimmed = ints[: n_samples * channels]
    rows = np.array(trimmed, dtype=np.int32).reshape(n_samples, channels)
    return {"raw": rows}


# ---------------------------
# Main delineator
# ---------------------------
def delineate_packet(raw: bytes, try_length_prefix: bool = True) -> Dict[str, Any]:
    """
    Parse a raw packet payload into subpackets where possible.
    Returns dict with:
      - 'top_level_packets': list of dicts (if length prefix used)
      - 'subpackets': list of parsed subpacket dicts:
            { 'pid_idx', 'pid_byte', 'freq_nibble', 'dtype_nibble', 'dtype_name',
              'unknown4', 'sample_region', 'bytes_per_block', 'num_blocks', 'decoded' }
      - 'leftover': trailing bytes not consumed
      - 'mode': which parsing mode was used
    Behaviour:
       - tries parsing using top-level length prefix first (if enabled).
       - falls back to a greedy in-packet scan that looks for pid-like bytes.
    """
    known_types = {
        1: "EEG_4ch",
        2: "EEG_8ch",
        3: "DRL_REF",
        4: "OPTICS_4ch",
        5: "OPTICS_8ch",
        6: "OPTICS_16ch",
        7: "IMU",
        8: "BATTERY",
    }
    result = {"top_level_packets": [], "subpackets": [], "leftover": b"", "mode": None}

    def parse_inner(pkt: bytes) -> Tuple[List[Dict[str, Any]], bytes]:
        """Parse inner bytes of a top-level packet (after its initial header), return subpkts + leftover."""
        subpkts = []
        i = 0
        L = len(pkt)
        # iterate while there are bytes - find pid and attempt to cut sample_region until next pid or end
        while i < L:
            pid = pkt[i]
            freq = (pid >> 4) & 0x0F
            dtype = pid & 0x0F
            dtype_name = known_types.get(dtype, f"UNKNOWN_{dtype}")
            # require at least 1 + 4 bytes for pid + unknown4
            if i + 1 + 4 >= L:
                # not enough bytes for header - treat remainder as leftover
                return subpkts, pkt[i:]
            unknown4 = pkt[i + 1 : i + 1 + 4]
            sample_start = i + 1 + 4
            # search for next pid candidate (first byte whose high nibble is a valid freq nibble AND low nibble a known dtype)
            next_candidate = None
            # compute expected bytes per block for this pid - may be None for battery/unknown
            bpb = bytes_per_block_for_pid(pid)
            # scan forward for the earliest plausible next pid such that the region between sample_start and that pid is a multiple of bpb (if bpb known)
            j = sample_start
            while j < L:
                b = pkt[j]
                high = (b >> 4) & 0x0F
                low = b & 0x0F
                is_pid_like = (high in {1, 2, 3, 4, 5, 6, 7, 8, 9}) and (
                    low in known_types.keys()
                )
                if is_pid_like:
                    # if this is a plausible boundary, test divisibility if possible
                    region_len = j - sample_start
                    if bpb is None:
                        next_candidate = j
                        break
                    else:
                        if region_len >= bpb and (region_len % bpb) == 0:
                            next_candidate = j
                            break
                j += 1
            if next_candidate is None:
                # No next candidate - take the rest as sample_region
                sample_region = pkt[sample_start:]
                i = L  # exit after processing
            else:
                sample_region = pkt[sample_start:next_candidate]
                i = next_candidate
            # interpret sample_region depending on dtype
            decoded = {"raw_bytes": sample_region}
            bpb_here = bytes_per_block_for_pid(pid)
            num_blocks = (
                (len(sample_region) // bpb_here)
                if (bpb_here and bpb_here > 0)
                else None
            )
            decoded["bytes_per_block"] = bpb_here
            decoded["num_blocks"] = num_blocks
            try:
                if dtype == 7:  # IMU
                    decoded.update(decode_imu_region(sample_region))
                elif dtype in (1, 2):  # EEG
                    decoded.update(decode_eeg_region(sample_region))
                elif dtype in (4, 5, 6):  # Optics (4/8/16)
                    ch = {4: 4, 5: 8, 6: 16}[dtype]
                    decoded.update(decode_optical_region(sample_region, channels=ch))
                elif dtype == 8:  # battery - store raw
                    # don't attempt to parse - leave raw bytes
                    decoded["note"] = "battery_or_small"
                else:
                    decoded["note"] = "unknown_dtype"
            except Exception as e:
                decoded["decode_error"] = str(e)
            subpkt = {
                "pid_idx": (
                    int(i - (len(sample_region) + 1 + 4))
                    if (len(sample_region) is not None)
                    else None
                ),
                "pid_byte": int(pid),
                "freq_nibble": int(freq),
                "dtype_nibble": int(dtype),
                "dtype_name": dtype_name,
                "unknown4": bytes(unknown4),
                "sample_region": bytes(sample_region),
                "decoded": decoded,
            }
            subpkts.append(subpkt)
        return subpkts, b""

    # First try length-prefixed top-level parsing
    if try_length_prefix and len(raw) >= 1:
        L = raw[0]
        if 1 <= L <= len(raw):
            # consume while possible: some dumps contain multiple top-level packets concatenated
            pos = 0
            parsed_any = False
            while pos < len(raw):
                if pos >= len(raw):
                    break
                if pos + 1 > len(raw):
                    break
                L = raw[pos]
                if L <= 0 or pos + L > len(raw):
                    # not a sensible length, break to fallback
                    break
                pkt = raw[pos : pos + L]
                # expected top-level header: length(1) + counter(1) + unknown7(7) -> header_len = 9
                if len(pkt) >= 9:
                    inner = pkt[1 + 1 + 7 :]  # skip length + counter + 7 unknown
                    subpkts, leftover = parse_inner(inner)
                    result["top_level_packets"].append(
                        {"raw": pkt, "subpackets": subpkts, "leftover": leftover}
                    )
                    parsed_any = True
                else:
                    result["top_level_packets"].append(
                        {"raw": pkt, "subpackets": [], "leftover": b""}
                    )
                    parsed_any = True
                pos += L
            if parsed_any:
                # flatten subpackets
                all_subs = []
                for t in result["top_level_packets"]:
                    all_subs.extend(t["subpackets"])
                result["subpackets"] = all_subs
                result["leftover"] = raw[pos:]
                result["mode"] = "length_prefixed"
                return result

    # Fallback greedy scan - treat raw as sequence possibly already without top-level length.
    # scan for pid-like bytes and attempt to parse from first candidate
    L = len(raw)
    i = 0
    subpkts = []
    while i < L:
        b = raw[i]
        high = (b >> 4) & 0x0F
        low = b & 0x0F
        if (high in {1, 2, 3, 4, 5, 6, 7, 8, 9}) and (low in known_types.keys()):
            # attempt to parse from here as a pid
            # ensure at least 1+4 available
            if i + 1 + 4 >= L:
                # not enough room - remainder is leftover
                break
            unknown4 = raw[i + 1 : i + 1 + 4]
            sample_start = i + 1 + 4
            # try to find next pid candidate as in parse_inner
            next_candidate = None
            bpb = bytes_per_block_for_pid(b)
            j = sample_start
            while j < L:
                bb = raw[j]
                hh = (bb >> 4) & 0x0F
                ll = bb & 0x0F
                is_pid_like = (hh in {1, 2, 3, 4, 5, 6, 7, 8, 9}) and (
                    ll in known_types.keys()
                )
                if is_pid_like:
                    region_len = j - sample_start
                    if bpb is None:
                        next_candidate = j
                        break
                    else:
                        if region_len >= bpb and (region_len % bpb) == 0:
                            next_candidate = j
                            break
                j += 1
            if next_candidate is None:
                sample_region = raw[sample_start:]
                i = L
            else:
                sample_region = raw[sample_start:next_candidate]
                i = next_candidate
            decoded = {"raw_bytes": sample_region}
            try:
                dtype = low
                if dtype == 7:
                    decoded.update(decode_imu_region(sample_region))
                elif dtype in (1, 2):
                    decoded.update(decode_eeg_region(sample_region))
                elif dtype in (4, 5, 6):
                    ch = {4: 4, 5: 8, 6: 16}[dtype]
                    decoded.update(decode_optical_region(sample_region, channels=ch))
                elif dtype == 8:
                    decoded["note"] = "battery_or_small"
                else:
                    decoded["note"] = "unknown_dtype"
            except Exception as e:
                decoded["decode_error"] = str(e)
            subpkt = {
                "pid_idx": (
                    i - (len(sample_region) + 1 + 4)
                    if sample_region is not None
                    else None
                ),
                "pid_byte": int(b),
                "freq_nibble": int(high),
                "dtype_nibble": int(low),
                "dtype_name": known_types.get(low, f"UNKNOWN_{low}"),
                "unknown4": bytes(unknown4),
                "sample_region": bytes(sample_region),
                "decoded": decoded,
            }
            subpkts.append(subpkt)
        else:
            i += 1  # skip non-pid byte
    result["subpackets"] = subpkts
    result["leftover"] = raw[i:] if i < L else b""
    result["mode"] = "greedy_scan"
    return result


def decode_to_signals(
    filename: str,
    parse_lines_fn=parse_lines,
    delineate_fn=delineate_packet,
) -> Dict[str, pd.DataFrame]:
    """
    Read `filename`, parse lines with parse_lines_fn, run delineate_fn on each raw packet,
    and return a dict of DataFrames:
      - meta: one row per subpacket with metadata and decode notes
      - imu: per-sample IMU rows (gyr0,gyr1,gyr2,acc0,acc1,acc2,acc_mag) - raw integer units
      - eeg: per-sample EEG rows (channel_0..channel_N) - raw integer units
      - optics: per-sample optical rows (channel_0..channel_N) - raw integer units
      - leftovers: per-packet leftover bytes / parsing mode
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    times, uuids, raws = parse_lines_fn(lines)

    meta_rows = []
    imu_rows = []
    eeg_rows = []
    opt_rows = []
    leftover_rows = []

    for pkt_idx, (pkt_time, uuid, raw) in enumerate(zip(times, uuids, raws)):
        packet_dt = pd.to_datetime(float(pkt_time), unit="s", utc=True)
        parsed = delineate_fn(raw)
        mode = parsed.get("mode")
        leftover = parsed.get("leftover", b"")
        leftover_rows.append(
            {
                "filename": filename,
                "packet_idx": int(pkt_idx),
                "packet_time": float(pkt_time),
                "packet_dt": packet_dt,
                "uuid": uuid,
                "mode": mode,
                "leftover_hex": (
                    leftover.hex()
                    if isinstance(leftover, (bytes, bytearray))
                    else str(leftover)
                ),
            }
        )

        for sp_idx, sp in enumerate(parsed.get("subpackets", [])):
            pid = int(sp.get("pid_byte", -1))
            freq = int(sp.get("freq_nibble", -1))
            dtype = int(sp.get("dtype_nibble", -1))
            dtype_name = sp.get("dtype_name", None)
            unknown4 = sp.get("unknown4", b"")
            sample_region = sp.get("sample_region", b"")
            decoded = sp.get("decoded", {}) or {}

            meta_rows.append(
                {
                    "filename": filename,
                    "packet_idx": int(pkt_idx),
                    "packet_time": float(pkt_time),
                    "uuid": uuid,
                    "subpacket_idx": int(sp_idx),
                    "pid": pid,
                    "freq": freq,
                    "dtype": dtype,
                    "dtype_name": dtype_name,
                    "unknown4_hex": (
                        unknown4.hex()
                        if isinstance(unknown4, (bytes, bytearray))
                        else str(unknown4)
                    ),
                    "sample_region_len": (
                        len(sample_region)
                        if isinstance(sample_region, (bytes, bytearray))
                        else None
                    ),
                    "bytes_per_block": decoded.get("bytes_per_block"),
                    "num_blocks": decoded.get("num_blocks"),
                    "note": decoded.get("note"),
                    "decode_error": decoded.get("decode_error"),
                }
            )

            # IMU: expected keys 'acc', 'gyro', 'acc_mag' (numpy arrays or lists) - already raw ints/float
            if "acc" in decoded and "gyro" in decoded:
                acc = np.asarray(decoded["acc"])
                gyr = np.asarray(decoded["gyro"])
                if "acc_mag" in decoded:
                    acc_mag = np.asarray(decoded["acc_mag"])
                else:
                    acc_mag = np.sqrt((acc**2).sum(axis=1))
                n_samples = acc.shape[0]
                for s in range(n_samples):
                    imu_rows.append(
                        {
                            "filename": filename,
                            "packet_idx": int(pkt_idx),
                            "packet_time": float(pkt_time),
                            "packet_dt": packet_dt,
                            "uuid": uuid,
                            "subpacket_idx": int(sp_idx),
                            "sample_in_subpkt": int(s),
                            "gyr0": float(gyr[s, 0]),
                            "gyr1": float(gyr[s, 1]),
                            "gyr2": float(gyr[s, 2]),
                            "acc0": float(acc[s, 0]),
                            "acc1": float(acc[s, 1]),
                            "acc2": float(acc[s, 2]),
                            "acc_mag": float(acc_mag[s]),
                        }
                    )

            # EEG: decoder now returns raw ints under 'raw' (shape: rows x 16 typically)
            if dtype in (1, 2) and "raw" in decoded:
                arr = np.asarray(decoded["raw"])
                # If arr has shape (-1,16) assume 2 samples x 8 channels per block
                if arr.ndim == 2 and arr.shape[1] == 16:
                    resh = arr.reshape(
                        -1, 2, 8
                    )  # blocks x samples_per_block x channels
                    for block_idx in range(resh.shape[0]):
                        for s in range(resh.shape[1]):
                            row = {
                                "filename": filename,
                                "packet_idx": int(pkt_idx),
                                "packet_time": float(pkt_time),
                                "packet_dt": packet_dt,
                                "uuid": uuid,
                                "subpacket_idx": int(sp_idx),
                                "block_idx": int(block_idx),
                                "sample_in_block": int(s),
                            }
                            for ch in range(8):
                                row[f"ch_{ch}"] = int(resh[block_idx, s, ch])
                            eeg_rows.append(row)
                else:
                    # Generic case: each row is one sample with N channels
                    if arr.ndim == 2:
                        for r_idx in range(arr.shape[0]):
                            row = {
                                "filename": filename,
                                "packet_idx": int(pkt_idx),
                                "packet_time": float(pkt_time),
                                "uuid": uuid,
                                "subpacket_idx": int(sp_idx),
                                "row_idx": int(r_idx),
                            }
                            for ch in range(arr.shape[1]):
                                row[f"ch_{ch}"] = int(arr[r_idx, ch])
                            eeg_rows.append(row)
                    elif arr.ndim == 1:
                        row = {
                            "filename": filename,
                            "packet_idx": int(pkt_idx),
                            "packet_time": float(pkt_time),
                            "uuid": uuid,
                            "subpacket_idx": int(sp_idx),
                            "row_idx": 0,
                        }
                        for ch in range(arr.shape[0]):
                            row[f"ch_{ch}"] = int(arr[ch])
                        eeg_rows.append(row)

            # Optics: decoder returns raw ints under 'raw'
            if dtype in (4, 5, 6) and "raw" in decoded:
                arr = np.asarray(decoded["raw"])
                # If 2D: rows x channels
                if arr.ndim == 2:
                    for r_idx in range(arr.shape[0]):
                        row = {
                            "filename": filename,
                            "packet_idx": int(pkt_idx),
                            "packet_time": float(pkt_time),
                            "uuid": uuid,
                            "subpacket_idx": int(sp_idx),
                            "sample_in_subpkt": int(r_idx),
                        }
                        for ch in range(arr.shape[1]):
                            row[f"ch_{ch}"] = int(arr[r_idx, ch])
                        opt_rows.append(row)
                elif arr.ndim == 1:
                    row = {
                        "filename": filename,
                        "packet_idx": int(pkt_idx),
                        "packet_time": float(pkt_time),
                        "uuid": uuid,
                        "subpacket_idx": int(sp_idx),
                        "sample_in_subpkt": 0,
                    }
                    for ch in range(arr.shape[0]):
                        row[f"ch_{ch}"] = int(arr[ch])
                    opt_rows.append(row)

    # Build DataFrames
    meta_df = pd.DataFrame(meta_rows)
    imu_df = pd.DataFrame(imu_rows)
    eeg_df = pd.DataFrame(eeg_rows)
    optics_df = pd.DataFrame(opt_rows)
    leftovers_df = pd.DataFrame(leftover_rows)

    # Ensure IMU frame has consistent columns even if empty
    if imu_df.empty:
        imu_df = pd.DataFrame(
            columns=[
                "filename",
                "packet_idx",
                "packet_time",
                "uuid",
                "subpacket_idx",
                "sample_in_subpkt",
                "gyr0",
                "gyr1",
                "gyr2",
                "acc0",
                "acc1",
                "acc2",
                "acc_mag",
            ]
        )

    return {
        "meta": meta_df,
        "imu": imu_df,
        "eeg": eeg_df,
        "optics": optics_df,
        "leftovers": leftovers_df,
    }


# ======================================================================
# Performance ==========================================================
# ======================================================================
def _lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x - x.mean()
    denom = (x * x).sum()
    if denom == 0:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)


def _low_freq_power_fraction(x: np.ndarray, frac: float = 0.1) -> float:
    n = x.size
    if n < 4:
        return 0.0
    X = np.fft.rfft(x - x.mean())
    P = (np.abs(X) ** 2).astype(float)
    cutoff = max(1, int(len(P) * frac))
    low = P[:cutoff].sum()
    total = P.sum()
    if total == 0:
        return 0.0
    return float(low / total)


def assess_decoding(
    filename: str,
    parse_lines_fn=parse_lines,
    delineate_fn=delineate_packet,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute decoding & signal-quality metrics for a single file.
    Returns (summary_dict, subpacket_df).
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    times, uuids, raws = parse_lines_fn(lines)

    total_raw_bytes = 0
    total_decoded_bytes = 0
    total_undecoded_bytes = 0
    decode_error_count = 0
    subpkt_records: List[Dict[str, Any]] = []

    for pkt_idx, (pkt_time, uuid, raw) in enumerate(zip(times, uuids, raws)):
        total_raw_bytes += len(raw)
        parsed = delineate_fn(raw)
        leftover = parsed.get("leftover", b"") or b""
        leftover_len = len(leftover)
        total_undecoded_bytes += leftover_len

        for sp_idx, sp in enumerate(parsed.get("subpackets", [])):
            pid = int(sp.get("pid_byte", -1))
            freq = int(sp.get("freq_nibble", -1))
            dtype = int(sp.get("dtype_nibble", -1))
            decoded = sp.get("decoded", {}) or {}
            sample_region = sp.get("sample_region", b"") or b""
            sample_len = len(sample_region)
            bpb = decoded.get("bytes_per_block")
            num_blocks = decoded.get("num_blocks")

            if bpb and isinstance(bpb, int) and bpb > 0:
                full_blocks = sample_len // bpb
                decoded_bytes_here = full_blocks * bpb
                remainder = sample_len - decoded_bytes_here
            else:
                decoded_bytes_here = 0
                remainder = sample_len

            total_decoded_bytes += decoded_bytes_here
            total_undecoded_bytes += remainder

            decode_err = False
            if decoded.get("decode_error"):
                decode_err = True
            if (bpb and bpb > 0) and (not num_blocks or num_blocks == 0):
                decode_err = True
            if decode_err:
                decode_error_count += 1

            # IMU metrics
            acc_mag_mean = acc_mag_std = gyro_mean_abs = autocorr1 = low_frac = np.nan
            raw_signal_score = np.nan
            if "acc" in decoded and "gyro" in decoded:
                acc = np.asarray(decoded["acc"])
                gyr = np.asarray(decoded["gyro"])
                if acc.size and acc.ndim == 2 and acc.shape[1] == 3:
                    acc_mag = np.sqrt((acc.astype(float) ** 2).sum(axis=1))
                    # safe means/stds
                    acc_mag_mean = (
                        float(np.nanmean(acc_mag)) if acc_mag.size else float("nan")
                    )
                    acc_mag_std = (
                        float(np.nanstd(acc_mag)) if acc_mag.size else float("nan")
                    )
                    autocorr1 = _lag1_autocorr(acc_mag)
                    gyro_mean_abs = (
                        float(np.nanmean(np.abs(gyr.astype(float))))
                        if gyr.size
                        else float("nan")
                    )
                    low_frac = _low_freq_power_fraction(acc_mag, frac=0.1)
                else:
                    acc_mag = np.array([], dtype=float)
                    acc_mag_mean = acc_mag_std = autocorr1 = gyro_mean_abs = (
                        low_frac
                    ) = np.nan

                # Compose a raw score from components. Keep it unnormalised for now.
                # Use 0 where component is nan, so components don't create nan score.
                c_autocorr = (
                    0.0 if np.isnan(autocorr1) else (max(0.0, (autocorr1 + 1.0) / 2.0))
                )
                c_lowfrac = 0.0 if np.isnan(low_frac) else float(low_frac)
                c_std = 0.0 if np.isnan(acc_mag_std) else float(acc_mag_std)
                raw_signal_score = c_autocorr + c_lowfrac + c_std

            rec = {
                "filename": filename,
                "packet_idx": int(pkt_idx),
                "packet_time": float(pkt_time),
                "uuid": uuid,
                "subpacket_idx": int(sp_idx),
                "pid": pid,
                "freq": freq,
                "dtype": dtype,
                "sample_region_len": int(sample_len),
                "bytes_per_block": int(bpb) if bpb else None,
                "num_blocks": int(num_blocks) if num_blocks else None,
                "decoded_bytes": int(decoded_bytes_here),
                "remainder_bytes": int(remainder),
                "leftover_len": int(leftover_len),
                "decode_error": bool(decode_err),
                "acc_mag_mean": acc_mag_mean,
                "acc_mag_std": acc_mag_std if "acc_mag_std" in locals() else np.nan,
                "gyro_mean_abs": gyro_mean_abs,
                "autocorr_lag1": autocorr1,
                "low_freq_power_frac": low_frac,
                "raw_signal_score_unscaled": raw_signal_score,
            }
            subpkt_records.append(rec)

    summary = {
        "filename": filename,
        "n_packets": len(raws),
        "total_raw_bytes": int(total_raw_bytes),
        "total_decoded_bytes": int(total_decoded_bytes),
        "total_undecoded_bytes": int(total_undecoded_bytes),
        "undecoded_pct": float(total_undecoded_bytes) / float(max(1, total_raw_bytes)),
        "decode_error_count": int(decode_error_count),
        "n_subpackets": len(subpkt_records),
    }

    df = pd.DataFrame(subpkt_records)

    # Normalise signal score across IMU subpackets safely (avoid mean/std on empty)
    df["signal_score"] = np.nan
    imu_mask = df["dtype"] == 7
    if imu_mask.any():
        vals = (
            df.loc[imu_mask, "raw_signal_score_unscaled"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )
        # if we have at least one value and non-zero std, z-score; otherwise give neutral zeros
        if vals.size > 0:
            std = np.nanstd(vals)
            mean = np.nanmean(vals)
            if std > 0:
                z = (vals - mean) / std
                norm_score = (np.tanh(z) + 1.0) / 2.0
            else:
                norm_score = np.zeros_like(vals, dtype=float)
            df.loc[imu_mask, "signal_score"] = norm_score
    return summary, df


def aggregate_subpackets(subpkt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-subpacket diagnostics by (pid, dtype).
    Returns DataFrame with counts, undecoded bytes, decode_error counts and IMU median quality.
    """
    if subpkt_df.empty:
        return pd.DataFrame()
    grp = subpkt_df.groupby(["pid", "dtype"], dropna=False)
    ag = grp.agg(
        n_subpkts=("subpacket_idx", "count"),
        total_sample_bytes=("sample_region_len", "sum"),
        total_decoded_bytes=(
            ("decoded_bytes", "sum")
            if "decoded_bytes" in subpkt_df.columns
            else ("sample_region_len", "sum")
        ),
        total_remainder=(
            ("remainder_bytes", "sum")
            if "remainder_bytes" in subpkt_df.columns
            else ("sample_region_len", "sum")
        ),
        decode_error_count=("decode_error", "sum"),
        median_signal_score=(
            ("signal_score", "median")
            if "signal_score" in subpkt_df.columns
            else ("acc_mag_mean", "median")
        ),
        median_acc_mag=(
            ("acc_mag_mean", "median")
            if "acc_mag_mean" in subpkt_df.columns
            else ("sample_region_len", "median")
        ),
    ).reset_index()
    # compute undecoded fraction per group
    ag["undecoded_pct"] = ag["total_remainder"] / ag["total_sample_bytes"].replace(
        {0: np.nan}
    )
    return ag


def decode_and_assess(filename):
    signals = decode_to_signals(filename)
    summary, subpkt = assess_decoding(filename)
    subpkt_summary = aggregate_subpackets(subpkt)
    return summary, subpkt_summary, signals


def _samples_and_channels_from_decoded(
    decoded: Dict[str, Any], dtype: int
) -> Tuple[int, Optional[int]]:
    """
    Return (n_samples_in_packet, observed_channels_per_sample_or_None)
    'decoded' is the dict produced by your delineate_packet decoder for that subpacket.
    """
    raw = decoded.get("raw")
    if raw is None:
        return 0, None
    arr = np.asarray(raw)
    # IMU: arr shape (n_samples, 6)
    if dtype == 7:
        if arr.ndim == 2 and arr.shape[1] == 6:
            return int(arr.shape[0]), 6
        # fallback: if flattened, try to infer
        flat_len = arr.size
        if flat_len % 6 == 0:
            return int(flat_len // 6), 6
        return int(arr.shape[0]) if arr.ndim == 2 else 0, (
            arr.shape[1] if arr.ndim == 2 else None
        )

    # EEG: often stored as rows x 16 (2 samples x 8 channels per row) or rows x 8
    if dtype in (1, 2):
        if arr.ndim == 2:
            cols = arr.shape[1]
            if cols == 16:
                # each row stores 2 samples each with 8 channels
                return int(arr.shape[0] * 2), 8
            if cols == 8:
                return int(arr.shape[0]), 8
            if cols == 4:  # unlikely but allow
                return int(arr.shape[0]), 4
            # fallback: unknown layout: treat each row as 1 sample with N channels
            return int(arr.shape[0]), int(cols)
        elif arr.ndim == 1:
            # 1D: ambiguous; treat as N channels x 1 sample
            return 1, int(arr.size)

    # Optics: arr shape (n_samples, channels)
    if dtype in (4, 5, 6):
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
        elif arr.ndim == 1:
            return 1, int(arr.size)

    # Battery / small: return number of bytes as sample length
    if dtype == 8:
        return int(arr.size), 1

    # Fallback generic
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 1:
        return 1, int(arr.size)
    return 0, None


# -----------------------
# Battery parsing helper
# -----------------------
def parse_battery_bytes(raw_bytes: bytes) -> Optional[float]:
    """
    Heuristic parsing of battery bytes:
     - if single byte 0..100 -> percent
     - if two bytes -> uint16 little-endian: try value/100 or value (give both heuristics)
     - else return None
    """
    if raw_bytes is None:
        return None
    if len(raw_bytes) == 0:
        return None
    if len(raw_bytes) == 1:
        v = raw_bytes[0]
        if 0 <= v <= 100:
            return float(v)
        return float(v)  # still return raw
    if len(raw_bytes) == 2:
        v = int.from_bytes(raw_bytes[:2], "little", signed=False)
        # heuristic: if <=100 -> direct percent; else if large try /100
        if v <= 100:
            return float(v)
        return float(v) / 100.0
    # longer: try first byte as percent
    first = raw_bytes[0]
    if 0 <= first <= 100:
        return float(first)
    return None


# -----------------------
# Aggregator: check observed vs expected
# -----------------------
def characterise_and_validate_file(
    filename: str,
    parse_lines_fn=parse_lines,
    delineate_fn=delineate_packet,
    rate_tolerance_frac: float = 0.15,
) -> Dict[str, Any]:
    """
    For one file, run delineation and return:
      - summary: high-level counts
      - groups_df: one row per (pid_byte, dtype) group with observed_rate, expected_rate, rate_match bool,
                   observed_channels, expected_channels_match bool, undecoded_pct, decode_error_rate, battery_level (if any)
      - subpkt_df: full subpacket diagnostics (as from assess_decoding with added samples/channels)
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    times, uuids, raws = parse_lines_fn(lines)

    # collect per-subpacket info including timestamp and sample counts
    records = []
    for pkt_idx, (pkt_time, uuid, raw) in enumerate(zip(times, uuids, raws)):
        parsed = delineate_fn(raw)
        leftover = parsed.get("leftover", b"") or b""
        for sp_idx, sp in enumerate(parsed.get("subpackets", [])):
            pid_byte = int(sp.get("pid_byte", -1))
            freq_nib = (pid_byte >> 4) & 0x0F
            dtype = int(sp.get("dtype_nibble", -1))
            decoded = sp.get("decoded", {}) or {}
            sample_region_len = len(sp.get("sample_region", b"") or b"")
            bpb = decoded.get("bytes_per_block")
            num_blocks = decoded.get("num_blocks")
            n_samples, observed_channels = _samples_and_channels_from_decoded(
                decoded, dtype
            )
            battery_value = None
            if dtype == 8:
                # try parsing battery from raw bytes
                battery_value = parse_battery_bytes(sp.get("sample_region", b"") or b"")
            rec = {
                "packet_idx": int(pkt_idx),
                "packet_time": float(pkt_time),
                "packet_dt": pd.to_datetime(float(pkt_time), unit="s", utc=True),
                "uuid": uuid,
                "pid_byte": pid_byte,
                "freq_nibble": freq_nib,
                "expected_rate_hz": FREQ_NIBBLE_TO_HZ.get(freq_nib),
                "dtype": dtype,
                "bytes_per_block": bpb,
                "num_blocks": num_blocks,
                "sample_region_len": sample_region_len,
                "n_samples": int(n_samples),
                "observed_channels": (
                    int(observed_channels) if observed_channels is not None else None
                ),
                "decode_error": bool(decoded.get("decode_error") is not None),
                "leftover_len": len(leftover),
                "battery_value": battery_value,
            }
            records.append(rec)

    if not records:
        return {
            "summary": {"n_subpackets": 0},
            "groups_df": pd.DataFrame(),
            "subpkt_df": pd.DataFrame(),
        }

    subpkt_df = pd.DataFrame.from_records(records)

    # Group by pid_byte and dtype
    # For each group compute:
    #  - median inter-packet interval (use packet_time sorted)
    #  - median samples per packet (n_samples)
    #  - observed_rate = median_samples / median_dt
    #  - expected_rate from freq nibble
    groups = []
    for (pid, dtype), g in subpkt_df.groupby(["pid_byte", "dtype"], sort=False):
        g_sorted = g.sort_values("packet_time")
        times_arr = g_sorted["packet_time"].to_numpy(dtype=float)
        if len(times_arr) >= 2:
            dt_median = float(np.median(np.diff(times_arr)))
            if dt_median <= 0:
                dt_median = np.median(np.diff(times_arr + 1e-9))  # fallback
        else:
            dt_median = np.nan
        median_samples = (
            float(np.median(g_sorted["n_samples"].to_numpy(dtype=float)))
            if not g_sorted["n_samples"].empty
            else 0.0
        )
        observed_rate = (
            float(median_samples / dt_median)
            if (not np.isnan(dt_median) and dt_median > 0)
            else np.nan
        )
        freq_nib = int(g_sorted["freq_nibble"].iloc[0])
        expected_rate = FREQ_NIBBLE_TO_HZ.get(freq_nib)
        rate_match = False
        if expected_rate is not None and not np.isnan(observed_rate):
            # relative difference
            rate_match = abs(observed_rate - expected_rate) <= (
                rate_tolerance_frac * expected_rate
            )
        # expected channels
        expected_ch_set = EXPECTED_CHANNELS_BY_DTYPE.get(dtype, set())
        # observed channel: take mode or median of non-null observed_channels
        obs_ch_vals = g_sorted["observed_channels"].dropna().astype(float)
        observed_channels = (
            int(obs_ch_vals.mode().iloc[0]) if not obs_ch_vals.empty else None
        )
        channels_match = False
        if observed_channels is not None and expected_ch_set:
            channels_match = observed_channels in expected_ch_set
        # undecoded fraction and decode error rate
        total_sample_bytes = int(g_sorted["sample_region_len"].sum())
        total_remainder = (
            int(
                (
                    g_sorted["sample_region_len"]
                    - (
                        g_sorted["n_samples"]
                        * (g_sorted["observed_channels"].fillna(1))
                    )
                )
                .clip(lower=0)
                .sum()
            )
            if "observed_channels" in g_sorted.columns
            else 0
        )
        # fallback undecoded estimation from leftover_len
        total_leftover = int(g_sorted["leftover_len"].sum())
        decode_error_count = int(g_sorted["decode_error"].sum())
        groups.append(
            {
                "pid_byte": int(pid),
                "dtype": int(dtype),
                "n_subpkts": int(len(g_sorted)),
                "median_samples_per_pkt": median_samples,
                "median_dt_s": dt_median,
                "observed_rate_hz": observed_rate,
                "expected_rate_hz": expected_rate,
                "rate_match": bool(rate_match),
                "observed_channels": observed_channels,
                "expected_channels": sorted(list(expected_ch_set)),
                "channels_match": bool(channels_match),
                "total_sample_bytes": total_sample_bytes,
                "total_leftover_bytes": total_leftover,
                "decode_error_count": decode_error_count,
                "battery_values": list(g_sorted["battery_value"].dropna().unique()),
            }
        )

    groups_df = pd.DataFrame.from_records(groups)
    # high-level summary
    summary = {
        "filename": filename,
        "n_packets": len(raws),
        "n_subpackets": len(subpkt_df),
        "n_groups": len(groups_df),
        "total_raw_bytes": sum(len(r) for r in raws),
    }
    return {"summary": summary, "groups_df": groups_df, "subpkt_df": subpkt_df}


# ======================================================================
# Main script ==========================================================
# ======================================================================
files = os.listdir("./data_raw/")
signals = {}
for f in files:
    f = "data_p1045.txt"  # for quick testing
    filename = os.path.join("./data_raw/", f)
    print(f"Processing {filename}...")
    preset = f.replace("data_", "").replace(".txt", "")
    summary, subpkt, signals[preset] = decode_and_assess(filename)
    res = characterise_and_validate_file(filename)
    print(
        res["groups_df"]
        .sort_values(["rate_match", "channels_match"], ascending=False)
        .to_markdown()
    )
    print(
        res["groups_df"][res["groups_df"]["dtype"] == 8]["battery_values"].to_markdown()
    )

# Plot
signals["p1034"]["imu"].plot(x="packet_time", y=["acc0", "acc1", "acc2"])
signals["p1034"]["imu"].plot(x="packet_time", y=["gyr0", "gyr1", "gyr2"])
signals["p1034"]["meta"]
