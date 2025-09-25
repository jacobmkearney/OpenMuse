import os
import struct
import datetime as _dt
from typing import Optional, Dict, List, Any, Tuple

# STARTING INFO ====================================================================


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

# Different presets may enable/disable some channels.

# Note: 1034 is the only preset for which a red LED brightly turned on during recording, suggesting the activation of OPTICS or PPG channels.

# Based on these specs, we can derive the following plausible expectations:


EXPECTED_RATES = {
    "EEG": 256.0,
    "AUX": 256.0,
    "ACC": 52.0,
    "GYRO": 52.0,
    "PPG": 64.0,
    "OPTICS": 64.0,
}
EXPECTED_CHANNELS = {
    "EEG": [0, 4],  #  256 Hz, 14 bits
    "AUX": [0, 1, 4],  # 256 Hz, 14 bits
    "ACC": [0, 3],  #  52 Hz, 16 bits
    "GYRO": [0, 3],  #  52 Hz, 16 bits
    "PPG": [0, 3],  # 64 Hz, 20 bits
    "OPTICS": [0, 1, 4, 5, 8, 16],  # 64 Hz, 20 bits
}

# Each data file should contain some combination of these channels.
# The exact combination depends on the preset used during recording.
# As some of these channels are likely indistinguishable in the raw data, we can group them based on their expected sampling rate.

EXPECTED_GROUPS = {
    "CH256": set(
        EXPECTED_CHANNELS["EEG"]
        + EXPECTED_CHANNELS["AUX"]
        + [
            i + j
            for i in EXPECTED_CHANNELS["EEG"]
            for j in EXPECTED_CHANNELS["AUX"]
            if j > 0
        ]
    ),
    "CH52": set(
        EXPECTED_CHANNELS["ACC"]
        + EXPECTED_CHANNELS["GYRO"]
        + [
            i + j
            for i in EXPECTED_CHANNELS["ACC"]
            for j in EXPECTED_CHANNELS["GYRO"]
            if j > 0
        ]
    ),
    "CH64": set(
        EXPECTED_CHANNELS["PPG"]
        + EXPECTED_CHANNELS["OPTICS"]
        + [
            i + j
            for i in EXPECTED_CHANNELS["PPG"]
            for j in EXPECTED_CHANNELS["OPTICS"]
            if j > 0
        ]
    ),
}


# ------------------------------------------------------------------------------

# Goals:
# - 1) Decode raw packets into structured data with channels labeled according to their type (group) and index.
# - 2) For each preset (i.e., each file), infer the most likely configuration of channels and groups based on the data.
# - 3) Make the decoding logic data-driven, naturally flowing from the prior knowledge about the channel counts and rates and little else.

# DECODING ====================================================================

# --- Named constants (replace magic numbers) ----------------------------------
# CH256: 18 bytes encode 12 12-bit samples (6 triplets -> 2 samples each).
CH256_SEG_BYTES = 18  # bytes per packed segment (3 bytes -> 2 samples)
CH256_PAIR_BYTES = 3  # number of bytes per packed pair within segment
CH256_SAMPLES_PER_SEG = 12  # decoded samples per segment (for reference)
CH256_SAMPLE_MAX = 4095  # max 12-bit value (0..4095) used for plausibility checks
CH256_CENTER = 2048  # center offset subtracted to produce signed values

# CH64: 20-byte-ish blocks and 5-byte 20-bit pairs
CH64_SEG_BYTES = 20  # approximate block size where CH64 heuristics are checked
CH64_20BIT_PAIR_BYTES = 5  # bytes per two 20-bit values
CH64_MIN_FRAME_BYTES = 24  # minimum frame length expected to contain CH64 data
CH64_PLAUSIBLE_MIN = 10000  # lower bound for plausible 20-bit CH64 values

# CH52 (IMU-like): 12 bytes per sample tuple (6 * int16 -> ax,ay,az,gx,gy,gz)
CH52_TUPLE_BYTES = 12
CH52_DEFAULT_START_MIN = 16  # where to start scanning for CH52 sequences
CH52_MIN_SAMPLES = 2  # minimum contiguous tuples to consider a valid batch
CH52_ACCEL_RANGE = 20000  # plausibility bound for accel values
CH52_GYRO_RANGE = 40000  # plausibility bound for gyro values

# Alignment / decoding heuristics
DEFAULT_FRAME_OFFSET = 4  # most packets seem to have payloads starting at offset 4
SKIP_STEP = CH256_SEG_BYTES  # step for tight packing
SKIP1 = SKIP_STEP * 2  # 36 = possible interleaved block skip
SKIP2 = SKIP_STEP * 3  # 54 = possible second interleaved block skip
MISSES_THRESHOLD = 3  # how many byte-wise misses to tolerate before ending run
GROUP4_ALIGNMENT_RADIUS = 128  # limited alignment window used for group4 frames
MIN_ALIGNED_RUN = 2  # minimum contiguous aligned CH256 segments to accept

# Consolidation heuristics
CONSOLIDATION_BALANCE_THRESHOLD = 0.5  # 50% imbalance tolerated between channels

ChannelConfig = Dict[str, int]


def _ch256_name(idx: int) -> str:
    return f"CH256_{idx + 1}"


def _ch64_name(idx: int) -> str:
    return f"CH64_{idx + 1}"


def _ch256_group_for_header(b0: int, config: Optional[ChannelConfig] = None) -> int:
    """Determine CH256 group size based on header or config."""
    if config and "EEG" in config and "AUX" in config:
        return config["EEG"] + config["AUX"]

    # Heuristic mapping: larger bulk frames likely carry 8-ch CH256 groups,
    # smaller D7/DB/D9/DA/CF flavors tend to 4.
    group8 = {0xE3, 0xEC, 0xF0, 0xF2, 0xEE, 0xE5, 0xE2, 0xEF, 0xDF}
    group4 = {0xD7, 0xDB, 0xD9, 0xDA, 0xCF}
    if b0 in group8:
        return 8
    if b0 in group4:
        return 4
    return 8  # default fallback


def _get_packet_type(b0: int) -> str:
    types = {
        0xDF: "EEG_PPG",
        0xF4: "CH52",
        0xDB: "MIXED_1",
        0xD9: "MIXED_2",
        0xD7: "CH52_BULK",
        0xDA: "CH52",
        0xE3: "PPG_CAL",
        0xEC: "PPG_CAL",
        0xF0: "PPG_CAL",
        0xE2: "BULK",
        0xE5: "BULK",
        0xEE: "BULK",
        0xEF: "BULK",
        0xF2: "BULK",
        0xCA: "BULK",
        0xCB: "BULK",
        0xCE: "BULK",
    }
    return types.get(b0, f"UNKNOWN_{b0:02X}")


def _looks_like_ch256(seg: bytes) -> bool:
    if len(seg) != CH256_SEG_BYTES:
        return False
    # A 12-bit sample must be in the range [0, 4095].
    # Check the first sample for validity.
    sample = (seg[0] << 4) | (seg[1] >> 4)
    return 0 <= sample <= CH256_SAMPLE_MAX


def _unpack_ch256_18b(seg: bytes) -> List[float]:
    out: List[float] = []
    # triplets of 3 bytes -> 2 samples per triplet
    for i in range(CH256_SEG_BYTES // CH256_PAIR_BYTES):
        b0, b1, b2 = seg[i * CH256_PAIR_BYTES : i * CH256_PAIR_BYTES + CH256_PAIR_BYTES]
        s1 = (b0 << 4) | (b1 >> 4)
        s2 = ((b1 & 0x0F) << 8) | b2
        out.append(s1 - CH256_CENTER)
        out.append(s2 - CH256_CENTER)
    return out


def _unpack_ch64_20b(seg: bytes) -> List[int]:
    if len(seg) < CH64_SEG_BYTES:
        return []
    samples: List[int] = []
    # Simplified: derive 16-bit values and filter to plausible range
    for i in range(0, CH64_SEG_BYTES - 2, CH256_PAIR_BYTES):
        if i + 2 < len(seg):
            val = (seg[i] << 8) | seg[i + 1]
            samples.append(val)
    return samples if len(samples) > 2 else []


def _unpack_ch64_20bit_pair(b: bytes, off: int) -> Optional[Tuple[int, int]]:
    if off + CH64_20BIT_PAIR_BYTES > len(b):
        return None
    b0, b1, b2, b3, b4 = b[off : off + CH64_20BIT_PAIR_BYTES]
    v1 = (b0 << 12) | (b1 << 4) | (b2 >> 4)
    v2 = ((b2 & 0x0F) << 16) | (b3 << 8) | b4
    return v1, v2


def _decode_ch64_calibrated(data: bytes) -> Dict[str, Any]:
    # Decoder for 20-bit, 3-channel PPG data found in E3/EC/F0 packets.
    # Data is stored in 5-byte chunks, encoding two 20-bit values.
    res: Dict[str, List[int]] = {
        _ch64_name(0): [],
        _ch64_name(1): [],
        _ch64_name(2): [],
    }
    if len(data) < CH64_MIN_FRAME_BYTES:
        return {}

    # The PPG data appears to start at a fixed offset.
    offset = DEFAULT_FRAME_OFFSET
    num_samples = 0
    while offset + CH64_20BIT_PAIR_BYTES <= len(data):
        pair = _unpack_ch64_20bit_pair(data, offset)
        if not pair:
            break
        v1, v2 = pair
        # Basic plausibility check for CH64 values
        if CH64_PLAUSIBLE_MIN <= v1 < (1 << 20) and CH64_PLAUSIBLE_MIN <= v2 < (
            1 << 20
        ):
            # Demultiplex the samples into 3 channels.
            # The pattern appears to be (ch1, ch2), (ch3, ch1), (ch2, ch3), etc.
            ch_idx1 = num_samples % 3
            ch_idx2 = (num_samples + 1) % 3
            res[_ch64_name(ch_idx1)].append(v1)
            res[_ch64_name(ch_idx2)].append(v2)
            num_samples += 2
            offset += CH64_20BIT_PAIR_BYTES
        else:
            # Stop if we hit a value that doesn't look like PPG data.
            break

    # Only return data if we found a reasonable number of samples.
    if num_samples < 6:
        return {}

    return {"ch64": res}


def _extract_imu16_bulk(
    data: bytes,
    start_min: int = CH52_DEFAULT_START_MIN,
    min_samples: int = CH52_MIN_SAMPLES,
) -> List[Dict[str, List[float]]]:
    n = len(data)
    best: List[Dict[str, List[float]]] = []
    # Search for the longest contiguous run of 12-byte IMU tuples (ax,ay,az,gx,gy,gz) int16 big-endian
    # Scan a wider window to catch later-offset IMU sequences
    for base in range(start_min, max(0, n - CH52_TUPLE_BYTES)):
        off = base
        items: List[Dict[str, List[float]]] = []
        while off + 12 <= n:
            try:
                ax, ay, az, gx, gy, gz = struct.unpack_from(">hhhhhh", data, off)
            except Exception:
                break
            # Light plausibility filters; allow wide ranges to avoid false negatives
            if not (
                -CH52_ACCEL_RANGE <= ax <= CH52_ACCEL_RANGE
                and -CH52_ACCEL_RANGE <= ay <= CH52_ACCEL_RANGE
                and -CH52_ACCEL_RANGE <= az <= CH52_ACCEL_RANGE
            ):
                break
            if not (
                -CH52_GYRO_RANGE <= gx <= CH52_GYRO_RANGE
                and -CH52_GYRO_RANGE <= gy <= CH52_GYRO_RANGE
                and -CH52_GYRO_RANGE <= gz <= CH52_GYRO_RANGE
            ):
                break
            items.append(
                {
                    "accel": [ax, ay, az],
                    "gyro": [gx, gy, gz],
                }
            )
            off += CH52_TUPLE_BYTES
        if len(items) > len(best):
            best = items
    return best if len(best) >= min_samples else []


def _decode_df(data: bytes, config: Optional[ChannelConfig] = None) -> Dict[str, Any]:
    res: Dict[str, Any] = {"ch256": {}, "ch64": {}}
    # Choose channel grouping based on header when possible
    b0 = data[0] if data else 0
    group_len = _ch256_group_for_header(b0, config=config)
    offset = 4
    ch_idx = 0
    iterations = 0
    while offset < len(data) and iterations < len(data):
        iterations += 1
        if offset + 18 <= len(data) and _looks_like_ch256(data[offset : offset + 18]):
            samples = _unpack_ch256_18b(data[offset : offset + 18])
            # Cap to detected group size to avoid proliferating keys
            name = _ch256_name(ch_idx % group_len)
            if name not in res["ch256"]:
                res["ch256"][name] = []
            res["ch256"][name].extend(samples)
            ch_idx += 1
            offset += 18
            continue
        if offset + 20 <= len(data):
            ppg_samples = _unpack_ch64_20b(data[offset : offset + 20])
            if ppg_samples:
                ch0 = _ch64_name(0)
                if ch0 not in res["ch64"]:
                    res["ch64"][ch0] = []
                res["ch64"][ch0].extend(ppg_samples)
                offset += 20
                continue
        offset += 1
    return res


def _decode_f4_ch52(data: bytes) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if len(data) < 16:
        return res
    try:
        ax, ay, az, gx, gy, gz = struct.unpack_from(">hhhhhh", data, 4)
        res["ch52"] = {
            "accel": [ax, ay, az],
            "gyro": [gx, gy, gz],
        }
    except Exception:
        pass
    return res


def _decode_da_ch52(data: bytes) -> Dict[str, Any]:
    # Decoder for packets starting with 0xDA, which appear to contain IMU data.
    res: Dict[str, Any] = {}
    if len(data) < 16:
        return res

    # The IMU data seems to be at a fixed offset, similar to 0xF4 packets.
    try:
        ax, ay, az, gx, gy, gz = struct.unpack_from(">hhhhhh", data, 4)
        res["ch52"] = {
            "accel": [ax, ay, az],
            "gyro": [gx, gy, gz],
        }
    except Exception:
        pass
    return res


def _decode_d7_ch52(data: bytes) -> Dict[str, Any]:
    # Decoder for packets starting with 0xD7, which appear to contain bulk CH52 data.
    # The standard bulk extractor doesn't work, so we search from an earlier offset.
    res: Dict[str, Any] = {}
    if len(data) < 16:
        return res

    # These packets can contain multiple IMU samples. Search starting from offset 4.
    ch52_batch = _extract_imu16_bulk(data, start_min=4)
    if ch52_batch:
        res["ch52_batch"] = ch52_batch
        res["imu_batch"] = ch52_batch
    return res


def _decode_dx_ch52_bulk(data: bytes) -> Dict[str, Any]:
    # Targeted CH52 batch extraction for D1/D5/DD-like frames carrying 12-byte tuples interleaved.
    res: Dict[str, Any] = {}
    if len(data) < 16:
        return res
    ch52_batch = _extract_imu16_bulk(data, start_min=4)
    if ch52_batch:
        res["ch52_batch"] = ch52_batch
        res["imu_batch"] = ch52_batch
    return res


def _decode_generic(data: bytes) -> Dict[str, Any]:
    res: Dict[str, Any] = {"ch256": {}}
    offset = 4
    while offset + 18 <= len(data):
        seg = data[offset : offset + 18]
        if _looks_like_ch256(seg):
            ch = len(res["ch256"])
            name = _ch256_name(ch)
            res["ch256"][name] = _unpack_ch256_18b(seg)
            offset += 18
        else:
            offset += 1
    return res


def _decode_bulk(data: bytes, config: Optional[ChannelConfig] = None) -> Dict[str, Any]:
    # Scan across the buffer; whenever a valid 18-byte CH256 block appears, unpack it
    res: Dict[str, Any] = {"ch256": {}, "ch64": {}}
    offset = 4
    ch_idx = 0
    b0 = data[0] if data else 0
    group_len = _ch256_group_for_header(b0, config=config)
    iterations = 0
    limit = len(data) + 1
    while offset < len(data) and iterations < limit:
        iterations += 1
        if offset + 18 <= len(data) and _looks_like_ch256(data[offset : offset + 18]):
            seg = data[offset : offset + 18]
            samples = _unpack_ch256_18b(seg)
            name = _ch256_name(ch_idx % group_len)
            if name not in res["ch256"]:
                res["ch256"][name] = []
            res["ch256"][name].extend(samples)
            ch_idx += 1
            offset += 18
            continue
        # Skip PPG heuristic in bulk to avoid overcount for now
        offset += 1
    return res


def _best_eeg_alignment(
    data: bytes, start_min: int = 4, start_max: Optional[int] = None
) -> Tuple[int, int]:
    # Find the base offset that yields the longest contiguous run of valid 18-byte EEG segments.
    n = len(data)
    best_off, best_run = -1, 0
    if start_max is None:
        start_max = n - 18
    upper = min(start_max, max(start_min, n - 18))
    for base in range(start_min, upper + 1):
        off = base
        run = 0
        while off + 18 <= n and _looks_like_ch256(data[off : off + 18]):
            run += 1
            off += 18
        if run > best_run:
            best_run = run
            best_off = base
    return best_off, best_run


def _decode_bulk_structured(
    data: bytes,
    config: Optional[ChannelConfig] = None,
    *,
    require_group_len: Optional[int] = None,
    alignment_search_radius: Optional[int] = None,
) -> Dict[str, Any]:
    """Parametric structured bulk decoder.

    - `require_group_len`: if set, only decode when header-derived group_len matches.
    - `alignment_search_radius`: if set, use `start_max = min(offset + radius, n-18)`
      when calling `_best_eeg_alignment`. This reproduces the group4 behavior
      (radius=128) while default behavior searches the full frame.
    """
    res: Dict[str, Any] = {"ch256": {}, "ch64": {}}
    if not data:
        return res
    b0 = data[0]
    group_len = _ch256_group_for_header(b0, config=config)
    if require_group_len is not None and group_len != require_group_len:
        return res
    n = len(data)
    ch_idx = 0
    imu_added = False

    offset = 4
    # iterate through the frame, finding multiple aligned runs
    while offset + 18 <= n:
        if alignment_search_radius is not None:
            start_max = min(offset + alignment_search_radius, n - 18)
        else:
            start_max = n - 18

        base, run = _best_eeg_alignment(data, start_min=offset, start_max=start_max)
        if base < 0 or run < 2:
            offset += 1
            continue

        off = base
        misses = 0
        # consume this aligned run
        while off + 18 <= n:
            seg = data[off : off + 18]
            if _looks_like_ch256(seg):
                samples = _unpack_ch256_18b(seg)
                name = _ch256_name(ch_idx % group_len)
                if name not in res["ch256"]:
                    res["ch256"][name] = []
                res["ch256"][name].extend(samples)
                ch_idx += 1
                # prefer tight packing first
                next_off = off + 18
                if next_off + 18 <= n and _looks_like_ch256(
                    data[next_off : next_off + 18]
                ):
                    off = next_off
                    misses = 0
                    continue
                # try skipping one block (36)
                next_off2 = off + 36
                if next_off2 + 18 <= n and _looks_like_ch256(
                    data[next_off2 : next_off2 + 18]
                ):
                    off = next_off2
                    misses = 0
                    continue
                # try skipping two blocks (54)
                next_off3 = off + 54
                if next_off3 + 18 <= n and _looks_like_ch256(
                    data[next_off3 : next_off3 + 18]
                ):
                    off = next_off3
                    misses = 0
                    continue
                # couldn't find continuation, end this run
                break
            else:
                misses += 1
                if misses >= 3:
                    break
                # scan forward byte-by-byte to re-align
                off += 1

        # opportunistically extract IMU batch once per frame
        if not imu_added:
            imu_batch = _extract_imu16_bulk(data, start_min=4)
            if imu_batch:
                res["imu_batch"] = imu_batch
                res["ch52_batch"] = imu_batch
                imu_added = True
        # move offset to end of the decoded run to avoid double counting
        offset = max(off, base + 18)

    # Consolidate per-frame CH256 channels to best group size
    res["ch256"] = (
        _consolidate_ch256_channels(res.get("ch256") or {}) if res.get("ch256") else {}
    )
    return res


def _total_ch256_samples(decoded: Dict[str, Any]) -> int:
    ch256 = decoded.get("ch256") or {}
    total = 0
    for samples in ch256.values():
        if isinstance(samples, list):
            total += len(samples)
    return total


def _consolidate_ch256_channels(eeg: Dict[str, List[float]]) -> Dict[str, List[float]]:
    # If the number of channels is already a plausible group size with balanced samples,
    # assume it's correct and don't consolidate.
    if not eeg:
        return eeg

    num_channels = len(eeg)
    if num_channels in list(filter(lambda x: x > 0, EXPECTED_GROUPS["CH256"])):
        totals = [len(v) for v in eeg.values()]
        if totals:
            mean = sum(totals) / len(totals)
            # Check if sample counts are reasonably balanced (e.g., within 50% of mean)
            if all(abs(t - mean) < 0.5 * mean for t in totals):
                return eeg  # Return as-is, no consolidation needed

    # Fallback to original logic for complex/unbalanced cases
    # Build ordered channels by index discovered
    items = sorted(
        ((int(k.split("_")[-1]) - 1, v) for k, v in eeg.items()), key=lambda x: x[0]
    )
    seq = [v for _, v in items]
    totals = [len(v) for v in seq]
    if not totals:
        return eeg
    candidates = list(filter(lambda x: x > 0, EXPECTED_GROUPS["CH256"]))
    # Score: variance of channel counts when folded into group size (lower is better)
    best = None
    for g in candidates:
        if g == 0:
            continue
        folded = [[] for _ in range(g)]
        for i, samples in enumerate(seq):
            folded[i % g].append(len(samples))
        sums = [sum(x) for x in folded]
        mean = sum(sums) / g
        var = sum((s - mean) ** 2 for s in sums) / g
        score = (var, -sum(sums))
        if best is None or score < best[0]:
            best = (score, g)
    if best is None:
        return eeg
    g = best[1]
    # Remap: interleave segments modulo g
    merged: List[List[float]] = [[] for _ in range(g)]
    for i, samples in enumerate(seq):
        merged[i % g].extend(samples)
    return {_ch256_name(i): merged[i] for i in range(g) if merged[i]}


# Note: _decode_bulk_structured_group4 functionality is now provided by
# _decode_bulk_structured(data, require_group_len=4, alignment_search_radius=128)


def _decode_bulk_best(
    data: bytes, config: Optional[ChannelConfig] = None
) -> Dict[str, Any]:
    # Choose structured strategy based on header group size
    b0 = data[0] if data else 0
    group_len = _ch256_group_for_header(b0, config=config)
    if group_len == 4:
        structured = _decode_bulk_structured(
            data, config=config, require_group_len=4, alignment_search_radius=128
        )
    else:
        structured = _decode_bulk_structured(data, config=config)
    heuristic = _decode_bulk(data, config=config)
    chosen = (
        structured
        if _total_ch256_samples(structured) >= _total_ch256_samples(heuristic)
        else heuristic
    )
    # Final safeguard consolidation
    if chosen.get("ch256"):
        chosen["ch256"] = _consolidate_ch256_channels(chosen["ch256"])  # idempotent
    return chosen


def decode_amused(
    data: bytes,
    timestamp: Optional[_dt.datetime] = None,
    config: Optional[ChannelConfig] = None,
) -> Dict[str, Any]:
    """
    Decode a raw Muse S BLE packet (amused-style).

    Returns a dict with keys: timestamp, packet_type, ch256, ch64, imu.
    CH256 values are centered raw integers, IMU are raw integers.
    """
    if isinstance(data, str):
        data = bytes.fromhex(data)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or hex string")
    if timestamp is None:
        timestamp = _dt.datetime.now()

    result: Dict[str, Any] = {
        "timestamp": timestamp,
        "packet_type": _get_packet_type(data[0]) if data else "EMPTY",
    }

    if not data:
        return result

    b0 = data[0]
    if b0 == 0xDF:
        decoded = _decode_df(data, config=config)
        result.update(decoded)
    elif b0 == 0xF4:
        decoded = _decode_f4_ch52(data)
        result.update(decoded)
    elif b0 == 0xDA:
        # This packet type seems to be another IMU packet.
        decoded = _decode_da_ch52(data)
        result.update(decoded)
    elif b0 == 0xD7:
        # This packet type seems to be another bulk IMU packet.
        decoded = _decode_d7_ch52(data)
        result.update(decoded)
    elif b0 in (0xE3, 0xEC, 0xF0):
        # These packets appear to contain calibrated CH64 (PPG/optics) data.
        decoded = _decode_ch64_calibrated(data)
        result.update(decoded)
    elif b0 in (0xE2, 0xE5, 0xEE, 0xEF, 0xF2, 0xD9, 0xDB, 0xCF, 0xCA, 0xCB, 0xCE):
        # Bulk frames: heuristic decoding
        decoded = _decode_bulk(data, config=config)
        if decoded.get("ch256") or decoded.get("ch64"):
            result.update(decoded)
    elif b0 in (0xD1, 0xD5, 0xDD):
        # Other D*-series packets that often embed IMU tuples
        decoded = _decode_dx_ch52_bulk(data)
        result.update(decoded)
    # Only decode known packet types for now; other unknowns are left as-is

    return result


# ------------------ Analyzer utilities ------------------


def _parse_record_line(line: str) -> Optional[Tuple[_dt.datetime, str, bytes]]:
    parts = line.strip().split("\t")
    if len(parts) != 3:
        return None
    ts_s, uuid, hex_payload = parts
    try:
        ts = _dt.datetime.fromisoformat(ts_s)
    except Exception:
        try:
            ts = _dt.datetime.strptime(ts_s, "%Y-%m-%dT%H:%M:%S.%f%z")
        except Exception:
            return None
    try:
        payload = bytes.fromhex(hex_payload)
    except Exception:
        return None
    return ts, uuid, payload


def analyze_file_with_config(
    lines: List[str],
    config: Optional[ChannelConfig],
    max_lines: Optional[int] = None,
    *,
    debug: bool = False,
    max_error_logs: int = 5,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "lines": 0,
        "decoded_lines": 0,
        "packet_type_counts": {},
        "uuid_counts": {},
        "b0_counts": {},
        "b0_len_sum": {},
        "b0_len_min": {},
        "b0_len_max": {},
        "b0_len_hist": {},  # per-b0 exact length histogram
        "ch256_samples": {},  # per-channel sample counts
        "ch64_samples": 0,
        "imu_samples": 0,
        "acc_samples": 0,
        "gyro_samples": 0,
        "start_ts": None,
        "end_ts": None,
        "duration_s": 0.0,
        "rates": {},
        "errors": 0,
        "unknown_lines": 0,
    }

    err_logs = 0
    for i, line in enumerate(lines):
        if max_lines is not None and i >= max_lines:
            break
        stats["lines"] += 1
        parsed = _parse_record_line(line)
        if not parsed:
            stats["unknown_lines"] += 1
            if debug and err_logs < max_error_logs:
                print(f"[parse] Unrecognized line format: {line.strip()[:160]}")
                err_logs += 1
            continue
        ts, uuid, payload = parsed
        # Count by UUID prefix (e.g., 273e0013 vs 273e0014)
        uuid_short = uuid.split("-")[0]
        stats["uuid_counts"][uuid_short] = stats["uuid_counts"].get(uuid_short, 0) + 1
        # First-byte histogram and payload length stats
        if payload:
            b0 = payload[0]
            stats["b0_counts"][b0] = stats["b0_counts"].get(b0, 0) + 1
            L = len(payload)
            stats["b0_len_sum"][b0] = stats["b0_len_sum"].get(b0, 0) + L
            # exact size histogram per header
            h = stats["b0_len_hist"].setdefault(b0, {})
            h[L] = h.get(L, 0) + 1
            stats["b0_len_min"][b0] = (
                L if b0 not in stats["b0_len_min"] else min(stats["b0_len_min"][b0], L)
            )
            stats["b0_len_max"][b0] = (
                L if b0 not in stats["b0_len_max"] else max(stats["b0_len_max"][b0], L)
            )
        if stats["start_ts"] is None:
            stats["start_ts"] = ts
        stats["end_ts"] = ts

        try:
            decoded = decode_amused(payload, ts, config=config)
            stats["decoded_lines"] += 1
            pt = decoded.get("packet_type", "UNKNOWN")
            stats["packet_type_counts"][pt] = stats["packet_type_counts"].get(pt, 0) + 1
            ch256 = decoded.get("ch256") or {}
            for ch, samples in ch256.items():
                stats["ch256_samples"][ch] = stats["ch256_samples"].get(ch, 0) + len(
                    samples
                )
            ch64 = decoded.get("ch64") or {}
            # CH64 may have named channels; sum lengths if lists
            for v in ch64.values():
                if isinstance(v, list):
                    stats["ch64_samples"] += len(v)
            imu = decoded.get("imu") or decoded.get("ch52") or {}
            if imu:
                # maintain legacy IMU packet count and add CH52 count
                stats["imu_samples"] += 1
                stats.setdefault("ch52_samples", 0)
                stats["ch52_samples"] += 1
                acc = imu.get("accel")
                gyro = imu.get("gyro")
                if acc is not None:
                    stats["acc_samples"] += 1
                if gyro is not None:
                    stats["gyro_samples"] += 1
            # count IMU batches if present
            imu_batch = decoded.get("imu_batch") or decoded.get("ch52_batch") or []
            if imu_batch:
                for item in imu_batch:
                    # each item has accel+gyro
                    stats["acc_samples"] += 1
                    stats["gyro_samples"] += 1
        except Exception as e:
            stats["errors"] += 1
            if debug and err_logs < max_error_logs:
                import binascii, traceback

                head = payload[:8]
                head_hex = binascii.hexlify(head).decode()
                b0 = head[0] if head else None
                print(
                    f"[decode] Exception: {type(e).__name__}: {e} | uuid={uuid} | b0={b0!r} head={head_hex} len={len(payload)}"
                )
                print(traceback.format_exc().splitlines()[-1])
                err_logs += 1
            continue

    if stats["start_ts"] and stats["end_ts"]:
        dur = (stats["end_ts"] - stats["start_ts"]).total_seconds()
        # Avoid zero duration for tiny files
        stats["duration_s"] = max(dur, 1e-6)
    else:
        stats["duration_s"] = 0.0

    dur = stats["duration_s"] or 1e-6
    # Compute inferred sampling rates
    if stats["ch256_samples"]:
        # average per-channel rate
        ch256_rates = {ch: cnt / dur for ch, cnt in stats["ch256_samples"].items()}
        if ch256_rates:
            stats["rates"]["CH256"] = sum(ch256_rates.values()) / max(
                1, len(ch256_rates)
            )
    if stats["ch64_samples"]:
        stats["rates"]["CH64"] = stats["ch64_samples"] / dur
    if stats["imu_samples"]:
        stats["rates"]["IMU"] = stats["imu_samples"] / dur
        stats["rates"]["CH52"] = stats["imu_samples"] / dur
    if stats["acc_samples"]:
        stats["rates"]["ACC"] = stats["acc_samples"] / dur
    if stats["gyro_samples"]:
        stats["rates"]["GYRO"] = stats["gyro_samples"] / dur

    return stats


if __name__ == "__main__":
    # Run (interactively)
    directory = "data_raw"
    # Each file was recorded using a different preset
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    results: List[Dict[str, Any]] = []
    for path in files:
        full_path = os.path.join(directory, path)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        stats = analyze_file_with_config(lines, config=None, max_lines=1000)
        stats["file"] = os.path.basename(path)  # Add file name to stats for printing
        results.append(stats)

        print(
            f"\nFile: {stats['file']} | Duration: {stats['duration_s']:.2f}s | Lines: {stats['lines']} (decoded {stats['decoded_lines']}, errors {stats['errors']})"
        )

        ch256_chs = list(stats.get("ch256_samples", {}).keys())
        if ch256_chs:
            ch256_rate = stats.get("rates", {}).get("CH256")
            ok = (
                abs(ch256_rate - EXPECTED_RATES["CH256"]) / EXPECTED_RATES["CH256"]
                < 0.25
                if ch256_rate
                else False
            )
            print(
                f"  CH256: {len(ch256_chs)} ch ({', '.join(ch256_chs)}), ~{ch256_rate:.1f} Hz {'OK' if ok else 'off'}"
            )

        ch64_rate = stats.get("rates", {}).get("CH64")
        if ch64_rate:
            ok = abs(ch64_rate - EXPECTED_RATES["CH64"]) / EXPECTED_RATES["CH64"] < 0.35
            print(f"  CH64: ~{ch64_rate:.1f} Hz {'OK' if ok else 'off'}")
        imu_rate = stats.get("rates", {}).get("IMU")
        if imu_rate:
            print(f"  CH52 packets: ~{imu_rate:.1f} Hz")
        acc_rate = stats.get("rates", {}).get("ACC")
        if acc_rate:
            ok = abs(acc_rate - EXPECTED_RATES["ACC"]) / EXPECTED_RATES["ACC"] < 0.35
            print(f"  ACC: ~{acc_rate:.1f} Hz {'OK' if ok else 'off'}")
        gyro_rate = stats.get("rates", {}).get("GYRO")
        if gyro_rate:
            ok = abs(gyro_rate - EXPECTED_RATES["GYRO"]) / EXPECTED_RATES["GYRO"] < 0.35
            print(f"  GYRO: ~{gyro_rate:.1f} Hz {'OK' if ok else 'off'}")
        if stats.get("packet_type_counts"):
            pts = ", ".join(
                f"{k}:{v}" for k, v in sorted(stats["packet_type_counts"].items())
            )
            print(f"  Packet types: {pts}")
        if stats.get("uuid_counts"):
            uu = ", ".join(f"{k}:{v}" for k, v in sorted(stats["uuid_counts"].items()))
            print(f"  UUID counts: {uu}")
        if stats.get("b0_counts"):
            items = sorted(
                stats["b0_counts"].items(), key=lambda kv: kv[1], reverse=True
            )[:8]
            parts = []
            for b0, cnt in items:
                s = stats["b0_len_sum"].get(b0, 0)
                avg = s / max(1, cnt)
                mn = stats["b0_len_min"].get(b0, 0)
                mx = stats["b0_len_max"].get(b0, 0)
                parts.append(f"{b0:02X}:{cnt}@{avg:.1f}[{mn}-{mx}]")
            print("  b0 histogram:", ", ".join(parts))
            # Extra debug: per-header exact size hist for unknown packet types
            try:
                unknown_hex = [
                    int(k.split("_")[-1], 16)
                    for k in stats.get("packet_type_counts", {})
                    if isinstance(k, str) and k.startswith("UNKNOWN_")
                ]
                unknown_hex = sorted(set(unknown_hex))
            except Exception:
                unknown_hex = []
            if unknown_hex:
                # limit to top few unknowns by count
                ranked = sorted(
                    ((b0, stats["b0_counts"].get(b0, 0)) for b0 in unknown_hex),
                    key=lambda x: x[1],
                    reverse=True,
                )[:6]
                for b0, _ in ranked:
                    hist = stats["b0_len_hist"].get(b0) or {}
                    if not hist:
                        continue
                    sizes = ", ".join(
                        f"{L}:{c}"
                        for L, c in sorted(
                            hist.items(), key=lambda x: x[1], reverse=True
                        )
                    )
                    print(f"    b0 {b0:02X} sizes -> {sizes}")
