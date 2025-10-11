from __future__ import annotations

import struct
from typing import List, Tuple


EEG_IDS = {0x11: 4, 0x12: 8}
NON_EEG_TAGS = {0x34, 0x35, 0x36, 0x47, 0x98}

# Frozen preset defaults (heuristic)
PRESET_DEFAULTS = {
    "p1035": {"bit_offset": 2, "channels": 4},
    "p1041": {"bit_offset": 0, "channels": 8},
    "p1045": {"bit_offset": 3, "channels": 8},
}


def _parse_packets_from_bin(data: bytes):
    off = 0
    n = len(data)
    while off < n:
        if off + 1 > n:
            break
        L = data[off]
        if L < 14:
            off += 1
            continue
        if off + L > n:
            break
        pkt = data[off : off + L]
        yield off, pkt
        off += L


def _unpack_14bit(buf: bytes, n_values: int) -> List[int]:
    values: List[int] = []
    bitpos = 0
    total_bits = len(buf) * 8
    for _ in range(n_values):
        if bitpos + 14 > total_bits:
            break
        acc = 0
        for b in range(14):
            byte_index = (bitpos + b) // 8
            bit_index = (bitpos + b) % 8
            bit = (buf[byte_index] >> bit_index) & 1
            acc |= (bit << b)
        if acc & (1 << 13):
            acc = acc - (1 << 14)
        values.append(acc)
        bitpos += 14
    return values


def _apply_bit_offset(block: bytes, bit_offset: int) -> bytes:
    if bit_offset == 0:
        return block
    data = bytearray(len(block))
    carry = 0
    for i, b in enumerate(block[::-1]):
        shifted = (b >> bit_offset) | (carry << (8 - bit_offset))
        data[len(block) - 1 - i] = shifted & 0xFF
        carry = b & ((1 << bit_offset) - 1)
    return bytes(data)


def _gather_eeg_bytes(payload: bytes) -> bytes:
    """Collect EEG bytes only.

    Primary EEG is assumed to be the untagged bytes from the start of the
    payload up to the first recognized secondary tag. Beyond that point,
    only explicitly tagged EEG chunks ([EEG_TAG, LEN, DATA]) are included.
    """
    eeg = bytearray()
    i = 0
    n = len(payload)
    first_secondary_seen = False
    while i < n:
        # Tagged chunk?
        if i + 2 <= n:
            tag = payload[i]
            ln = payload[i + 1]
            if tag in NON_EEG_TAGS and i + 2 + ln <= n:
                first_secondary_seen = True
                i += 2 + ln
                continue
            if tag in EEG_IDS and i + 2 + ln <= n:
                eeg.extend(payload[i + 2 : i + 2 + ln])
                i += 2 + ln
                continue
        # Untagged byte: include only before first secondary tag
        if not first_secondary_seen:
            eeg.append(payload[i])
        i += 1
    return bytes(eeg)


def _score_offset(blocks: List[bytes], channels: int, bit_offset: int) -> float:
    """Lower is better: mean absolute successive diff across unpacked values."""
    diffs_sum = 0.0
    diffs_cnt = 0
    for blk in blocks:
        b = _apply_bit_offset(blk, bit_offset)
        vals = _unpack_14bit(b, 2 * channels)
        if len(vals) < 2:
            continue
        for i in range(len(vals) - 1):
            diffs_sum += abs(vals[i + 1] - vals[i])
            diffs_cnt += 1
    return diffs_sum / diffs_cnt if diffs_cnt else float("inf")


def decode_file(
    path: str,
    preset: str,
) -> Tuple[List[List[float]], int]:
    """Decode a full .bin file into samples×channels with stitched EEG bytes.

    Returns (samples, channels) where samples is a list of per-channel lists.
    """
    defaults = PRESET_DEFAULTS.get(preset)
    if not defaults:
        raise ValueError(f"Unknown preset {preset}")
    channels = int(defaults["channels"])
    bit_offset = int(defaults["bit_offset"])

    with open(path, "rb") as f:
        blob = f.read()

    rolling = bytearray()
    per_ch: List[List[float]] = [[] for _ in range(channels)]

    for _, pkt in _parse_packets_from_bin(blob):
        if len(pkt) < 14:
            continue
        L = pkt[0]
        pid = pkt[9]
        if pid not in EEG_IDS:
            continue
        payload = pkt[14:L]
        eeg_bytes = _gather_eeg_bytes(payload)
        if not eeg_bytes:
            continue
        rolling.extend(eeg_bytes)
        # If we have at least a few blocks, sweep offsets 0..7 and lock best
        if len(rolling) >= 28 * 4:
            blocks = [bytes(rolling[i:i+28]) for i in range(0, 28*4, 28)]
            best_off = bit_offset
            best_score = _score_offset(blocks, channels, bit_offset)
            for off in range(8):
                if off == bit_offset:
                    continue
                sc = _score_offset(blocks, channels, off)
                if sc < best_score:
                    best_score = sc
                    best_off = off
            bit_offset = best_off

        # process full 28B blocks
        while len(rolling) >= 28:
            block = bytes(rolling[:28])
            rolling = rolling[28:]
            vals = _unpack_14bit(_apply_bit_offset(block, bit_offset), 2 * channels)
            if len(vals) < 2 * channels:
                continue
            s1 = vals[:channels]
            s2 = vals[channels: 2 * channels]
            for ch in range(channels):
                per_ch[ch].append(float(s1[ch]))
                per_ch[ch].append(float(s2[ch]))

    return per_ch, channels


