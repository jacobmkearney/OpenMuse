import os
import sys
import argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from muse_eeg.core import (
    _parse_packets_from_bin,
    _gather_eeg_bytes,
    _gather_eeg_bytes_strict,
    _apply_bit_offset,
    _unpack_14bit,
    _score_alignment,
    PRESET_DEFAULTS,
    EEG_IDS,
)


def collect_blocks(infile: str, preset: str, max_blocks: int = 200) -> list[bytes]:
    with open(infile, 'rb') as f:
        blob = f.read()

    mode = os.environ.get('MUSE_EEG_MODE', 'until_tag')  # all | until_tag | tagged
    pkt_align = os.environ.get('MUSE_EEG_PKT_ALIGN', '0') == '1'
    defaults = PRESET_DEFAULTS.get(preset, {})
    channels = int(defaults.get('channels', 8))
    bit_offset = int(defaults.get('bit_offset', 0))

    rolling = bytearray()
    for _, pkt in _parse_packets_from_bin(blob):
        if len(pkt) < 14:
            continue
        L = pkt[0]
        pid = pkt[9]
        if pid not in EEG_IDS:
            continue
        payload = pkt[14:L]
        if mode == 'all':
            eeg_bytes = payload
        elif mode == 'tagged':
            eeg_bytes = _gather_eeg_bytes_strict(payload)
        else:
            eeg_bytes = _gather_eeg_bytes(payload)
        if not eeg_bytes:
            continue
        if pkt_align and len(eeg_bytes) >= 28 * 2:
            best_slide = 0
            best_score = _score_alignment(eeg_bytes, channels, bit_offset)
            for slide in range(1, 28):
                sc = _score_alignment(eeg_bytes[slide:], channels, bit_offset)
                if sc < best_score:
                    best_score = sc
                    best_slide = slide
            if best_slide:
                eeg_bytes = eeg_bytes[best_slide:]
        if len(eeg_bytes) % 28 != 0:
            eeg_bytes = eeg_bytes[: len(eeg_bytes) - (len(eeg_bytes) % 28)]
        rolling.extend(eeg_bytes)
        if len(rolling) >= 28 * max_blocks:
            break

    blocks = [bytes(rolling[i:i+28]) for i in range(0, len(rolling) - (len(rolling) % 28), 28)]
    return blocks[:max_blocks]


def score_layout(blocks: list[bytes], bit_offset: int, layout: str) -> tuple[float, float]:
    # returns (score, sat_frac); lower is better
    sat_thresh = int(os.environ.get('MUSE_EEG_SAT_THRESH', '7000'))
    sat_weight = float(os.environ.get('MUSE_EEG_SAT_WEIGHT', '5.0'))
    diffs_sum = 0.0
    diffs_cnt = 0
    sat_cnt = 0
    val_cnt = 0
    for blk in blocks:
        vals16 = _unpack_14bit(_apply_bit_offset(blk, bit_offset), 16)
        if len(vals16) < 2:
            continue
        # layout influences how we consider continuity; we just use raw successive diffs here
        val_cnt += len(vals16)
        sat_cnt += sum(1 for v in vals16 if abs(v) >= sat_thresh)
        for i in range(len(vals16) - 1):
            diffs_sum += abs(vals16[i + 1] - vals16[i])
            diffs_cnt += 1
    if diffs_cnt == 0:
        return float('inf'), 1.0
    base = diffs_sum / diffs_cnt
    sat_frac = (sat_cnt / val_cnt) if val_cnt else 1.0
    return base + sat_weight * sat_frac, sat_frac


def find_best_offset(blocks: list[bytes]) -> int:
    best_off = 0
    best_score = float('inf')
    for off in range(8):
        sc, _ = score_layout(blocks[:64], off, 'raw')
        if sc < best_score:
            best_score = sc
            best_off = off
    return best_off


def main():
    p = argparse.ArgumentParser(description='Test 28B decode layouts and offsets')
    p.add_argument('--preset', required=True)
    p.add_argument('--infile', required=True)
    args = p.parse_args()

    blocks = collect_blocks(args.infile, args.preset, max_blocks=400)
    if not blocks:
        print('No blocks collected')
        return
    best_off = find_best_offset(blocks)
    layouts = ['2x8', '4x4', '1x16']
    results = []
    for lay in layouts:
        sc, sat = score_layout(blocks, best_off, lay)
        results.append((lay, sc, sat))
    print(f'best bit_offset: {best_off}')
    print('layout, score, sat_frac')
    for lay, sc, sat in results:
        print(f'{lay}, {sc:.2f}, {sat:.3f}')


if __name__ == '__main__':
    main()


