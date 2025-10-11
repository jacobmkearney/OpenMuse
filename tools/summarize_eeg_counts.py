import argparse
import sys, os
import numpy as np
from typing import Optional

# Ensure repo root on path when run from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from muse_eeg import decode_file


def summarize(infile: str, preset: str, seconds: Optional[float], rate: Optional[float]):
    samples, ch = decode_file(infile, preset)
    if ch == 0 or not samples:
        return None, 0
    n = None
    if seconds and rate:
        n = min(len(samples[0]), int(seconds * rate))
    arr = np.stack([np.asarray(s[:n], dtype=np.float32) if n else np.asarray(s, dtype=np.float32) for s in samples], axis=1)
    stats = []
    for i in range(ch):
        x = arr[:, i]
        if x.size == 0:
            stats.append((i, 0, 0, 0, 0, 0))
            continue
        vmin = float(np.min(x))
        vmax = float(np.max(x))
        vmean = float(np.mean(x))
        vstd = float(np.std(x))
        vrms = float(np.sqrt(np.mean(x**2)))
        p95 = float(np.percentile(np.abs(x), 95))
        stats.append((i, vmin, vmax, vmean, vstd, vrms, p95))
    return stats, ch


def main():
    p = argparse.ArgumentParser(description='Summarize EEG counts per channel (min/max/mean/std/RMS/p95)')
    p.add_argument('--preset', required=True)
    p.add_argument('--infile', required=True)
    p.add_argument('--seconds', type=float, default=None, help='Optional window length (s)')
    p.add_argument('--rate', type=float, default=256.0, help='Sampling rate for windowing')
    p.add_argument('--csvout', default=None, help='Optional CSV output path')
    args = p.parse_args()

    stats, ch = summarize(args.infile, args.preset, args.seconds, args.rate)
    if stats is None:
        print('No EEG decoded')
        return
    print(f'Channels: {ch}')
    print('ch, min, max, mean, std, rms, p95_abs')
    for row in stats:
        i, vmin, vmax, vmean, vstd, vrms, p95 = row
        print(f'{i}, {vmin:.1f}, {vmax:.1f}, {vmean:.1f}, {vstd:.1f}, {vrms:.1f}, {p95:.1f}')

    if args.csvout:
        import csv
        os.makedirs(os.path.dirname(args.csvout) or '.', exist_ok=True)
        with open(args.csvout, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ch','min','max','mean','std','rms','p95_abs'])
            for row in stats:
                i, vmin, vmax, vmean, vstd, vrms, p95 = row
                w.writerow([i, vmin, vmax, vmean, vstd, vrms, p95])
        print(f'Wrote {args.csvout}')


if __name__ == '__main__':
    main()


