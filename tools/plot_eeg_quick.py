import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from muse_eeg import PRESET_DEFAULTS, decode_file
import numpy as np


EEG_IDS = {0x11: 4, 0x12: 8}
NON_EEG_TAGS = {0x34, 0x35, 0x36, 0x47, 0x98}


def parse_packets_from_bin(data: bytes):
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


def first_secondary_offset(payload: bytes):
    i = 0
    n = len(payload)
    while i + 2 <= n:
        tag = payload[i]
        ln = payload[i + 1]
        if tag in NON_EEG_TAGS and i + 2 + ln <= n:
            return i
        i += 1
    return None


def collect_series_via_core(infile: str, seconds: float, rate: float, preset: str):
    samples, ch = decode_file(infile, preset)
    if not samples or ch == 0:
        return [], 0, None
    n = min(len(samples[0]), int(seconds * rate)) if seconds and rate else len(samples[0])
    series = [list(map(float, s[:n])) for s in samples]
    bitoff = PRESET_DEFAULTS.get(preset, {}).get('bit_offset')
    return series, ch, bitoff


def alpha_ratio(x, fs):
    if len(x) < 4:
        return 0.0
    xf = np.array(x, dtype=np.float32)
    # Detrend
    xf = xf - np.mean(xf)
    # Simple Welch-like PSD via rFFT
    N = len(xf)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    psd = np.abs(np.fft.rfft(xf))**2
    def band(f1,f2):
        idx = (freqs>=f1) & (freqs<=f2)
        return float(np.sum(psd[idx]))
    alpha = band(8,12)
    broad = band(1,40) + 1e-9
    return alpha/broad


def main():
    p = argparse.ArgumentParser(description='Plot ~3s EEG window and alpha ratios')
    p.add_argument('--preset', required=True)
    p.add_argument('--infile', required=True)
    p.add_argument('--seconds', type=float, default=3.0)
    p.add_argument('--rate', type=float, default=256.0)
    p.add_argument('--scale', type=float, default=1.0, help='Optional scale factor for display (e.g., 0.0885 to convert counts→µV)')
    p.add_argument('--out', default=None)
    p.add_argument('--detect-blinks', action='store_true', help='Annotate likely blink events')
    args = p.parse_args()

    series, ch, bitoff = collect_series_via_core(args.infile, args.seconds, args.rate, args.preset)
    if not series:
        print('No data decoded')
        return

    t = np.arange(len(series[0]))/args.rate
    # Plot all channels in separate subplots for clarity
    sel_idx = list(range(ch))
    fig_height = max(6, 1.6 * len(sel_idx))
    fig, axes = plt.subplots(len(sel_idx), 1, figsize=(11, fig_height), sharex=True)
    if len(sel_idx) == 1:
        axes = [axes]

    eff_scale = args.scale

    # Optional blink annotation indices computed once (using first two channels)
    blink_idx = None
    if args.detect_blinks and ch >= 2:
        def zscore(x):
            x = np.asarray(x, dtype=np.float32)
            m = np.mean(x)
            s = np.std(x) if np.std(x) > 1e-6 else 1.0
            return (x - m) / s
        z = np.abs(zscore(series[0])) + np.abs(zscore(series[1]))
        thr = np.percentile(z, 99.5)
        blink_idx = np.where(z >= thr)[0]

    for ax, i in zip(axes, sel_idx):
        y = np.array(series[i], dtype=np.float32) * eff_scale
        ax.plot(t, y, lw=0.8)
        if blink_idx is not None and y.size:
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            ax.vlines(t[blink_idx], ymin=ymin, ymax=ymax, colors='r', alpha=0.15)
        ax.set_ylabel(f'ch{i}\n' + ('µV' if args.scale != 1.0 else 'counts'))
        ax.grid(True, which='both', alpha=0.2)

    axes[0].set_title(f'{args.preset} EEG (~{args.seconds}s), bit_off={bitoff}')
    axes[-1].set_xlabel('Time (s)')

    out = args.out or f'{args.preset}_quick.png'
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()


