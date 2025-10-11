import argparse
import os
from collections import Counter, defaultdict
import numpy as np

from muse_eeg_unpack import decode_eeg_payload, find_best_bit_offset

EEG_IDS = {0x11: 4, 0x12: 8}
NON_EEG_TAGS = {0x34, 0x35, 0x36, 0x47, 0x98}


def parse_packets_from_bin(data: bytes):
	pos = 0
	N = len(data)
	while pos < N:
		if pos + 1 > N:
			break
		L = data[pos]
		if L < 14:
			pos += 1
			continue
		if pos + L > N:
			break
		yield pos, data[pos : pos + L]
		pos += L


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


def alpha_ratio(x: np.ndarray, fs: float) -> float:
	if len(x) < 8:
		return 0.0
	x = x.astype(np.float32)
	x = x - x.mean()
	freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
	psd = np.abs(np.fft.rfft(x))**2
	def band(f1,f2):
		idx = (freqs>=f1) & (freqs<=f2)
		return float(psd[idx].sum())
	alpha = band(8,12)
	broad = band(1,40) + 1e-9
	return alpha / broad


def blink_strength(x: np.ndarray) -> float:
	if len(x) < 8:
		return 0.0
	x = x.astype(np.float32)
	z = (x - x.mean()) / (x.std() + 1e-6)
	thr = np.percentile(np.abs(z), 99.5)
	idx = np.where(np.abs(z) >= thr)[0]
	return float(np.median(np.abs(x[idx]))) if idx.size else 0.0


def collect_window_series(blob: bytes, seconds: float, rate: float, start_packet: int):
	needed = int(seconds * rate)
	locked = None
	series = None
	channels = None
	collected = 0
	pkt_idx = 0
	for _, pkt in parse_packets_from_bin(blob):
		if pkt_idx < start_packet:
			pkt_idx += 1
			continue
		if len(pkt) < 14:
			pkt_idx += 1
			continue
		pid = pkt[9]
		if pid not in EEG_IDS:
			pkt_idx += 1
			continue
		channels = EEG_IDS[pid]
		payload = pkt[14:]
		off = first_secondary_offset(payload)
		primary = payload if off is None else payload[:off]
		if len(primary) < 28:
			pkt_idx += 1
			continue
		if locked is None:
			locked = find_best_bit_offset(primary[:28], channels=channels)
		samples, _ = decode_eeg_payload(primary, channels=channels, block_bytes=28, forced_bit_offset=locked)
		if not samples:
			pkt_idx += 1
			continue
		if series is None:
			series = [list() for _ in range(channels)]
		for s in samples:
			for ch in range(channels):
				series[ch].append(int(s[ch]))
				collected += 1
				if collected >= needed:
					return [np.asarray(c, dtype=np.int32) for c in series], channels, locked
		pkt_idx += 1
	return [np.asarray(c, dtype=np.int32) for c in (series or [])], channels or 0, locked or 0


def correlate_pairs(series: list[np.ndarray]) -> list[tuple[int,int,float]]:
	ch = len(series)
	arr = [s.astype(np.float32) for s in series]
	corr = np.zeros((ch, ch), dtype=np.float32)
	for i in range(ch):
		for j in range(i+1, ch):
			a = arr[i] - arr[i].mean()
			b = arr[j] - arr[j].mean()
			den = (a.std() * b.std() + 1e-9)
			r = float((a*b).sum() / (len(a) * den))
			corr[i,j] = corr[j,i] = r
	pairs = []
	used = set()
	while len(used) < ch-1:
		best = None
		for i in range(ch):
			if i in used: continue
			for j in range(i+1, ch):
				if j in used: continue
				v = corr[i,j]
				if best is None or v > best[2]:
					best = (i,j,v)
		if best is None:
			break
		i,j,v = best
		pairs.append(best)
		used.add(i); used.add(j)
	return pairs


def main():
	p = argparse.ArgumentParser(description='Stabilize channel mapping across files by aggregating windows')
	p.add_argument('--preset', required=True)
	p.add_argument('--files', nargs='+', required=True, help='List of .bin files (e.g., eyes_open.bin eyes_closed.bin blinks.bin)')
	p.add_argument('--seconds', type=float, default=5.0)
	p.add_argument('--rate', type=float, default=256.0)
	p.add_argument('--stride', type=int, default=10, help='Window start step in packets')
	p.add_argument('--csvout', default=None, help='Optional combined CSV output for pairs and per-channel metrics')
	p.add_argument('--mdout', default=None, help='Optional markdown file to append a summary section')
	args = p.parse_args()

	pair_counts = Counter()
	blink_scores = defaultdict(list)
	alpha_scores = defaultdict(list)

	for fpath in args.files:
		with open(fpath, 'rb') as f:
			blob = f.read()
		start = 0
		for _ in range(10):  # up to 10 windows per file
			series, ch, bitoff = collect_window_series(blob, args.seconds, args.rate, start)
			if ch == 0 or not series:
				break
			for (i,j,r) in correlate_pairs(series):
				pair_counts[(i,j)] += 1
			# metrics
			for k, s in enumerate(series):
				blink_scores[k].append(blink_strength(s))
				alpha_scores[k].append(alpha_ratio(s.astype(np.float32), args.rate))
			start += args.stride

	print(f'{args.preset}: aggregated over {len(args.files)} files')
	print('Top pairs (i,j)->count:')
	for (i,j), c in pair_counts.most_common(8):
		print(f'  ({i},{j}) -> {c}')
	# Summaries per channel
	print('Blink medians per ch:')
	for k in sorted(blink_scores.keys()):
		vals = blink_scores[k]
		print(f'  ch{k}: med={np.median(vals):.1f} (n={len(vals)})')
	print('Alpha medians per ch:')
	for k in sorted(alpha_scores.keys()):
		vals = alpha_scores[k]
		print(f'  ch{k}: med={np.median(vals):.3f} (n={len(vals)})')
	# Heuristic mapping
	frontal = sorted(blink_scores.items(), key=lambda kv: -np.median(kv[1]))[:2]
	posterior = sorted(alpha_scores.items(), key=lambda kv: -np.median(kv[1]))[:2]
	print('Suggested frontal:', [k for k,_ in frontal])
	print('Suggested posterior:', [k for k,_ in posterior])

	# Optional CSV output
	if args.csvout:
		import csv, os
		os.makedirs(os.path.dirname(args.csvout) or '.', exist_ok=True)
		with open(args.csvout, 'w', newline='') as f:
			w = csv.writer(f)
			w.writerow(['kind','i','j','count','ch','blink_median','alpha_median'])
			for (i,j), c in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
				w.writerow(['pair', i, j, c, '', '', ''])
			for k in sorted(set(list(blink_scores.keys())+list(alpha_scores.keys()))):
				bm = float(np.median(blink_scores.get(k, [0.0])))
				am = float(np.median(alpha_scores.get(k, [0.0])))
				w.writerow(['channel', '', '', '', k, f'{bm:.3f}', f'{am:.5f}'])

	# Optional markdown append
	if args.mdout:
		from datetime import datetime
		with open(args.mdout, 'a', encoding='utf-8') as f:
			f.write(f"\n\n### Channel mapping (heuristic) — {args.preset}\n")
			f.write(f"Aggregated {len(args.files)} files at {datetime.utcnow().isoformat()}Z\n\n")
			f.write("Top pairs (i,j)->count:\n\n")
			for (i,j), c in pair_counts.most_common(8):
				f.write(f"- ({i},{j}) -> {c}\n")
			f.write("\nPer-channel medians:\n\n")
			f.write("ch | blink_median | alpha_median\n")
			f.write(":-:|:-:|:-:\n")
			for k in sorted(set(list(blink_scores.keys())+list(alpha_scores.keys()))):
				bm = float(np.median(blink_scores.get(k, [0.0])))
				am = float(np.median(alpha_scores.get(k, [0.0])))
				f.write(f"{k} | {bm:.1f} | {am:.3f}\n")
			f.write("\nSuggested frontal: " + ", ".join(str(k) for k,_ in frontal) + "\n")
			f.write("Suggested posterior: " + ", ".join(str(k) for k,_ in posterior) + "\n")


if __name__ == '__main__':
	main()
