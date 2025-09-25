"""
Conclusion from this analysis:

Extract the packet timestamp by reading the 32‑bit unsigned little‑endian field located at bytes 3–6 of the payload (0‑based indexing), interpret that integer as milliseconds, and convert to seconds by multiplying by 1e‑3; use this value as the packet’s authoritative time (it aligns with the file timestamps).
"""

import os
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = "./data_raw/"
FILES = sorted(p for p in os.listdir(DATA_DIR) if p.endswith(".txt"))

# reuse the robust scorer from earlier (paste the function block here)
import struct, numpy as np
from datetime import datetime

MAX_OFFSET = 12
WIDTHS = [2, 4, 8]
ENDIANS = ["<", ">"]
SIGNED = [False, True]
UNITS = [("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)]


def load(path, max_lines=None):
    times = []
    payloads = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) < 3:
                continue
            times.append(
                datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
            )
            payloads.append(bytes.fromhex(parts[2]))
            if max_lines and len(times) >= max_lines:
                break
    return np.array(times, dtype=np.float64), payloads


def unpack_at(payloads, offset, width, endian, signed):
    fmt = {2: "h", 4: ("I" if not signed else "i"), 8: ("Q" if not signed else "q")}[
        width
    ]
    s = endian + fmt
    out = np.full(len(payloads), np.nan, dtype=np.float64)
    for i, p in enumerate(payloads):
        if len(p) >= offset + width:
            out[i] = struct.unpack(s, p[offset : offset + width])[0]
    return out


def candidate_metrics(raw, file_times):
    mask = np.isfinite(raw)
    if mask.sum() < 4:
        return None
    vals = raw[mask]
    fts = file_times[mask]
    dv = np.diff(vals)
    mono = float(np.mean(dv >= 0))
    if np.all(dv == 0):
        return None
    med_dv = (
        float(np.median(np.abs(dv[dv != 0])))
        if np.any(dv != 0)
        else float(np.median(np.abs(dv) + 1e-12))
    )
    med_df = float(np.median(np.diff(fts)))
    return dict(vals=vals, fts=fts, mono=mono, med_dv=med_dv, med_df=med_df)


def best_candidate_for_file(path, max_lines=3000):
    file_times, payloads = load(path, max_lines=max_lines)
    best = None
    for offset in range(0, MAX_OFFSET + 1):
        for width in WIDTHS:
            for endian in ENDIANS:
                for signed in SIGNED:
                    raw = unpack_at(payloads, offset, width, endian, signed)
                    info = candidate_metrics(raw, file_times)
                    if info is None:
                        continue
                    dv = info["med_dv"]
                    for uname, unit in UNITS:
                        cvals = (info["vals"] - info["vals"][0]) * unit
                        y = info["fts"] - info["fts"][0]
                        if len(cvals) < 4:
                            continue
                        mask = np.isfinite(cvals) & np.isfinite(y)
                        if mask.sum() < 4:
                            continue
                        x = cvals[mask]
                        yy = y[mask]
                        A = np.vstack([x, np.ones_like(x)]).T
                        sol, residuals, *_ = np.linalg.lstsq(A, yy, rcond=None)
                        slope = sol[0]
                        y_pred = A @ sol
                        ss_res = np.sum((yy - y_pred) ** 2)
                        ss_tot = np.sum((yy - np.mean(yy)) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                        px = (
                            np.cov(x, yy, bias=True)[0, 1] / (np.std(x) * np.std(yy))
                            if np.std(x) > 0 and np.std(yy) > 0
                            else 0.0
                        )
                        slope_err = abs(slope - 1.0) / (abs(slope) + 1e-12)
                        quality = (
                            0.45 * info["mono"]
                            + 0.45 * max(0.0, r2)
                            - 0.10 * min(1.0, slope_err)
                        )
                        candidate = {
                            "offset": offset,
                            "width": width,
                            "endian": endian,
                            "signed": signed,
                            "unit": uname,
                            "mono": info["mono"],
                            "med_dv": info["med_dv"],
                            "med_df": info["med_df"],
                            "tick_s": info["med_dv"] * unit,
                            "slope": slope,
                            "r2": r2,
                            "pearson": px,
                            "quality": quality,
                        }
                        if best is None or candidate["quality"] > best["quality"]:
                            best = candidate
    return best


# loop through files, collect bests
summary = {}
for fn in FILES:
    path = os.path.join(DATA_DIR, fn)
    best = best_candidate_for_file(path, max_lines=3000)
    summary[fn] = best
    if best is None:
        print(f"{fn}: no candidate")
    else:
        print(
            f"{fn}: off={best['offset']} w={best['width']} end={best['endian']} "
            f"{'signed' if best['signed'] else 'unsigned'} unit={best['unit']} "
            f"mono={best['mono']:.3f} r2={best['r2']:.3f} slope={best['slope']:.3f} quality={best['quality']:.3f}"
        )

# aggregate consistency checks
agg_offset = Counter()
agg_width = Counter()
agg_endian = Counter()
agg_signed = Counter()
agg_unit = Counter()
for v in summary.values():
    if not v:
        continue
    agg_offset[v["offset"]] += 1
    agg_width[v["width"]] += 1
    agg_endian[v["endian"]] += 1
    agg_signed["signed" if v["signed"] else "unsigned"] += 1
    agg_unit[v["unit"]] += 1

print("\nAggregate top-candidate distribution across files:")
print(" offsets:", dict(agg_offset))
print(" widths :", dict(agg_width))
print(" endian :", dict(agg_endian))
print(" signed :", dict(agg_signed))
print(" units  :", dict(agg_unit))


# report files that disagree with the modal choice
def modal(counter):
    return counter.most_common(1)[0][0] if counter else None


modal_off = modal(agg_offset)
modal_w = modal(agg_width)
modal_end = modal(agg_endian)
modal_signed = modal(agg_signed)
modal_unit = modal(agg_unit)

print("\nFiles disagreeing with modal candidate:")
for fn, v in summary.items():
    if not v:
        print(f" {fn}: no candidate")
        continue
    if (
        v["offset"],
        v["width"],
        v["endian"],
        ("signed" if v["signed"] else "unsigned"),
        v["unit"],
    ) != (modal_off, modal_w, modal_end, modal_signed, modal_unit):
        print(
            f" {fn}: off={v['offset']} w={v['width']} end={v['endian']} "
            f"{'signed' if v['signed'] else 'unsigned'} unit={v['unit']} r2={v['r2']:.3f} mono={v['mono']:.3f}"
        )

# =========================
# VISUAL TEST
# =========================
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

DATA_DIR = "./data_raw/"
FILES = sorted(p for p in os.listdir(DATA_DIR) if p.endswith(".txt"))
OFFSET = 3  # discovered payload offset (0-based) for 4-byte unsigned ms
WIDTH = 4
ENDIAN = "<"  # little-endian
UNIT_SCALE = 1e-3  # ms -> s


def load_file(path):
    file_times = []
    payloads = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) < 3:
                continue
            file_times.append(
                datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
            )
            payloads.append(bytes.fromhex(parts[2]))
    return np.array(file_times, dtype=np.float64), payloads


def payload_ts_seconds(payloads):
    fmt = ENDIAN + ("I" if WIDTH == 4 else "H" if WIDTH == 2 else "Q")
    out = []
    for p in payloads:
        if len(p) >= OFFSET + WIDTH:
            out.append(struct.unpack(fmt, p[OFFSET : OFFSET + WIDTH])[0] * UNIT_SCALE)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


plt.figure(figsize=(7, 7))
cmap = plt.get_cmap("tab20")
for i, fn in enumerate(FILES):
    path = os.path.join(DATA_DIR, fn)
    file_times, payloads = load_file(path)
    if file_times.size == 0:
        continue
    p_ts = payload_ts_seconds(payloads)

    # keep only finite pairs and rebase to zero origin per file
    mask = np.isfinite(p_ts) & np.isfinite(file_times)
    if mask.sum() < 4:
        continue
    x = p_ts[mask] - p_ts[mask][0]
    y = file_times[mask] - file_times[mask][0]

    plt.scatter(x, y, s=6, color=cmap(i % 20), label=fn, alpha=0.8)

# plot identity line and cosmetics
lims = plt.gca().get_xlim()
minv = min(lims[0], plt.gca().get_ylim()[0])
maxv = max(lims[1], plt.gca().get_ylim()[1])
plt.plot([minv, maxv], [minv, maxv], color="k", linestyle="--", linewidth=1)
plt.xlabel("Payload timestamp (s) relative to first packet")
plt.ylabel("File line timestamp (s) relative to first packet")
plt.title("Payload ms timestamps vs file timestamps (one color per file)")
plt.legend(fontsize="small", ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.grid(alpha=0.3)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()




# Test
import urllib.request 

url = "https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_p1034.txt"

lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

times, t = decode_rawdata(lines)
# Correlation between extracted payload times and file times
scipy.stats.pearsonr(np.array(times), np.array(t))
plt.plot(times, t, ".")