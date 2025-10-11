## Muse EEG decoding (Athena) — notes and current status

This folder contains the consolidated EEG decoder for the Muse Athena. It stitches EEG bytes across co‑packed payload chunks and packet boundaries, unpacks 14‑bit samples, and exposes a simple API and CLI.

### Exploration

1) Packet framing and IDs
- Each BLE message contains one or more packets; every packet is `[LEN, counter, time(4), unk(3), pkt_id, unk(3), 0x00, data…]` with a fixed 14‑byte header.
- The first byte is the length, and the `pkt_id` sits at header byte 9.
- Using `inspect_packets.py` and segmentation scripts, confirmed EEG packet IDs:
  - `0x11` → EEG4 (e.g., preset `p1035`)
  - `0x12` → EEG8 (e.g., presets `p1041`, `p1045`)
  - Co‑packed tags frequently observed alongside EEG: `0x34/0x36` (optics/PPG), `0x47` (IMU), `0x98` (battery).

2) Base units and payload structure
- `segment_packets.py` and `inspect_eeg_units.py`, computed payload histograms and remainder distributions.
- EEG payloads are composed of repeated 28‑byte base units (or 56 byte pairs). Across different presets you can see either 8x28B per packet, plus extras from co-paced chunks
- Dominant payload sizes matched that expectation: ~222 B (EEG4) and ~226 B (EEG8). Estimated samples/packet: EEG4 ≈ 32, EEG8 ≈ 16.

3) 14‑bit unpacking + bit offset
- We implemented a 14‑bit two’s‑complement unpacker and a data‑driven bit‑offset search (offset ∈ 0..7) that minimizes outlier continuity.
- Stable preset offsets emerged:
  - `p1035` (EEG4): bit_offset ≈ 2
  - `p1041` (EEG8): bit_offset ≈ 0
  - `p1045` (EEG8): bit_offset ≈ 3

4) Stitching EEG across tags and packets
- only taking bytes up to the first secondary tag undercounted samples.
- Final approach scans the entire payload, skipping non‑EEG chunks and collecting EEG bytes; plus a rolling buffer to join 28B blocks across packet boundaries. This recovers nearly all samples.

5) Rate verification and physiology checks
- Wall‑clock estimator on text recordings shows ≈246–247 Hz over ~30 s (expected 256 Hz; minor shortfall due to conservative scanning and short windows).

### What’s in here now

- `core.py` — stitched decoder
  - `decode_file(path, preset) -> (per_channel_samples, channels)`
  - Scans full payloads, skips non‑EEG tags, stitches across packets, applies frozen preset bit offsets.
- `__init__.py` — re‑exports
  - `decode_file`, `PRESET_DEFAULTS`

Related CLIs (under `tools/`):
- `muse_eeg_decode.py` — decodes a `.bin` to an NPZ with `(samples × channels)` EEG array.
- `plot_eeg_quick.py` — plots ~3 s window with channel labels, alpha bars, optional blink markers.
- `segment_packets.py`, `inspect_eeg_units.py` — segmentation and base‑unit inspection.
- `estimate_eeg_rate_wallclock.py` — empirical rate from text recordings (wall‑clock timestamps).

### Current status and assumptions

- Preset defaults (heuristic, stable in our tests):
  - `p1035`: 4 channels, bit_offset=2
  - `p1041`: 8 channels, bit_offset=0
  - `p1045`: 8 channels, bit_offset=3
- EEG base unit: 28 bytes per 16 values → treated as `2 samples × channels` per 28B block.
- Co‑packed data is present and skipped automatically (optics/IMU/battery).
- Channel order is currently heuristic (estimated via correlations/blink/alpha features) and may be refined.

### Quick usage

Decode `.bin` to NPZ (replace paths/preset as needed):
```bash
PYTHONPATH=/path/to/MuseLSL3 \
python3 tools/muse_eeg_decode.py \
  --preset p1045 \
  --infile data/p1045/session.bin \
  --out p1045_eeg.npz
```

Plot a short window with alpha bars and blink markers:
```bash
PYTHONPATH=/path/to/MuseLSL3 \
python3 tools/plot_eeg_quick.py \
  --preset p1045 \
  --infile data/p1045/session.bin \
  --seconds 3 --rate 256 --detect-blinks \
  --out p1045_quick.png
```

### Next steps
- Confirm channel order per preset (pairing symmetry, alpha/bink features) and freeze a canonical ordering.
- Expose scaling (counts → µV) when gain is confirmed.
- Add an LSL EEG outlet for real‑time streaming.


