# MuseLSL3

Minimal utilities to discover and stream Muse EEG devices via Bluetooth LE.

> [!CAUTION]
> **STATUS: EXPERIMENTAL**
> 
> Accelerometer and gyroscope decoding plus LSL streaming are freshly implemented. Expect rough edges and please report any issues you encounter.

## Install

From the project root:

```powershell
pip install -e .
```

## Usage

### EEG decoding quickstart (prototype)

Decode stitched EEG from a `.bin` into a NumPy NPZ (samples×channels), using frozen preset defaults (bit offset, channels):

```bash
PYTHONPATH=/Users/jacobkearney/Documents/personal/MuseLSL3 \
python3 /Users/jacobkearney/Documents/personal/MuseLSL3/tools/muse_eeg_decode.py \
  --preset p1045 \
  --infile /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/session.bin \
  --out /Users/jacobkearney/Documents/personal/MuseLSL3/p1045_eeg.npz
```

Plot a ~3 s window with alpha bars and blink markers (uses locked bit offsets per preset):

```bash
PYTHONPATH=/Users/jacobkearney/Documents/personal/MuseLSL3 \
python3 /Users/jacobkearney/Documents/personal/MuseLSL3/tools/plot_eeg_quick.py \
  --preset p1045 \
  --infile /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/session.bin \
  --seconds 3 --rate 256 --detect-blinks \
  --out /Users/jacobkearney/Documents/personal/MuseLSL3/p1045_quick.png
```

Verify ~256 Hz empirically (wall-clock from text recording):

```bash
MuseLSL3 record --preset p1045 --duration 30 --outfile /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/rate_check_30s.txt

PYTHONPATH=/Users/jacobkearney/Documents/personal/MuseLSL3 \
python3 /Users/jacobkearney/Documents/personal/MuseLSL3/tools/estimate_eeg_rate_wallclock.py \
  --infile /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/rate_check_30s.txt
```

Segment packets and inspect base units (28/56 B):

```bash
PYTHONPATH=/Users/jacobkearney/Documents/personal/MuseLSL3 \
python3 /Users/jacobkearney/Documents/personal/MuseLSL3/tools/segment_packets.py \
  --preset p1045 \
  --infile /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/session.bin \
  --csvout /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/segmented.csv

PYTHONPATH=/Users/jacobkearney/Documents/personal/MuseLSL3 \
python3 /Users/jacobkearney/Documents/personal/MuseLSL3/tools/inspect_eeg_units.py \
  --preset p1045 \
  --csv /Users/jacobkearney/Documents/personal/MuseLSL3/data/p1045/segmented.csv
```

Notes:
- Preset defaults (bit offsets, channel count) are heuristic but stable in our tests: `p1035=2/4ch`, `p1041=0/8ch`, `p1045=3/8ch`.
- The stitched decoder scans full payloads, skips non-EEG chunks, and joins EEG units across packet boundaries.

### Find Muse devices

```powershell
MuseLSL3 find --timeout 10
```

### Record raw data from the Muse S Athena

```powershell
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 20 --outfile data.txt --preset p1041
```

Presets to test:

```powershell
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p20.txt --preset p20
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p21.txt --preset p21
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p50.txt --preset p50 
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p51.txt --preset p51
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p60.txt --preset p60
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p61.txt --preset p61
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1034.txt --preset p1034
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1035.txt --preset p1035
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1041.txt --preset p1041
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1042.txt --preset p1042
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1043.txt --preset p1043
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1044.txt --preset p1044
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1045.txt --preset p1045
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p1046.txt --preset p1046
MuseLSL3 record --address 00:55:DA:B9:FA:20 --duration 60 --outfile data_raw/data_p4129.txt --preset p4129
```

### Stream ACC/GYRO over LSL

```powershell
# Stream only (no file save)
MuseLSL3 stream --address 00:55:DA:B9:FA:20 --preset p1041 --duration 20

# Stream AND save to JSON file
MuseLSL3 stream --address 00:55:DA:B9:FA:20 --preset p1041 --duration 20 --outfile data.json
```

This creates a single LSL outlet named **"MuseAccGyro"** with 6 channels: `ACC_X`, `ACC_Y`, `ACC_Z`, `GYRO_X`, `GYRO_Y`, `GYRO_Z` at 52 Hz. Omit `--duration` to stream until you press `Ctrl+C`, or use `-d 30` to stream for 30 seconds.

**Optional file saving:** Use `--outfile` or `-o` to save decoded samples to a JSON file with structure:
```json
{
  "ACC": {"time": [1728067845.123, ...], "time_lsl": [1234567890.456, ...], "ACC_X": [...], "ACC_Y": [...], "ACC_Z": [...]},
  "GYRO": {"time": [1728067845.123, ...], "time_lsl": [1234567890.456, ...], "GYRO_X": [...], "GYRO_Y": [...], "GYRO_Z": [...]}
}
```
Both `time` (device timestamp) and `time_lsl` (LSL timestamp) are floats in seconds since epoch for easy synchronization.

You can also call the Python API directly:

```python
import MuseLSL3

# Stream only
MuseLSL3.stream(
    address="00:55:DA:B9:FA:20",
    preset="p1035",
)

# Stream and save
MuseLSL3.stream(
    address="00:55:DA:B9:FA:20",
    preset="p1035",
    outfile="session1.json",
)
```

**Load saved JSON data:**
```python
import json
from datetime import datetime

with open("data.json") as f:
    data = json.load(f)

acc_x = data["ACC"]["ACC_X"]  # List of accelerometer X values
gyro_z = data["GYRO"]["GYRO_Z"]  # List of gyroscope Z values
times = data["ACC"]["time"]  # Float timestamps (device time, seconds since epoch)
times_lsl = data["ACC"]["time_lsl"]  # LSL timestamps (seconds since epoch)

# Convert to datetime if needed
times_dt = [datetime.fromtimestamp(t) for t in times]
```

### Visualize ACC/GYRO data in real-time

While streaming in one terminal, open another terminal to visualize the data:

```powershell
# Terminal 1: Start streaming
MuseLSL3 stream --address 00:55:DA:B9:FA:20 --preset p1041

# Terminal 2: View the live data
MuseLSL3 view
```

Optional parameters:
- `--window 10.0` or `-w 10.0`: Set time window in seconds (default: 10.0)
- `--duration 30` or `-d 30`: Auto-close after 30 seconds
- `--stream-name MuseAccGyro`: Specify LSL stream name (default: MuseAccGyro)

Python API:

```python
import MuseLSL3

# View the stream (blocks until window closed)
MuseLSL3.view(
    stream_name="MuseAccGyro",
    window_size=10.0,
)
```



## Decoding

### Constructor Information

Muse S Athena specs (From the [Muse website](https://eu.choosemuse.com/products/muse-s-athena) - note that these info might not be up to date or fully accurate):
- Wireless Connection: BLE 5.3, 2.4 GHz
- EEG Channels: 4 EEG channels (TP9, AF7, AF8, TP10) + 1 (or 4?) amplified Aux channels
  - Sample Rate: 256 Hz
  - Sample Resolution: 14 bits / sample
- Accelerometer: Three-axis at 52Hz, 16-bit resolution, range +/- 2G
- Gyroscope: Three-axis at 52Hz, 16-bit resolution, range +/- 250dps
- PPG Sensor: Triple wavelength: IR (850nm), Near-IR (730nm), Red (660nm), 64 Hz sample rate, 20-bit resolution
- fNIRS Sensor: 5-optode bilateral frontal cortex hemodynamics, 64 Hz sample rate, 20-bit resolution

### Presets

Different presets enable/disable some channels, but the exact combinations are not fully documented.

| Presets                                 | EEG   | REF     | PPG     | Optics   | ACC/GYRO | Battery | Red LED |
|-----------------------------------------|:-----:|:-------:|:-------:|:--------:|:--------:|:-------:|:--------|
| p20, p21, p50, p51, p60, p61            | EEG4  | DRL_REF |         |          |    X     |    X    |   off   |
| p1034, p1043                            | EEG8  | DRL_REF |    X    | Optics8  |    X     |    X    | bright  |
| p1044                                   | EEG8  | DRL_REF |    X    | Optics8  |    X     |    X    |  dim    |
| p1035                                   | EEG4  | DRL_REF |    X    | Optics4  |    X     |    X    |  dim    |
| p1041, p1042                            | EEG8  | DRL_REF |    X    | Optics16 |    X     |    X    | bright  |
| p1045                                   | EEG8  | DRL_REF |    X    | Optics4  |    X     |    X    |  dim    |
| p1046                                   | EEG8  | DRL_REF |    X    | Optics4  |    X     |    X    |   —     |
| p4129                                   | EEG8  | DRL_REF |    X    | Optics4  |    X     |    X    |  dim    |

*Table derived from the signature of the data packets present in the data.*



## Related Projects

- Amused: https://github.com/Amused-EEG/amused-py
- brainflow PR #779: https://github.com/brainflow-dev/brainflow/pull/779
- MuseLSL2 PR #3: https://github.com/DominiqueMakowski/MuseLSL2/pull/3
- https://mind-monitor.com/FAQ.php#oscspec
- https://github.com/AbosaSzakal/MuseAthenaDataformatParser
- neuralencoding (uses an other app, but might contain useful information): https://github.com/BrianMohseni/neuralencoding
