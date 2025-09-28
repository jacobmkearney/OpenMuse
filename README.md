# MuseLSL3

Minimal utilities to discover and stream Muse EEG devices via Bluetooth LE.

> [!CAUTION]
> **STATUS: FAILED**
> 
> This is my attempt at enabling LSL streaming for the Muse S Athena. But I have not been able to decode the data packets properly. Please get in touch if you can help.

## Install

From the project root:

```powershell
pip install -e .
```

## Usage

### Find Muse devices

```powershell
MuseLSL3 find --timeout 10
```

### Record data from the Muse S Athena

```powershell
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 10 --outfile data.txt --preset p1035
```

Presets to test:

```powershell
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p20.txt --preset p20
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p21.txt --preset p21
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p50.txt --preset p50 
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p51.txt --preset p51
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p60.txt --preset p60
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p61.txt --preset p61
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1034.txt --preset p1034
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1035.txt --preset p1035
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1041.txt --preset p1041
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1042.txt --preset p1042
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1043.txt --preset p1043
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1044.txt --preset p1044
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1045.txt --preset p1045
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p1046.txt --preset p1046
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 60 --outfile data_raw/data_p4129.txt --preset p4129
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
  - Might result in 1, 4, 5, 8, 16 OPTICS channels

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

### Attempt 

- [Validate ID-info byte](./decoding_attempts/validate_IDByte.py):

I started by validating the byte that allegedly contained information, which strongly suggests that the 10th byte (index 9) contains the ID-info byte, namely info about the frequency and channel type.


### Successful Parsing

✅ [Timestamp](./decoding_attempts/validate_Timestamp.py):

```python
import urllib.request
import matplotlib.pyplot as plt
import MuseLSL3

url = "https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_p1034.txt"

lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

times, t = MuseLSL3.decode_rawdata(lines)
plt.plot(times, t, ".")
```

## Related Projects

- Amused: https://github.com/Amused-EEG/amused-py
- brainflow PR #779: https://github.com/brainflow-dev/brainflow/pull/779
- MuseLSL2 PR #3: https://github.com/DominiqueMakowski/MuseLSL2/pull/3
- https://mind-monitor.com/FAQ.php#oscspec
- https://github.com/AbosaSzakal/MuseAthenaDataformatParser
- neuralencoding (uses an other app, but might contain useful information): https://github.com/BrianMohseni/neuralencoding
