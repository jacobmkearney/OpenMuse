# MuseLSL3

Minimal utilities to discover and stream Muse EEG devices via Bluetooth LE.

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
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p20.txt --preset p20
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p21.txt --preset p21
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p50.txt --preset p50 
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p51.txt --preset p51
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p61.txt --preset p61
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p1034.txt --preset p1034
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p1035.txt --preset p1035
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p1044.txt --preset p1044
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p1045.txt --preset p1045
MuseLSL3 record --address 00:55:DA:B9:FA:20 --timeout 30 --outfile data_raw/data_p4129.txt --preset p4129
```

Notes:
- p20-p61: no red light is turned on
- p1034: red light in the centre is turned on
- p1035, p1044, p1045, p4129: red light in the centre is turned on but dimmer than p1034


## Decoding Information

Muse S Athena specs (From the [Muse website](https://eu.choosemuse.com/products/muse-s-athena) - note that these info might not be up to date or fully accurate):
- Wireless Connection: BLE 5.3, 2.4 GHz
- EEG Channels: 4 EEG channels (TP9, AF7, AF8, TP10) + 1 (or 4?) amplified Aux channels
  - Sample Rate: 256 Hz
  - Sample Resolution: 14 bits / sample
- Accelerometer: Three-axis at 52Hz, 16-bit resolution, range +/- 2G
- Gyroscope: Three-axis at 52Hz, 16-bit resolution, range +/- 250dps
- PPG Sensor: Triple wavelength: IR (850nm), Near-IR (730nm), Red (660nm), 64 Hz sample rate, 20-bit resolution
- fNIRS Sensor: 5-optode bilateral frontal cortex hemodynamics, 64 Hz sample rate, 20-bit resolution

This means that there are potentially:
- 5 channels sampled at 256Hz (4 EEG + 1 Aux) (possibly 8 if 4 Aux channels, to be confirmed)
- 6 channels sampled at 52Hz (3 Acc + 3 Gyro)
- 8 channels sampled at 64Hz (3 PPG, 5 fNIRS)
Different presets may enable/disable some channels.


## Related Projects

- Amused: https://github.com/Amused-EEG/amused-py
- brainflow PR #779: https://github.com/brainflow-dev/brainflow/pull/779
- MuseLSL2 PR #3: https://github.com/DominiqueMakowski/MuseLSL2/pull/3
- https://mind-monitor.com/FAQ.php#oscspec
- neuralencoding (uses an other app, but might contain useful information): https://github.com/BrianMohseni/neuralencoding
