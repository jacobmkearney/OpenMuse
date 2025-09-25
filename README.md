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


- p20-p61: no red light
- p1034, p1041, p1042, p1043: red light in the centre is brightly on
- p1035, p1044, p1045, p4129: red light in the centre is turned on but dimmer 


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
  - Might result in 1, 4, 5, 8, 16 OPTICS channels

Different presets enable/disable some channels, but the exact combinations are not fully documented.
- p20-p61: Red LED in the centre is off
- p1034, p1041, p1042, p1043: red LED in the centre is brightly on (suggesting the activation of OPTICS or PPG channels)
- p1035, p1044, p1045, p4129: red LED in the centre is dimmer

I have attempted to decode the raw data files recorded with different presets in [`decoding_attempts/analyze_rawdata.py`](decoding_attempts/analyze_rawdata.py).


| Header   |   FileCount | BestGuessCH   |   BlocksPerPkt_Median |   BlocksPerPkt_SD |   ObsHz_Median |   ObsHz_SD |   RelErr_Median |   SequencePurity_Median |
|:---------|------------:|:--------------|----------------------:|------------------:|---------------:|-----------:|----------------:|------------------------:|
| 0xca     |           5 | CH256         |                227.00 |            219.20 |         256.01 |       0.05 |          inf    |                    0.27 |
| 0xe4     |          11 | CH256         |                301.00 |            288.54 |         255.99 |     102.32 |            0.00 |                    0.50 |
| 0xe5     |           4 | uncertain     |                 16.00 |            nan    |         255.69 |     nan    |          inf    |                    0.12 |
| 0xef     |           5 | CH256         |                 10.00 |              0.89 |         236.90 |     102.35 |            0.03 |                    0.13 |
| 0xd9     |           4 | CH256         |                 10.00 |              1.00 |         230.88 |      93.32 |            0.04 |                    0.12 |
| 0xf2     |           4 | CH256         |                 11.00 |              4.76 |         161.72 |     115.69 |            0.03 |                    0.11 |
| 0xe8     |           8 | CH256         |                164.50 |            109.18 |         159.94 |     114.34 |            0.00 |                    0.50 |
| 0xce     |           5 | CH256         |                 22.50 |             16.58 |         159.62 |     113.25 |            0.01 |                    0.18 |
| 0xcd     |           4 | CH256         |                122.50 |            226.75 |         153.89 |     117.76 |            0.00 |                    0.27 |
| 0xd7     |          15 | CH256         |                 10.00 |              3.67 |         149.65 |     101.59 |            0.02 |                    0.13 |
| 0xdd     |           4 | CH64          |                 44.00 |              5.50 |          67.37 |       5.06 |            0.07 |                    0.17 |
| 0xd6     |           5 | uncertain     |                 35.00 |             28.00 |          66.21 |     113.85 |            0.03 |                    0.18 |
| 0xe2     |           5 | CH64          |                  9.00 |              2.71 |          65.86 |      96.87 |            0.11 |                    0.10 |
| 0xda     |           5 | CH64          |                 14.00 |              6.26 |          64.27 |      79.16 |            0.02 |                    0.14 |
| 0xee     |           4 | CH64          |                 23.00 |              5.69 |          64.16 |     110.67 |            0.00 |                    0.18 |
| 0xe1     |           4 | CH64          |                 26.00 |              8.62 |          64.11 |     113.56 |            0.07 |                    0.14 |
| 0xd2     |           5 | CH64          |                 42.00 |             75.01 |          64.03 |      76.15 |            0.01 |                    0.18 |
| 0xcf     |          15 | uncertain     |                 34.00 |             32.00 |          64.02 |      69.19 |            0.06 |                    0.20 |
| 0xeb     |           9 | uncertain     |                 93.00 |             62.72 |          63.98 |       0.15 |          inf    |                    0.18 |
| 0xd1     |           4 | CH64          |                 22.00 |             49.15 |          63.79 |       6.43 |            0.01 |                    0.15 |
| 0xea     |           6 | CH52          |                 26.00 |            159.32 |          63.79 |      72.63 |            0.00 |                    0.29 |
| 0xe7     |          15 | CH64          |                 32.00 |            138.06 |          63.59 |      68.84 |            0.00 |                    0.18 |
| 0xf0     |          15 | CH64          |                  2.00 |              7.19 |          63.55 |       7.18 |            0.04 |                    0.11 |
| 0xf3     |           5 | CH64          |                 28.00 |             22.39 |          63.39 |       4.75 |            0.09 |                    0.17 |
| 0xec     |          15 | CH64          |                  4.00 |              2.39 |          62.90 |      93.64 |            0.02 |                    0.13 |
| 0xe3     |          15 | CH64          |                 15.00 |             50.16 |          62.83 |      54.50 |            0.02 |                    0.15 |
| 0xf4     |          15 | CH64          |                 40.00 |             20.52 |          61.53 |      79.22 |            0.04 |                    0.16 |
| 0xd5     |           4 | CH256         |                 40.00 |             11.59 |          59.62 |     115.36 |            0.04 |                    0.22 |
| 0xed     |           4 | CH52          |                168.00 |             82.02 |          58.03 |       8.51 |            0.00 |                    0.45 |
| 0xdf     |          15 | CH52          |                 21.00 |             28.62 |          58.02 |      93.17 |            0.02 |                    0.17 |
| 0xe6     |           6 | CH52          |                  7.50 |              4.65 |          53.90 |       4.12 |            0.09 |                    0.16 |
| 0xcb     |           5 | CH52          |                 27.50 |             11.05 |          52.10 |       2.68 |            0.00 |                    0.17 |
| 0xd3     |          15 | CH52          |                 27.00 |             47.81 |          52.05 |      60.89 |            0.00 |                    0.20 |
| 0xde     |           2 | CH52          |                 26.00 |            nan    |          51.87 |     nan    |          inf    |                    0.25 |
| 0xdb     |          15 | CH52          |                  6.00 |              8.21 |          50.05 |      71.86 |            0.04 |                    0.12 |
| 0xe9     |           1 | uncertain     |                nan    |            nan    |         nan    |     nan    |          nan    |                    1.00 |

![](decoding_attempts/header_histogram.png)

## Related Projects

- Amused: https://github.com/Amused-EEG/amused-py
- brainflow PR #779: https://github.com/brainflow-dev/brainflow/pull/779
- MuseLSL2 PR #3: https://github.com/DominiqueMakowski/MuseLSL2/pull/3
- https://mind-monitor.com/FAQ.php#oscspec
- https://github.com/AbosaSzakal/MuseAthenaDataformatParser
- neuralencoding (uses an other app, but might contain useful information): https://github.com/BrianMohseni/neuralencoding
