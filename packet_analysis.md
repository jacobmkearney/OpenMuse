

### p1035

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1035 | 0x11 | 302 | 228.85 | 120.89 | 214.85 | variable
p1035 | 0x34 | 89 | 228.38 | 86.35 | 214.38 | variable
p1035 | 0x47 | 110 | 224.52 | 74.74 | 174.52 | variable
p1035 | 0x53 | 20 | 226.05 | 115.75 | 212.05 | variable
p1035 | 0x98 | 1 | 219.00 | 0.00 | 203.00 | stable


### p1041

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1041 | 0x12 | 558 | 231.94 | 146.61 | 217.94 | variable
p1041 | 0x36 | 414 | 228.95 | 62.68 | 214.95 | variable
p1041 | 0x47 | 105 | 224.10 | 63.87 | 174.10 | variable
p1041 | 0x53 | 21 | 224.24 | 217.42 | 210.24 | variable
p1041 | 0x98 | 6 | 232.17 | 106.81 | 216.17 | variable


### p1045

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1045 | 0x12 | 584 | 233.93 | 91.58 | 219.93 | variable
p1045 | 0x34 | 94 | 228.11 | 89.58 | 214.11 | variable
p1045 | 0x47 | 102 | 223.04 | 55.02 | 173.04 | variable
p1045 | 0x53 | 22 | 231.36 | 90.05 | 217.36 | variable


### p1035

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1035 | 0x11 | 302 | 228.85 | 120.89 | 214.85 | base 28B: 5% multiples; mode 222B → ~7.9 units
p1035 | 0x34 | 89 | 228.38 | 86.35 | 214.38 | variable
p1035 | 0x47 | 110 | 224.52 | 74.74 | 174.52 | variable
p1035 | 0x53 | 20 | 226.05 | 115.75 | 212.05 | variable
p1035 | 0x98 | 1 | 219.00 | 0.00 | 203.00 | stable


### p1041

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1041 | 0x12 | 558 | 231.94 | 146.61 | 217.94 | base 28B: 5% multiples; mode 226B → ~8.1 units
p1041 | 0x36 | 414 | 228.95 | 62.68 | 214.95 | variable
p1041 | 0x47 | 105 | 224.10 | 63.87 | 174.10 | variable
p1041 | 0x53 | 21 | 224.24 | 217.42 | 210.24 | variable
p1041 | 0x98 | 6 | 232.17 | 106.81 | 216.17 | variable


### p1045

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1045 | 0x12 | 584 | 233.93 | 91.58 | 219.93 | base 28B: 2% multiples; mode 226B → ~8.1 units
p1045 | 0x34 | 94 | 228.11 | 89.58 | 214.11 | variable
p1045 | 0x47 | 102 | 223.04 | 55.02 | 173.04 | variable
p1045 | 0x53 | 22 | 231.36 | 90.05 | 217.36 | variable


### p1035

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1035 | 0x11 | 302 | 228.85 | 120.89 | 214.85 | base 28B: 5% multiples; mode 222B → ~7.9 units (~16 samples); top remainders [26:56, 2:51, 17:40]
p1035 | 0x34 | 89 | 228.38 | 86.35 | 214.38 | variable
p1035 | 0x47 | 110 | 224.52 | 74.74 | 174.52 | variable
p1035 | 0x53 | 20 | 226.05 | 115.75 | 212.05 | variable
p1035 | 0x98 | 1 | 219.00 | 0.00 | 203.00 | stable

Remainders for 0x11: mod28 [0:15, 1:7, 2:51, 3:4, 4:13, 5:9, 6:7, 7:9, 9:26, 11:6], mod56 [0:15, 2:51, 4:13, 6:7, 23:2, 25:10, 27:17, 29:7, 31:4, 33:9]

Detected co-packed chunks (TAG:count) by primary packet_id:

  - 0x11: 0x11:265, 0x98:82, 0x34:74, 0x47:64, 0x12:52
  - 0x34: 0x11:58, 0x34:44, 0x47:37, 0x12:25, 0x35:11
  - 0x47: 0x34:77, 0x11:75, 0x98:25, 0x47:24, 0x36:17
  - 0x53: 0x11:22, 0x34:6, 0x98:6, 0x47:5, 0x12:2
  - 0x98: 0x34:1, 0x11:1


### p1041

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1041 | 0x12 | 558 | 231.94 | 146.61 | 217.94 | base 28B: 5% multiples; mode 226B → ~8.1 units (~16 samples); top remainders [2:251, 26:87, 5:48]
p1041 | 0x36 | 414 | 228.95 | 62.68 | 214.95 | variable
p1041 | 0x47 | 105 | 224.10 | 63.87 | 174.10 | variable
p1041 | 0x53 | 21 | 224.24 | 217.42 | 210.24 | variable
p1041 | 0x98 | 6 | 232.17 | 106.81 | 216.17 | variable

Remainders for 0x12: mod28 [0:28, 1:14, 2:251, 4:1, 5:48, 6:3, 8:6, 9:23, 12:4, 13:1], mod56 [1:13, 2:251, 5:29, 6:3, 21:3, 24:17, 25:32, 28:28, 29:1, 32:1]

Detected co-packed chunks (TAG:count) by primary packet_id:

  - 0x12: 0x12:646, 0x98:177, 0x36:101, 0x47:92, 0x35:91
  - 0x36: 0x35:241, 0x36:195, 0x12:167, 0x47:91, 0x34:66
  - 0x47: 0x36:88, 0x12:56, 0x11:29, 0x35:27, 0x47:17
  - 0x53: 0x12:27, 0x98:5, 0x36:5, 0x47:2, 0x35:1
  - 0x98: 0x36:11, 0x47:2, 0x98:1


### p1045

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1045 | 0x12 | 584 | 233.93 | 91.58 | 219.93 | base 28B: 2% multiples; mode 226B → ~8.1 units (~16 samples); top remainders [2:266, 26:89, 17:50]
p1045 | 0x34 | 94 | 228.11 | 89.58 | 214.11 | variable
p1045 | 0x47 | 102 | 223.04 | 55.02 | 173.04 | variable
p1045 | 0x53 | 22 | 231.36 | 90.05 | 217.36 | variable

Remainders for 0x12: mod28 [0:10, 1:4, 2:266, 3:3, 4:41, 5:12, 6:11, 7:26, 9:21, 11:5], mod56 [0:10, 2:266, 4:41, 6:11, 23:4, 25:5, 27:13, 29:4, 31:3, 33:12]

Detected co-packed chunks (TAG:count) by primary packet_id:

  - 0x12: 0x12:670, 0x98:255, 0x34:100, 0x47:97, 0x11:91
  - 0x34: 0x12:78, 0x47:42, 0x34:32, 0x98:30, 0x11:13
  - 0x47: 0x12:92, 0x34:70, 0x11:26, 0x47:22, 0x98:20
  - 0x53: 0x12:23, 0x98:6, 0x11:4, 0x34:2, 0x47:1


### p1035

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1035 | 0x11 | 302 | 228.85 | 120.89 | 214.85 | base 28B: 5% multiples; mode 222B → ~7.9 units (~16 samples); top remainders [26:56, 2:51, 17:40]
p1035 | 0x34 | 89 | 228.38 | 86.35 | 214.38 | variable
p1035 | 0x47 | 110 | 224.52 | 74.74 | 174.52 | variable
p1035 | 0x53 | 20 | 226.05 | 115.75 | 212.05 | variable
p1035 | 0x98 | 1 | 219.00 | 0.00 | 203.00 | stable

Remainders for 0x11: mod28 [0:15, 1:7, 2:51, 3:4, 4:13, 5:9, 6:7, 7:9, 9:26, 11:6], mod56 [0:15, 2:51, 4:13, 6:7, 23:2, 25:10, 27:17, 29:7, 31:4, 33:9]

Detected co-packed chunks (TAG:count) by primary packet_id:

  - 0x11: 0x11:265, 0x98:82, 0x34:74, 0x47:64, 0x12:52
  - 0x34: 0x11:58, 0x34:44, 0x47:37, 0x12:25, 0x35:11
  - 0x47: 0x34:77, 0x11:75, 0x98:25, 0x47:24, 0x36:17
  - 0x53: 0x11:22, 0x34:6, 0x98:6, 0x47:5, 0x12:2
  - 0x98: 0x34:1, 0x11:1


EEG summary (dominant payload and samples-per-packet estimate)

preset | EEG id | dominant payload length | multiple-of-28/56 | est samples/packet
:- | :-: | :-: | :-: | :-:
p1035 | 0x11 | 222 | — | 32


### p1041

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1041 | 0x12 | 558 | 231.94 | 146.61 | 217.94 | base 28B: 5% multiples; mode 226B → ~8.1 units (~16 samples); top remainders [2:251, 26:87, 5:48]
p1041 | 0x36 | 414 | 228.95 | 62.68 | 214.95 | variable
p1041 | 0x47 | 105 | 224.10 | 63.87 | 174.10 | variable
p1041 | 0x53 | 21 | 224.24 | 217.42 | 210.24 | variable
p1041 | 0x98 | 6 | 232.17 | 106.81 | 216.17 | variable

Remainders for 0x12: mod28 [0:28, 1:14, 2:251, 4:1, 5:48, 6:3, 8:6, 9:23, 12:4, 13:1], mod56 [1:13, 2:251, 5:29, 6:3, 21:3, 24:17, 25:32, 28:28, 29:1, 32:1]

Detected co-packed chunks (TAG:count) by primary packet_id:

  - 0x12: 0x12:646, 0x98:177, 0x36:101, 0x47:92, 0x35:91
  - 0x36: 0x35:241, 0x36:195, 0x12:167, 0x47:91, 0x34:66
  - 0x47: 0x36:88, 0x12:56, 0x11:29, 0x35:27, 0x47:17
  - 0x53: 0x12:27, 0x98:5, 0x36:5, 0x47:2, 0x35:1
  - 0x98: 0x36:11, 0x47:2, 0x98:1


EEG summary (dominant payload and samples-per-packet estimate)

preset | EEG id | dominant payload length | multiple-of-28/56 | est samples/packet
:- | :-: | :-: | :-: | :-:
p1041 | 0x12 | 226 | — | 16


### p1045

preset | packet_id | count | mean_len | var_len | leftover_mean | notes
:- | :-: | :-: | :-: | :-: | :-: | -
p1045 | 0x12 | 584 | 233.93 | 91.58 | 219.93 | base 28B: 2% multiples; mode 226B → ~8.1 units (~16 samples); top remainders [2:266, 26:89, 17:50]
p1045 | 0x34 | 94 | 228.11 | 89.58 | 214.11 | variable
p1045 | 0x47 | 102 | 223.04 | 55.02 | 173.04 | variable
p1045 | 0x53 | 22 | 231.36 | 90.05 | 217.36 | variable

Remainders for 0x12: mod28 [0:10, 1:4, 2:266, 3:3, 4:41, 5:12, 6:11, 7:26, 9:21, 11:5], mod56 [0:10, 2:266, 4:41, 6:11, 23:4, 25:5, 27:13, 29:4, 31:3, 33:12]

Detected co-packed chunks (TAG:count) by primary packet_id:

  - 0x12: 0x12:670, 0x98:255, 0x34:100, 0x47:97, 0x11:91
  - 0x34: 0x12:78, 0x47:42, 0x34:32, 0x98:30, 0x11:13
  - 0x47: 0x12:92, 0x34:70, 0x11:26, 0x47:22, 0x98:20
  - 0x53: 0x12:23, 0x98:6, 0x11:4, 0x34:2, 0x47:1


EEG summary (dominant payload and samples-per-packet estimate)

preset | EEG id | dominant payload length | multiple-of-28/56 | est samples/packet
:- | :-: | :-: | :-: | :-:
p1045 | 0x12 | 226 | — | 16


## Conclusion

- EEG4 (id 0x11, p1035): dominant payload ≈ 222 B → ~32 samples/packet.
- EEG8 (id 0x12, p1041/p1045): dominant payload ≈ 226 B → ~16 samples/packet.
- Remainder histograms (mod 28/56) show small, consistent offsets, suggesting a stable EEG core with co-packed chunks (optics/IMU/battery) varying per packet.
- Co-packed tags detected (0x34/0x36 optics, 0x47 IMU, 0x98 battery) corroborate that leftover bytes hold additional sensor data alongside EEG.

These observations support a primary+leftovers packet model and provide preliminary samples-per-packet counts to guide EEG/optics decoder design.
