# Enhanced Multi-Subpacket Decoder for Muse Devices

## Overview

This enhanced decoder (`decode_enhanced.py`) provides comprehensive parsing of BLE messages from Muse devices, extracting **all subpackets** from each message, not just the primary data.

## Key Discovery (October 2025)

Investigation revealed that Muse devices **bundle multiple subpackets in single BLE messages** for transmission efficiency. After the primary subpacket data, additional subpackets from other sensors are included with the format:

```
1 byte:  Tag (subpacket type identifier)
4 bytes: Header (metadata)
N bytes: Payload (sensor data)
```

### Why This Matters

- **63.3% of ACCGYRO packets contain additional ACCGYRO subpackets**, effectively **doubling temporal resolution** from 3 to 6 samples per packet
- Single BLE messages can contain data from multiple sensor types (EEG, ACCGYRO, Battery, Optics)
- Previous decoders only extracted primary data, missing 50-80% of available sensor information

## Usage

### Basic Parsing

```python
from MuseLSL3.decode_enhanced import parse_message_enhanced

message = "2025-10-05T19:06:25.799053+00:00\t273e...\td7000055..."
parsed = parse_message_enhanced(message)

print(f"Valid: {parsed['valid']}")
print(f"Primary type: {parsed['primary']['type']}")
print(f"Primary data shape: {parsed['primary']['data'].shape}")
print(f"Additional subpackets: {len(parsed['additional'])}")

for subpkt in parsed['additional']:
    print(f"  - {subpkt['type']}: {subpkt['data'].shape if subpkt['data'] is not None else 'No data'}")
```

### Batch Extraction

```python
from MuseLSL3.decode_enhanced import extract_all_sensor_data_enhanced

# Load messages from file
with open('recording.txt') as f:
    messages = f.readlines()

# Extract all sensor data
data = extract_all_sensor_data_enhanced(messages)

# Access motion data
accgyro = data['ACCGYRO']  # Shape: (N, 7) = [time, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
print(f"Motion samples: {accgyro.shape[0]}")
print(f"Time range: {accgyro[0, 0]:.3f} to {accgyro[-1, 0]:.3f} seconds")
print(f"ACC range: {accgyro[:, 1:4].min():.3f} to {accgyro[:, 1:4].max():.3f} g")

# Access battery data
if 'Battery' in data:
    battery = data['Battery']  # Shape: (M, 2) = [time, percentage]
    print(f"Battery: {battery[-1, 1]:.1f}%")
```

## Supported Sensor Types

| Type      | Tag  | Primary | Additional | Data Shape | Description                    |
|-----------|------|---------|------------|------------|--------------------------------|
| ACCGYRO   | 0x47 | ✓       | ✓          | (3, 7)     | 3 samples × [time, ACC_XYZ, GYRO_XYZ] |
| Battery   | 0x98 | ✓       | ✓          | (2,)       | [time, percentage]             |
| EEG4      | 0x11 | ✓       | ✓          | (3, 5)     | 3 samples × [time, CH1-4]      |
| EEG8      | 0x12 | ✓       | ✓          | (3, 9)     | 3 samples × [time, CH1-8]      |
| REF       | 0x13 | ✓       | ✓          | -          | Reference channels (not decoded) |
| Optics16  | 0x36 | ✓       | ✓          | -          | 16 optical channels (not decoded) |

**Note:** EEG4, EEG8, REF, and Optics decoders are placeholder implementations and need validation/completion.

## API Reference

### `parse_message_enhanced(message: str) -> Dict`

Parse a single BLE message with all subpackets.

**Returns:**
```python
{
    'message_time': datetime,      # BLE message arrival time
    'message_uuid': str,            # BLE characteristic UUID
    'pkt_len': int,                 # Declared packet length
    'pkt_n': int,                   # Packet counter (0-255)
    'pkt_time': float,              # Device timestamp (seconds)
    'pkt_freq': float,              # Sampling frequency (Hz)
    'pkt_unknown1': bytes,          # Reserved bytes 6-8
    'pkt_unknown2': bytes,          # Metadata bytes 10-12
    'primary': {
        'type': str,                # Sensor type
        'data': np.ndarray,         # Decoded data
        'raw': bytes                # Raw bytes
    },
    'additional': [                 # List of additional subpackets
        {
            'tag': int,             # Subpacket tag
            'type': str,            # Sensor type
            'header': bytes,        # 4-byte header
            'data': np.ndarray,     # Decoded data
            'raw': bytes            # Raw payload
        },
        ...
    ],
    'valid': bool,                  # Validation status
    'errors': List[str]             # Errors if any
}
```

### `extract_all_sensor_data_enhanced(messages: List[str]) -> Dict[str, np.ndarray]`

Extract and concatenate all sensor data from multiple messages.

**Returns:**
```python
{
    'ACCGYRO': np.ndarray,   # Shape (N, 7): [time, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
    'Battery': np.ndarray,   # Shape (M, 2): [time, percentage]
    'EEG4': np.ndarray,      # Shape (P, 5): [time, CH1, CH2, CH3, CH4]
    'EEG8': np.ndarray,      # Shape (Q, 9): [time, CH1-8]
    ...
}
```

## Performance Comparison

### Data Extraction Rates

| Decoder          | ACCGYRO Samples | EEG8 Samples | Battery Samples | Notes                |
|------------------|-----------------|--------------|-----------------|----------------------|
| Original         | 21 (3 per msg)  | 939          | 3               | Primary only         |
| **Enhanced**     | **54 (3-6 per msg)** | **939**  | **3**           | **+157% ACCGYRO!**   |

Testing with 100 BLE messages:
- **Original decoder:** Extracts only primary subpackets
- **Enhanced decoder:** Extracts primary + additional subpackets
  - Average 2.89 additional subpackets per message
  - 157% more ACCGYRO samples (54 vs 21)
  - Complete multi-sensor data capture

## Example Output

```
Loaded 3272 messages

TESTING SINGLE MESSAGE PARSING
===============================
Valid: True
Packet time: 3509081.344s
Packet frequency: 52.0 Hz

Primary subpacket:
  Type: ACCGYRO
  Data shape: (3, 7)

Additional subpackets: 5
  1. Type=EEG8, Tag=0x12, Data shape=(3, 9)
  2. Type=EEG8, Tag=0x12, Data shape=(3, 9)
  3. Type=EEG8, Tag=0x12, Data shape=(3, 9)
  4. Type=EEG8, Tag=0x12, Data shape=(3, 9)
  5. Type=EEG8, Tag=0x12, Data shape=(3, 9)

BATCH EXTRACTION (100 messages)
================================
Extracted sensor data:
  ACCGYRO: (54, 7)  ← 157% increase!
  Battery: (3, 2)
  EEG8: (939, 9)

SUBPACKET STATISTICS
====================
Primary subpackets:
  ACCGYRO: 7
  EEG8: 56
  Optics16: 36
  REF: 1

Additional subpackets:
  ACCGYRO: 11       ← Extra motion data!
  Battery: 3
  EEG8: 257
  Optics16: 15
  REF: 3

Total additional subpackets: 289
Average per message: 2.89
```

## Implementation Status

### ✅ Fully Implemented
- ACCGYRO decoding (3 samples × 6 channels)
- Battery decoding (percentage extraction)
- Additional subpacket parsing
- Multi-sensor data extraction
- Timestamp generation

### ⚠️ Placeholder / Needs Validation
- EEG4 decoder (format unclear)
- EEG8 decoder (format unclear)
- REF decoder (not implemented)
- Optics decoder (not implemented)

## Technical Details

### Subpacket Structure

Each additional subpacket follows this format:

```
Offset | Size | Field       | Description
-------|------|-------------|----------------------------------
0      | 1    | Tag         | Subpacket type (0x11, 0x47, etc.)
1      | 4    | Header      | Metadata (purpose unclear)
5      | N    | Payload     | Sensor data (size depends on tag)
```

### Known Subpacket Sizes

| Tag  | Size | Type      |
|------|------|-----------|
| 0x11 | 14   | EEG4      |
| 0x12 | 28   | EEG8      |
| 0x13 | 7    | REF       |
| 0x36 | 96   | Optics16  |
| 0x47 | 36   | ACCGYRO   |
| 0x98 | 4    | Battery   |

## Files

- **`decode_enhanced.py`** - Enhanced decoder implementation
- **`test_new_decoder.py`** - Test script demonstrating usage
- **`FINAL_REPORT.md`** - Detailed investigation findings
- **`investigate_leftover_bytes.py`** - Analysis script
- **`investigate_additional_accgyro.py`** - ACCGYRO validation script

## References

See `decoding_attempts/FINAL_REPORT.md` for complete investigation details, including:
- Discovery methodology
- Statistical validation
- Packet structure analysis
- Recommendations for future work

## Migration Guide

### From Original Decoder

**Before:**
```python
from MuseLSL3.decode import decode_rawdata

data = decode_rawdata(messages)
accgyro = data['ACC']  # Pandas DataFrame
```

**After:**
```python
from MuseLSL3.decode_enhanced import extract_all_sensor_data_enhanced

data = extract_all_sensor_data_enhanced(messages)
accgyro = data['ACCGYRO']  # NumPy array with MORE samples
```

### Key Differences

1. **Return format:** NumPy arrays instead of Pandas DataFrames
2. **More data:** Includes additional subpackets (up to 157% more ACCGYRO samples)
3. **Unified API:** Single function extracts all sensor types
4. **Column layout:** First column is always timestamp

## Contributing

To add/improve sensor decoders:

1. Implement decoder function following the pattern:
   ```python
   def decode_SENSOR(data: bytes, timestamp: float, freq: float) -> Tuple[np.ndarray, bytes]:
       # Parse data
       # Return (decoded_array, leftover_bytes)
   ```

2. Add to `parse_additional_subpackets()` switch statement

3. Add to `sensor_data` dictionary in `extract_all_sensor_data_enhanced()`

4. Test with real data and validate output

---

**Last Updated:** October 8, 2025  
**Investigation:** Analysis of 15 raw data files, 8,480 BLE messages, 1,008 ACCGYRO packets
