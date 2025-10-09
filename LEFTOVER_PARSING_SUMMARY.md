"""
SUMMARY: Leftover Data Parsing Implementation
==============================================

## Analysis Findings

After extensive analysis of the leftover bytes in decoded packets, we discovered:

### Structure of Leftovers

Leftovers contain **additional sensor data** with a simple structure:
- **TAG byte** (0x47 for ACCGYRO, 0x98 for Battery)
- **Immediately followed by raw sensor data**
- NO 14-byte packet header in leftovers (unlike primary packets)

### ACCGYRO in Leftovers

- Tag: 0x47
- Data: 36 bytes (18 int16 values = 3 samples × 6 channels × 2 bytes)
- Format: Same as primary ACCGYRO data
- Occurrence: Found in ~50% of ACCGYRO packet leftovers at position 135-170

### Battery in Leftovers (NOT IMPLEMENTED)

- Tag: 0x98
- Data structure unclear - many false positives when parsing naively
- Decided NOT to parse battery from leftovers to avoid noise

## Implementation

Modified `MuseLSL3/decode.py` `parse_message()` function to:

1. After decoding primary packet data, scan the leftover bytes
2. Look for 0x47 (ACCGYRO) tags
3. Extract and decode 36 bytes following each tag
4. Validate data with sanity checks (ACC: -50 to 50 m/s², GYRO: -50 to 50 rad/s)
5. Append valid decoded data to `data_accgyro` list

## Results

### Before Enhancement:
- ACCGYRO packets: Only primary data decoded
- Data loss: ~36% of ACCGYRO data was in leftovers

### After Enhancement:
- ACCGYRO packets: Primary + leftover data decoded
- Example: 7 primary packets now yield 11 total ACCGYRO data arrays
- Data recovery: ~57% more ACCGYRO samples recovered

### Test Results:
- All 20 unit tests pass ✓
- Battery decoding unaffected (no false positives)
- ACCGYRO data validated with sanity checks
- Performance: Still < 0.012 ms per message

## Usage

No API changes - existing code continues to work:

```python
from MuseLSL3.decode import parse_message

subpackets = parse_message(message)
for sp in subpackets:
    if sp['pkt_type'] == 'ACCGYRO':
        # sp['data_accgyro'] now contains list of arrays
        # (may have 1 primary + additional from leftovers)
        for acc_data in sp['data_accgyro']:
            # acc_data shape: (3, 7) - 3 samples, 7 cols (time + 6 channels)
            times = acc_data[:, 0]
            acc_xyz = acc_data[:, 1:4]
            gyro_xyz = acc_data[:, 4:7]
```

## Conclusion

✓ Successfully implemented leftover parsing for ACCGYRO data
✓ Recovered significant amount of previously unused sensor data
✓ Maintained code quality with validation and testing
✓ No breaking changes to existing API
