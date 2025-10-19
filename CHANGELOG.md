# Changelog

## [0.1.2]

### Added

### Changed
- **Stream-Relative Timestamps**: Timestamps are now generated relative to stream start (base_time = 0.0) instead of device boot time, eliminating the need for complex re-anchoring in LSL streaming while maintaining device timing precision.
- **Conditional Re-Anchoring**: LSL streaming now only applies timestamp re-anchoring for edge cases (timestamps >30s in past), providing better synchronization with other LSL streams.
- **Global Timestamping**: Replaced per-message timestamping with global subpacket sorting and timestamping in `decode_rawdata()` to account for cross-message timing inversions.
- **Output Change**: The `parse_message()` function now always returns raw subpackets (Dict[str, List[Dict]]) for flexible processing. Users should call `make_timestamps()` explicitly on the subpackets to get numpy arrays.

### Deprecated

### Removed

### Fixed

### Security

---

## [0.1.0] - 2025-10-16

Initial release.
