# Changelog

## [Unreleased]

### Added

### Changed
- **Global Timestamping**: Replaced per-message timestamping with global subpacket sorting and timestamping in `decode_rawdata()` to account for cross-message timing inversions.
- **Output Change**: The `parse_message()` function now always returns raw subpackets (Dict[str, List[Dict]]) for flexible processing. Users should call `make_timestamps()` explicitly on the subpackets to get numpy arrays.

### Deprecated

### Removed

### Fixed

### Security

---

## [0.1.0] - 2025-10-16

Initial release.
