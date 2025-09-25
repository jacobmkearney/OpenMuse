"""MuseLSL3: Minimal utilities for Muse EEG devices."""

from .find import find_devices
from .record import record
from .decode import decode_rawdata

__version__ = "0.1.0"
