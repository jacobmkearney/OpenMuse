"""OpenMuse: Minimal utilities for Muse EEG devices."""

from .find import find_devices
from .record import record
from .muse import MuseS
from .decode import parse_message, decode_rawdata

__version__ = "0.1.0"


def stream(*args, **kwargs):
    """Lazy import for stream to avoid requiring mne-lsl for non-streaming use."""
    from .stream import stream as _stream

    return _stream(*args, **kwargs)


def view(*args, **kwargs):
    """Lazy import for view to avoid requiring mne-lsl and matplotlib for non-viewing use."""
    from .view import view as _view

    return _view(*args, **kwargs)
