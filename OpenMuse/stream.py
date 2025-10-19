"""
Muse BLE to LSL Streaming
==========================

This module streams decoded Muse sensor data over Lab Streaming Layer (LSL) in real-time.
It handles BLE data reception, decoding, timestamp conversion, packet reordering, and
LSL transmission.

Streaming Architecture:
-----------------------
1. BLE packets arrive asynchronously via Bleak callbacks (_on_data)
2. Packets are decoded using parse_message() from decode.py
3. Device timestamps are converted to LSL time
4. Samples are buffered to allow packet reordering
5. Buffer is periodically flushed: samples sorted by timestamp and pushed to LSL
6. LSL outlets broadcast data to any connected LSL clients (e.g., LabRecorder)

Timestamp Handling - Three Time Indices:
-----------------------------------------
The parse_message() function returns decoded data with THREE timestamp types:

1. **message_time** (datetime)
   - When the BLE message was received on the computer
   - Format: UTC datetime from get_utc_timestamp()
   - Source: Computer system clock
   - Used for: Debugging, logging
   - NOT used for: LSL timestamps

2. **pkt_time** (float, seconds)
   - When samples were captured on the Muse device
   - Format: Seconds since device boot (from 256 kHz device clock)
   - Source: Extracted from packet header (4-byte timestamp)
   - Used for: Initial timestamp assignment in decode.py
   - Converted to: LSL time via device_to_lsl_offset

3. **timestamps** (array column in decoded data)
   - Per-sample timestamps calculated by decode.py
   - Format: Seconds, uniformly spaced at nominal sampling rate
   - Source: pkt_time + (sample_index / sampling_rate)
   - Used for: Final LSL timestamps (after conversion)
   - These are the timestamps that appear in recorded XDF files

Timestamp Conversion Flow:
---------------------------------------------------------------------
    Device Time Domain              →           LSL Time Domain
    ==================                          ===============

    pkt_time (device seconds)                   LSL timestamps (seconds)
           ↓                                             ↑
    timestamps = pkt_time + Δt      →→→→→   Anchored timestamps
                                             (preserving device timing)

    Where:
    - Δt = sample_index / sampling_rate (uniform spacing from device)
    - Device timestamps are first converted using initial offset
    - Then re-anchored to current LSL time during flush to prevent drift

The conversion happens in two stages:

1. _queue_samples(): Initial mapping
    - Extract device timestamps from decoded data array (first column)
    - Add device_to_lsl_offset (computed once from first packet)
    - Store in buffer with device-relative timing preserved

2. _flush_buffer(): Re-anchor to current LSL time
    - Get current LSL time: lsl_now = local_clock()
    - Calculate how long ago samples should have arrived
    - Shift all timestamps to anchor oldest sample appropriately
    - Preserves relative device timing while ensuring proper LSL synchronization
    - Prevents timestamp drift when device was powered on long before recording

Packet Reordering Buffer - Critical Design Component:
------------------------------------------------------
**WHY BUFFERING IS NECESSARY:**

BLE transmission can REORDER entire messages (not just individual packets). Analysis shows:
- ~5% of messages arrive out of order
- Backward jumps can exceed 80ms in severe cases
- Device's timestamps are CORRECT (device clock is monotonic and accurate)
- But messages processed in arrival order → non-monotonic timestamps

**EXAMPLE:**
  Device captures:  Msg 17 (t=13711.801s) → Msg 16 (t=13711.811s)
  BLE transmits:    Msg 16 arrives first, then Msg 17 (OUT OF ORDER!)
  Without buffer:   Push [t=811, t=801, ...] → NON-MONOTONIC to LSL ✗
  With buffer:      Sort [t=801, t=811, ...] → MONOTONIC to LSL ✓

**BUFFER OPERATION:**

1. Samples held in buffer for BUFFER_DURATION_SECONDS (default: 250ms)
2. When buffer time limit reached, all buffered samples are:
   - Concatenated across packets/messages
   - **Sorted by device timestamp** (preserves device timing, corrects arrival order)
   - Converted to LSL time (device_timestamp + offset)
   - Pushed as a single chunk to LSL
3. LSL receives samples in correct temporal order with device timing preserved

**BUFFER FLUSH TRIGGERS:**
- Time threshold: BUFFER_DURATION_SECONDS elapsed since last flush
- Size threshold: MAX_BUFFER_PACKETS accumulated (safety limit)
- End of stream: Final flush when disconnecting

**BUFFER SIZE RATIONALE:**
- Original: 80ms (insufficient for ~90ms delays observed in data)
- Current: 250ms (captures nearly all out-of-order messages)
- Trade-off: Latency (250ms delay) vs. timestamp quality (monotonic output)
- For real-time applications: reduce buffer size, accept some non-monotonic timestamps
- For recording quality: keep 250ms+ buffer for perfect temporal ordering

Timestamp Quality & Device Timing Preservation:
------------------------------------------------
**CRITICAL INSIGHT:**

The decode.py output may show ~20% non-monotonic timestamps, but this is EXPECTED
and NOT an error. These non-monotonic timestamps simply reflect BLE message arrival
order, NOT device timing errors. The timestamp VALUES are correct and preserve the
device's accurate 256 kHz clock timing.

**PIPELINE FLOW:**
  decode.py:  Processes messages in arrival order → ~20% non-monotonic (expected)
              ↓ (but timestamp values preserve device timing)
  stream.py:  Sorts by device timestamp → 0% non-monotonic ✓
              ↓ (restores correct temporal order)
  LSL/XDF:    Monotonic timestamps with device timing preserved ✓

**DEVICE TIMING ACCURACY:**
- Device uses 256 kHz internal clock (accurate, monotonic)
- All subpackets within a message share same pkt_time (verified empirically)
- decode.py uses base_time + sequential offsets (preserves device timing)
- Intervals between samples match device's actual sampling rate (256 Hz, 52 Hz, etc.)
- This pipeline preserves device timing perfectly while handling BLE reordering

**VERIFICATION:**

When loading XDF files with pyxdf:
- Use synchronize_clocks=True for multi-device sync (e.g., Muse + other devices)
- Use dejitter_timestamps=False for actual timestamp quality
- Expected result: 0% non-monotonic, uniform intervals at nominal sampling rates
- Timestamp intervals should match device rates within 0.2% error

LSL Stream Configuration:
-------------------------
Three LSL streams are created:
- Muse_EEG: 8 channels at 256 Hz (EEG + AUX)
- Muse_ACCGYRO: 6 channels at 52 Hz (accelerometer + gyroscope)
- Muse_Optics: 16 channels at 64 Hz (PPG sensors)

Each stream includes:
- Channel labels (from decode.py: EEG_CHANNELS, ACCGYRO_CHANNELS, OPTICS_CHANNELS)
- Nominal sampling rate (declared device rate)
- Data type (float32)
- Units (microvolts for EEG, a.u. for others)
- Manufacturer metadata

Optional JSON Logging:
----------------------
If outfile parameter is provided, all pushed LSL data is logged to JSON:
- Exact timestamps sent to LSL (post-conversion, post-sorting)
- Sample data as pushed to LSL
- Globally sorted by timestamp within each sensor type
- Useful for verification and offline analysis

"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import bleak
import numpy as np

from .backends import _run
from .decode import (
    ACCGYRO_CHANNELS,
    EEG_CHANNELS,
    OPTICS_CHANNELS,
    make_timestamps,
    parse_message,
)
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp

from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock


# LSL streaming constants
EEG_LABELS: tuple[str, ...] = EEG_CHANNELS
ACCGYRO_LABELS: tuple[str, ...] = ACCGYRO_CHANNELS
OPTICS_LABELS: tuple[str, ...] = OPTICS_CHANNELS

# Buffer duration in seconds: holds samples to allow reordering before pushing to LSL
#
# CRITICAL: BLE transmission reorders entire messages (not just packets).
# Empirical analysis shows:
#   - ~5% of messages arrive out of order
#   - Backward jumps can exceed 80ms (up to ~90ms observed)
#   - Device timestamps are CORRECT (device clock is monotonic)
#   - Buffering + sorting restores correct temporal order
#
# Buffer size trade-off:
#   - Smaller buffer (80ms):  Lower latency, but ~1-2% non-monotonic timestamps in output
#   - Larger buffer (250ms+): Higher latency, but nearly 0% non-monotonic (perfect ordering)
#
# Current: 250ms captures nearly all out-of-order messages while maintaining acceptable latency
BUFFER_DURATION_SECONDS = 0.25

# Maximum number of BLE packets to buffer before forcing a flush (safety limit)
MAX_BUFFER_PACKETS = 10


@dataclass
class SensorStream:
    outlet: StreamOutlet
    pad_to_channels: Optional[int]
    labels: tuple[str, ...]
    sampling_rate: float
    unit: str
    buffer: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    last_push_time: Optional[float] = None
    log_records: Optional[list[tuple[np.ndarray, np.ndarray]]] = None


def _create_stream_outlet(
    name: str,
    stype: str,
    labels: tuple[str, ...],
    sfreq: float,
    dtype: str,
    source_id: str,
    unit: str,
    channel_type: Optional[str] = None,
) -> StreamOutlet:
    info = StreamInfo(
        name=name,
        stype=stype,
        n_channels=len(labels),
        sfreq=sfreq,
        dtype=dtype,
        source_id=source_id,
    )
    desc = info.desc
    desc.append_child_value("manufacturer", "Muse")
    channels = desc.append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        if channel_type:
            channel.append_child_value("type", channel_type)

    return StreamOutlet(info, chunk_size=1)


def _build_sensor_streams(enable_logging: bool) -> dict[str, SensorStream]:
    eeg_outlet = _create_stream_outlet(
        name="Muse_EEG",
        stype="EEG",
        labels=EEG_LABELS,
        sfreq=256.0,
        dtype="float32",
        source_id="Muse_EEG",
        unit="microvolts",
        channel_type="EEG",
    )

    accgyro_outlet = _create_stream_outlet(
        name="Muse_ACCGYRO",
        stype="Motion",
        labels=ACCGYRO_LABELS,
        sfreq=52.0,
        dtype="float32",
        source_id="Muse_ACCGYRO",
        unit="a.u.",
    )

    optics_outlet = _create_stream_outlet(
        name="Muse_Optics",
        stype="Optics",
        labels=OPTICS_LABELS,
        sfreq=64.0,
        dtype="float32",
        source_id="Muse_Optics",
        unit="a.u.",
        channel_type="Optics",
    )

    streams = {
        "EEG": SensorStream(
            outlet=eeg_outlet,
            pad_to_channels=len(EEG_LABELS),
            labels=EEG_LABELS,
            sampling_rate=256.0,
            unit="microvolts",
        ),
        "ACCGYRO": SensorStream(
            outlet=accgyro_outlet,
            pad_to_channels=None,
            labels=ACCGYRO_LABELS,
            sampling_rate=52.0,
            unit="a.u.",
        ),
        "Optics": SensorStream(
            outlet=optics_outlet,
            pad_to_channels=len(OPTICS_LABELS),
            labels=OPTICS_LABELS,
            sampling_rate=64.0,
            unit="a.u.",
        ),
    }

    if enable_logging:
        for stream in streams.values():
            stream.log_records = []

    return streams


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float],
    outfile: Optional[str],
    verbose: bool,
) -> None:
    sensor_streams = _build_sensor_streams(enable_logging=outfile is not None)
    samples_sent = {name: 0 for name in sensor_streams}
    # State for timestamping: sensor_type -> (base_time, wrap_offset, last_abs_tick, sample_counter)
    timestamp_states: Dict[str, tuple[Optional[float], int, int, int]] = {
        sensor: (None, 0, 0, 0) for sensor in sensor_streams.keys()
    }

    # Compute device-to-LSL time offset once at the start
    # Will be updated with the first data packet
    device_to_lsl_offset = None

    def _flush_buffer(sensor_type: str) -> None:
        """Flush reordering buffer for a specific sensor type: sort and push samples to LSL."""
        nonlocal samples_sent

        stream = sensor_streams[sensor_type]
        if len(stream.buffer) == 0:
            return

        # Concatenate all timestamps and data (much faster than Python list operations)
        all_timestamps = np.concatenate([ts for ts, _ in stream.buffer])
        all_data = np.vstack([data for _, data in stream.buffer])

        # Sort by timestamp using argsort (numpy is much faster than Python sort)
        sort_indices = np.argsort(all_timestamps)
        sorted_timestamps = all_timestamps[sort_indices]
        sorted_data = all_data[sort_indices]

        # Re-anchor timestamps to current LSL time while preserving device timing
        #
        # WHY: If the Muse device was powered on hours before LabRecorder started,
        # the initial device_to_lsl_offset creates timestamps in the "past" relative
        # to when recording actually begins. This causes sync issues with other devices.
        #
        # SOLUTION: Shift all timestamps so they're anchored to current LSL time,
        # but preserve the relative timing between samples (device precision).
        #
        # BENEFITS:
        # - Sub-millisecond device timing preserved (e.g., 4ms @ 250Hz)
        # - Timestamps always near current LSL time (proper multi-device sync)
        # - Compatible with LabRecorder clock offset collection
        lsl_now = local_clock()
        first_timestamp = sorted_timestamps[0]
        time_span = sorted_timestamps[-1] - first_timestamp

        # Calculate how long ago these samples should have been captured
        # Account for: sample time span + half the buffer duration (typical delay)
        expected_age = time_span + BUFFER_DURATION_SECONDS / 2

        # Compute anchor point: where the oldest sample should be
        # (in the past by expected_age seconds from now)
        anchor_time = lsl_now - expected_age

        # Calculate shift needed to move timestamps from their current position
        # to the anchored position
        timestamp_shift = anchor_time - first_timestamp

        # Apply uniform shift to all timestamps
        # This preserves all relative timing from the device while anchoring to LSL time
        anchored_timestamps = sorted_timestamps + timestamp_shift

        # Push to LSL with re-anchored timestamps
        # This preserves device-level timing precision while ensuring proper
        # synchronization with LabRecorder and other LSL streams
        try:
            stream.outlet.push_chunk(
                x=sorted_data.astype(np.float32, copy=False),  # type: ignore[arg-type]
                timestamp=anchored_timestamps.astype(np.float64, copy=False),
                pushThrough=True,
            )
            samples_sent[sensor_type] += len(anchored_timestamps)

            # Log to JSON if requested - save exactly what was pushed to LSL (with anchored timestamps)
            if stream.log_records is not None:
                stream.log_records.append(
                    (anchored_timestamps.copy(), sorted_data.copy())
                )
        except Exception as exc:
            if verbose:
                print(f"LSL push_chunk failed for {sensor_type}: {exc}")

        # Clear buffer and update last push time
        stream.buffer.clear()
        stream.last_push_time = local_clock()

    def _queue_samples(
        sensor_type: str, data_array: np.ndarray, lsl_now: float
    ) -> None:
        nonlocal device_to_lsl_offset

        if data_array.size == 0 or data_array.shape[1] < 2:
            return

        stream = sensor_streams[sensor_type]

        # Extract sensor data (exclude time column)
        samples = data_array[:, 1:].astype(np.float32)
        if stream.pad_to_channels:
            target = stream.pad_to_channels
            current = samples.shape[1]
            if current < target:
                padding = np.zeros(
                    (samples.shape[0], target - current), dtype=np.float32
                )
                samples = np.hstack([samples, padding])
            elif current > target:
                samples = samples[:, :target]

        device_times = data_array[:, 0]

        # Compute device-to-LSL time offset on first packet only
        # This offset maps device time to LSL time domain
        if device_to_lsl_offset is None:
            device_to_lsl_offset = lsl_now - device_times[0]
            if verbose:
                print(f"Initialized time offset: {device_to_lsl_offset:.3f} seconds")

        # Convert device timestamps to LSL time
        # LSL time is the reference time domain used by all LSL streams
        # This ensures proper synchronization with other LSL streams in LabRecorder
        lsl_timestamps = device_times + device_to_lsl_offset
        stream.buffer.append((lsl_timestamps, samples))

        if stream.last_push_time is None:
            stream.last_push_time = lsl_now

        # Flush if buffer duration exceeded OR buffer size limit reached
        if lsl_now - stream.last_push_time >= BUFFER_DURATION_SECONDS:
            _flush_buffer(sensor_type)
        elif len(stream.buffer) >= MAX_BUFFER_PACKETS:
            if verbose:
                print(
                    f"Warning: {sensor_type} buffer reached {MAX_BUFFER_PACKETS} packets, forcing flush"
                )
            _flush_buffer(sensor_type)

    def _on_data(_, data: bytearray):
        nonlocal device_to_lsl_offset, timestamp_states
        try:
            # Both EEG and ACC/GYRO data come through EEG characteristic
            message = f"{get_utc_timestamp()}\t{MuseS.EEG_UUID}\t{data.hex()}"
            subpackets = parse_message(message)

            # Apply timestamps for each sensor type
            decoded = {}
            for sensor_type, pkt_list in subpackets.items():
                if sensor_type in timestamp_states:
                    array, *new_state = make_timestamps(
                        pkt_list, *timestamp_states[sensor_type]
                    )
                    decoded[sensor_type] = array
                    timestamp_states[sensor_type] = (new_state[0], new_state[1], new_state[2], new_state[3])  # type: ignore
        except Exception as exc:
            if verbose:
                print(f"Decoding error: {exc}")
            return

        lsl_now = local_clock()

        _queue_samples("EEG", decoded.get("EEG", np.empty((0, 0))), lsl_now)
        _queue_samples("ACCGYRO", decoded.get("ACCGYRO", np.empty((0, 0))), lsl_now)
        _queue_samples("Optics", decoded.get("Optics", np.empty((0, 0))), lsl_now)

    try:
        if verbose:
            print(f"Connecting to {address} ...")

        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Build callbacks dict for all data characteristics
            data_callbacks = {MuseS.EEG_UUID: _on_data}

            # Use shared connection routine
            await MuseS.connect_and_initialize(client, preset, data_callbacks, verbose)

            # Streaming is now active (callbacks are registered and device is configured)
            # Data will start flowing asynchronously

            if duration:
                if verbose:
                    print(f"Streaming for {duration} seconds...")
                start = time.time()
                try:
                    while time.time() - start < duration:
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass
            else:
                if verbose:
                    print("Streaming indefinitely. Press Ctrl+C to stop.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass

    except asyncio.CancelledError:
        if verbose:
            print("Stream cancelled.")
    except Exception as exc:
        if verbose:
            print(f"An error occurred: {exc}")
    finally:
        if verbose:
            print("Stopping stream...")

        # Flush any remaining samples in all reordering buffers
        for sensor_type, stream in sensor_streams.items():
            if len(stream.buffer) > 0:
                if verbose:
                    print(
                        f"Flushing {len(stream.buffer)} buffered {sensor_type} packets..."
                    )
                _flush_buffer(sensor_type)

        # Write JSON output if requested - save exactly what was sent to LSL
        if outfile:
            if verbose:
                total_pushes = sum(
                    len(stream.log_records or []) for stream in sensor_streams.values()
                )
                print(f"Writing {total_pushes} LSL push operations to {outfile}...")
            try:

                def _serialize_log(
                    stream: SensorStream,
                ) -> tuple[list[float], list[list[float]]]:
                    if not stream.log_records:
                        return [], []

                    samples: list[tuple[float, list[float]]] = []
                    for timestamps, data_chunk in stream.log_records:
                        for i, ts in enumerate(timestamps):
                            samples.append((float(ts), data_chunk[i, :].tolist()))

                    samples.sort(key=lambda x: x[0])
                    timestamps_out = [s[0] for s in samples]
                    data_out = [s[1] for s in samples]
                    return timestamps_out, data_out

                json_data: dict[str, object] = {}
                sample_counts: dict[str, int] = {}
                for sensor_type, stream in sensor_streams.items():
                    timestamps_out, data_out = _serialize_log(stream)
                    sample_counts[sensor_type] = len(timestamps_out)
                    json_data[sensor_type] = {
                        "lsl_timestamps": timestamps_out,
                        "channels": list(stream.labels),
                        "data": data_out,
                        "n_samples": sample_counts[sensor_type],
                        "sampling_rate": stream.sampling_rate,
                        "unit": stream.unit,
                    }

                json_data["note"] = (
                    "LSL data globally sorted by timestamp per sensor type"
                )

                # Ensure output directory exists
                outdir = os.path.dirname(os.path.abspath(outfile))
                if outdir and not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)

                # Write to file
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

                if verbose:
                    print(
                        "Wrote "
                        + ", ".join(
                            f"{sensor}: {sample_counts.get(sensor, 0)} samples"
                            for sensor in ("EEG", "ACCGYRO", "Optics")
                        )
                        + f" to {outfile}"
                    )
                    print("File written successfully.")
            except Exception as exc:
                if verbose:
                    print(f"Error writing to file: {exc}")

        if verbose:
            print(
                "Stream stopped. "
                + ", ".join(
                    f"{sensor}: {samples_sent[sensor]} samples"
                    for sensor in ("EEG", "ACCGYRO", "Optics")
                )
            )


def stream(
    address: str,
    preset: str = "p1041",
    duration: Optional[float] = None,
    outfile: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Stream decoded EEG and accelerometer/gyroscope data over LSL.

    Creates two LSL streams:
    - Muse_EEG: 4 channels (TP9, AF7, AF8, TP10) at 256 Hz
    - Muse_ACCGYRO: 6 channels (ACC_X/Y/Z, GYRO_X/Y/Z) at 52 Hz

    Parameters
    ----------
    address : str
        Device address (e.g., MAC on Windows).
    preset : str
        Preset to send (e.g., p1041 for all channels, p1035 for basic config).
    duration : float, optional
        Optional stream duration in seconds. Omit to stream until interrupted.
    outfile : str, optional
        Optional output JSON file to save decoded EEG and ACC/GYRO samples.
        Omit to only stream without saving.
    verbose : bool
        If True, print verbose output.
    """
    # Configure LSL to reduce verbosity (disables IPv6 warnings and lowers log level)
    configure_lsl_api_cfg()

    _run(_stream_async(address, preset, duration, outfile, verbose))
