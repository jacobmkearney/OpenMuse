import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import bleak
import numpy as np

from .backends import _run
from .decode import ACCGYRO_CHANNELS, EEG_CHANNELS, OPTICS_CHANNELS, parse_message
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp

from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock


# LSL streaming constants
EEG_LABELS: tuple[str, ...] = EEG_CHANNELS
ACCGYRO_LABELS: tuple[str, ...] = ACCGYRO_CHANNELS
OPTICS_LABELS: tuple[str, ...] = OPTICS_CHANNELS

# Buffer duration in seconds: hold samples to allow reordering
# (BLE transmission can be delayed up to ~40ms)
BUFFER_DURATION_SECONDS = 0.08

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

        # Push to LSL - ensure correct dtype (float32 for data, float64 for timestamps)
        try:
            stream.outlet.push_chunk(
                x=sorted_data.astype(np.float32, copy=False),
                timestamp=sorted_timestamps.astype(np.float64, copy=False),
                pushThrough=True,
            )
            samples_sent[sensor_type] += len(sorted_timestamps)

            # Log to JSON if requested - save exactly what was pushed to LSL
            if stream.log_records is not None:
                stream.log_records.append(
                    (sorted_timestamps.copy(), sorted_data.copy())
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
        if device_to_lsl_offset is None:
            device_to_lsl_offset = lsl_now - device_times[0]
            if verbose:
                print(f"Initialized time offset: {device_to_lsl_offset:.3f} seconds")

        # Store as numpy arrays (no conversion to list needed)
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
        nonlocal device_to_lsl_offset
        try:
            # Both EEG and ACC/GYRO data come through EEG characteristic
            message = f"{get_utc_timestamp()}\t{MuseS.EEG_UUID}\t{data.hex()}"
            decoded = parse_message(message)
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
