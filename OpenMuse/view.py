"""Real-time visualization of ACC/GYRO data from LSL stream."""

import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def view(
    stream_name: str = "MuseAccGyro",
    duration: Optional[float] = None,
    window_size: float = 10.0,
    update_interval: float = 0.05,
    verbose: bool = True,
) -> None:
    """
    Visualize ACC/GYRO data from an LSL stream in real-time.

    Parameters:
    - stream_name: Name of the LSL stream to connect to (default: "MuseAccGyro")
    - duration: Optional viewing duration in seconds. Omit to view until window closed.
    - window_size: Time window to display in seconds (default: 10.0)
    - update_interval: Update interval in seconds (default: 0.05 = 20 Hz)
    - verbose: Print progress messages
    """
    from mne_lsl.stream import StreamLSL

    if verbose:
        print(f"Looking for LSL stream '{stream_name}'...")

    # Connect to the LSL stream
    try:
        stream = StreamLSL(bufsize=window_size, name=stream_name)
        stream.connect(timeout=5.0)  # 5 second timeout for better error messages
    except Exception as exc:
        print(f"Error: Could not connect to LSL stream '{stream_name}': {exc}")
        print("\nMake sure the stream is running in another terminal:")
        print(f"  OpenMuse stream --address <MAC> --preset <PRESET>")
        return

    if verbose:
        print(f"Connected to stream: {stream_name}")
        print(f"  Channels: {stream.info.ch_names}")
        print(f"  Sampling rate: {stream.info['sfreq']} Hz")
        print(f"  Window size: {window_size} seconds")
        print("\nStarting visualization... Close the window to stop.")

    # Channel names
    ch_names = stream.info.ch_names
    n_channels = len(ch_names)
    sfreq = stream.info["sfreq"]

    # Setup the plot
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8), sharex=True)
    if n_channels == 1:
        axes = [axes]

    fig.suptitle(f"Real-time ACC/GYRO Data - {stream_name}", fontsize=14)

    lines = []
    rate_texts = []  # Text annotations for sampling rates
    for i, (ax, ch_name) in enumerate(zip(axes, ch_names)):
        # Initialize with empty data - will be updated with actual timestamps
        # Use '-' line style explicitly and no markers
        (line,) = ax.plot([], [], "-", lw=1, marker="")
        lines.append(line)
        ax.set_ylabel(ch_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-window_size, 0)

        # Add text annotation for sampling rate in top-right corner
        rate_text = ax.text(
            0.98,
            0.95,
            "",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        rate_texts.append(rate_text)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    plt.tight_layout()

    start_time = time.time()
    first_timestamp = None
    last_timestamp = None
    total_samples = 0  # Track unique samples based on timestamp range

    def update(frame):
        nonlocal first_timestamp, last_timestamp, total_samples

        # Get the last window_size seconds of data directly from the stream buffer
        # This is the cleanest approach - let MNE-LSL handle the buffering
        try:
            # Get data from the stream's internal buffer
            data, timestamps = stream.get_data(winsize=window_size)
        except Exception:
            return lines + rate_texts

        if data.shape[1] == 0:
            return lines + rate_texts

        # Track first and last timestamps for final statistics
        if len(timestamps) > 0:
            if first_timestamp is None:
                first_timestamp = timestamps[0]
            last_timestamp = timestamps[-1]

            # Calculate total unique samples from timestamp range and sampling rate
            if first_timestamp is not None and last_timestamp is not None:
                time_span = last_timestamp - first_timestamp
                if sfreq > 0:
                    total_samples = int(time_span * sfreq) + 1
                else:
                    # Fallback: count unique timestamps seen so far
                    total_samples = len(timestamps)

        # Calculate instantaneous sampling rate from the window data
        # Use last ~1 second of data (or all available data if less)
        current_rate = 0.0
        if len(timestamps) >= 2:
            # For rate calculation, use recent data (last 1 second or 52 samples)
            rate_window_size = min(1.0, window_size)  # Use 1 second or less
            rate_window_samples = int(sfreq * rate_window_size) if sfreq > 0 else 52
            recent_start_idx = max(0, len(timestamps) - rate_window_samples)
            recent_timestamps = timestamps[recent_start_idx:]

            if len(recent_timestamps) >= 2:
                time_span = recent_timestamps[-1] - recent_timestamps[0]
                num_intervals = len(recent_timestamps) - 1
                if time_span > 0:
                    current_rate = num_intervals / time_span

        # Update plots and rate displays
        for i, (line, rate_text) in enumerate(zip(lines, rate_texts)):
            if len(timestamps) > 0:
                # Convert to relative times (most recent sample = 0)
                latest_time = timestamps[-1]
                relative_times = timestamps - latest_time

                # Get this channel's data
                channel_data = data[i, :]

                # Plot the data
                line.set_data(relative_times, channel_data)

                # Auto-scale y-axis
                if len(channel_data) > 0:
                    y_min, y_max = channel_data.min(), channel_data.max()
                    y_range = y_max - y_min
                    if y_range > 0:
                        axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                    else:
                        axes[i].set_ylim(y_min - 0.1, y_max + 0.1)
            else:
                # No data, clear the line
                line.set_data([], [])

            # Update rate text
            rate_text.set_text(f"{current_rate:.1f} Hz")

        # Check duration limit
        if duration is not None and (time.time() - start_time) >= duration:
            plt.close(fig)

        return lines + rate_texts

    # Start animation
    anim = FuncAnimation(
        fig,
        update,
        interval=int(update_interval * 1000),
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stream.disconnect()
        if verbose:
            elapsed = time.time() - start_time
            print(f"\nVisualization stopped.")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Total samples: {total_samples}")

            # Calculate mean effective sampling rate from first to last timestamp
            if first_timestamp is not None and last_timestamp is not None:
                time_span = last_timestamp - first_timestamp
                if time_span > 0 and total_samples > 0:
                    avg_rate = (total_samples - 1) / time_span
                    print(f"  Mean effective rate: {avg_rate:.1f} Hz")
                    print(f"  Data time span: {time_span:.1f} seconds")
                else:
                    print(f"  Average rate: N/A (insufficient data)")
            else:
                print(f"  Average rate: N/A (no timestamps received)")
