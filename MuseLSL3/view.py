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
        stream.connect()
    except Exception as exc:
        print(f"Error: Could not connect to LSL stream '{stream_name}': {exc}")
        print("\nMake sure the stream is running in another terminal:")
        print(f"  MuseLSL3 stream --address <MAC> --preset <PRESET>")
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

    # Data buffer
    max_samples = int(window_size * sfreq)
    data_buffer = np.zeros((n_channels, max_samples))
    time_buffer = np.arange(-max_samples, 0) / sfreq

    # Setup the plot
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8), sharex=True)
    if n_channels == 1:
        axes = [axes]

    fig.suptitle(f"Real-time ACC/GYRO Data - {stream_name}", fontsize=14)

    lines = []
    rate_texts = []  # Text annotations for sampling rates
    for i, (ax, ch_name) in enumerate(zip(axes, ch_names)):
        (line,) = ax.plot(time_buffer, data_buffer[i, :], lw=1)
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
    samples_received = 0

    # Track ALL timestamps for final statistics (never pruned)
    all_timestamps = []

    # Track recent timestamps for real-time rate display
    recent_timestamps = []
    rate_window = 1.0  # Calculate rate over last 1 second

    def update(frame):
        nonlocal samples_received, recent_timestamps, all_timestamps

        # Pull new data from the stream
        try:
            data, timestamps = stream.get_data()
        except Exception:
            return lines + rate_texts

        if data.shape[1] == 0:
            return lines + rate_texts

        samples_received += data.shape[1]

        # Update timestamp tracking
        if len(timestamps) > 0:
            ts_list = timestamps.tolist()

            # Keep ALL timestamps for final statistics
            all_timestamps.extend(ts_list)

            # Keep recent timestamps for real-time rate display
            recent_timestamps.extend(ts_list)

            # Remove timestamps older than rate_window for real-time display
            if len(recent_timestamps) > 0:
                cutoff_time = recent_timestamps[-1] - rate_window
                recent_timestamps = [t for t in recent_timestamps if t >= cutoff_time]

        # Update buffer with new data
        n_new = data.shape[1]
        if n_new >= max_samples:
            # If more data than buffer, just take the last max_samples
            data_buffer[:] = data[:, -max_samples:]
        else:
            # Shift old data and append new
            data_buffer[:, :-n_new] = data_buffer[:, n_new:]
            data_buffer[:, -n_new:] = data

        # Calculate current sampling rate
        if len(recent_timestamps) > 1:
            time_span = recent_timestamps[-1] - recent_timestamps[0]
            if time_span > 0:
                current_rate = (len(recent_timestamps) - 1) / time_span
            else:
                current_rate = 0.0
        else:
            current_rate = 0.0

        # Update plots and rate displays
        for i, (line, rate_text) in enumerate(zip(lines, rate_texts)):
            line.set_ydata(data_buffer[i, :])
            # Auto-scale y-axis
            y_min, y_max = data_buffer[i, :].min(), data_buffer[i, :].max()
            y_range = y_max - y_min
            if y_range > 0:
                axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

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
            print(f"  Samples received: {samples_received}")

            # Calculate average rate from all collected LSL timestamps
            if len(all_timestamps) > 1:
                lsl_duration = all_timestamps[-1] - all_timestamps[0]
                if lsl_duration > 0:
                    avg_rate = (len(all_timestamps) - 1) / lsl_duration
                    print(f"  Average rate: {avg_rate:.1f} Hz (from LSL timestamps)")
                    print(f"  Data time span: {lsl_duration:.1f} seconds")
                else:
                    print(f"  Average rate: N/A (insufficient data)")
            else:
                print(f"  Average rate: N/A (no timestamps received)")
