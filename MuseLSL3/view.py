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
    samples_received = 0

    # Track ALL timestamps for final statistics
    all_timestamps = []

    # Track recent timestamps for rate calculation (keep last 52 samples)
    recent_timestamps = []
    max_timestamps_for_rate = 52

    def update(frame):
        nonlocal samples_received, recent_timestamps, all_timestamps

        # Get the last window_size seconds of data directly from the stream buffer
        # This is the cleanest approach - let MNE-LSL handle the buffering
        try:
            # Get data from the stream's internal buffer
            data, timestamps = stream.get_data(winsize=window_size)
        except Exception:
            return lines + rate_texts

        if data.shape[1] == 0:
            return lines + rate_texts

        samples_received += data.shape[1]

        # Update timestamp tracking for rate calculation
        if len(timestamps) > 0:
            # For final statistics, keep all unique timestamps we've seen
            all_timestamps.extend(timestamps.tolist())

            # For rate calculation, keep only the most recent timestamps
            recent_timestamps.extend(timestamps.tolist())
            if len(recent_timestamps) > max_timestamps_for_rate:
                recent_timestamps = recent_timestamps[-max_timestamps_for_rate:]

        # Calculate mean effective sampling rate from last 52 samples
        if len(recent_timestamps) >= 2:
            time_span = recent_timestamps[-1] - recent_timestamps[0]
            num_intervals = len(recent_timestamps) - 1
            if time_span > 0:
                current_rate = num_intervals / time_span
            else:
                current_rate = 0.0
        else:
            current_rate = 0.0

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
            print(f"  Samples received: {samples_received}")

            # Calculate mean effective sampling rate (includes gaps)
            if len(all_timestamps) >= 2:
                # Mean rate = total samples / total time span
                time_span = all_timestamps[-1] - all_timestamps[0]
                num_samples = len(all_timestamps)
                if time_span > 0:
                    avg_rate = (num_samples - 1) / time_span
                    print(
                        f"  Mean effective rate: {avg_rate:.1f} Hz (includes gaps between packets)"
                    )
                    print(f"  Data time span: {time_span:.1f} seconds")

                    # Also show instantaneous rate (median of differences, ignores gaps)
                    all_diffs = [
                        all_timestamps[i + 1] - all_timestamps[i]
                        for i in range(len(all_timestamps) - 1)
                    ]
                    all_diffs_sorted = sorted(all_diffs)
                    median_diff = all_diffs_sorted[len(all_diffs_sorted) // 2]
                    if median_diff > 0:
                        instant_rate = 1.0 / median_diff
                        print(
                            f"  Instantaneous rate: {instant_rate:.1f} Hz (median, ignores gaps)"
                        )
                else:
                    print(f"  Average rate: N/A (zero time span)")
            else:
                print(f"  Average rate: N/A (no timestamps received)")
