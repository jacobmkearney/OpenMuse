import io
import urllib

import numpy as np
import pandas as pd
import requests
import pyxdf
import warnings
import neurokit2 as nk


# Contains .xdf files recorded with LabRecorder, containing Muse data streams (recorded using the stream function, preset 1041) and a Bitalino stream with ECG and Photosensor data.
# Was recorded during a face presentation experiment.
filename = "./faces1.xdf"
filename = "./test-06.xdf"
upsample = 2.0
fillmissing = None

streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=False,
)


# Get smaller time stamp to later use as offset (zero point)
min_ts = min([min(s["time_stamps"]) for s in streams if len(s["time_stamps"]) > 0])

# Get range of timestamps for each stream
for i, stream in enumerate(streams):
    name = stream["info"].get("name", ["Unnamed"])[0]
    if len(stream["time_stamps"]) == 0:
        print(f"{i} - Stream {name}: empty")
        continue
    ts_min = stream["time_stamps"].min()
    ts_max = stream["time_stamps"].max()
    duration = ts_max - ts_min
    n_samples = len(stream["time_stamps"])
    nominal_srate = float(stream["info"]["nominal_srate"][0])
    effective_srate = float(stream["info"]["effective_srate"])
    print(
        f"{i} - Stream {name}: {n_samples} samples, duration {duration:.2f} s (from {ts_min:.2f} to {ts_max:.2f}), nominal srate {nominal_srate:.2f} Hz, effective srate {effective_srate:.2f} Hz"
    )


# ========================================================================================
# Test the proximity of eletronic markers vs. Photosensor ones
# ========================================================================================
stream = streams[3]
jspsych_events = np.array([float(val[0]) for val in stream["time_series"]])
jspsych_onsets = jspsych_events == 1
jspsych_ts = np.array(stream["time_stamps"])
jspsych_ts_onset = jspsych_ts[jspsych_onsets]
jspsych_duration = jspsych_ts[jspsych_events == 0] - jspsych_ts_onset[:-1]


stream = streams[4]
# LUX is 4th column
# stream["info"]["desc"][0]["channels"][0]["channel"]
lux_signal = stream["time_series"][:, 3]
# nk.signal_plot(lux_signal)
lux_events = nk.events_find(
    lux_signal, threshold_keep="below", duration_min=1500, duration_max=2500
)
lux_ts = stream["time_stamps"]
lux_ts_onset = lux_ts[lux_events["onset"].astype(int)]

# lot a vertical lines from 0 to 0.5 and from 0.5 to 1 for each jspsych and lux onset
plt.figure()
for i in range(20):
    plt.vlines(
        (jspsych_ts_onset[i] - min_ts), 0.5, 1, color="red", label="JSpsych onsets"
    )
    plt.vlines(
        (lux_ts_onset[i] - min_ts), 0, 1, color="blue", label="Photosensor onsets"
    )

delays = jspsych_ts_onset[0:-1] - lux_ts_onset[1::]
plt.hist(delays, bins=50)

# ========================================================================================
# Test GYRO sync
# ========================================================================================
# LUX is 4th column
# streams[4]["info"]["desc"][0]["channels"][0]["channel"]
lux_signal = streams[4]["time_series"][:, 3]
# nk.signal_plot(lux_signal)
lux_events = nk.events_find(
    lux_signal, threshold_keep="below", duration_min=1500, duration_max=2500
)
lux_ts = streams[4]["time_stamps"]
lux_ts_onset = lux_ts[lux_events["onset"].astype(int)]

# GYRO
accgyro = pd.DataFrame(
    streams[1]["time_series"],
    columns=[
        ch["label"][0] for ch in streams[1]["info"]["desc"][0]["channels"][0]["channel"]
    ],
)
accgyro.index = streams[1]["time_stamps"]

accgyro.iloc[:3000].plot(y=["GYRO_X"], subplots=True)
for ts in lux_ts_onset[0:11]:
    plt.axvline(x=ts, color="red", linestyle="--")

# ========================================================================================
# PRECISION-PRESERVING APPROACH: Work with numpy arrays and float timestamps
# Only convert to pandas at the very end
# ========================================================================================

# Store stream data with full float precision
stream_data = []

for stream in streams:
    if len(stream["time_stamps"]) == 0:
        continue

    # Get columns names
    try:
        channels_info = stream["info"]["desc"][0]["channels"][0]["channel"]
        cols = [channels_info[i]["label"][0] for i in range(len(channels_info))]
    except (KeyError, TypeError) as e:
        warnings.warn(f"Missing channel names in stream metadata: {e}")
        cols = [f"ch{i}" for i in range(stream["time_series"].shape[1])]

    # Store as numpy arrays with full precision
    timestamps = stream["time_stamps"] - min_ts  # Float64 precision maintained
    data = stream["time_series"]  # Already numpy array

    stream_data.append(
        {
            "timestamps": timestamps,
            "data": data,
            "columns": cols,
            "nominal_srate": float(stream["info"]["nominal_srate"][0]),
            "effective_srate": float(stream["info"]["effective_srate"]),
        }
    )

# Compute target sampling rate (for later resampling)
info = {
    "sampling_rate": int(np.max([s["nominal_srate"] for s in stream_data]) * upsample)
}

# ========================================================================================
# Synchronize with full precision
# ========================================================================================

# Collect all unique timestamps from all streams (preserving full float64 precision)
all_timestamps = np.concatenate([s["timestamps"] for s in stream_data])
unique_timestamps = np.unique(all_timestamps)
print(f"Total unique timestamps across all streams: {len(unique_timestamps)}")

# Check for duplicates in individual streams (at full precision)
for i, stream in enumerate(stream_data):
    unique_in_stream, inverse_indices = np.unique(
        stream["timestamps"], return_inverse=True
    )
    n_duplicates = len(stream["timestamps"]) - len(unique_in_stream)
    if n_duplicates > 0:
        print(
            f"Stream {i} has {n_duplicates} duplicate timestamps at float64 precision"
        )
        # Average duplicates
        # inverse_indices maps each original timestamp to its unique timestamp index
        n_unique = len(unique_in_stream)
        n_channels = stream["data"].shape[1]
        averaged_data = np.zeros((n_unique, n_channels))
        counts = np.bincount(inverse_indices)

        # Fully vectorized accumulation using np.add.at
        np.add.at(averaged_data, inverse_indices, stream["data"])

        # Divide by counts to get averages
        averaged_data /= counts[:, np.newaxis]

        stream["timestamps"] = unique_in_stream
        stream["data"] = averaged_data
        print(f"  -> Averaged to {len(unique_in_stream)} unique timestamps")

# Rebuild unique timestamps after deduplication
all_timestamps = np.concatenate([s["timestamps"] for s in stream_data])
unique_timestamps = np.unique(all_timestamps)
print(f"After deduplication: {len(unique_timestamps)} unique timestamps")

# Create synchronized data matrix
# Each row corresponds to a unique timestamp, columns are all channels
n_samples = len(unique_timestamps)
n_channels = sum(len(s["columns"]) for s in stream_data)

# Initialize with NaN
synchronized_data = np.full((n_samples, n_channels), np.nan)
all_columns = []

# Fill in data from each stream
col_offset = 0
for stream in stream_data:
    all_columns.extend(stream["columns"])
    n_stream_channels = len(stream["columns"])

    # Find indices where this stream's timestamps exist in the global timeline
    indices = np.searchsorted(unique_timestamps, stream["timestamps"])

    # Fill in the data
    synchronized_data[indices, col_offset : col_offset + n_stream_channels] = stream[
        "data"
    ]
    col_offset += n_stream_channels


# ========================================================================================
# Resample and Interpolate with full precision
# ========================================================================================

# Create evenly-spaced timestamp array with float precision
dt = 1.0 / info["sampling_rate"]  # Time step in seconds
n_resampled = int((unique_timestamps[-1] - unique_timestamps[0]) / dt) + 1
resampled_timestamps = np.linspace(
    unique_timestamps[0], unique_timestamps[-1], n_resampled
)

print(f"\nResampling to {info['sampling_rate']} Hz")
print(f"Resampled data will have {n_resampled} samples")

# Interpolate each channel independently
resampled_data = np.full((n_resampled, n_channels), np.nan)

for ch_idx in range(n_channels):
    channel_data = synchronized_data[:, ch_idx]

    # Get non-NaN values for interpolation
    valid_mask = ~np.isnan(channel_data)

    if valid_mask.sum() > 1:  # Need at least 2 points to interpolate
        valid_timestamps = unique_timestamps[valid_mask]
        valid_data = channel_data[valid_mask]

        # Interpolate with full precision
        if fillmissing is not None:
            # Limit interpolation gap
            max_gap_samples = int(info["sampling_rate"] * fillmissing)
            max_gap_time = max_gap_samples * dt

            # Only interpolate where gaps are within limit
            for i, ts in enumerate(resampled_timestamps):
                # Find nearest valid data points
                if ts >= valid_timestamps[0] and ts <= valid_timestamps[-1]:
                    idx_right = np.searchsorted(valid_timestamps, ts)
                    if idx_right == 0:
                        idx_right = 1
                    idx_left = idx_right - 1

                    gap = valid_timestamps[idx_right] - valid_timestamps[idx_left]
                    if gap <= max_gap_time:
                        # Linear interpolation
                        t_left = valid_timestamps[idx_left]
                        t_right = valid_timestamps[idx_right]
                        v_left = valid_data[idx_left]
                        v_right = valid_data[idx_right]

                        alpha = (ts - t_left) / (t_right - t_left)
                        resampled_data[i, ch_idx] = v_left + alpha * (v_right - v_left)
        else:
            # Interpolate all gaps
            resampled_data[:, ch_idx] = np.interp(
                resampled_timestamps,
                valid_timestamps,
                valid_data,
                left=np.nan,
                right=np.nan,
            )

print(f"Interpolation complete")


# ========================================================================================
# Convert to pandas DataFrame at the very end (optional, using float timestamps)
# ========================================================================================

# Option 1: Keep timestamps as float index (maximum precision)
df = pd.DataFrame(resampled_data, columns=all_columns, index=resampled_timestamps)
df.index.name = "time"  # in seconds

# Option 2: Convert to datetime if needed for compatibility
# df = pd.DataFrame(resampled_data, columns=all_columns)
# df.index = pd.to_datetime(resampled_timestamps, unit='s')

# =======================================================================================
# Visual test
# =======================================================================================

max(df.index) / 60

# df.plot(subplots=True)

# # Visualize chunk of EEG (where index > 10000 & < 10001)
# df.loc[1000:5020, ["EEG_AF7", "EEG_AF8"]].plot(subplots=True)
# df.loc[0:18008, ["EEG_AF7", "EEG_AF8"]].plot(subplots=True)
