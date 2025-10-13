"""Real-time visualization using GLOO (OpenGL) for maximum performance with many channels."""

import time
from typing import Optional

import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho
from vispy.visuals import TextVisual

# Vertex shader - processes each vertex position
VERT_SHADER = """
#version 120

// Vertex attributes
attribute float a_position;      // Y value of the data point
attribute vec3 a_index;           // (channel_index, sample_index, unused)
attribute vec3 a_color;           // RGB color for this channel

// Uniforms (constants for all vertices)
uniform vec2 u_scale;             // (x_scale, y_scale) for zooming
uniform vec2 u_size;              // (n_channels, 1)
uniform float u_n_samples;        // Number of samples per channel
uniform mat4 u_projection;        // Projection matrix

// Output to fragment shader
varying vec4 v_color;

void main() {
    // Calculate normalized coordinates
    float channel_idx = a_index.x;
    float sample_idx = a_index.y;
    
    // X position: Leave space on left for channel names (start at 0.08 instead of 0)
    float x_margin = 0.08;  // 8% of width reserved for labels
    float x = x_margin + (1.0 - x_margin) * (sample_idx / u_n_samples);
    
    // Y position: stack channels vertically
    // Each channel gets 1/n_channels of vertical space
    float y_offset = channel_idx / u_size.x;
    float y_scale = 1.0 / u_size.x;  // Height allocated to each channel
    
    // Center the signal within its allocated space
    // a_position is normalized to [-1, 1], scale by 0.35 to use more space
    float y = y_offset + y_scale * 0.5 + (a_position * y_scale * 0.35);
    
    // Apply projection
    gl_Position = u_projection * vec4(x * u_scale.x, y, 0.0, 1.0);
    
    // Pass color to fragment shader
    v_color = vec4(a_color, 1.0);
}
"""

# Fragment shader - determines pixel colors
FRAG_SHADER = """
#version 120

varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""


class RealtimeViewer:
    """High-performance real-time signal viewer using GLOO."""

    def __init__(
        self,
        streams,
        window_size=10.0,
        update_interval=0.005,
        duration=None,
        verbose=True,
    ):
        self.streams = streams
        self.window_size = window_size
        self.update_interval = update_interval
        self.duration = duration
        self.verbose = verbose
        self.start_time = time.time()

        # Collect channel info from all streams
        self.channel_info = []  # List of (stream_idx, ch_idx, ch_name, color, y_range)
        self.total_channels = 0

        # Color schemes
        eeg_colors = [
            (0 / 255, 206 / 255, 209 / 255),  # Cyan
            (65 / 255, 105 / 255, 225 / 255),  # Royal Blue
            (30 / 255, 144 / 255, 255 / 255),  # Dodger Blue
            (0 / 255, 191 / 255, 255 / 255),  # Deep Sky Blue
        ]
        acc_color = (141 / 255, 182 / 255, 0 / 255)  # Apple green
        gyro_color = (152 / 255, 255 / 255, 152 / 255)  # Mint green
        optics_colors = [
            (255 / 255, 165 / 255, 0 / 255),  # Orange
            (255 / 255, 99 / 255, 71 / 255),  # Tomato
            (220 / 255, 20 / 255, 60 / 255),  # Crimson
            (255 / 255, 69 / 255, 0 / 255),  # Red-Orange
        ]

        for stream_idx, stream in enumerate(streams):
            stream_name = stream.name
            is_eeg = "EEG" in stream_name.upper()
            is_optics = "OPTICS" in stream_name.upper()

            for ch_idx, ch_name in enumerate(stream.info.ch_names):
                # Determine color and y-range (for proper scaling)
                if is_eeg:
                    color = eeg_colors[ch_idx % len(eeg_colors)]
                    y_range = 1000.0  # 0 to 1000 raw EEG units
                    y_min, y_max = 0.0, 1000.0
                    y_ticks = [0, 500, 1000]
                elif is_optics:
                    color = optics_colors[ch_idx % len(optics_colors)]
                    y_range = 1.0  # Normalized
                    y_min, y_max = -1.0, 1.0
                    y_ticks = [-1, 0, 1]
                elif "ACC" in ch_name.upper():
                    color = acc_color
                    y_range = 1.0  # ±1g
                    y_min, y_max = -1.0, 1.0
                    y_ticks = [-1, 0, 1]
                else:  # GYRO
                    color = gyro_color
                    y_range = 245.0  # ±245 deg/s
                    y_min, y_max = -245.0, 245.0
                    y_ticks = [-245, 0, 245]

                self.channel_info.append(
                    {
                        "stream_idx": stream_idx,
                        "ch_idx": ch_idx,
                        "name": ch_name,
                        "color": color,
                        "y_range": y_range,
                        "y_min": y_min,
                        "y_max": y_max,
                        "y_ticks": y_ticks,
                    }
                )
                self.total_channels += 1

        # Determine number of samples to display
        # Use the highest sampling rate to ensure smooth display
        max_sfreq = max(stream.info["sfreq"] for stream in streams)
        self.n_samples = int(max_sfreq * window_size)

        # Initialize data buffer (n_samples x n_channels)
        self.data = np.zeros((self.n_samples, self.total_channels), dtype=np.float32)

        # Create window title
        if len(streams) == 1:
            window_title = f"OpenMuse - {streams[0].name}"
        else:
            window_title = f"OpenMuse - {self.total_channels} channels"

        # Create canvas
        self.canvas = app.Canvas(
            title=window_title,
            keys="interactive",
            size=(1400, 900),
            position=(100, 100),
        )

        # Create GLOO program for signals
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Build vertex data (needs self.program to exist)
        self._create_vertex_data()

        # Create grid lines for better readability
        self._create_grid_lines()

        # Set uniforms
        self.program["u_scale"] = (1.0, 1.0)
        self.program["u_size"] = (self.total_channels, 1.0)
        self.program["u_n_samples"] = float(self.n_samples)
        self.program["u_projection"] = ortho(0, 1, 0, 1, -1, 1)

        # Create text labels for channel names (y-axis, left side)
        self.channel_labels = []
        for ch_idx, ch_info in enumerate(self.channel_info):
            text = TextVisual(
                ch_info["name"],
                pos=(10, 0),  # Will be positioned in on_draw
                color="white",
                font_size=10,
                anchor_x="left",
                anchor_y="center",
                bold=True,
            )
            text.transforms.configure(
                canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
            )
            self.channel_labels.append(text)

        # Create EEG standard deviation labels (impedance indicator)
        self.eeg_std_labels = []
        for ch_idx, ch_info in enumerate(self.channel_info):
            # Only create std labels for EEG channels
            stream_name = streams[ch_info["stream_idx"]].name
            if "EEG" in stream_name.upper():
                text = TextVisual(
                    "σ: ---",
                    pos=(0, 0),  # Will be positioned in on_draw
                    color="yellow",
                    font_size=8,
                    anchor_x="right",
                    anchor_y="center",
                )
                text.transforms.configure(
                    canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
                )
                self.eeg_std_labels.append((ch_idx, text))

        # Buffer for calculating EEG std (last 1 second of data)
        self.eeg_std_buffer = {}
        for ch_idx, ch_info in enumerate(self.channel_info):
            stream_name = streams[ch_info["stream_idx"]].name
            if "EEG" in stream_name.upper():
                sfreq = streams[ch_info["stream_idx"]].info["sfreq"]
                buffer_size = int(sfreq * 1.0)  # 1 second of data
                self.eeg_std_buffer[ch_idx] = np.zeros(buffer_size)

        # Create y-axis tick labels for each channel
        self.y_tick_labels = []
        for ch_idx, ch_info in enumerate(self.channel_info):
            tick_texts = []
            for tick_val in ch_info["y_ticks"]:
                text = TextVisual(
                    str(int(tick_val)),
                    pos=(0, 0),  # Will be positioned in on_draw
                    color="gray",
                    font_size=7,
                    anchor_x="right",
                    anchor_y="center",
                )
                text.transforms.configure(
                    canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
                )
                tick_texts.append((tick_val, text))
            self.y_tick_labels.append(tick_texts)

        # Create text labels for time axis (x-axis)
        self.time_labels = []
        n_time_ticks = int(self.window_size) + 1  # One per second
        for i in range(n_time_ticks):
            time_val = -self.window_size + i  # From -window_size to 0
            text = TextVisual(
                f"{time_val:.0f}s",
                pos=(0, 0),  # Will be positioned in on_draw
                color="white",
                font_size=9,
                anchor_x="center",
                anchor_y="top",
            )
            text.transforms.configure(
                canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
            )
            self.time_labels.append((time_val, text))

        # Connect events
        self.canvas.events.draw.connect(self.on_draw)
        self.canvas.events.resize.connect(self.on_resize)
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)

        # Setup timer for updates
        self.timer = app.Timer(self.update_interval, connect=self.on_timer, start=True)

        # Stats tracking
        self.stream_stats = [
            {"first_timestamp": None, "last_timestamp": None, "total_samples": 0}
            for _ in streams
        ]

        if verbose:
            print(f"\nGLOO-based visualization initialized:")
            print(f"  Total channels: {self.total_channels}")
            print(f"  Samples per channel: {self.n_samples}")
            print(f"  Window size: {window_size}s")
            print(f"  Update rate: {1/update_interval:.0f} Hz")
            print("\nPress '+' or '-' to zoom time axis")
            print("Scroll mouse wheel to zoom amplitude")
            print("Close window to stop.\n")

    def _create_vertex_data(self):
        """Create vertex arrays for all channels and samples."""
        # Create index arrays
        # a_index: (channel_idx, sample_idx, 0) for each vertex
        channel_indices = np.repeat(np.arange(self.total_channels), self.n_samples)
        sample_indices = np.tile(np.arange(self.n_samples), self.total_channels)

        a_index = np.c_[
            channel_indices, sample_indices, np.zeros(len(channel_indices))
        ].astype(np.float32)

        # Create color array (each channel has its own color, repeated for all samples)
        colors = np.array([ch["color"] for ch in self.channel_info], dtype=np.float32)
        a_color = np.repeat(colors, self.n_samples, axis=0)

        # Set attributes
        self.program["a_index"] = a_index
        self.program["a_color"] = a_color
        # a_position will be updated each frame with actual data
        self.program["a_position"] = self.data.T.ravel().astype(np.float32)

        # Create index buffers for each channel to draw them separately
        self.index_buffers = []
        for ch_idx in range(self.total_channels):
            start = ch_idx * self.n_samples
            end = start + self.n_samples
            indices = np.arange(start, end, dtype=np.uint32)
            index_buffer = gloo.IndexBuffer(indices)
            self.index_buffers.append(index_buffer)

    def _create_grid_lines(self):
        """Create grid lines for horizontal separators and y-axis zero lines."""
        # Simple shader for drawing lines
        grid_vert = """
        attribute vec2 a_position;
        uniform mat4 u_projection;
        void main() {
            gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
        }
        """
        grid_frag = """
        uniform vec4 u_color;
        void main() {
            gl_FragColor = u_color;
        }
        """

        self.grid_program = gloo.Program(grid_vert, grid_frag)
        self.grid_program["u_projection"] = ortho(0, 1, 0, 1, -1, 1)

        # Create horizontal grid lines at y-limits for each channel
        # These correspond to where the min/max ticks are shown
        # Signal uses 35% (0.35) of channel height on each side of center
        y_limit_lines = []
        for ch_idx in range(self.total_channels):
            y_offset = ch_idx / self.total_channels
            channel_height = 1.0 / self.total_channels
            y_center = y_offset + 0.5 * channel_height

            # Lines at ±35% from center (matching the 0.35 scale in shader)
            y_top = y_center + 0.35 * channel_height  # Upper y-limit
            y_bottom = y_center - 0.35 * channel_height  # Lower y-limit

            y_limit_lines.extend([[0.08, y_top], [1.0, y_top]])
            y_limit_lines.extend([[0.08, y_bottom], [1.0, y_bottom]])

        # Add zero lines for each channel (drawn separately with thicker line)
        self.zero_lines = []
        for ch_idx in range(self.total_channels):
            y_offset = ch_idx / self.total_channels
            y_center = y_offset + 0.5 / self.total_channels
            self.zero_lines.extend([[0.08, y_center], [1.0, y_center]])

        self.y_limit_lines = np.array(y_limit_lines, dtype=np.float32)
        self.zero_lines = np.array(self.zero_lines, dtype=np.float32)

    def on_draw(self, event):
        """Render the scene."""
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0))

        # Draw y-limit lines (horizontal lines at min/max of each channel)
        if len(self.y_limit_lines) > 0:
            gloo.set_line_width(1.0)
            self.grid_program["a_position"] = self.y_limit_lines
            self.grid_program["u_color"] = (0.25, 0.25, 0.25, 1.0)  # Dark gray
            self.grid_program.draw("lines")

        # Draw zero lines (center of each channel) - thicker and lighter
        gloo.set_line_width(2.0)
        self.grid_program["a_position"] = self.zero_lines
        self.grid_program["u_color"] = (0.35, 0.35, 0.35, 1.0)  # Lighter gray, thicker
        self.grid_program.draw("lines")

        # Reset line width for signal drawing
        gloo.set_line_width(1.0)

        # Draw each channel separately to avoid connecting them
        for index_buffer in self.index_buffers:
            self.program.draw("line_strip", indices=index_buffer)

        # Draw channel labels and y-ticks
        width, height = self.canvas.size
        for ch_idx, (text_visual, ch_info) in enumerate(
            zip(self.channel_labels, self.channel_info)
        ):
            # Calculate channel's vertical bounds
            y_offset = ch_idx / self.total_channels
            channel_height = height / self.total_channels
            y_bottom = height - (y_offset * height)
            y_top = y_bottom - channel_height
            y_center = (y_bottom + y_top) / 2

            # Draw channel name on the left (moved right to make room for ticks)
            text_visual.pos = (75, y_center)
            text_visual.draw()

            # Draw y-tick labels for this channel
            # Position ticks to align with grid lines (at ±35% from center)
            y_range = ch_info["y_max"] - ch_info["y_min"]
            for idx, (tick_val, tick_text) in enumerate(self.y_tick_labels[ch_idx]):
                # Normalize tick value to [-1, 1] (same as shader normalization)
                normalized_signal = 2.0 * (tick_val - ch_info["y_min"]) / y_range - 1.0

                # Apply the same 0.35 scale and positioning as in shader
                # This ensures ticks align exactly with grid lines
                tick_y_normalized = (
                    normalized_signal * 0.35
                )  # Scale by 0.35 like in shader
                tick_y = y_center - (
                    tick_y_normalized * channel_height
                )  # Convert to pixels

                tick_text.pos = (40, tick_y)
                tick_text.draw()

            # Draw EEG standard deviation (impedance indicator) on the right side
            for eeg_ch_idx, std_text in self.eeg_std_labels:
                if eeg_ch_idx == ch_idx:
                    std_text.pos = (width - 10, y_center)
                    std_text.draw()
                    break

        # Draw time labels (x-axis)
        x_margin = 0.08  # Same as in shader
        signal_width = width * (1.0 - x_margin)
        x_start = width * x_margin

        for time_val, text_visual in self.time_labels:
            # Calculate x position (time_val is negative, from -window_size to 0)
            # Map to signal area only (after the left margin)
            x_fraction = (time_val + self.window_size) / self.window_size
            x_pos = x_start + (x_fraction * signal_width)
            text_visual.pos = (x_pos, height - 10)
            text_visual.draw()

    def on_resize(self, event):
        """Handle window resize."""
        gloo.set_viewport(0, 0, *event.size)

        # Update text transforms
        for text in self.channel_labels:
            text.transforms.configure(canvas=self.canvas, viewport=(0, 0, *event.size))
        for _, text in self.eeg_std_labels:
            text.transforms.configure(canvas=self.canvas, viewport=(0, 0, *event.size))
        for ch_ticks in self.y_tick_labels:
            for _, text in ch_ticks:
                text.transforms.configure(
                    canvas=self.canvas, viewport=(0, 0, *event.size)
                )
        for _, text in self.time_labels:
            text.transforms.configure(canvas=self.canvas, viewport=(0, 0, *event.size))

    def on_key_press(self, event):
        """Handle keyboard input for zooming."""
        if event.key.name in ["+", "-", "="]:
            # Time axis zoom
            dx = -0.1 if event.key.name in ["+", "="] else 0.1
            scale_x, scale_y = self.program["u_scale"]
            new_scale_x = max(0.5, min(5.0, scale_x * np.exp(dx)))
            self.program["u_scale"] = (new_scale_x, scale_y)
            self.canvas.update()

    def on_mouse_wheel(self, event):
        """Handle mouse wheel for amplitude zoom."""
        dy = np.sign(event.delta[1]) * 0.1
        scale_x, scale_y = self.program["u_scale"]
        new_scale_y = max(0.5, min(5.0, scale_y * np.exp(dy)))
        self.program["u_scale"] = (scale_x, new_scale_y)
        self.canvas.update()

    def on_timer(self, event):
        """Update data from streams."""
        # Update each channel
        for ch_info in self.channel_info:
            stream_idx = ch_info["stream_idx"]
            ch_idx = ch_info["ch_idx"]
            stream = self.streams[stream_idx]
            stats = self.stream_stats[stream_idx]

            # Get new data from stream
            try:
                data, timestamps = stream.get_data(winsize=self.window_size)
            except Exception:
                continue

            if data.shape[1] == 0:
                continue

            # Update stats
            if len(timestamps) > 0:
                if stats["first_timestamp"] is None:
                    stats["first_timestamp"] = timestamps[0]
                stats["last_timestamp"] = timestamps[-1]

                time_span = stats["last_timestamp"] - stats["first_timestamp"]
                sfreq = stream.info["sfreq"]
                if sfreq > 0:
                    stats["total_samples"] = int(time_span * sfreq) + 1
                else:
                    stats["total_samples"] = len(timestamps)

            # Get this channel's data
            channel_data = data[ch_idx, :]

            # Update EEG std buffer and calculate std for impedance indicator
            channel_global_idx = self.channel_info.index(ch_info)
            if channel_global_idx in self.eeg_std_buffer:
                # Update rolling buffer with most recent data
                buffer_size = len(self.eeg_std_buffer[channel_global_idx])
                if len(channel_data) > 0:
                    # Take last samples to fill buffer
                    samples_to_use = min(buffer_size, len(channel_data))
                    self.eeg_std_buffer[channel_global_idx] = np.roll(
                        self.eeg_std_buffer[channel_global_idx], -samples_to_use
                    )
                    self.eeg_std_buffer[channel_global_idx][-samples_to_use:] = (
                        channel_data[-samples_to_use:]
                    )

                    # Calculate std and update label with color-coded impedance
                    std_val = np.std(self.eeg_std_buffer[channel_global_idx])
                    for eeg_ch_idx, std_text in self.eeg_std_labels:
                        if eeg_ch_idx == channel_global_idx:
                            std_text.text = f"σ: {std_val:.1f}"
                            # Color-code based on impedance quality
                            if std_val < 50:
                                std_text.color = "green"  # Good impedance
                            elif std_val > 100:
                                std_text.color = "red"  # Poor impedance
                            else:
                                std_text.color = "yellow"  # Acceptable impedance
                            break

            # Normalize to [-1, 1] range for consistent display
            # Map [y_min, y_max] to [-1, 1]
            y_min = ch_info["y_min"]
            y_max = ch_info["y_max"]
            y_range = y_max - y_min
            normalized_data = 2.0 * (channel_data - y_min) / y_range - 1.0
            normalized_data = np.clip(normalized_data, -1.0, 1.0)

            # Resample to fit display buffer if needed
            if len(normalized_data) != self.n_samples:
                # Simple interpolation
                x_old = np.linspace(0, 1, len(normalized_data))
                x_new = np.linspace(0, 1, self.n_samples)
                normalized_data = np.interp(x_new, x_old, normalized_data)

            # Update data buffer for this channel
            channel_global_idx = self.channel_info.index(ch_info)
            self.data[:, channel_global_idx] = normalized_data

        # Update vertex positions
        self.program["a_position"].set_data(self.data.T.ravel().astype(np.float32))
        self.canvas.update()

        # Check duration limit
        if (
            self.duration is not None
            and (time.time() - self.start_time) >= self.duration
        ):
            self.canvas.close()

    def show(self):
        """Show the canvas and start the event loop."""
        self.canvas.show()

        @self.canvas.connect
        def on_close(event):
            self.timer.stop()
            for stream in self.streams:
                stream.disconnect()

            if self.verbose:
                elapsed = time.time() - self.start_time
                print(f"\nVisualization stopped.")
                print(f"  Duration: {elapsed:.1f} seconds")
                for stream_idx, stream in enumerate(self.streams):
                    stats = self.stream_stats[stream_idx]
                    print(f"  {stream.name}: {stats['total_samples']} samples")

                    if (
                        stats["first_timestamp"] is not None
                        and stats["last_timestamp"] is not None
                    ):
                        time_span = stats["last_timestamp"] - stats["first_timestamp"]
                        if time_span > 0 and stats["total_samples"] > 0:
                            avg_rate = (stats["total_samples"] - 1) / time_span
                            print(
                                f"    Mean rate: {avg_rate:.1f} Hz (expected: {stream.info['sfreq']} Hz)"
                            )

        try:
            app.run()
        except KeyboardInterrupt:
            pass


def view(
    stream_name: Optional[str] = None,
    duration: Optional[float] = None,
    window_size: float = 10.0,
    update_interval: float = 0.005,
    verbose: bool = True,
) -> None:
    """
    Visualize EEG and/or ACC/GYRO data from LSL streams in real-time using GLOO (OpenGL).

    This uses direct OpenGL rendering via GLOO for maximum performance with many channels.
    Optimized for 20+ channels (EEG + Motion + Optics).

    By default, displays ALL available Muse streams (EEG + ACCGYRO + Optics if available).

    Supported streams:
    - Muse_EEG: 4 EEG channels (TP9, AF7, AF8, TP10) at 256 Hz
    - Muse_ACCGYRO: 6 motion channels (ACC_X/Y/Z, GYRO_X/Y/Z) at 52 Hz
    - Muse_Optics: Up to 16 optical channels at 64 Hz

    Parameters:
    - stream_name: Name of specific LSL stream to connect to (default: None = show all available)
    - duration: Optional viewing duration in seconds. Omit to view until window closed.
    - window_size: Time window to display in seconds (default: 10.0)
    - update_interval: Update interval in seconds (default: 0.005 = 200 Hz)
    - verbose: Print progress messages
    """
    from mne_lsl.stream import StreamLSL

    # If no stream specified, try to connect to all available Muse streams
    streams = []
    if stream_name is None:
        # Try all possible Muse streams
        for name in ["Muse_EEG", "Muse_ACCGYRO", "Muse_Optics"]:
            if verbose:
                print(f"Looking for LSL stream '{name}'...")
            try:
                stream = StreamLSL(bufsize=window_size, name=name)
                stream.connect(timeout=2.0)
                streams.append(stream)
                if verbose:
                    print(f"  ✓ Connected to {name}")
            except Exception:
                if verbose:
                    print(f"  ✗ {name} not found")

        if len(streams) == 0:
            print("\nError: No Muse streams found!")
            print("Make sure streaming is running in another terminal:")
            print("  OpenMuse stream --address <MAC>")
            return
    else:
        # Connect to specific stream
        if verbose:
            print(f"Looking for LSL stream '{stream_name}'...")
        try:
            stream = StreamLSL(bufsize=window_size, name=stream_name)
            stream.connect(timeout=5.0)
            streams.append(stream)
            if verbose:
                print(f"  ✓ Connected to {stream_name}")
        except Exception as exc:
            print(f"Error: Could not connect to LSL stream '{stream_name}': {exc}")
            print("\nMake sure the stream is running in another terminal:")
            print("  OpenMuse stream --address <MAC> --preset <PRESET>")
            return

    # Create and show viewer
    viewer = RealtimeViewer(
        streams=streams,
        window_size=window_size,
        update_interval=update_interval,
        duration=duration,
        verbose=verbose,
    )
    viewer.show()
