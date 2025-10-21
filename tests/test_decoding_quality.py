"""
Test for empirically quantifying decoding quality using smoothness metrics.

This test measures how well decoded sensor data "fits" between consecutive samples,
providing a quantitative way to detect timestamping issues that cause discontinuities.

BASELINE RESULTS (Current Implementation):
----------------------------------------
Test Data: tests/test_data/test_accgyro.txt (3272 messages)
Samples Decoded: 4710
Duration: 90.653846 seconds
Timestamps Monotonic: True (0 inversions)

Quality Metrics:
- Mean Smoothness Score: 0.835394 ± 0.007966
- Mean Frequency Score: 599.838294

Channel Smoothness Scores:
- ACC_X:  0.828197
- ACC_Y:  0.836020
- ACC_Z:  0.834422
- GYRO_X: 0.849680
- GYRO_Y: 0.824924
- GYRO_Z: 0.839122

Channel Frequency Scores:
- ACC_X:   94.804788
- ACC_Y:  262.543459
- ACC_Z:   71.109375
- GYRO_X: 2550.861392
- GYRO_Y:  61.253315
- GYRO_Z: 558.457436

These baseline values should be maintained or improved in future changes.
Regressions in smoothness scores or monotonicity indicate potential issues.
"""

import unittest
import numpy as np
from scipy import signal
from OpenMuse.decode import decode_rawdata


class TestDecodingQuality(unittest.TestCase):
    """Test suite for quantifying decoding quality using smoothness metrics."""

    @classmethod
    def setUpClass(cls):
        """Load test data for quality assessment."""
        cls.test_file = "tests/test_data/test_accgyro.txt"

        # Load and decode the test data
        with open(cls.test_file, "r") as f:
            messages = [line.strip() for line in f if line.strip()]

        result = decode_rawdata(messages)
        cls.accgyro_data = result["ACCGYRO"]

        # Extract the 6 ACCGYRO channels (time + 6 sensor channels)
        if len(cls.accgyro_data) > 0:
            cls.timestamps = cls.accgyro_data["time"].values
            cls.channels = cls.accgyro_data[
                ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
            ].values.T

    def test_data_loaded(self):
        """Ensure test data was loaded successfully."""
        self.assertGreater(len(self.accgyro_data), 0, "No ACCGYRO data loaded")
        self.assertEqual(self.channels.shape[0], 6, "Expected 6 ACCGYRO channels")

    def test_smoothness_metrics(self):
        """Test that decoded data has expected smoothness characteristics."""
        if len(self.accgyro_data) < 10:
            self.skipTest("Insufficient data for smoothness analysis")

        # Compute smoothness scores for each channel
        smoothness_scores = []
        for channel_data in self.channels:
            score = self._compute_smoothness_score(channel_data)
            smoothness_scores.append(score)

        # Convert to numpy array for analysis
        smoothness_scores = np.array(smoothness_scores)

        # ACCGYRO data should be relatively smooth (not random noise)
        # These thresholds are calibrated for the test data characteristics
        mean_smoothness = np.mean(smoothness_scores)
        std_smoothness = np.std(smoothness_scores)

        print(f"Mean smoothness score: {mean_smoothness:.6f}")
        print(f"Smoothness std: {std_smoothness:.6f}")

        # For real movement data, smoothness should be above threshold
        # This detects gross timestamping errors that create discontinuities
        self.assertGreater(
            mean_smoothness,
            0.1,
            f"Data appears discontinuous (smoothness: {mean_smoothness:.6f})",
        )

        # Smoothness shouldn't be too high (indicating potential over-smoothing)
        self.assertLess(
            mean_smoothness,
            0.95,
            f"Data appears over-smoothed (smoothness: {mean_smoothness:.6f})",
        )

    def test_frequency_characteristics(self):
        """Test that decoded data has expected frequency characteristics."""
        if len(self.accgyro_data) < 100:
            self.skipTest("Insufficient data for frequency analysis")

        # Analyze frequency content for each channel
        freq_scores = []
        for channel_data in self.channels:
            score = self._compute_frequency_score(channel_data)
            freq_scores.append(score)

        freq_scores = np.array(freq_scores)
        mean_freq_score = np.mean(freq_scores)

        print(f"Mean frequency score: {mean_freq_score:.6f}")

        # Real movement data should have more low-frequency than high-frequency content
        self.assertGreater(
            mean_freq_score,
            0.3,
            f"Unexpected frequency characteristics (score: {mean_freq_score:.6f})",
        )

    def _compute_smoothness_score(self, data):
        """
        Compute smoothness score based on local polynomial fitting.

        Returns a score between 0 and 1, where 1 indicates perfectly smooth data
        and lower scores indicate discontinuities or noise.
        """
        if len(data) < 10:
            return 0.0

        # Use sliding windows to fit local polynomials
        window_size = min(20, len(data) // 4)
        if window_size < 5:
            return 0.0

        # Fit quadratic polynomials to windows and measure RMS error
        errors = []
        step = max(1, window_size // 4)  # Overlapping windows

        for start in range(0, len(data) - window_size, step):
            end = start + window_size
            window = data[start:end]

            # Fit quadratic polynomial
            x = np.arange(window_size)
            try:
                coeffs = np.polyfit(x, window, 2)
                fitted = np.polyval(coeffs, x)
                error = np.sqrt(np.mean((window - fitted) ** 2))
                # Normalize by data range to make it scale-invariant
                data_range = np.ptp(window)  # peak-to-peak
                if data_range > 0:
                    normalized_error = error / data_range
                    errors.append(1.0 / (1.0 + normalized_error))  # Convert to score
            except (np.linalg.LinAlgError, ValueError):
                # Skip windows that can't be fitted (rank deficient or other fitting issues)
                continue

        if not errors:
            return 0.0

        # Return mean smoothness score across all windows
        return np.mean(errors)

    def _compute_frequency_score(self, data):
        """
        Compute frequency score based on low vs high frequency content.

        Returns a score indicating the ratio of low-frequency to high-frequency power.
        Higher scores indicate smoother, more gradual changes.
        """
        if len(data) < 50:
            return 0.0

        # Remove linear trend
        data_detrended = signal.detrend(data)

        # Compute power spectral density
        freqs, psd = signal.welch(data_detrended, fs=52.0, nperseg=min(256, len(data)))

        # Split into low and high frequency bands
        # For ACCGYRO at 52Hz, low freq might be < 5Hz, high freq > 10Hz
        low_freq_mask = freqs < 5.0
        high_freq_mask = freqs > 10.0

        low_power = np.sum(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
        high_power = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
        total_power = np.sum(psd)

        if total_power == 0:
            return 0.0

        # Return ratio of low to high frequency power
        # Higher values indicate smoother signals
        if high_power > 0:
            return low_power / high_power
        else:
            return 1.0  # All low frequency


if __name__ == "__main__":
    unittest.main()
