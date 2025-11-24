"""
SR71 Nanoradar (77 GHz) demonstration utilities.

This module provides a minimal interface around the SR71 radar and a set of helper
functions for processing and visualising radar data.  The goal of this project
is to demonstrate how to acquire raw frames from the SR71 radar, compute range
profiles using a Fast Fourier Transform (FFT), detect peaks corresponding to
targets and display the results in a human‑friendly way.  Because the real
hardware and SDK are proprietary, this implementation includes a `MockSR71`
class that generates synthetic data for demonstration purposes.  When you have
access to the official SDK, you can replace the mock implementation with
appropriate API calls.

Example usage:

.. code-block:: python

    from sr71_demo import SR71Radar, compute_range_profile, detect_peaks, plot_range_profile

    # Create radar instance with desired configuration (placeholder values)
    radar = SR71Radar(config={"sample_rate": 2_000_000, "chirp_duration": 0.001})

    # Configure the radar (does nothing in mock)
    radar.configure()

    # Read a single frame of raw IQ samples from the radar
    frame = radar.read_frame()

    # Compute range profile using FFT
    range_bins, range_profile = compute_range_profile(frame, radar.sample_rate)

    # Detect peaks above a simple threshold
    peak_indices = detect_peaks(range_profile, threshold=0.5)

    # Plot the result
    plot_range_profile(range_bins, range_profile, peaks=peak_indices)

"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


@dataclass
class SR71Radar:
    """High‑level interface to an SR71 radar sensor.

    This class encapsulates the configuration and data acquisition from the SR71
    radar.  In production you would wrap calls to the Nanoradar SDK here.  The
    default implementation uses a mock interface that returns synthetic data so
    that the processing pipeline can be developed and tested without hardware.
    """

    config: Dict[str, Any]
    use_mock: bool = True

    def __post_init__(self) -> None:
        # Set default configuration values if not provided
        self.sample_rate: float = self.config.get("sample_rate", 2_000_000.0)  # Hz
        self.chirp_duration: float = self.config.get("chirp_duration", 1e-3)    # seconds
        self.num_samples: int = int(self.sample_rate * self.chirp_duration)
        # Additional SR71 specific parameters can be added here

        # Initialise hardware connection here if use_mock is False
        if not self.use_mock:
            # TODO: replace with actual SDK initialisation
            raise NotImplementedError(
                "Real SR71 hardware support is not implemented; set use_mock=True "
                "to generate synthetic data."
            )

    def configure(self) -> None:
        """Configure the radar with the specified chirp parameters.

        In a real implementation this would send configuration commands to the
        hardware.  The mock implementation simply prints a message.
        """
        if self.use_mock:
            print(
                f"[SR71Radar] Mock configuration: sample_rate={self.sample_rate} Hz, "
                f"chirp_duration={self.chirp_duration} s, num_samples={self.num_samples}"
            )
        else:
            # TODO: call into SR71 SDK to configure chirp parameters
            pass

    def read_frame(self) -> np.ndarray:
        """Read a single frame of raw IQ data from the radar.

        Returns
        -------
        np.ndarray
            Complex array of samples representing one chirp.  The array shape
            will be (num_samples,) for a single receiver channel.  If multiple
            channels are available, you can extend this function to return a 2D
            array of shape (num_rx, num_samples).
        """
        if self.use_mock:
            # Generate a synthetic chirp with a couple of sinusoidal targets and noise.
            time = np.arange(self.num_samples) / self.sample_rate
            # Two targets at 3 m and 8 m equivalent beat frequencies
            f1 = 30_000  # Hz
            f2 = 80_000  # Hz
            signal = (np.exp(2j * np.pi * f1 * time) + 0.5 * np.exp(2j * np.pi * f2 * time))
            # Add white noise
            noise = 0.2 * (np.random.randn(self.num_samples) + 1j * np.random.randn(self.num_samples))
            return signal + noise
        else:
            # TODO: call SR71 SDK to read raw IQ samples
            raise NotImplementedError(
                "Hardware capture not implemented; set use_mock=True to generate synthetic data."
            )

    def stream_frames(self) -> Iterable[np.ndarray]:
        """Generator that yields continuous frames from the radar.

        Yields
        ------
        np.ndarray
            Complex raw IQ data for each frame.
        """
        while True:
            yield self.read_frame()

    def close(self) -> None:
        """Clean up any resources associated with the radar connection."""
        if not self.use_mock:
            # TODO: close hardware connection
            pass


def compute_range_profile(iq_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 1D range profile from raw IQ data using FFT.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex baseband samples for a single chirp.
    sample_rate : float
        Sampling rate in Hz used during acquisition.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing (range_bins, magnitude_spectrum).  `range_bins`
        represents the relative range values (in metres) for each FFT bin and
        `magnitude_spectrum` contains the corresponding magnitude response.
    """
    n_fft = len(iq_data)
    # Perform FFT and take the magnitude
    spectrum = np.fft.fft(iq_data, n=n_fft)
    magnitude = np.abs(spectrum)
    # Use only the positive frequencies (first half of the FFT)
    half = n_fft // 2
    magnitude = magnitude[:half]
    # Generate range bins (assumes a simple linear relationship; for real radar
    # systems you must incorporate chirp slope and propagation speed)
    c = 3e8  # Speed of light in m/s
    # Chirp bandwidth (placeholder 200 MHz).  Replace with actual SR71 bandwidth
    bandwidth = 200e6
    # Range resolution = c/(2*bandwidth)
    range_res = c / (2 * bandwidth)
    range_bins = np.arange(half) * (sample_rate / n_fft) * range_res
    return range_bins, magnitude


def detect_peaks(magnitude_spectrum: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Detect peaks in the magnitude spectrum above a given threshold.

    Parameters
    ----------
    magnitude_spectrum : np.ndarray
        Magnitude spectrum from the FFT.
    threshold : float, optional
        Minimum relative amplitude (0–1) to consider a peak significant.  The
        threshold is applied after normalising the spectrum.

    Returns
    -------
    np.ndarray
        Indices of the detected peaks.
    """
    if len(magnitude_spectrum) == 0:
        return np.array([])
    # Normalise magnitude to [0, 1]
    norm = magnitude_spectrum / np.max(magnitude_spectrum)
    # Find peaks above the threshold using scipy.signal.find_peaks
    peaks, _ = find_peaks(norm, height=threshold)
    return peaks


def plot_range_profile(range_bins: np.ndarray, magnitude_spectrum: np.ndarray, peaks: Iterable[int] = None, ax: plt.Axes = None) -> plt.Axes:
    """Plot the range profile and annotate detected peaks.

    Parameters
    ----------
    range_bins : np.ndarray
        Range values for each FFT bin (in metres).
    magnitude_spectrum : np.ndarray
        Magnitude spectrum corresponding to `range_bins`.
    peaks : Iterable[int], optional
        Indices of peaks to highlight.  If provided, vertical lines will be
        drawn at these positions.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.  If None a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(range_bins, magnitude_spectrum, label="Range profile")
    if peaks is not None:
        for p in peaks:
            ax.axvline(range_bins[p], color="r", linestyle="--", alpha=0.6)
            ax.text(range_bins[p], magnitude_spectrum[p], f"{range_bins[p]:.1f} m", color="r")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude")
    ax.set_title("SR71 Range Profile")
    ax.grid(True)
    ax.legend()
    return ax


def generate_range_time_heatmap(frames: Iterable[np.ndarray], sample_rate: float, num_frames: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a range‑time heatmap from a sequence of frames.

    Parameters
    ----------
    frames : Iterable[np.ndarray]
        Generator or list of raw IQ frames.
    sample_rate : float
        Sampling rate of the data in Hz.
    num_frames : int, optional
        Number of frames to include in the heatmap.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing (range_bins, heatmap) where `heatmap` is a 2D array
        shaped (num_frames, len(range_bins)) representing the magnitude over
        time.
    """
    magnitude_matrix: List[np.ndarray] = []
    range_bins: np.ndarray = np.array([])
    for i, frame in enumerate(frames):
        if i >= num_frames:
            break
        rng, mag = compute_range_profile(frame, sample_rate)
        range_bins = rng  # same for each frame
        magnitude_matrix.append(mag)
    # Stack into 2D array
    heatmap = np.vstack(magnitude_matrix)
    return range_bins, heatmap


def plot_heatmap(range_bins: np.ndarray, heatmap: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    """Plot a range‑time heatmap.

    Parameters
    ----------
    range_bins : np.ndarray
        The range bin values.
    heatmap : np.ndarray
        2D matrix of magnitude values where each row corresponds to a frame.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.  If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(
        heatmap,
        aspect="auto",
        extent=[range_bins[0], range_bins[-1], 0, heatmap.shape[0]],
        origin="lower",
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Frame index")
    ax.set_title("Range‑Time Heatmap")
    plt.colorbar(im, ax=ax, label="Magnitude")
    return ax


__all__ = [
    "SR71Radar",
    "compute_range_profile",
    "detect_peaks",
    "plot_range_profile",
    "generate_range_time_heatmap",
    "plot_heatmap",
]