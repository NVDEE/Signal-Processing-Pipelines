"""
Standalone demonstration script for the SR71 radar demo project.

This script initialises an SR71 radar object (using the mock interface by default),
acquires a single frame, computes a range profile, detects peaks, and displays the
result.  It also generates a range‑time heatmap using a small number of frames.

Run this script from the repository root with:

    python -m sr71_demo.src.run_demo

If you wish to use a real SR71 sensor instead of the mock, set `use_mock=False` when
instantiating ``SR71Radar`` and implement the hardware calls in the ``sr71_demo`` module.
"""

import argparse
import matplotlib.pyplot as plt

from .sr71_demo import (
    SR71Radar,
    compute_range_profile,
    detect_peaks,
    plot_range_profile,
    generate_range_time_heatmap,
    plot_heatmap,
)


def main(num_frames: int = 30, threshold: float = 0.3, use_mock: bool = True, save_figures: bool = False, non_blocking: bool = False) -> None:
    # Create radar with default configuration
    radar = SR71Radar(config={'sample_rate': 2e6, 'chirp_duration': 1e-3}, use_mock=use_mock)
    radar.configure()

    # Acquire one frame for range profile
    frame = radar.read_frame()
    range_bins, magnitude = compute_range_profile(frame, radar.sample_rate)
    peaks = detect_peaks(magnitude, threshold=threshold)

    # Plot range profile with detected peaks
    plt.figure()
    ax = plot_range_profile(range_bins, magnitude, peaks)
    if save_figures:
        plt.savefig("range_profile.png")
        plt.close()
    else:
        if non_blocking:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.show()

    # Generate and plot range‑time heatmap
    frames = [radar.read_frame() for _ in range(num_frames)]
    range_bins, heatmap = generate_range_time_heatmap(frames, radar.sample_rate, num_frames=num_frames)
    plt.figure()
    plot_heatmap(range_bins, heatmap)
    if save_figures:
        plt.savefig("range_time_heatmap.png")
        plt.close()
    else:
        if non_blocking:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR71 radar demo script")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames for heatmap")
    parser.add_argument("--threshold", type=float, default=0.3, help="Peak detection threshold (0-1)")
    parser.add_argument(
        "--no-mock",
        action="store_false",
        dest="use_mock",
        help="Disable mock interface and use real hardware (requires SR71 SDK)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        dest="save_figures",
        help="Save figures to files instead of showing them interactively",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        dest="non_blocking",
        help="Show figures with non-blocking mode (windows won't stop execution)",
    )
    args = parser.parse_args()
    main(
        num_frames=args.frames,
        threshold=args.threshold,
        use_mock=args.use_mock,
        save_figures=getattr(args, "save_figures", False),
        non_blocking=getattr(args, "non_blocking", False),
    )