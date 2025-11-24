# SR71 Nanoradar Demo

This repository provides a quick start demonstration for working with the
Nanoradar **SR71** 77 GHz radar sensor.  It shows how to configure the radar,
acquire raw frames, compute a **range profile** using a Fast Fourier
Transform (FFT), detect prominent targets and display a **range–time heatmap**.

The code is written in Python and uses a **mock interface** by default so that
you can experiment without having access to the hardware.  When you connect
your SR71 sensor via a CAN/TTL interface and have the official SDK
installed, you can extend the interface in `sr71_demo/src/sr71_demo.py` to call
the real API.

## Features

- High‑level `SR71Radar` class for configuration and frame acquisition.
- Helpers to compute range profiles and detect peaks in the spectrum.
- Functions to generate and plot a range–time heatmap.
- Example script (`src/run_demo.py`) and Jupyter notebook (`examples/sr71_demo_notebook.ipynb`).
- Sample output images and GIF illustrating the processing pipeline.

## Repository structure

```
sr71_demo/
├── src/
│   ├── sr71_demo.py          # Core processing and mock radar interface
│   └── run_demo.py           # Stand‑alone demo script
├── examples/
│   └── sr71_demo_notebook.ipynb # Jupyter notebook version of the demo
├── docs/
│   ├── images/
│   │   └── heatmap.png       # Example heatmap generated using mock data
│   └── demo.gif              # Animated GIF showing range profile over time
├── requirements.txt          # Python dependencies
└── README.md                 # This document
```

## Hardware setup

1. **Sensor:** Nanoradar SR71 radar.  Connect the sensor to your host computer
   using an appropriate CAN/TTL interface (e.g. a USB‑to‑CAN adapter).  Consult
   the SR71 documentation for wiring and power requirements.
2. **SDK:** Install the official SR71 SDK from Nanoradar and ensure the
   libraries are available on your system.  Edit the `SR71Radar` class in
   `src/sr71_demo.py` to call the SDK’s initialisation, configuration and data
   acquisition functions instead of the mock implementation.
3. **Python environment:** Create a virtual environment (optional) and install
   the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the demo script

To run the stand‑alone demonstration using mock data:


```bash
python -m pip install -r requirements.txt
```

```bash
cd RadarSignalProcessing\
python -m sr71_demo.src.run_demo
```

This will display a range profile plot with detected peaks and a range–time
heatmap generated from multiple frames.  To use the real radar, add
`--no-mock` and modify the `SR71Radar` class accordingly.

You can adjust the number of frames and peak detection threshold:

```bash
python -m sr71_demo.src.run_demo --frames 50 --threshold 0.4
```

## Using the Jupyter notebook

Launch Jupyter (or VS Code) and open `examples/sr71_demo_notebook.ipynb`.  The
notebook walks through initialising the radar, acquiring data, computing a
range profile and generating a heatmap.  Because the notebook uses the same
mock interface, it will run out‑of‑the‑box.  You can copy and modify the code
cells to work with real data once you integrate the SDK.

## Modifying chirp parameters

The `SR71Radar` class accepts a configuration dictionary with keys such as
`sample_rate` (Hz) and `chirp_duration` (seconds).  To change the chirp
parameters, pass different values when instantiating the radar:

```python
radar = SR71Radar(config={
    'sample_rate': 3e6,     # 3 MSps
    'chirp_duration': 2e-3  # 2 ms
}, use_mock=True)
```

When using the real SDK you should also adjust the chirp slope and bandwidth
according to your application.  See the SR71 documentation for details.

## Sample output

Below is a heatmap generated using the mock interface (range on the x‑axis and
frame index on the y‑axis).  Brighter colours indicate stronger reflections.

![Range–time heatmap](docs/images/heatmap.png)

The repository also includes an animated GIF (`docs/demo.gif`) that shows how
the range profile evolves over multiple frames.  This can be useful for
demonstrating presence detection or tracking moving targets.

## Contributing

Feel free to open issues or submit pull requests if you extend the hardware
interface, improve the signal processing algorithms or add additional
visualisations.  Contributions are welcome!