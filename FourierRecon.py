#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_top_frequencies(freqs, magnitude, phase, N, signal_length):
    half = signal_length // 2

    mag = magnitude[:half].copy()
    phs = phase[:half]
    freqs = freqs[:half]

    mag[0] = 0  # remove DC

    top_idx = np.argsort(mag)[-N:]

    n = np.arange(signal_length)

    components = []
    reconstruction = np.zeros(signal_length, dtype=np.float32)

    for k in top_idx:
        amp = 2 * mag[k] / signal_length
        wave = amp * np.cos(2 * np.pi * freqs[k] * n + phs[k])

        components.append(wave)
        reconstruction += wave

    return np.array(components), reconstruction

def components_to_image(components, height_per=40):
    comps = components.copy()

    # Normalize each row independently
    for i in range(len(comps)):
        c = comps[i]
        c -= c.min()
        c /= c.max() - c.min() + 1e-8
        comps[i] = c * 255

    img = np.repeat(comps.astype(np.uint8), height_per, axis=0)
    return img

def signal_to_image(signal, height=60):
    s = signal.copy()
    s -= s.min()
    s /= s.max() - s.min() + 1e-8
    s = (s * 255).astype(np.uint8)

    return np.repeat(s[np.newaxis, :], height, axis=0)


def draw_fft_spectrum(freqs, magnitude, width=600, height=200):
    N = len(magnitude)
    half = N // 2

    mag = magnitude[:half].copy()
    mag[0] = 0

    # Log scale
    # mag = np.log(mag + 1e-8)

    # Normalize to [0, height]
    mag -= mag.min()
    mag /= mag.max() + 1e-8
    mag *= height - 1

    
    # Create blank canvas
    canvas = np.zeros((height, width), dtype=np.uint8)

    top = np.argsort(mag)[-5:]
    for i in top:
        x = int(i * width / half)
        cv.circle(canvas, (x, height - int(mag[i]) - 1), 3, 255, -1)
        

    # Resample spectrum to window width
    xs = np.linspace(0, half - 1, width).astype(int)
    ys = mag[xs].astype(int)

    for i in range(1, width):
        cv.line(
            canvas,
            (i - 1, height - ys[i - 1] - 1),
            (i,     height - ys[i]     - 1),
            255,
            1
        )

    return canvas


def plot_fft_magnitude(freqs, magnitude):
    N = len(magnitude)

    # Use only positive frequencies (real signal symmetry)
    half = N // 2
    freqs = freqs[:half]
    mag = magnitude[:half]

    # Remove DC spike for visibility
    mag = mag.copy()
    mag[0] = 0

    # Log scale for dynamic range
    mag_log = np.log(mag + 1e-8)

    plt.figure(figsize=(8, 3))
    # plt.plot(freqs, mag_log)
    plt.plot(freqs, mag)
    plt.xlabel("Frequency (cycles per pixel)")
    plt.ylabel("Log Magnitude")
    plt.title("1D Fourier Magnitude Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def top_frequencies(freqs, magnitude, N, skip_dc=False):
    mag = magnitude.copy()

    if skip_dc:
        mag[0] = 0  # remove zero-frequency (mean brightness)

    idx = np.argsort(mag)[-N:]   # top N bins
    return freqs[idx], idx

def get_fourier_features_1d(img, print_data=True,draw_data=True):
    """
    img: shape (1, W, 3) BGR OpenCV image
    returns: frequencies, magnitude, phase, complex_fft
    """

    # Convert to grayscale
    print(f"SHAPE: {img.shape}")
    img = img.astype(np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Flatten to 1D signal (length W)
    signal = gray.flatten().astype(np.float32)
    # Perform FFT
    fft_vals = np.fft.fft(signal)
    # Frequency bins
    freqs = np.fft.fftfreq(len(signal))

    # Magnitude and phase
    magnitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)

    if print_data:
        print(f"Signal Shape: {signal.shape}")
        print(f"Returned fft shape: {fft_vals.shape} - {fft_vals.dtype}")
        # print("Signal (first 10):", signal[:10])
        print_top = 10
        print(f"N    FFT\t\t\tMagnitude\tFrequencies\tPhase")
        for i in range(print_top):
            fft = f"({fft_vals[i].real:5.2f} + {fft_vals[i].imag:5.2f}j)"
            mag = f"{magnitude[i]:.2f}"
            fq = f"{freqs[i]:.3f}"
            ph = f"{phase[i]:.3f}"
            print(f"[{i}] ", f"{fft:<{20}}\t{mag:<10}\t{fq:<10}\t{ph:<10}")

    if draw_data:
        #Frequency Spectrum
        fft_img = draw_fft_spectrum(freqs,magnitude)
        cv.imshow("FFT",fft_img)

    return signal, freqs, magnitude, phase, fft_vals

def fft_1d_rgb(img):
    """
    img: (1, W, 3) BGR OpenCV image

    returns dict with per-channel:
        freqs, magnitude, phase, fft_vals
    """

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Split BGR (OpenCV order)
    b, g, r = cv.split(img)

    channels = {"b": b, "g": g, "r": r}
    out = {}

    for name, ch in channels.items():
        signal = ch.flatten().astype(np.float32)

        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))

        magnitude = np.abs(fft_vals)
        phase = np.angle(fft_vals)

        out[name] = {
            "freqs": freqs,
            "magnitude": magnitude,
            "phase": phase,
            "fft": fft_vals
        }

    return out


def reconstruct_rgb(fft_data, N, height=50):
    length = len(fft_data["r"]["magnitude"])

    recon = {}

    # Reconstruct each channel in 1D
    for c in ["b", "g", "r"]:
        d = fft_data[c]
        recon[c] = reconstruct_channel(
            d["freqs"],
            d["magnitude"],
            d["phase"],
            N,
            length
        )

    vis_rows = []
    rgb_channels = []
    
    for c in ["b", "g", "r"]:
        x = recon[c].astype(np.float32)

        # normalize 0–255
        x -= x.min()
        x /= x.max() - x.min() + 1e-8
        x = (x * 255).astype(np.uint8)

        # make it a tall strip for viewing
        row = np.repeat(x[np.newaxis, :], height, axis=0)

        vis_rows.append(row)     # grayscale row
        rgb_channels.append(x)  # save for RGB combine

    # build RGB reconstruction
    rgb = np.stack(rgb_channels, axis=-1)      # (W, 3)
    rgb = np.repeat(rgb[np.newaxis, :, :], height, axis=0)

    color_rows = []

    # Blue row
    b_row = np.zeros((height, length, 3), dtype=np.uint8)
    b_row[..., 0] = vis_rows[0]
    color_rows.append(b_row)

    # Green row
    g_row = np.zeros((height, length, 3), dtype=np.uint8)
    g_row[..., 1] = vis_rows[1]
    color_rows.append(g_row)

    # Red row
    r_row = np.zeros((height, length, 3), dtype=np.uint8)
    r_row[..., 2] = vis_rows[2]
    color_rows.append(r_row)

    # build RGB reconstruction
    # rgb = np.stack(color_rows, axis=-1)      # (W, 3)
    # rgb = np.repeat(rgb[np.newaxis, :, :], height, axis=0)


    stacked = np.vstack([
        color_rows[0],
        color_rows[1],
        color_rows[2],
        rgb
    ])

    return stacked

def reconstruct_channel(freqs, magnitude, phase, N, length):
    half = length // 2

    mag = magnitude[:half].copy()
    phs = phase[:half]
    freqs = freqs[:half]

    # mag[0] = 0  # drop DC

    top_idx = np.argsort(mag)[-N:]

    n = np.arange(length)

    recon = np.zeros(length, dtype=np.float32)

    for k in top_idx:
        amp = 2 * mag[k] / length
        recon += amp * np.cos(2 * np.pi * freqs[k] * n + phs[k])

    return recon
