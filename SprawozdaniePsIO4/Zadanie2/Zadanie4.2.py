from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FS = 8000
N = 1024
F1 = 1000.0
F2 = 2000.0


def generate_signals() -> dict[str, np.ndarray]:
	rng = np.random.default_rng(12345)
	n = np.arange(N)
	w = rng.normal(0.0, 1.0, size=N)
	s = 0.5 * np.sin(2.0 * np.pi * F1 * n / FS) + np.sin(2.0 * np.pi * F2 * n / FS)
	y = s + 0.1 * w
	return {"w[n]": w, "s[n]": s, "y[n]": y}


def safe_db(values: np.ndarray, reference: float = 1.0) -> np.ndarray:
	return 10.0 * np.log10(np.maximum(values, 1e-15) / reference)


def one_sided_frequency_axis(length: int, fs: float = FS) -> np.ndarray:
	return np.fft.rfftfreq(length, d=1.0 / fs)


def periodogram(signal: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
	length = len(signal)
	spectrum = np.fft.rfft(signal)
	power = (np.abs(spectrum) ** 2) / length
	if power.size > 2:
		power[1:-1] *= 2.0
	return one_sided_frequency_axis(length, fs), power


def fft_power_spectrum(signal: np.ndarray, fs: float = FS, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
	if nfft is None:
		nfft = len(signal)
	spectrum = np.fft.rfft(signal, n=nfft)
	power = (np.abs(spectrum) ** 2) / len(signal)
	if power.size > 2:
		power[1:-1] *= 2.0
	return one_sided_frequency_axis(nfft, fs), power


def hann_window_spectrum(signal: np.ndarray, window_length: int, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
	window = np.hanning(window_length)
	windowed = signal[:window_length] * window
	coherent_gain = np.sum(window) / window_length
	if coherent_gain == 0:
		coherent_gain = 1.0
	spectrum = np.fft.rfft(windowed, n=window_length)
	power = (np.abs(spectrum) ** 2) / (window_length * coherent_gain**2)
	if power.size > 2:
		power[1:-1] *= 2.0
	return one_sided_frequency_axis(window_length, fs), power


def plot_spectrum(ax, freqs: np.ndarray, power: np.ndarray, label: str, style: str = "-", linewidth: float = 1.2) -> None:
	ax.plot(freqs, safe_db(power), style, linewidth=linewidth, label=label)
	ax.set_xlim(0, FS / 2)
	ax.set_xlabel("częstotliwość [Hz]")
	ax.set_ylabel("poziom [dB]")
	ax.grid(True, alpha=0.25)


def analyze_signal(signal_name: str, signal: np.ndarray) -> tuple[plt.Figure, plt.Figure]:
	fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
	axes = axes.ravel()

	freqs_p, power_p = periodogram(signal)
	plot_spectrum(axes[0], freqs_p, power_p, "periodogram")
	axes[0].set_title(f"{signal_name} - periodogram")
	axes[0].legend(loc="upper right")

	freqs_f, power_f = fft_power_spectrum(signal)
	plot_spectrum(axes[1], freqs_f, power_f, "FFT", linewidth=1.0)
	axes[1].set_title(f"{signal_name} - FFT")
	axes[1].legend(loc="upper right")

	for ax, window_length in zip(axes[2:], [256, 128]):
		freqs_w, power_w = hann_window_spectrum(signal, window_length)
		plot_spectrum(ax, freqs_w, power_w, f"Hann {window_length}")
		ax.set_title(f"{signal_name} - FFT z oknem Hanna {window_length}")
		ax.legend(loc="upper right")

	extra_fig, extra_ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
	freqs_w64, power_w64 = hann_window_spectrum(signal, 64)
	plot_spectrum(extra_ax, freqs_w64, power_w64, "Hann 64")
	extra_ax.set_title(f"{signal_name} - FFT z oknem Hanna 64")
	extra_ax.legend(loc="upper right")

	fig.suptitle(f"Zadanie 4.2 - analiza widmowa: {signal_name}", fontsize=16)
	extra_fig.suptitle(f"Zadanie 4.2 - analiza widmowa: {signal_name} (okno 64)", fontsize=16)
	return fig, extra_fig


def save_or_show(figures: list[tuple[str, plt.Figure]]) -> None:
	backend = plt.get_backend().lower()
	if "agg" in backend or "inline" in backend:
		output_dir = Path(__file__).resolve().parent
		for name, fig in figures:
			output_path = output_dir / f"zad4_2_{name}.png"
			fig.savefig(output_path, dpi=200)
			print(f"Zapisano wykres: {output_path}")
	else:
		plt.show()


def main() -> None:
	signals = generate_signals()
	figures: list[tuple[str, plt.Figure]] = []

	for name, signal in signals.items():
		fig_main, fig_64 = analyze_signal(name, signal)
		safe_name = name.replace("[", "").replace("]", "").replace(" ", "_")
		figures.append((f"{safe_name}_main", fig_main))
		figures.append((f"{safe_name}_hann64", fig_64))

	save_or_show(figures)

if __name__ == "__main__":
	main()
