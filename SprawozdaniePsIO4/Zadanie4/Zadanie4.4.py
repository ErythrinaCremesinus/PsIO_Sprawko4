from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
	from scipy.io import wavfile
	from scipy.signal import spectrogram as scipy_spectrogram
except Exception as exc:  # pragma: no cover - environment dependent
	raise RuntimeError("Do uruchomienia wymagane są pakiety scipy i matplotlib.") from exc


WINDOW_LENGTH = 1024
OVERLAP = 768
HOP = WINDOW_LENGTH - OVERLAP
DEFAULT_FALLBACK_FS = 8000


def normalize_signal(signal: np.ndarray) -> np.ndarray:
	signal = signal.astype(float)
	peak = np.max(np.abs(signal))
	if peak > 0:
		signal = signal / peak
	return signal


def generate_demo_signal(fs: int = DEFAULT_FALLBACK_FS, duration: float = 3.0) -> tuple[int, np.ndarray]:
	t = np.arange(int(fs * duration)) / fs
	signal = np.zeros_like(t)

	first = t < 1.0
	second = (t >= 1.0) & (t < 2.0)
	third = t >= 2.0

	signal[first] = 0.9 * np.sin(2.0 * np.pi * 220.0 * t[first])
	signal[second] = 0.6 * np.sin(2.0 * np.pi * (300.0 * t[second] + 250.0 * t[second] ** 2))
	rng = np.random.default_rng(11)
	signal[third] = 0.12 * rng.normal(0.0, 1.0, size=np.count_nonzero(third))

	return fs, signal


def find_default_wav(script_dir: Path) -> Path | None:
	candidates = sorted(script_dir.glob("*.wav"))
	return candidates[0] if candidates else None


def load_signal(script_dir: Path) -> tuple[int, np.ndarray, str]:
	path = find_default_wav(script_dir)
	if path is None:
		fs, signal = generate_demo_signal()
		return fs, signal, "sygnał demonstracyjny"

	fs, data = wavfile.read(path)
	if data.ndim > 1:
		data = np.mean(data, axis=1)
	signal = normalize_signal(data)
	return int(fs), signal, str(path)


def safe_db(values: np.ndarray) -> np.ndarray:
	return 10.0 * np.log10(np.maximum(values, 1e-15))


def one_sided_fft(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
	spectrum = np.fft.rfft(signal)
	freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
	amplitude = np.abs(spectrum) / len(signal)
	if amplitude.size > 2:
		amplitude[1:-1] *= 2.0
	return freqs, amplitude


def custom_smoothed_periodogram_spectrogram(
	signal: np.ndarray,
	fs: int,
	window_length: int = WINDOW_LENGTH,
	overlap: int = OVERLAP,
	time_smoothing: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	hop = window_length - overlap
	if hop <= 0:
		raise ValueError("overlap musi być mniejszy od window_length")

	window = np.hamming(window_length)
	window_energy = np.sum(window**2)
	if window_energy == 0:
		window_energy = 1.0

	if len(signal) < window_length:
		signal = np.pad(signal, (0, window_length - len(signal)))

	starts = np.arange(0, len(signal) - window_length + 1, hop)
	if starts.size == 0:
		starts = np.array([0])

	spectrum_columns = []
	times = []

	for start in starts:
		frame = signal[start : start + window_length]
		if len(frame) < window_length:
			frame = np.pad(frame, (0, window_length - len(frame)))
		windowed = frame * window
		spectrum = np.fft.rfft(windowed, n=window_length)
		periodogram = (np.abs(spectrum) ** 2) / (fs * window_energy)
		if periodogram.size > 2:
			periodogram[1:-1] *= 2.0
		spectrum_columns.append(periodogram)
		times.append((start + window_length / 2.0) / fs)

	power = np.stack(spectrum_columns, axis=1)

	if time_smoothing > 1:
		kernel = np.ones(time_smoothing) / time_smoothing
		smoothed = np.empty_like(power)
		for bin_index in range(power.shape[0]):
			smoothed[bin_index, :] = np.convolve(power[bin_index, :], kernel, mode="same")
		power = smoothed

	freqs = np.fft.rfftfreq(window_length, d=1.0 / fs)
	return freqs, np.asarray(times), power


def reference_spectrogram(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	freqs, times, sxx = scipy_spectrogram(
		signal,
		fs=fs,
		window="hamming",
		nperseg=WINDOW_LENGTH,
		noverlap=OVERLAP,
		nfft=WINDOW_LENGTH,
		scaling="density",
		mode="psd",
	)
	return freqs, times, sxx


def plot_time_domain(ax, signal: np.ndarray, fs: int, title: str) -> None:
	time = np.arange(len(signal)) / fs
	ax.plot(time, signal, color="#1f77b4", linewidth=0.9)
	ax.set_title(title)
	ax.set_xlabel("czas [s]")
	ax.set_ylabel("amplituda")
	ax.grid(True, alpha=0.25)


def plot_fft(ax, signal: np.ndarray, fs: int, title: str) -> None:
	freqs, amplitude = one_sided_fft(signal, fs)
	ax.plot(freqs, safe_db(amplitude), color="#d62728", linewidth=0.9)
	ax.set_title(title)
	ax.set_xlabel("częstotliwość [Hz]")
	ax.set_ylabel("amplituda [dB]")
	ax.set_xlim(0, fs / 2)
	ax.grid(True, alpha=0.25)


def plot_3d_mesh(ax, freqs: np.ndarray, times: np.ndarray, power: np.ndarray, title: str) -> None:
	time_grid, freq_grid = np.meshgrid(times, freqs)
	surface = ax.plot_surface(
		time_grid,
		freq_grid,
		safe_db(power),
		cmap="magma",
		linewidth=0,
		antialiased=True,
		rcount=min(80, power.shape[0]),
		ccount=min(80, power.shape[1]),
	)
	ax.set_title(title)
	ax.set_xlabel("czas [s]")
	ax.set_ylabel("częstotliwość [Hz]")
	ax.set_zlabel("gęstość mocy [dB]")
	ax.set_ylim(0, freqs.max())
	return surface


def save_figures(figures: list[tuple[str, plt.Figure]]) -> None:
	for name, fig in figures:
		output = Path(__file__).with_name(f"zad4_4_{name}.png")
		fig.savefig(output, dpi=200)
		print(f"Zapisano wykres: {output}")




def create_main_figure(signal: np.ndarray, fs: int, source_name: str) -> plt.Figure:
	fig = plt.figure(figsize=(16, 8), constrained_layout=True)
	grid = fig.add_gridspec(2, 1)

	ax_time = fig.add_subplot(grid[0, 0])
	ax_fft = fig.add_subplot(grid[1, 0])

	plot_time_domain(ax_time, signal, fs, f"{source_name} - dziedzina czasu")
	plot_fft(ax_fft, signal, fs, f"{source_name} - FFT")
	fig.suptitle("Zadanie 4.4 - sygnał i FFT", fontsize=16)
	return fig


def create_3d_figure(signal: np.ndarray, fs: int, source_name: str) -> plt.Figure:
	fig = plt.figure(figsize=(16, 9), constrained_layout=True)
	ax_mesh = fig.add_subplot(111, projection="3d")
	freqs, times, power = custom_smoothed_periodogram_spectrogram(signal, fs)
	surface = plot_3d_mesh(ax_mesh, freqs, times, power, f"{source_name} - własny spektrogram 3D")
	fig.colorbar(surface, ax=ax_mesh, shrink=0.8, pad=0.1, label="gęstość mocy [dB]")
	fig.suptitle("Zadanie 4.4 - własny spektrogram 3D", fontsize=16)
	return fig


def create_comparison_figure(signal: np.ndarray, fs: int, source_name: str) -> plt.Figure:
	custom_freqs, custom_times, custom_power = custom_smoothed_periodogram_spectrogram(signal, fs)
	ref_freqs, ref_times, ref_power = reference_spectrogram(signal, fs)

	fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
	mesh0 = axes[0].pcolormesh(custom_times, custom_freqs, safe_db(custom_power), shading="auto", cmap="magma")
	axes[0].set_title("Własny spektrogram")
	axes[0].set_xlabel("czas [s]")
	axes[0].set_ylabel("częstotliwość [Hz]")
	axes[0].set_ylim(0, fs / 2)
	fig.colorbar(mesh0, ax=axes[0], label="gęstość mocy [dB]")

	mesh1 = axes[1].pcolormesh(ref_times, ref_freqs, safe_db(ref_power), shading="auto", cmap="magma")
	axes[1].set_title("Referencyjny spektrogram SciPy")
	axes[1].set_xlabel("czas [s]")
	axes[1].set_ylabel("częstotliwość [Hz]")
	axes[1].set_ylim(0, fs / 2)
	fig.colorbar(mesh1, ax=axes[1], label="gęstość mocy [dB]")

	fig.suptitle(f"Zadanie 4.4 - porównanie spektrogramów: {source_name}", fontsize=16)
	return fig


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	fs, signal, source_name = load_signal(script_dir)

	main_fig = create_main_figure(signal, fs, source_name)
	three_d_fig = create_3d_figure(signal, fs, source_name)
	comparison_fig = create_comparison_figure(signal, fs, source_name)
	safe_name = Path(source_name).stem.replace(".", "_").replace(" ", "_")
	save_figures([
		(f"{safe_name}_main", main_fig),
		(f"{safe_name}_3d", three_d_fig),
		(f"{safe_name}_compare", comparison_fig),
	])
	plt.show()

	duration = len(signal) / fs

if __name__ == "__main__":
	main()
