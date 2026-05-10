from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
	from scipy.io import wavfile
	from scipy.signal import spectrogram
except Exception as exc:  # pragma: no cover - depends on environment
	raise RuntimeError(
		"Do uruchomienia tego skryptu potrzebne są pakiety scipy i matplotlib."
	) from exc


WINDOW_LENGTH = 1024
OVERLAP = 768
DEFAULT_FALLBACK_FS = 8000


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Zadanie 4.3 - analiza sygnału")
	parser.add_argument(
		"input",
		nargs="?",
		default=None,
		help="Ścieżka do pliku WAV. Jeśli brak, skrypt użyje pierwszego WAV z katalogu lub sygnału demonstracyjnego.",
	)
	return parser.parse_args()


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

	signal[first] = 0.9 * np.sin(2.0 * np.pi * 200.0 * t[first])
	signal[second] = 0.6 * np.sin(2.0 * np.pi * (400.0 * t[second] + 200.0 * t[second] ** 2))
	rng = np.random.default_rng(7)
	signal[third] = 0.15 * rng.normal(0.0, 1.0, size=np.count_nonzero(third))

	return fs, signal


def find_default_wav(script_dir: Path) -> Path | None:
	preferred = script_dir / "4.3.wav"
	if preferred.exists():
		return preferred

	candidates = sorted(script_dir.glob("*.wav"))
	return candidates[0] if candidates else None


def load_signal(input_path: str | None, script_dir: Path) -> tuple[int, np.ndarray, str]:
	if input_path:
		path = Path(input_path)
	else:
		path = find_default_wav(script_dir)

	if path is None:
		fs, signal = generate_demo_signal()
		return fs, signal, "sygnał demonstracyjny"

	fs, data = wavfile.read(path)
	if data.ndim > 1:
		data = np.mean(data, axis=1)
	signal = normalize_signal(data)
	return int(fs), signal, str(path)


def safe_db(values: np.ndarray, reference: float = 1.0) -> np.ndarray:
	return 10.0 * np.log10(np.maximum(values, 1e-15) / reference)


def one_sided_fft(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
	spectrum = np.fft.rfft(signal)
	freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
	amplitude = np.abs(spectrum) / len(signal)
	if amplitude.size > 2:
		amplitude[1:-1] *= 2.0
	return freqs, amplitude


def compute_spectrogram(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	freqs, times, sxx = spectrogram(
		signal,
		fs=fs,
		window="hamming",
		nperseg=WINDOW_LENGTH,
		noverlap=OVERLAP,
		nfft=WINDOW_LENGTH,
		scaling="density",
		mode="magnitude",
	)
	power = sxx**2
	return freqs, times, power


def plot_time_domain(ax, signal: np.ndarray, fs: int, title: str) -> None:
	time = np.arange(len(signal)) / fs
	ax.plot(time, signal, color="#1f77b4", linewidth=1.0)
	ax.set_title(title)
	ax.set_xlabel("czas [s]")
	ax.set_ylabel("amplituda")
	ax.grid(True, alpha=0.25)


def plot_fft(ax, signal: np.ndarray, fs: int, title: str) -> None:
	freqs, amplitude = one_sided_fft(signal, fs)
	ax.plot(freqs, safe_db(amplitude), color="#d62728", linewidth=1.0)
	ax.set_title(title)
	ax.set_xlabel("częstotliwość [Hz]")
	ax.set_ylabel("amplituda [dB]")
	ax.grid(True, alpha=0.25)
	ax.set_xlim(0, fs / 2)


def plot_spectrogram(ax, signal: np.ndarray, fs: int, title: str) -> None:
	freqs, times, power = compute_spectrogram(signal, fs)
	pcm = ax.pcolormesh(times, freqs, safe_db(power), shading="auto", cmap="magma")
	ax.set_title(title)
	ax.set_xlabel("czas [s]")
	ax.set_ylabel("częstotliwość [Hz]")
	ax.set_ylim(0, fs / 2)
	return pcm


def save_or_show(figures: list[tuple[str, plt.Figure]]) -> None:
	for name, fig in figures:
		output = Path(__file__).with_name(f"zad4_3_{name}.png")
		fig.savefig(output, dpi=200)
		print(f"Zapisano wykres: {output}")

	backend = plt.get_backend().lower()
	if "agg" not in backend and "inline" not in backend:
		plt.show()


def create_analysis_figure(signal: np.ndarray, fs: int, source_name: str) -> plt.Figure:
	fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)

	plot_time_domain(axes[0], signal, fs, f"{source_name} - dziedzina czasu")
	plot_fft(axes[1], signal, fs, f"{source_name} - FFT")
	pcm = plot_spectrogram(axes[2], signal, fs, f"{source_name} - spektrogram Hamming {WINDOW_LENGTH}")

	fig.colorbar(pcm, ax=axes[2], label="poziom mocy [dB]")
	fig.suptitle("Zadanie 4.3 - analiza sygnału w czasie i częstotliwości", fontsize=16)
	return fig


def main() -> None:
	args = parse_args()
	script_dir = Path(__file__).resolve().parent
	fs, signal, source_name = load_signal(args.input, script_dir)

	fig = create_analysis_figure(signal, fs, source_name)
	save_or_show([(Path(source_name).stem.replace(".", "_").replace(" ", "_"), fig)])

	duration = len(signal) / fs
	print("Wnioski:")
	print(f"- sygnał ma długość {duration:.3f} s i jest próbkowany z fs = {fs} Hz")
	print("- FFT pokazuje globalny rozkład energii w całym sygnale")
	print("- spektrogram pokazuje, jak energia zmienia się w czasie i częstotliwości")
	print("- dla modułu STFT i widma mocy w dB obowiązuje zależność 10log10(|X|^2) = 20log10(|X|)")
	print("- jeśli sygnał jest mowy lub modulowany, energia zwykle skupia się w pasmach formantowych i w krótkich odcinkach czasu")


if __name__ == "__main__":
	main()
