from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


FS = 8000
DT = 1.0 / FS


def recursive_power_envelope(signal: np.ndarray, alpha: float) -> np.ndarray:
	power = np.empty_like(signal, dtype=float)
	power[0] = signal[0] ** 2
	for index in range(1, len(signal)):
		power[index] = alpha * power[index - 1] + (1.0 - alpha) * signal[index] ** 2
	return power


def time_axis(duration: float) -> np.ndarray:
	return np.arange(int(duration * FS)) / FS


def generate_gaussian_noise(duration: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(42)
	t = time_axis(duration)
	signal = rng.normal(0.0, 1.0, size=t.shape)
	return t, signal


def generate_sine(duration: float = 3.0, frequency: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
	t = time_axis(duration)
	signal = np.sin(2.0 * np.pi * frequency * t)
	return t, signal


def generate_chirp(duration: float = 5.0, f0: float = 0.0, f1: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
	t = time_axis(duration)
	k = (f1 - f0) / duration
	phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t**2)
	signal = np.sin(phase)
	return t, signal


def generate_speech_like(duration: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
	t = time_axis(duration)
	signal = np.zeros_like(t)

	segments = [
		(0.0, 0.8, 120.0, 0.20, 700.0),
		(0.8, 1.6, 140.0, 0.30, 900.0),
		(1.6, 2.1, None, 0.12, None),
		(2.1, 3.0, 110.0, 0.25, 500.0),
	]

	for start, end, pitch, amplitude, formant in segments:
		mask = (t >= start) & (t < end)
		local_t = t[mask] - start
		if pitch is None:
			rng = np.random.default_rng(7)
			noise = rng.normal(0.0, 1.0, size=local_t.shape)
			envelope = 0.7 + 0.3 * np.sin(2.0 * np.pi * 3.0 * local_t) ** 2
			signal[mask] = amplitude * envelope * noise
			continue

		glottal = (
			np.sin(2.0 * np.pi * pitch * local_t)
			+ 0.5 * np.sin(2.0 * np.pi * 2.0 * pitch * local_t)
			+ 0.25 * np.sin(2.0 * np.pi * 3.0 * pitch * local_t)
		)
		modulator = 0.6 + 0.4 * np.sin(2.0 * np.pi * 1.5 * local_t) ** 2
		if formant is not None:
			resonance = 1.0 + 0.4 * np.sin(2.0 * np.pi * formant * local_t / FS)
		else:
			resonance = 1.0
		signal[mask] = amplitude * modulator * resonance * glottal

	return t, signal


def load_or_generate_speech() -> tuple[np.ndarray, np.ndarray]:
	script_dir = Path(__file__).resolve().parent
	candidates = sorted(script_dir.glob("*.wav"))
	if candidates:
		try:
			from scipy.io import wavfile

			sample_rate, data = wavfile.read(candidates[0])
			if data.ndim > 1:
				data = data[:, 0]
			data = data.astype(float)
			if np.max(np.abs(data)) > 0:
				data /= np.max(np.abs(data))
			if sample_rate != FS:
				duration = len(data) / sample_rate
				old_t = np.linspace(0.0, duration, len(data), endpoint=False)
				new_t = time_axis(duration)
				data = np.interp(new_t, old_t, data)
				return new_t, data
			return time_axis(len(data) / sample_rate), data
		except Exception:
			pass

	return generate_speech_like()


def plot_signal_with_envelopes(ax, t: np.ndarray, signal: np.ndarray, title: str, alphas: list[float]) -> None:
	ax.plot(t, signal, color="0.7", linewidth=0.8, label="sygnał")
	for alpha in alphas:
		envelope = recursive_power_envelope(signal, alpha)
		ax.plot(t, envelope, linewidth=1.4, label=f"P[n], α={alpha}")
	ax.set_title(title)
	ax.set_xlabel("czas [s]")
	ax.set_ylabel("amplituda / moc")
	ax.grid(True, alpha=0.25)


def main() -> None:
	alpha_values = [0.9, 0.99, 0.999]

	signals = [
		("Szum gaussowski", *generate_gaussian_noise(3.0)),
		("Sinusoida 1 kHz", *generate_sine(3.0, 1000.0)),
		("Chirp 0 Hz -> 1 kHz", *generate_chirp(5.0, 0.0, 1000.0)),
		("Sygnał mowy", *load_or_generate_speech()),
	]

	fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
	axes = axes.ravel()

	for ax, (title, t, signal) in zip(axes, signals):
		plot_signal_with_envelopes(ax, t, signal, title, alpha_values)
		ax.legend(loc="upper right", fontsize=9)

	fig.suptitle("Zadanie 4.1 - obwiednia mocy liczona rekurencyjnie", fontsize=16)
	backend = plt.get_backend().lower()
	if "agg" in backend or "inline" in backend:
		output_path = Path(__file__).with_name("zad4_1_wykresy.png")
		fig.savefig(output_path, dpi=200)
		print(f"Zapisano wykres do: {output_path}")
	else:
		plt.show()

if __name__ == "__main__":
	main()
