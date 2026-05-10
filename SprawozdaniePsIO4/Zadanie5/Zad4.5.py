from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from noise import pnoise2
    _USE_LIBRARY = True
except ModuleNotFoundError:
    _USE_LIBRARY = False

def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _build_permutation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    return np.tile(p, 2)


def _gradient(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    vectors = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                        [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    g = vectors[h % 8]
    return g[..., 0] * x + g[..., 1] * y


def _perlin2(x: np.ndarray, y: np.ndarray, perm: np.ndarray) -> np.ndarray:
    xi = np.floor(x).astype(int) & 255
    yi = np.floor(y).astype(int) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = perm[perm[xi    ] + yi    ]
    ab = perm[perm[xi    ] + yi + 1]
    ba = perm[perm[xi + 1] + yi    ]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = _lerp(_gradient(aa, xf,     yf    ),
                _gradient(ba, xf - 1, yf    ), u)
    x2 = _lerp(_gradient(ab, xf,     yf - 1),
                _gradient(bb, xf - 1, yf - 1), u)
    return _lerp(x1, x2, v)


def generate_noise_map_numpy(
    width: int,
    height: int,
    scale: float = 4.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generuje mapę szumu Perlin przez sumowanie oktaw (fBm)."""
    perm = _build_permutation(seed)
    xs = np.linspace(0, scale, width, endpoint=False)
    ys = np.linspace(0, scale, height, endpoint=False)
    xg, yg = np.meshgrid(xs, ys)

    noise_map = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0

    for _ in range(octaves):
        noise_map += amplitude * _perlin2(xg * frequency, yg * frequency, perm)
        max_val += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return noise_map / max_val


def generate_noise_map_library(
    width: int,
    height: int,
    scale: float = 4.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generuje mapę szumu Perlin z użyciem pnoise2."""
    noise_map = np.empty((height, width))
    for row in range(height):
        for col in range(width):
            x = col / width * scale
            y = row / height * scale
            noise_map[row, col] = pnoise2(
                x, y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=seed,
            )
    return noise_map


def generate_noise_map(
    width: int = 512,
    height: int = 512,
    scale: float = 4.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    if _USE_LIBRARY:
        print("Użyto: pnoise2 z biblioteki noise")
        return generate_noise_map_library(width, height, scale, octaves, persistence, lacunarity, seed)
    else:
        print("Użyto: własna implementacja Perlin 2D (numpy)")
        return generate_noise_map_numpy(width, height, scale, octaves, persistence, lacunarity, seed)

TERRAIN_LEVELS = [
    (-1.00, -0.10, ( 2,  50, 127)),
    (-0.10,  0.00, ( 0, 128, 205)),
    ( 0.00,  0.06, (231, 203, 153)),
    ( 0.06,  0.34, (101, 179,  22)),
    ( 0.34,  0.44, (  2,  75,  26)),
    ( 0.44,  1.01, (203, 178,  78)),
]


def colorize(noise_map: np.ndarray) -> np.ndarray:
    height, width = noise_map.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for lo, hi, color in TERRAIN_LEVELS:
        mask = (noise_map >= lo) & (noise_map < hi)
        for channel, value in enumerate(color):
            rgb[..., channel][mask] = value
    return rgb

def main() -> None:
    WIDTH, HEIGHT = 512, 512

    noise_map = generate_noise_map(
        width=WIDTH,
        height=HEIGHT,
        scale=4.0,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        seed=42,
    )

    rgb = colorize(noise_map)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    fig.suptitle("Zadanie 4.5 - mapa terenu z szumu Perlin 2D", fontsize=14)

    output = Path(__file__).with_name("zad4_5_mapa.png")
    fig.savefig(output, dpi=200)
    print(f"Zapisano: {output}")

    plt.show()


if __name__ == "__main__":
    main()
