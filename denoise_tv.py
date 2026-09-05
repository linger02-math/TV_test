"""NumPy implementation of ROF total-variation image denoising with PDHG."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib

import numpy as np


def make_test_image(size: int = 256) -> np.ndarray:
    """Create a reproducible grayscale test image with smooth and sharp features."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    x /= size - 1
    y /= size - 1

    image = 0.12 + 0.25 * x + 0.12 * y
    image += 0.34 * ((x - 0.30) ** 2 + (y - 0.34) ** 2 < 0.14**2)
    image += 0.28 * ((x > 0.56) & (x < 0.86) & (y > 0.16) & (y < 0.43))
    image -= 0.18 * ((x > 0.16) & (x < 0.43) & (y > 0.65) & (y < 0.84))
    image += 0.19 * np.exp(-((x - 0.72) ** 2 + (y - 0.73) ** 2) / 0.018)
    # Thin structures make loss of detail visible in the comparison.
    image += 0.12 * (np.abs(y - (0.53 + 0.06 * np.sin(8 * np.pi * x))) < 0.008)
    return np.clip(image, 0.0, 1.0)


def gradient(u: np.ndarray) -> np.ndarray:
    """Forward finite differences (Neumann boundary), shape (H, W, 2)."""
    g = np.zeros((*u.shape, 2), dtype=u.dtype)
    g[:-1, :, 0] = u[1:, :] - u[:-1, :]
    g[:, :-1, 1] = u[:, 1:] - u[:, :-1]
    return g


def gradient_adjoint(p: np.ndarray) -> np.ndarray:
    """Adjoint of ``gradient``; satisfies <grad u,p> = <u,grad* p>."""
    out = np.zeros(p.shape[:2], dtype=p.dtype)
    out[:-1, :] -= p[:-1, :, 0]
    out[1:, :] += p[:-1, :, 0]
    out[:, :-1] -= p[:, :-1, 1]
    out[:, 1:] += p[:, :-1, 1]
    return out


def tv_denoise_pdhg(
    noisy: np.ndarray,
    weight: float = 0.105,
    iterations: int = 500,
    tau: float = 0.35,
    sigma: float = 0.35,
    theta: float = 1.0,
) -> np.ndarray:
    """Solve 0.5*||u-f||^2 + weight*TV(u) by Chambolle-Pock PDHG."""
    if tau * sigma * 8.0 >= 1.0:
        raise ValueError("PDHG requires tau * sigma * ||grad||^2 < 1 (use < 1/8).")
    u = noisy.astype(np.float64, copy=True)
    u_bar = u.copy()
    p = np.zeros((*u.shape, 2), dtype=np.float64)

    for _ in range(iterations):
        p += sigma * gradient(u_bar)
        norm = np.sqrt(np.sum(p * p, axis=2, keepdims=True))
        p /= np.maximum(1.0, norm / weight)  # projection onto |p_ij| <= weight

        u_old = u
        # Proximal map of 0.5*||u-f||^2.
        u = (u - tau * gradient_adjoint(p) + tau * noisy) / (1.0 + tau)
        u_bar = u + theta * (u - u_old)

    return np.clip(u, 0.0, 1.0)


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((reference - test) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    axis = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-(axis**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    tmp = np.pad(image, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, tmp)
    tmp = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, tmp)


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Standard local-window SSIM for images in [0, 1]."""
    kernel = _gaussian_kernel()
    mu_x = _gaussian_filter(reference, kernel)
    mu_y = _gaussian_filter(test, kernel)
    var_x = _gaussian_filter(reference * reference, kernel) - mu_x * mu_x
    var_y = _gaussian_filter(test * test, kernel) - mu_y * mu_y
    cov_xy = _gaussian_filter(reference * test, kernel) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )
    return float(np.mean(score))


def save_comparison_png(output: Path, rows: list[tuple[np.ndarray, ...]], description: str) -> None:
    """Save rows of grayscale panels using only the Python standard library."""
    gap = 8
    h, w = rows[0][0].shape
    nrows, ncols = len(rows), len(rows[0])
    canvas = np.full((nrows * h + (nrows - 1) * gap, ncols * w + (ncols - 1) * gap), 255, dtype=np.uint8)
    for row_index, row in enumerate(rows):
        for col_index, image in enumerate(row):
            top, left = row_index * (h + gap), col_index * (w + gap)
            canvas[top : top + h, left : left + w] = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row.tobytes() for row in canvas)
    header = struct.pack(">IIBBBBB", canvas.shape[1], canvas.shape[0], 8, 0, 0, 0, 0)
    text_data = b"Description\x00" + description.encode("latin-1", errors="replace")
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"tEXt", text_data)
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description="TV image denoising using NumPy PDHG")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=[10, 25, 50], help="Noise sigma(s), in 0-255 pixel units")
    parser.add_argument("--weight", type=float, default=None, help="TV weight; default is chosen from the noise sigma")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("denoise_result.png"))
    args = parser.parse_args()

    clean = make_test_image(args.size)
    rows = []
    descriptions = []
    print("sigma | noisy PSNR | noisy SSIM | denoised PSNR | denoised SSIM")
    print("------|------------|------------|---------------|--------------")
    result_psnrs = []
    for index, sigma_255 in enumerate(args.noise_sigmas):
        rng = np.random.default_rng(args.seed + index)
        sigma = sigma_255 / 255.0
        noisy = np.clip(clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)
        weight = args.weight if args.weight is not None else 1.25 * sigma
        denoised = tv_denoise_pdhg(noisy, weight=weight, iterations=args.iterations)
        noisy_psnr, noisy_ssim = psnr(clean, noisy), ssim(clean, noisy)
        result_psnr, result_ssim = psnr(clean, denoised), ssim(clean, denoised)
        result_psnrs.append(result_psnr)
        rows.append((clean, noisy, denoised))
        descriptions.append(
            f"sigma={sigma_255:g}: noisy {noisy_psnr:.2f}dB/{noisy_ssim:.4f}, "
            f"denoised {result_psnr:.2f}dB/{result_ssim:.4f}"
        )
        print(f"{sigma_255:5g} | {noisy_psnr:10.2f} | {noisy_ssim:10.4f} | {result_psnr:13.2f} | {result_ssim:13.4f}")

    save_comparison_png(args.output, rows, "Rows: sigma 10/25/50; columns: original/noisy/TV-PDHG. " + "; ".join(descriptions))
    print(f"Saved comparison to: {args.output.resolve()}")

    if min(result_psnrs) < 25.0:
        raise RuntimeError(f"Denoising failed: minimum PSNR {min(result_psnrs):.2f} dB < 25 dB")


if __name__ == "__main__":
    main()
