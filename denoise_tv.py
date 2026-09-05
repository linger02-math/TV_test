"""
NumPy 实现 ROF 全变分图像去噪（使用 PDHG 原始对偶混合梯度法）
本模块提供完整的图像去噪流程：生成测试图、添加噪声、TV 去噪、
质量评估（PSNR / SSIM）以及结果保存为 PNG。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib

import numpy as np


# ========== 1. 生成测试图像 ==========
def make_test_image(size: int = 256) -> np.ndarray:
    """
    生成一个可复现的灰度测试图像（值域 [0,1]），包含平滑区域和尖锐边缘。

    参数:
        size: 图像尺寸（size x size），默认 256。

    返回:
        np.ndarray: 形状 (size, size)，数据类型 float64，数值在 0~1 之间。
    """
    # 生成网格坐标，并归一化到 [0,1]
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    x /= size - 1
    y /= size - 1

    # 基础渐变背景（灰度缓慢变化）
    image = 0.12 + 0.25 * x + 0.12 * y

    # 添加一个圆形亮斑
    image += 0.34 * ((x - 0.30) ** 2 + (y - 0.34) ** 2 < 0.14**2)

    # 添加一个矩形亮块
    image += 0.28 * ((x > 0.56) & (x < 0.86) & (y > 0.16) & (y < 0.43))

    # 添加一个矩形暗块
    image -= 0.18 * ((x > 0.16) & (x < 0.43) & (y > 0.65) & (y < 0.84))

    # 添加一个高斯模糊的亮点（类似点光源）
    image += 0.19 * np.exp(-((x - 0.72) ** 2 + (y - 0.73) ** 2) / 0.018)

    # 添加一条正弦曲线形状的细线（用于测试去噪对细节的保留能力）
    image += 0.12 * (np.abs(y - (0.53 + 0.06 * np.sin(8 * np.pi * x))) < 0.008)

    # 截断保证数值在合法范围 [0,1]
    return np.clip(image, 0.0, 1.0)


# ========== 2. 梯度算子及其伴随 ==========
def gradient(u: np.ndarray) -> np.ndarray:
    """
    前向有限差分梯度（Neumann 边界条件）。
    返回形状 (H, W, 2)，其中第 0 通道为 x 方向差分，第 1 通道为 y 方向差分。

    参数:
        u: 输入图像，形状 (H, W)

    返回:
        np.ndarray: 梯度场，形状 (H, W, 2)
    """
    g = np.zeros((*u.shape, 2), dtype=u.dtype)
    # x 方向：右边减左边（最后一行保持 0，即 Neumann 边界）
    g[:-1, :, 0] = u[1:, :] - u[:-1, :]
    # y 方向：下边减上边（最后一列保持 0）
    g[:, :-1, 1] = u[:, 1:] - u[:, :-1]
    return g


def gradient_adjoint(p: np.ndarray) -> np.ndarray:
    """
    梯度算子的伴随算子（离散散度，负号形式）。
    满足恒等式 <grad u, p> = <u, grad* p>，用于 PDHG 迭代。

    参数:
        p: 对偶变量（梯度场），形状 (H, W, 2)

    返回:
        np.ndarray: 散度场，形状 (H, W)
    """
    out = np.zeros(p.shape[:2], dtype=p.dtype)
    # 根据伴随关系，对 x 方向差分做转置
    out[:-1, :] -= p[:-1, :, 0]
    out[1:, :] += p[:-1, :, 0]
    # 对 y 方向差分做转置
    out[:, :-1] -= p[:, :-1, 1]
    out[:, 1:] += p[:, :-1, 1]
    return out


# ========== 3. ROF 全变分去噪（PDHG 求解） ==========
def tv_denoise_pdhg(
    noisy: np.ndarray,
    weight: float = 0.105,
    iterations: int = 500,
    tau: float = 0.35,
    sigma: float = 0.35,
    theta: float = 1.0,
) -> np.ndarray:
    """
    使用 Chambolle-Pock 原始对偶混合梯度（PDHG）算法求解 ROF 模型：
         minimize  0.5 * ||u - f||^2  +  weight * TV(u)

    参数:
        noisy: 含噪图像，形状 (H, W)，值域任意（建议 [0,1]）
        weight: 全变分正则化权重，越大去噪越强
        iterations: 迭代次数，默认 500
        tau: 原始变量步长
        sigma: 对偶变量步长
        theta: 外推参数（通常设为 1.0）

    返回:
        np.ndarray: 去噪后图像，值域被裁剪到 [0,1]
    """
    # 稳定性条件：tau * sigma * ||grad||^2 < 1，对于 2D 图像 ||grad||^2 <= 8
    if tau * sigma * 8.0 >= 1.0:
        raise ValueError("PDHG requires tau * sigma * ||grad||^2 < 1 (use < 1/8).")

    u = noisy.astype(np.float64, copy=True)   # 原始变量（去噪图像）
    u_bar = u.copy()                         # 外推后的原始变量
    p = np.zeros((*u.shape, 2), dtype=np.float64)  # 对偶变量（梯度场）

    for _ in range(iterations):
        # ---- 对偶更新 ----
        p += sigma * gradient(u_bar)          # 梯度上升
        norm = np.sqrt(np.sum(p * p, axis=2, keepdims=True))  # 每个像素点处的模长
        # 投影到范数球：|p_ij| <= weight
        p /= np.maximum(1.0, norm / weight)

        # ---- 原始更新 ----
        u_old = u
        # 近端映射：求解 0.5*||u-f||^2 + tau * div(p) 的解析解
        u = (u - tau * gradient_adjoint(p) + tau * noisy) / (1.0 + tau)

        # ---- 外推（加速收敛） ----
        u_bar = u + theta * (u - u_old)

    # 最终裁剪到有效范围
    return np.clip(u, 0.0, 1.0)


# ========== 4. 图像质量评估 ==========
def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """
    计算峰值信噪比（PSNR），假设图像值域为 [0,1]。
    PSNR = 10 * log10(1 / MSE)

    返回:
        float: PSNR 值（dB），若 MSE=0 则返回 inf。
    """
    mse = float(np.mean((reference - test) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """生成一维高斯核（用于 SSIM 计算中的加权窗口）。"""
    axis = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-(axis**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    使用一维高斯核对图像进行二维滤波（通过行列分离卷积实现）。
    边界使用镜像反射填充。
    """
    pad = len(kernel) // 2
    # 水平方向卷积
    tmp = np.pad(image, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, tmp)
    # 垂直方向卷积
    tmp = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, tmp)


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """
    计算结构相似性指数（SSIM），基于局部窗口（默认高斯窗口 11x11）。
    图像值域应为 [0,1]，SSIM 值越接近 1 表示结构越相似。
    """
    kernel = _gaussian_kernel()
    # 均值
    mu_x = _gaussian_filter(reference, kernel)
    mu_y = _gaussian_filter(test, kernel)
    # 方差
    var_x = _gaussian_filter(reference * reference, kernel) - mu_x * mu_x
    var_y = _gaussian_filter(test * test, kernel) - mu_y * mu_y
    # 协方差
    cov_xy = _gaussian_filter(reference * test, kernel) - mu_x * mu_y

    # 常数项（防止分母为零）
    c1, c2 = 0.01**2, 0.03**2
    # SSIM 公式
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )
    return float(np.mean(score))


# ========== 5. 保存结果图像（纯标准库生成 PNG） ==========
def save_comparison_png(output: Path, rows: list[tuple[np.ndarray, ...]], description: str) -> None:
    """
    将多行多列图像组合成一张对比图，并保存为 PNG 文件（不依赖 PIL 等第三方库）。

    参数:
        output: 输出文件路径
        rows: 行列表，每行是一个元组，包含该行的若干图像（形状相同）
        description: 写入 PNG 的 tEXt 块中的描述文字
    """
    gap = 8  # 图像之间的间隔像素
    h, w = rows[0][0].shape
    nrows, ncols = len(rows), len(rows[0])

    # 创建画布（初始为白色 255）
    canvas = np.full(
        (nrows * h + (nrows - 1) * gap, ncols * w + (ncols - 1) * gap),
        255,
        dtype=np.uint8,
    )
    # 将每张图像放到画布对应位置
    for row_index, row in enumerate(rows):
        for col_index, image in enumerate(row):
            top, left = row_index * (h + gap), col_index * (w + gap)
            canvas[top : top + h, left : left + w] = np.rint(
                np.clip(image, 0, 1) * 255
            ).astype(np.uint8)

    # ---------- 构造 PNG 文件（手动打包） ----------
    def chunk(kind: bytes, data: bytes) -> bytes:
        """生成 PNG 的 chunk 块：长度 + 类型 + 数据 + CRC32"""
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    # 像素数据（每行前加 0 过滤字节）
    raw = b"".join(b"\x00" + row.tobytes() for row in canvas)
    # IHDR 头信息
    header = struct.pack(">IIBBBBB", canvas.shape[1], canvas.shape[0], 8, 0, 0, 0, 0)
    # 描述文本块
    text_data = b"Description\x00" + description.encode("latin-1", errors="replace")
    # 拼接完整 PNG
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"tEXt", text_data)
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)


# ========== 6. 命令行主程序 ==========
def main() -> None:
    """
    主函数：解析命令行参数，生成测试图，叠加高斯噪声，执行 PDHG 去噪，
    计算 PSNR/SSIM，并保存对比图。若去噪效果低于阈值则抛出异常（用于 CI 测试）。
    """
    parser = argparse.ArgumentParser(description="TV image denoising using NumPy PDHG")
    parser.add_argument("--size", type=int, default=256, help="图像尺寸")
    parser.add_argument(
        "--noise-sigmas",
        type=float,
        nargs="+",
        default=[10, 25, 50],
        help="噪声标准差（以 0-255 像素值为单位）",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="TV 正则化权重，若不指定则根据噪声标准差自动选择（1.25*sigma）",
    )
    parser.add_argument("--iterations", type=int, default=500, help="PDHG 迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=Path, default=Path("denoise_result.png"), help="输出图片路径")

    args = parser.parse_args()

    # 生成干净图像
    clean = make_test_image(args.size)
    rows = []
    descriptions = []

    print("sigma | noisy PSNR | noisy SSIM | denoised PSNR | denoised SSIM")
    print("------|------------|------------|---------------|--------------")

    result_psnrs = []

    for index, sigma_255 in enumerate(args.noise_sigmas):
        # 添加高斯噪声（标准差转换为 [0,1] 范围）
        rng = np.random.default_rng(args.seed + index)
        sigma = sigma_255 / 255.0
        noisy = np.clip(clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)

        # 自动选择权重（经验公式）
        weight = args.weight if args.weight is not None else 1.25 * sigma

        # 去噪
        denoised = tv_denoise_pdhg(noisy, weight=weight, iterations=args.iterations)

        # 计算指标
        
        noisy_psnr, noisy_ssim = psnr(clean, noisy), ssim(clean, noisy)
        result_psnr, result_ssim = psnr(clean, denoised), ssim(clean, denoised)
        

        result_psnrs.append(result_psnr)

        # 保存该行图像（原始干净、含噪、去噪）
        rows.append((clean, noisy, denoised))
        descriptions.append(
            f"sigma={sigma_255:g}: noisy {noisy_psnr:.2f}dB/{noisy_ssim:.4f}, "
            f"denoised {result_psnr:.2f}dB/{result_ssim:.4f}"
        )

        # 打印表格行
        print(
            f"{sigma_255:5g} | {noisy_psnr:10.2f} | {noisy_ssim:10.4f} | "
            f"{result_psnr:13.2f} | {result_ssim:13.4f}"
        )

    
    save_comparison_png(
        args.output,
        rows,
        "Rows: sigma 10/25/50; columns: original/noisy/TV-PDHG. " + "; ".join(descriptions),
    )
    print(f"Saved comparison to: {args.output.resolve()}")

    
    if min(result_psnrs) < 25.0:
        raise RuntimeError(f"Denoising failed: minimum PSNR {min(result_psnrs):.2f} dB < 25 dB")


if __name__ == "__main__":
    main()