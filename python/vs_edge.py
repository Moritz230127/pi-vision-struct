#!/usr/bin/env python3
"""vs_edge.py — V3 亚像素边缘检测传感器（Devernay 亚像素 + Hough 线段拟合）。

纯算法，零模型。输出 schema v3 元素：
  - 亚像素边缘点（type=edge_point）
  - Hough 拟合线段（type=line_segment）
  - 轮廓（type=contour）

用法:
  vs_edge.py --image PATH [--canny-low 50] [--canny-high 150]
             [--hough-threshold 40] [--max-lines 50]
"""
import argparse
import json
import sys
from typing import Any

import numpy as np

import vs_schema as S


def devernay_subpixel(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Devernay 亚像素边缘：Canny 梯度方向上的二次插值。

    返回亚像素边缘点 (N, 2) 数组（x, y 浮点坐标）。
    """
    from scipy import ndimage  # type: ignore[import-not-found]

    # 高斯平滑
    g = ndimage.gaussian_filter(gray.astype(np.float64), sigma)
    # 梯度
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    # 非极大值抑制（沿梯度方向）
    h, w = g.shape
    suppressed = np.zeros_like(mag)
    # 梯度方向量化到 4 方向
    angle = np.arctan2(gy, gx)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]
            # 量化方向
            if -np.pi / 8 <= a < np.pi / 8 or a >= 7 * np.pi / 8 or a < -7 * np.pi / 8:
                n1, n2 = mag[y, x - 1], mag[y, x + 1]
            elif np.pi / 8 <= a < 3 * np.pi / 8:
                n1, n2 = mag[y - 1, x - 1], mag[y + 1, x + 1]
            elif 3 * np.pi / 8 <= a < 5 * np.pi / 8 or -5 * np.pi / 8 <= a < -3 * np.pi / 8:
                n1, n2 = mag[y - 1, x], mag[y + 1, x]
            else:
                n1, n2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            if mag[y, x] >= n1 and mag[y, x] >= n2:
                suppressed[y, x] = mag[y, x]
    # 细化：梯度正方向相邻像素若 mag >= 当前 → 本像素非真峰（对称峰只留一个）
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if suppressed[y, x] == 0:
                continue
            a = angle[y, x]
            dxn, dyn = np.cos(a), np.sin(a)
            x2n, y2n = int(x + dxn), int(y + dyn)
            if 0 <= x2n < w and 0 <= y2n < h and suppressed[y2n, x2n] >= suppressed[y, x]:
                suppressed[y, x] = 0
    # 亚像素插值（像素值半值法）：沿梯度方向找亮度半值位置
    # 比 mag 质心/抛物线更精确（mag 峰值 ≠ 边缘位置，半值才是）
    pts = []
    ys, xs = np.nonzero(suppressed > 0)
    for y, x in zip(ys, xs):
        a = angle[y, x]
        dx, dy = np.cos(a), np.sin(a)
        # 沿梯度方向采样亮度（±3 像素，位置用像素中心坐标）
        samples = []
        for k in range(-3, 4):
            sx = int(round(x + k * dx))
            sy = int(round(y + k * dy))
            if 0 <= sx < w and 0 <= sy < h:
                samples.append((x + k * dx + 0.5, gray[sy, sx]))
        if len(samples) < 2:
            continue
        # 找亮度跨越半值的位置（暗→亮方向）
        v_min = min(v for _, v in samples)
        v_max = max(v for _, v in samples)
        if v_max - v_min < 1e-6:
            continue
        half = (v_min + v_max) / 2.0
        for i in range(len(samples) - 1):
            p1, v1 = samples[i]
            p2, v2 = samples[i + 1]
            if (v1 - half) * (v2 - half) <= 0 and v2 != v1:
                t = (half - v1) / (v2 - v1)
                pts.append((p1 + t * (p2 - p1), y))
                break
    return np.array(pts) if pts else np.zeros((0, 2))


def hough_lines(edge_pts: np.ndarray, img_shape: tuple[int, int],
                threshold: int = 40, max_lines: int = 50) -> list[dict[str, Any]]:
    """Hough 变换线段拟合（极坐标累加器）。"""
    h, w = img_shape
    if len(edge_pts) < 2:
        return []
    # 极坐标参数空间
    diag = int(np.hypot(h, w))
    theta_res = 1  # 度
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rho_max = diag
    accumulator = np.zeros((2 * rho_max + 1, len(thetas)), dtype=np.int32)
    xs = edge_pts[:, 0]
    ys = edge_pts[:, 1]
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        for j, th in enumerate(thetas):
            rho = x * np.cos(th) + y * np.sin(th)
            rho_idx = int(round(rho)) + rho_max
            if 0 <= rho_idx < accumulator.shape[0]:
                accumulator[rho_idx, j] += 1
    # 找峰值
    lines = []
    acc_flat = accumulator.flatten()
    for _ in range(max_lines):
        idx = int(np.argmax(acc_flat))
        if acc_flat[idx] < threshold:
            break
        rho_idx, th_idx = divmod(idx, len(thetas))
        rho = rho_idx - rho_max
        theta = thetas[th_idx]
        # 抑制邻域
        r0, r1 = max(0, rho_idx - 5), min(accumulator.shape[0], rho_idx + 6)
        t0, t1 = max(0, th_idx - 5), min(len(thetas), th_idx + 6)
        acc_flat = acc_flat.reshape(accumulator.shape)
        acc_flat[r0:r1, t0:t1] = 0
        acc_flat = acc_flat.flatten()
        # 线段端点（沿直线找边缘点范围）
        votes = acc_flat[idx] if False else accumulator[rho_idx, th_idx]
        lines.append({
            "rho": float(rho), "theta": float(np.rad2deg(theta)),
            "votes": int(votes),
        })
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--hough-threshold", type=int, default=40)
    ap.add_argument("--max-lines", type=int, default=50)
    ap.add_argument("--max-points", type=int, default=2000)
    args = ap.parse_args()

    try:
        from PIL import Image
        im = Image.open(args.image).convert("L")
        w, h = im.size
        gray = np.array(im, dtype=np.float64)

        # 亚像素边缘
        pts = devernay_subpixel(gray, args.sigma)
        if len(pts) > args.max_points:
            # 均匀下采样
            step = len(pts) // args.max_points
            pts = pts[::step]

        # Hough 线段
        lines = hough_lines(pts, (h, w), args.hough_threshold, args.max_lines)

        elements = []
        # 边缘点元素（下采样到 ≤200 个）
        for i, (x, y) in enumerate(pts[:200]):
            elements.append(S.element(i, "edge_point", [round(x), round(y), round(x) + 1, round(y) + 1],
                                      conf=1.0, source=["edge"], coordsys="image_px"))
        # 线段元素
        for i, ln in enumerate(lines):
            elements.append(S.element(200 + i, "line_segment",
                                      [0, 0, 0, 0],  # bbox 由 rho/theta 表示
                                      conf=min(1.0, ln["votes"] / 100.0),
                                      source=["edge"], coordsys="image_px"))
            elements[-1]["line"] = ln

        report = S.envelope(task="edge", sensors=["edge"], coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h], "sigma": args.sigma})
        report["schema"] = "vision-report/v3"
        report["elements"] = elements
        report["metrics"] = {
            "edge_points": len(pts),
            "lines": len(lines),
            "subpixel": True,
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_edge failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
