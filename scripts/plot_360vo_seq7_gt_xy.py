#!/usr/bin/env python3
"""
Plot 360VO seq7 GT trajectory on the XY plane.

GT format (datasets/360VO/seq7/groundtruth.txt):
  #frame name x y z qx qy qz qw
  0 Frame_00000_FinalColor.png 0.0 0.0 0.0 0.0 0.0 0.0 1.0
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # safe default (no GUI needed)

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_360vo_groundtruth_xy(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # expected: frame_id, name, x, y, z, qx, qy, qz, qw
            if len(parts) < 9:
                continue
            try:
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Plot 360VO seq7 GT XY trajectory.")
    parser.add_argument(
        "--gt",
        type=str,
        default="datasets/360VO/seq7/groundtruth.txt",
        help="Path to groundtruth.txt (default: datasets/360VO/seq7/groundtruth.txt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/gt_xy_seq7.png",
        help="Output PNG path (default: results/gt_xy_seq7.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (uses current matplotlib backend).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="360VO seq7 GT trajectory (XY plane)",
        help="Plot title",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    x, y = load_360vo_groundtruth_xy(gt_path)
    if x.size < 2:
        raise RuntimeError(f"Not enough GT points parsed from {gt_path} (got {x.size})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y, linewidth=1.5)
    ax.scatter([x[0]], [y[0]], s=30, label="start")
    ax.scatter([x[-1]], [y[-1]], s=30, label="end")
    ax.set_title(args.title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()


