"""比较 95° 渲染裁剪后的 90° 与原 90° 渲染的差异，并按 left/front/right/back 顺序拼接为 panorama 条带（横排或竖排见 --strip_layout）。

对单帧执行：
  1. 把 render_fov95/<face>/frame_XXXXXX_render.png (例如 559x559) 居中裁剪到 target_size (默认 512x512)
  2. 与 render/<face>/frame_XXXXXX_render.png (512x512) 做 absdiff，三通道求均值得到灰度差，
     使用全局 vmax 归一化后用 cv2.COLORMAP_JET 上色（保证 4 个面共享同一色尺）
  3. 将 4 个面拼接成 3 张 panorama 输出（cropped90 / render90 / diff_jet；顺序 left→front→right→back，
     布局由 --strip_layout 控制：horizontal 为左到右，vertical 为自上而下）
  4. （默认）将四面按与 utils/cubemap.EquirectangularToCubemapConverter 一致的几何逆投影到 ERP：
     上下极区（对应立方体 top/bottom 面主导的方向）留黑，输出 3 张等距柱状图（*_erp_*.png）

仅输出上述 PNG，不写其他中间产物（可用 --no_erp 关闭第 4 步）。

Usage:
    python scripts/render_compare_panorama.py \
        --results_dir results/SynPano_apartment/2026-05-07-12-03-57 \
        --frame 40
"""

import argparse
import os
import re
from glob import glob

import cv2
import numpy as np


FACES = ["left", "front", "right", "back"]
FRAME_REGEX = re.compile(r"frame_(\d+)_render\.png$")
FRAME_DIR_REGEX = re.compile(r"frame_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare cropped 95° vs 90° renders, build 4-face strip panoramas "
            "(horizontal or vertical layout via --strip_layout), and optional "
            "lateral-only ERP reprojection (poles black)."
        )
    )
    parser.add_argument(
        "--results_dir",
        default="results/SynPano_apartment/2026-05-07-12-03-57",
        help="SLAM 结果根目录（包含 render/ 和 render_fov95/ 子目录）。",
    )
    parser.add_argument(
        "--input_layout",
        choices=("legacy", "cubemap"),
        default="legacy",
        help="输入目录结构：legacy=render/<face>/frame_xxxxxx_render.png，cubemap=frame_xxxxxx/<face>.png",
    )
    parser.add_argument(
        "--render90_dir",
        default=None,
        help="cubemap 模式的 90° 根目录（包含 frame_xxxxxx/ 子目录）。",
    )
    parser.add_argument(
        "--render95_dir",
        default=None,
        help="cubemap 模式的 95° 根目录（包含 frame_xxxxxx/ 子目录）。",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="要处理的帧索引（如 40）。默认自动取第一帧：legacy 扫 render/front，cubemap 扫 frame_xxxxxx/front.png。",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="输出目录，默认 <results_dir>/panorama_compare/。",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="裁剪后每个面的边长（与 render/ 中 90° 面分辨率一致），默认 512。",
    )
    parser.add_argument(
        "--strip_layout",
        choices=("horizontal", "vertical"),
        default="vertical",
        help=(
            "四面 panorama 拼接方向：horizontal=左→右（4F×F 像素），"
            "vertical=上→下（F×4F）。顺序均为 L/F/R/B。"
        ),
    )
    parser.add_argument(
        "--diff_max",
        default="auto",
        help="JET 归一化的最大灰度值。'auto' 用当前帧 4 个面灰度差的全局最大值；也可传整数（如 50）固定。",
    )
    parser.add_argument(
        "--title_cropped90",
        default="",
        help="叠加在 cropped90 全景图顶部的英文标题（可选）。默认不写标题。",
    )
    parser.add_argument(
        "--title_render90",
        default="",
        help="叠加在 render90 全景图顶部的英文标题（可选）。默认不写标题。",
    )
    parser.add_argument(
        "--title_diff_jet",
        default="",
        help="叠加在 diff_jet 全景图顶部的英文标题（可选）。默认不写标题。",
    )
    parser.add_argument(
        "--erp_width",
        type=int,
        default=1920,
        help="ERP 输出宽度（默认 1920，与 SynPano Calibration 一致）。",
    )
    parser.add_argument(
        "--erp_height",
        type=int,
        default=960,
        help="ERP 输出高度（默认 960）。",
    )
    parser.add_argument(
        "--no_erp",
        action="store_true",
        help="不写出四面逆投影的 ERP 图像（仅保留 panorama 条带）。",
    )
    parser.add_argument(
        "--title_cropped90_erp",
        default="",
        help="叠加在 cropped90 ERP 图顶部的英文标题（可选）。",
    )
    parser.add_argument(
        "--title_render90_erp",
        default="",
        help="叠加在 render90 ERP 图顶部的英文标题（可选）。",
    )
    parser.add_argument(
        "--title_diff_jet_erp",
        default="",
        help="叠加在 diff_jet ERP 图顶部的英文标题（可选）。",
    )
    return parser.parse_args()


def auto_pick_first_frame(render_face_dir):
    files = sorted(glob(os.path.join(render_face_dir, "frame_*_render.png")))
    frame_ids = []
    for f in files:
        m = FRAME_REGEX.search(os.path.basename(f))
        if m:
            frame_ids.append(int(m.group(1)))
    if not frame_ids:
        raise FileNotFoundError(
            f"在 {render_face_dir} 下没有找到任何 frame_XXXXXX_render.png 文件。"
        )
    return min(frame_ids)


def auto_pick_first_frame_cubemap(render90_dir):
    frame_dirs = sorted(glob(os.path.join(render90_dir, "frame_*")))
    frame_ids = []
    for d in frame_dirs:
        if not os.path.isdir(d):
            continue
        m = FRAME_DIR_REGEX.search(os.path.basename(d))
        if not m:
            continue
        front_path = os.path.join(d, "front.png")
        if os.path.isfile(front_path):
            frame_ids.append(int(m.group(1)))
    if not frame_ids:
        raise FileNotFoundError(
            f"在 {render90_dir} 下未找到任何 frame_xxxxxx/front.png。"
        )
    return min(frame_ids)


def imread_or_raise(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return img


def center_crop(img, target_size):
    h, w = img.shape[:2]
    if h < target_size or w < target_size:
        raise ValueError(
            f"目标尺寸 {target_size} 大于源图尺寸 ({h}x{w})，无法居中裁剪。"
        )
    sh = (h - target_size) // 2
    sw = (w - target_size) // 2
    return img[sh : sh + target_size, sw : sw + target_size, :]


def draw_title_top_banner(img, title):
    """在图像顶部绘制英文标题横幅；title 为空或仅空白时不绘制。"""
    label = (title or "").strip()
    if not label:
        return img

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.55, min(1.05, w / 2100.0))
    thickness = 1
    min_pad = 16

    while font_scale >= 0.38:
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
        if tw <= w - min_pad:
            break
        font_scale *= 0.9

    (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
    pad_y = 10
    bar_h = max(34, min(56, int(round(h * 0.078))), th + bl + pad_y)
    bar_h = min(bar_h, max(48, h // 5))

    text_x = max(0, (w - tw) // 2)
    # putText 的 y 为文字基线：在顶栏内垂直居中
    text_y = int((bar_h + th - bl) / 2)

    bg = (46, 46, 52)
    cv2.rectangle(img, (0, 0), (w - 1, bar_h - 1), bg, thickness=-1)

    shadow = (28, 28, 32)
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        cv2.putText(
            img,
            label,
            (text_x + dx, text_y + dy),
            font,
            font_scale,
            shadow,
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (248, 248, 252),
        thickness,
        cv2.LINE_AA,
    )
    return img


def lateral_faces_to_erp(face_bgr, erp_w, erp_h, face_size):
    """
    将 left/front/right/back 四面贴回等距柱状图（与 EquirectangularToCubemapConverter 轴向一致）。

    逆映射：列 i、行 j 使用与 grid_sample(align_corners=True) 及 cubemap 中 map1 相同的连续坐标 u=i, v=j；
    theta = (u/W - 0.5)*2*pi, phi = (v/H - 0.5)*pi，单位方向 x=cos(phi)cos(theta), y=cos(phi)sin(theta), z=sin(phi)（与
    _vector_to_ERP_pixel 的 atan2(y,x)、asin(z) 一致）。

    若 |z| 在 |x|,|y|,|z| 中为最大（含并列优先 z），该方向归 top/bottom 面，此脚本不采样，输出为黑。
    否则在对应侧面求与立方体面相交得到 (nx,ny)，再映射到 [0,F-1] 像素并双线性采样。
    """
    H, W = int(erp_h), int(erp_w)
    F = int(face_size)
    if W < 2 or H < 2:
        raise ValueError(f"ERP 尺寸过小：{W}x{H}")
    if F < 2:
        raise ValueError(f"face_size 过小：{F}")

    jj, ii = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )
    u = ii
    v = jj
    theta = (u / W - 0.5) * (2.0 * np.pi)
    phi = (v / H - 0.5) * np.pi
    cp = np.cos(phi)
    x = cp * np.cos(theta)
    y = cp * np.sin(theta)
    z = np.sin(phi)

    ax = np.abs(x)
    ay = np.abs(y)
    az = np.abs(z)

    pole = (az >= ax) & (az >= ay)
    x_side = (~pole) & (ax >= ay)
    y_side = (~pole) & (ax < ay)

    right_m = x_side & (x > 0.0)
    left_m = x_side & (x <= 0.0)
    front_m = y_side & (y > 0.0)
    back_m = y_side & (y <= 0.0)

    eps = 1e-9
    y_front = np.where(y > eps, y, eps)
    front_nx = x / y_front
    front_ny = z / y_front

    y_back = np.where(y < -eps, y, -eps)
    back_nx = x / y_back
    back_ny = -z / y_back

    x_right = np.where(x > eps, x, eps)
    right_nx = -y / x_right
    right_ny = z / x_right

    x_left = np.where(x < -eps, x, -eps)
    left_nx = -y / x_left
    left_ny = -z / x_left

    s = 0.5 * float(F - 1)
    px_front = (front_nx + 1.0) * s
    py_front = (front_ny + 1.0) * s
    px_back = (back_nx + 1.0) * s
    py_back = (back_ny + 1.0) * s
    px_right = (right_nx + 1.0) * s
    py_right = (right_ny + 1.0) * s
    px_left = (left_nx + 1.0) * s
    py_left = (left_ny + 1.0) * s

    out = np.zeros((H, W, 3), dtype=np.uint8)
    face_maps = [
        ("front", front_m, px_front, py_front),
        ("back", back_m, px_back, py_back),
        ("right", right_m, px_right, py_right),
        ("left", left_m, px_left, py_left),
    ]
    for name, mask, pxi, pyi in face_maps:
        img = face_bgr[name]
        if img.shape[0] != F or img.shape[1] != F:
            raise ValueError(f"面 {name} 尺寸应为 {F}x{F}，实为 {img.shape[:2]}")
        map_x = np.where(mask, pxi, 0.0).astype(np.float32)
        map_y = np.where(mask, pyi, 0.0).astype(np.float32)
        warped = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        out[mask] = warped[mask]
    return out


def resolve_diff_max(arg_value, diff_grays):
    if isinstance(arg_value, str) and arg_value.lower() == "auto":
        vmax = float(max(g.max() for g in diff_grays))
        if vmax < 1e-6:
            vmax = 1.0
        return vmax, "auto"
    try:
        vmax = float(arg_value)
    except (TypeError, ValueError):
        raise ValueError(f"--diff_max 应为 'auto' 或数字，收到：{arg_value!r}")
    if vmax <= 0:
        raise ValueError(f"--diff_max 必须为正数，收到：{vmax}")
    return vmax, "fixed"


def main():
    args = parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if args.input_layout == "legacy":
        render_root = os.path.join(results_dir, "render")
        render_fov95_root = os.path.join(results_dir, "render_fov95")
    else:
        if args.render90_dir is None or args.render95_dir is None:
            raise ValueError(
                "cubemap 模式必须同时指定 --render90_dir 与 --render95_dir。"
            )
        render_root = os.path.abspath(args.render90_dir)
        render_fov95_root = os.path.abspath(args.render95_dir)

    if not os.path.isdir(render_root):
        raise FileNotFoundError(f"找不到 90° 输入目录：{render_root}")
    if not os.path.isdir(render_fov95_root):
        raise FileNotFoundError(f"找不到 95° 输入目录：{render_fov95_root}")

    if args.input_layout == "legacy":
        frame_id = (
            args.frame
            if args.frame is not None
            else auto_pick_first_frame(os.path.join(render_root, "front"))
        )
    else:
        frame_id = (
            args.frame
            if args.frame is not None
            else auto_pick_first_frame_cubemap(render_root)
        )
    fname = f"frame_{frame_id:06d}_render.png"

    out_dir = args.out_dir or os.path.join(results_dir, "panorama_compare")
    os.makedirs(out_dir, exist_ok=True)

    cropped_panels = []
    render_panels = []
    diff_grays = []

    for face in FACES:
        if args.input_layout == "legacy":
            p95_path = os.path.join(render_fov95_root, face, fname)
            p90_path = os.path.join(render_root, face, fname)
        else:
            frame_dir = f"frame_{frame_id:06d}"
            p95_path = os.path.join(render_fov95_root, frame_dir, f"{face}.png")
            p90_path = os.path.join(render_root, frame_dir, f"{face}.png")

        p95 = imread_or_raise(p95_path)
        p90 = imread_or_raise(p90_path)

        cropped = center_crop(p95, args.target_size)
        if cropped.shape[:2] != p90.shape[:2]:
            raise ValueError(
                f"裁剪后尺寸 {cropped.shape[:2]} 与 render 尺寸 {p90.shape[:2]} 不一致 (face={face})。"
            )

        cropped_panels.append(cropped)
        render_panels.append(p90)

        diff_uint8 = cv2.absdiff(cropped, p90)
        diff_gray = diff_uint8.astype(np.float32).mean(axis=2)
        diff_grays.append(diff_gray)

    vmax, vmax_mode = resolve_diff_max(args.diff_max, diff_grays)

    diff_panels = []
    for diff_gray in diff_grays:
        scaled = np.clip(diff_gray * (255.0 / vmax), 0.0, 255.0).astype(np.uint8)
        diff_panels.append(cv2.applyColorMap(scaled, cv2.COLORMAP_JET))

    strip_axis = 1 if args.strip_layout == "horizontal" else 0
    cropped_pano = np.concatenate(cropped_panels, axis=strip_axis)
    render_pano = np.concatenate(render_panels, axis=strip_axis)
    diff_pano = np.concatenate(diff_panels, axis=strip_axis)

    cropped_path = os.path.join(out_dir, f"cropped90_panorama_frame_{frame_id:06d}.png")
    render_path = os.path.join(out_dir, f"render90_panorama_frame_{frame_id:06d}.png")
    diff_path = os.path.join(out_dir, f"diff_jet_panorama_frame_{frame_id:06d}.png")

    cropped_out = cropped_pano.copy()
    render_out = render_pano.copy()
    diff_out = diff_pano.copy()
    draw_title_top_banner(cropped_out, args.title_cropped90)
    draw_title_top_banner(render_out, args.title_render90)
    draw_title_top_banner(diff_out, args.title_diff_jet)

    cv2.imwrite(cropped_path, cropped_out)
    cv2.imwrite(render_path, render_out)
    cv2.imwrite(diff_path, diff_out)

    print(f"frame_id        : {frame_id}")
    print(f"face order      : {FACES}")
    print(f"target_size     : {args.target_size} (per face)")
    print(f"strip_layout    : {args.strip_layout}")
    print(f"panorama size   : {cropped_pano.shape[1]} x {cropped_pano.shape[0]}")
    print(f"diff vmax       : {vmax:.4f} ({vmax_mode})")
    print(f"saved cropped90 : {cropped_path}")
    print(f"saved render90  : {render_path}")
    print(f"saved diff_jet  : {diff_path}")

    if not args.no_erp:
        face_cropped = dict(zip(FACES, cropped_panels))
        face_render = dict(zip(FACES, render_panels))
        face_diff = dict(zip(FACES, diff_panels))

        ew, eh = args.erp_width, args.erp_height
        erp_cropped = lateral_faces_to_erp(face_cropped, ew, eh, args.target_size)
        erp_render = lateral_faces_to_erp(face_render, ew, eh, args.target_size)
        erp_diff = lateral_faces_to_erp(face_diff, ew, eh, args.target_size)

        cropped_erp_path = os.path.join(
            out_dir, f"cropped90_erp_frame_{frame_id:06d}.png"
        )
        render_erp_path = os.path.join(
            out_dir, f"render90_erp_frame_{frame_id:06d}.png"
        )
        diff_erp_path = os.path.join(
            out_dir, f"diff_jet_erp_frame_{frame_id:06d}.png"
        )

        erp_c_out = erp_cropped.copy()
        erp_r_out = erp_render.copy()
        erp_d_out = erp_diff.copy()
        draw_title_top_banner(erp_c_out, args.title_cropped90_erp)
        draw_title_top_banner(erp_r_out, args.title_render90_erp)
        draw_title_top_banner(erp_d_out, args.title_diff_jet_erp)

        cv2.imwrite(cropped_erp_path, erp_c_out)
        cv2.imwrite(render_erp_path, erp_r_out)
        cv2.imwrite(diff_erp_path, erp_d_out)

        print(f"erp size        : {ew} x {eh} (lateral faces only, poles black)")
        print(f"saved cropped90 erp : {cropped_erp_path}")
        print(f"saved render90 erp  : {render_erp_path}")
        print(f"saved diff_jet erp  : {diff_erp_path}")


if __name__ == "__main__":
    main()
