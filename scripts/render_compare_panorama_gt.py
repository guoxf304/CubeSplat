#!/usr/bin/env python3
"""将 GT ERP 全景拆成立方体四面（左/前/右/后），与 SLAM 保存的 render 四面做对比，输出条带与可选 ERP 拼图。

GT 路径：<results_dir>/gt/rgb/<rgb_dir_name>/frame_XXXXXX.png
Render 路径：<results_dir>/render/<face>/frame_XXXXXX_render.png

立方体拆分使用 py360convert.e2c（与 utils/cubemap.ERPToCube 一致的字典键 F/B/L/R/U/D），仅取侧面 L/F/R/B。

Usage:
    python scripts/render_compare_panorama_gt.py \\
        --results_dir results/SynPano_apartment/2026-05-07-12-03-57 \\
        --rgb_dir_name apartment \\
        --frame 40
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
from glob import glob

import cv2
import numpy as np
import py360convert


FACES = ["left", "front", "right", "back"]
# render 结果帧文件名
FRAME_RENDER_RE = re.compile(r"frame_(\d+)_render\.png$")


def _load_render_compare_panorama():
    """同目录下加载 render_compare_panorama.py 中的共用函数（避免重复维护几何代码）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "render_compare_panorama.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"需要同目录脚本：{path}")
    spec = importlib.util.spec_from_file_location("render_compare_panorama", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare GT ERP (e2c lateral faces) vs render/<face>/ outputs: "
            "panorama strips + optional lateral-only ERP (same helpers as render_compare_panorama)."
        )
    )
    parser.add_argument(
        "--results_dir",
        default="results/SynPano_apartment/2026-05-07-12-03-57",
        help="SLAM 结果根目录（需含 render/ 与 gt/rgb/<name>/）。",
    )
    parser.add_argument(
        "--rgb_dir_name",
        required=True,
        help="gt/rgb/ 下子目录名，例如 apartment（完整路径为 gt/rgb/<name>/）。",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="帧索引。默认取 render/front/ 下最小 frame 号。",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="输出目录，默认 <results_dir>/panorama_compare_gt/",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="立方体面边长，须与 render 中一面分辨率一致，默认 512。",
    )
    parser.add_argument(
        "--strip_layout",
        choices=("horizontal", "vertical"),
        default="vertical",
        help="四面条带拼接方向：horizontal=左→右，vertical=上→下（与 render_compare_panorama 相同）。",
    )
    parser.add_argument(
        "--diff_max",
        default="auto",
        help="JET 归一化：'auto' 或正数（与 render_compare_panorama 相同）。",
    )
    parser.add_argument(
        "--title_gt",
        default="",
        help="叠加在 GT 条带图顶部的英文标题（可选）。",
    )
    parser.add_argument(
        "--title_render",
        default="",
        help="叠加在 render 条带图顶部的英文标题（可选）。",
    )
    parser.add_argument(
        "--title_diff_jet",
        default="",
        help="叠加在 diff_jet 条带图顶部的英文标题（可选）。",
    )
    parser.add_argument("--erp_width", type=int, default=1920, help="ERP 宽度。")
    parser.add_argument("--erp_height", type=int, default=960, help="ERP 高度。")
    parser.add_argument(
        "--no_erp",
        action="store_true",
        help="不写出三面 ERP 拼图。",
    )
    parser.add_argument("--title_gt_erp", default="", help="GT 拼图顶栏标题（可选）。")
    parser.add_argument("--title_render_erp", default="", help="render 拼图顶栏标题（可选）。")
    parser.add_argument("--title_diff_jet_erp", default="", help="diff 拼图顶栏标题（可选）。")
    args = parser.parse_args()
    args._rcp = _load_render_compare_panorama()
    return args


def auto_pick_first_frame_render(render_face_dir):
    files = sorted(glob(os.path.join(render_face_dir, "frame_*_render.png")))
    ids = []
    for f in files:
        m = FRAME_RENDER_RE.search(os.path.basename(f))
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise FileNotFoundError(f"{render_face_dir} 下无 frame_*_render.png")
    return min(ids)


def imread_or_raise(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取：{path}")
    return img


def erp_bgr_to_lateral_faces(erp_bgr, face_w):
    """ERP BGR -> 四面 dict，键为 left/front/right/back。与 ERPToCube 使用相同 e2c 映射。"""
    erp_rgb = cv2.cvtColor(erp_bgr, cv2.COLOR_BGR2RGB)
    c_img = py360convert.e2c(
        erp_rgb, face_w=int(face_w), mode="bicubic", cube_format="dict"
    )
    return {
        "left": cv2.cvtColor(c_img["L"], cv2.COLOR_RGB2BGR),
        "front": cv2.cvtColor(c_img["F"], cv2.COLOR_RGB2BGR),
        "right": cv2.cvtColor(c_img["R"], cv2.COLOR_RGB2BGR),
        "back": cv2.cvtColor(c_img["B"], cv2.COLOR_RGB2BGR),
    }


def main():
    args = parse_args()
    rcp = args._rcp

    results_dir = os.path.abspath(args.results_dir)
    render_root = os.path.join(results_dir, "render")
    gt_rgb_dir = os.path.join(results_dir, "gt", "rgb", args.rgb_dir_name)

    if not os.path.isdir(render_root):
        raise FileNotFoundError(f"缺少 render 目录：{render_root}")
    if not os.path.isdir(gt_rgb_dir):
        raise FileNotFoundError(f"缺少 GT 目录：{gt_rgb_dir}")

    frame_id = (
        args.frame
        if args.frame is not None
        else auto_pick_first_frame_render(os.path.join(render_root, "front"))
    )
    gt_name = f"frame_{frame_id:06d}.png"
    render_name = f"frame_{frame_id:06d}_render.png"

    gt_path = os.path.join(gt_rgb_dir, gt_name)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"缺少 GT ERP：{gt_path}")

    erp_gt = imread_or_raise(gt_path)
    gt_faces = erp_bgr_to_lateral_faces(erp_gt, args.target_size)

    out_dir = args.out_dir or os.path.join(results_dir, "panorama_compare_gt")
    os.makedirs(out_dir, exist_ok=True)

    gt_panels = []
    render_panels = []
    diff_grays = []

    for face in FACES:
        gt_face = gt_faces[face]
        if gt_face.shape[0] != args.target_size or gt_face.shape[1] != args.target_size:
            raise ValueError(
                f"GT 面 {face} 尺寸 {gt_face.shape[:2]} 与 target_size={args.target_size} 不符"
            )

        r_path = os.path.join(render_root, face, render_name)
        if not os.path.isfile(r_path):
            raise FileNotFoundError(f"缺少 render：{r_path}")
        rend = imread_or_raise(r_path)
        if rend.shape[:2] != gt_face.shape[:2]:
            raise ValueError(
                f"render {face} 尺寸 {rend.shape[:2]} 与 GT 面 {gt_face.shape[:2]} 不一致"
            )

        gt_panels.append(gt_face)
        render_panels.append(rend)

        d = cv2.absdiff(gt_face, rend)
        diff_grays.append(d.astype(np.float32).mean(axis=2))

    vmax, vmax_mode = rcp.resolve_diff_max(args.diff_max, diff_grays)

    diff_panels = []
    for g in diff_grays:
        s = np.clip(g * (255.0 / vmax), 0.0, 255.0).astype(np.uint8)
        diff_panels.append(cv2.applyColorMap(s, cv2.COLORMAP_JET))

    strip_axis = 1 if args.strip_layout == "horizontal" else 0
    gt_pano = np.concatenate(gt_panels, axis=strip_axis)
    render_pano = np.concatenate(render_panels, axis=strip_axis)
    diff_pano = np.concatenate(diff_panels, axis=strip_axis)

    gt_p_path = os.path.join(out_dir, f"gt_face_panorama_frame_{frame_id:06d}.png")
    r_p_path = os.path.join(out_dir, f"render_face_panorama_frame_{frame_id:06d}.png")
    d_p_path = os.path.join(out_dir, f"diff_jet_panorama_frame_{frame_id:06d}.png")

    o1, o2, o3 = gt_pano.copy(), render_pano.copy(), diff_pano.copy()
    rcp.draw_title_top_banner(o1, args.title_gt)
    rcp.draw_title_top_banner(o2, args.title_render)
    rcp.draw_title_top_banner(o3, args.title_diff_jet)

    cv2.imwrite(gt_p_path, o1)
    cv2.imwrite(r_p_path, o2)
    cv2.imwrite(d_p_path, o3)

    print(f"gt erp path     : {gt_path}")
    print(f"frame_id        : {frame_id}")
    print(f"face order      : {FACES}")
    print(f"target_size     : {args.target_size}")
    print(f"strip_layout    : {args.strip_layout}")
    print(f"panorama size   : {gt_pano.shape[1]} x {gt_pano.shape[0]}")
    print(f"diff vmax       : {vmax:.4f} ({vmax_mode})")
    print(f"saved gt strip  : {gt_p_path}")
    print(f"saved render    : {r_p_path}")
    print(f"saved diff_jet  : {d_p_path}")

    if not args.no_erp:
        fg = dict(zip(FACES, gt_panels))
        fr = dict(zip(FACES, render_panels))
        fd = dict(zip(FACES, diff_panels))

        ew, eh = args.erp_width, args.erp_height
        e_gt = rcp.lateral_faces_to_erp(fg, ew, eh, args.target_size)
        e_rd = rcp.lateral_faces_to_erp(fr, ew, eh, args.target_size)
        e_df = rcp.lateral_faces_to_erp(fd, ew, eh, args.target_size)

        p_gt = os.path.join(out_dir, f"gt_face_erp_frame_{frame_id:06d}.png")
        p_rd = os.path.join(out_dir, f"render_face_erp_frame_{frame_id:06d}.png")
        p_df = os.path.join(out_dir, f"diff_jet_erp_frame_{frame_id:06d}.png")

        a, b, c = e_gt.copy(), e_rd.copy(), e_df.copy()
        rcp.draw_title_top_banner(a, args.title_gt_erp)
        rcp.draw_title_top_banner(b, args.title_render_erp)
        rcp.draw_title_top_banner(c, args.title_diff_jet_erp)

        cv2.imwrite(p_gt, a)
        cv2.imwrite(p_rd, b)
        cv2.imwrite(p_df, c)

        print(f"erp size        : {ew} x {eh} (lateral only, poles black)")
        print(f"saved gt erp    : {p_gt}")
        print(f"saved render erp: {p_rd}")
        print(f"saved diff erp  : {p_df}")


if __name__ == "__main__":
    main()
