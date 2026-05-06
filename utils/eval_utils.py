# 在导入matplotlib之前设置环境变量，避免Tkinter问题
import os
os.environ['MPLBACKEND'] = 'Agg'  # matplotlib使用非交互式后端
os.environ['PIL_USE_TKINTER'] = '0'  # 禁用PIL的Tkinter支持

import json

import cv2
import evo
import numpy as np
import torch
from bitarray import bitarray
from os.path import join
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import matplotlib
matplotlib.use('Agg')  # 显式设置非交互式后端，确保不打开窗口
from matplotlib import pyplot as plt
import math
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix
from utils.logging_utils import Log


def dec2binary(x, n_bits=None):
    """将十进制整数x转换为二进制表示
    
    代码来源: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    
    Args:
        x: 输入的十进制整数张量
        n_bits: 二进制位数，如果为None则自动计算
        
    Returns:
        二进制表示的张量
    """
    if x.numel() == 0:
        return torch.empty((0, 0), dtype=torch.bool, device=x.device)
    if n_bits is None:
        max_val = x.max()
        if max_val == 0:
            n_bits = 1
        else:
            n_bits = torch.ceil(torch.log2(max_val + 1)).type(torch.int64)
    if n_bits == 0:
        n_bits = 1
    mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)


def save_kmeans(gaussians, quantized_params, out_dir):
    """保存k-means的码本和索引
    
    Args:
        gaussians: GaussianModel对象
        quantized_params: 量化参数类型列表
        out_dir: 输出目录
    """
    mkdir_p(out_dir)
    
    # 转换为bitarray对象以保存压缩版本
    # 保存为npy或pth格式会为每个索引使用8位（或布尔值）
    # 转换为二进制，连接所有参数的索引并保存
    bitarray_all = bitarray([])
    param_info = {}
    current_offset = 0
    
    for param in quantized_params:
        # 检查该参数是否真的被量化了（centers不为None）
        if (gaussians.quantization_centers.get(param) is None or 
            gaussians.quantization_indices.get(param) is None):
            continue
            
        indices = gaussians.quantization_indices[param]
        n_clusters = gaussians.quantization_centers[param].shape[0]
        n_bits = int(np.ceil(np.log2(n_clusters)))
        num_indices = indices.shape[0]
        
        # 将索引转换为二进制
        assignments = dec2binary(indices.cpu(), n_bits)
        bitarr = bitarray(list(assignments.numpy().flatten()))
        bitarray_all.extend(bitarr)
        
        # 记录参数信息
        param_info[param] = {
            'offset': current_offset,
            'n_bits': n_bits,
            'num_indices': num_indices,
            'n_clusters': n_clusters
        }
        current_offset += num_indices * n_bits
    
    # 保存索引的二进制文件
    if len(bitarray_all) > 0:
        with open(join(out_dir, 'kmeans_inds.bin'), 'wb') as file:
            bitarray_all.tofile(file)
    
    # 保存加载所需的详细信息
    args_dict = {
        'params': list(param_info.keys()),  # 只保存实际被量化的参数
        'param_info': param_info,
        'total_len': len(bitarray_all)
    }
    np.save(join(out_dir, 'kmeans_args.npy'), args_dict)
    
    # 保存码本
    centers_dict = {}
    for param in param_info.keys():  # 只保存实际被量化的参数
        if gaussians.quantization_centers.get(param) is not None:
            centers_dict[param] = gaussians.quantization_centers[param].cpu()
    
    if len(centers_dict) > 0:
        torch.save(centers_dict, join(out_dir, 'kmeans_centers.pth'))


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, coordinate_transform=False):
    # 检查输入数据是否有效
    if len(poses_gt) < 2 or len(poses_est) < 2:
        Log(f"Warning: Not enough poses for evaluation (gt: {len(poses_gt)}, est: {len(poses_est)}, need at least 2)")
        return None
    
    if len(poses_gt) != len(poses_est):
        Log(f"Warning: Mismatch in pose counts (gt: {len(poses_gt)}, est: {len(poses_est)})")
        # 取较小的长度
        min_len = min(len(poses_gt), len(poses_est))
        poses_gt = poses_gt[:min_len]
        poses_est = poses_est[:min_len]
    
    ## 坐标系转换：将实验轨迹从左手坐标系转换为右手坐标系
    # 只有在需要时才进行转换（例如360VO需要，PAL不需要）
    if coordinate_transform:
        # 对Z轴取负（常见的左手到右手坐标系转换）
        # 如果需要对其他轴取负，可以修改这里的索引：0=X轴, 1=Y轴, 2=Z轴
        poses_est_converted = []
        for pose in poses_est:
            pose_converted = pose.copy()
            # 对平移部分的Z轴取负
            pose_converted[2, 3] = - pose_converted[2, 3]
            # 对旋转矩阵的Z轴相关部分取负（保持旋转一致性）
            # pose_converted[0:3, 2] = -pose_converted[0:3, 2]  # Z轴列向量取负
            poses_est_converted.append(pose_converted)
        poses_est = poses_est_converted
    
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE [m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    # 使用非交互式后端创建图形，确保不打开窗口
    fig = plt.figure()
    # 确保matplotlib不会尝试显示图形
    plt.ioff()  # 关闭交互模式
    
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    
    # 尝试使用 traj_colormap，如果失败则使用简单的轨迹绘制
    try:
        evo.tools.plot.traj_colormap(
            ax,
            traj_est_aligned,
            ape_metric.error,
            plot_mode,
            min_map=ape_stats["min"],
            max_map=ape_stats["max"],
        )
    except ValueError as e:
        print(f"Warning: traj_colormap failed ({e}), using simple trajectory plot")
        evo.tools.plot.traj(ax, plot_mode, traj_est_aligned, "-", "blue", "est")
    
    ax.legend()
    # 保存图片，不显示
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)  # 关闭图形以释放内存

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, coordinate_transform=False):
    # 处理 frames 可能是字典或列表的情况
    if isinstance(frames, dict):
        # 如果是字典，获取所有帧索引并排序
        all_frame_indices = sorted(frames.keys())
        latest_frame_idx = max(all_frame_indices) if all_frame_indices else -1
    else:
        # 如果是列表，使用原来的逻辑
        all_frame_indices = list(range(len(frames)))
        latest_frame_idx = len(frames) - 1

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    
    # 评估关键帧（使用原始命名）
    trj_data_kf = dict()
    trj_id_kf, trj_est_kf, trj_gt_kf = [], [], []
    trj_est_np_kf, trj_gt_np_kf = [], []
    
    for idx in kf_ids:
        # 检查索引是否在 frames 中存在
        if isinstance(frames, dict):
            if idx not in frames:
                continue
            frame = frames[idx]
        else:
            if idx >= len(frames):
                continue
            frame = frames[idx]
        
        pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(frame.R_gt, frame.T_gt))

        trj_id_kf.append(frame.uid)
        trj_est_kf.append(pose_est.tolist())
        trj_gt_kf.append(pose_gt.tolist())

        trj_est_np_kf.append(pose_est)
        trj_gt_np_kf.append(pose_gt)

    # 检查是否有足够的数据进行评估（至少需要2个点）
    if len(trj_est_np_kf) < 2:
        Log(f"Warning: Not enough keyframes for ATE evaluation (got {len(trj_est_np_kf)}, need at least 2)")
        ate_kf = None
    else:
        trj_data_kf["trj_id"] = trj_id_kf
        trj_data_kf["trj_est"] = trj_est_kf
        trj_data_kf["trj_gt"] = trj_gt_kf

        with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(trj_data_kf, f, indent=4)

        ate_kf = evaluate_evo(
            poses_gt=trj_gt_np_kf,
            poses_est=trj_est_np_kf,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
            coordinate_transform=coordinate_transform,
        )
    
    # 评估所有帧（使用带_all后缀的命名）
    trj_data_all = dict()
    trj_id_all, trj_est_all, trj_gt_all = [], [], []
    trj_est_np_all, trj_gt_np_all = [], []
    
    # 遍历所有实际存在的帧
    for idx in all_frame_indices:
        if isinstance(frames, dict):
            frame = frames[idx]
        else:
            frame = frames[idx]
        
        pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(frame.R_gt, frame.T_gt))

        trj_id_all.append(frame.uid)
        trj_est_all.append(pose_est.tolist())
        trj_gt_all.append(pose_gt.tolist())

        trj_est_np_all.append(pose_est)
        trj_gt_np_all.append(pose_gt)

    # 检查是否有足够的数据进行评估（至少需要2个点）
    if len(trj_est_np_all) < 2:
        Log(f"Warning: Not enough frames for ATE evaluation (got {len(trj_est_np_all)}, need at least 2)")
        ate_all = None
    else:
        trj_data_all["trj_id"] = trj_id_all
        trj_data_all["trj_est"] = trj_est_all
        trj_data_all["trj_gt"] = trj_gt_all

        label_evo_all = f"{label_evo}_all"
        with open(
            os.path.join(plot_dir, f"trj_{label_evo_all}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(trj_data_all, f, indent=4)

        ate_all = evaluate_evo(
            poses_gt=trj_gt_np_all,
            poses_est=trj_est_np_all,
            plot_dir=plot_dir,
            label=label_evo_all,
            monocular=monocular,
            coordinate_transform=coordinate_transform,
        )
    
    # 记录到wandb（使用关键帧的ATE作为主要指标，保持向后兼容）
    wandb.log({
        "frame_idx": latest_frame_idx, 
        "ate": ate_kf,
        "ate_all": ate_all
    })
    
    Log(f"ATE (keyframes): {ate_kf}, ATE (all frames): {ate_all}", tag="Eval")
    
    # 返回包含关键帧和所有帧评估结果的字典
    result = {
        "ate_kf": ate_kf,
        "ate_all": ate_all,
        "kf_ids": kf_ids,
        "kf_count": len(kf_ids),
        "total_frames": len(frames)
    }
    
    return result


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    cube=None,
):
    """
    评估渲染质量（cubemap模式，每个面分别计算指标）
    
    Args:
        frames: 相机帧列表或字典
        gaussians: 高斯点云模型
        dataset: 数据集
        save_dir: 保存目录
        pipe: 渲染管道参数
        background: 背景颜色
        kf_indices: 关键帧索引列表
        iteration: 迭代标识（"before_opt", "after_opt", "final"）
    """
    # 处理 frames 可能是字典或列表的情况
    if isinstance(frames, dict):
        # 如果是字典，获取所有帧索引并排序
        all_frame_indices = sorted(frames.keys())
        if len(all_frame_indices) == 0:
            Log("Warning: No frames available for rendering evaluation")
            return {}
        # 获取第一个帧来确定 faces
        first_frame_idx = all_frame_indices[0]
        first_frame = frames[first_frame_idx]
    else:
        # 如果是列表，使用原来的逻辑
        all_frame_indices = list(range(len(frames)))
        if len(frames) == 0:
            Log("Warning: No frames available for rendering evaluation")
            return {}
        first_frame = frames[0]
    
    faces = first_frame.keep_faces

    # 为每个面初始化指标数组
    face_metrics = {face: {"psnr": [], "ssim": [], "lpips": []} for face in faces}
    # 为每个面初始化每帧指标记录（用于after_opt时保存详细指标）
    face_frame_metrics = {face: [] for face in faces}
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    # 如果是after_opt，创建图像保存目录
    save_images = (iteration == "after_opt")
    if save_images:
        # 创建render和gt两个文件夹
        render_save_dir = os.path.join(save_dir, "render")
        gt_save_dir = os.path.join(save_dir, "gt")
        mkdir_p(render_save_dir)
        mkdir_p(gt_save_dir)
        # 为每个面在render和gt文件夹下创建子文件夹
        for face_key in faces:
            render_face_dir = os.path.join(render_save_dir, face_key)
            gt_face_dir = os.path.join(gt_save_dir, face_key)
            mkdir_p(render_face_dir)
            mkdir_p(gt_face_dir)

    # 确定结束索引
    if iteration == "final" or iteration == "before_opt" or iteration == "after_opt":
        end_idx = all_frame_indices[-1] if all_frame_indices else 0
    else:
        # 如果 iteration 是整数，使用它作为结束索引
        try:
            end_idx = int(iteration)
        except (ValueError, TypeError):
            # 如果无法转换为整数，使用最后一个帧索引
            end_idx = all_frame_indices[-1] if all_frame_indices else 0
    
    # 确保 end_idx 是整数类型
    end_idx = int(end_idx)
    
    # 遍历所有帧索引
    for idx in all_frame_indices:
        # 跳过超过结束索引的帧
        if idx > end_idx:
            break
        # 跳过关键帧
        if idx in kf_indices:
            continue
        # 按间隔采样
        if (idx - all_frame_indices[0]) % 5 != 0:
            continue
        
        # 获取帧
        if isinstance(frames, dict):
            if idx not in frames:
                continue
            frame = frames[idx]
        else:
            if idx >= len(frames):
                continue
            frame = frames[idx]
        
        frame_uid = frame.uid if hasattr(frame, "uid") else idx

        # 从数据集读取GT，并转换为cubemap
        try:
            gt_image_full = dataset[frame_uid][0]
            cubemap_gt = cube.convert(gt_image_full)
        except Exception:
            continue
        if cubemap_gt is None or not isinstance(cubemap_gt, dict):
            continue

        # 对每个面分别计算指标
        for face_key in faces:
            if face_key not in cubemap_gt:
                continue

            # 获取该面的真实图像，并确保范围在[0,1]
            gt_image = cubemap_gt[face_key]
            if isinstance(gt_image, np.ndarray):
                gt_image = torch.from_numpy(gt_image)
            gt_image = gt_image.to(device=background.device if isinstance(background, torch.Tensor) else "cuda", dtype=torch.float32)

            gt_image = torch.clamp(gt_image, 0.0, 1.0)

            rendering = render(frame, gaussians, pipe, background, face_key=face_key)["render"]
            image = torch.clamp(rendering, 0.0, 1.0)

            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
            lpips_score = cal_lpips(image.unsqueeze(0), gt_image.unsqueeze(0))

            psnr_value = psnr_score.item()
            ssim_value = ssim_score.item()
            lpips_value = lpips_score.item()

            face_metrics[face_key]["psnr"].append(psnr_value)
            face_metrics[face_key]["ssim"].append(ssim_value)
            face_metrics[face_key]["lpips"].append(lpips_value)
            
            # 如果是after_opt，保存图像和记录每帧指标
            if save_images:
                # 保存渲染图像和GT图像
                # 将tensor转换为numpy: (C, H, W) -> (H, W, C)
                # 使用detach()分离计算图，避免梯度计算
                image_np = image.detach().cpu().permute(1, 2, 0).numpy()
                gt_image_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
                
                # 转换为uint8格式
                image_uint8 = (image_np * 255.0).clip(0, 255).astype(np.uint8)
                gt_image_uint8 = (gt_image_np * 255.0).clip(0, 255).astype(np.uint8)
                
                # 转换为BGR格式（cv2使用BGR）
                image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
                gt_image_bgr = cv2.cvtColor(gt_image_uint8, cv2.COLOR_RGB2BGR)
                
                # 保存图像：渲染图像保存到render文件夹，GT图像保存到gt文件夹
                render_face_dir = os.path.join(render_save_dir, face_key)
                gt_face_dir = os.path.join(gt_save_dir, face_key)
                render_path = os.path.join(render_face_dir, f"frame_{frame_uid:06d}_render.png")
                gt_path = os.path.join(gt_face_dir, f"frame_{frame_uid:06d}_gt.png")
                cv2.imwrite(render_path, image_bgr)
                cv2.imwrite(gt_path, gt_image_bgr)
                
                # 记录该帧的指标
                frame_metric = {
                    "frame_uid": int(frame_uid),
                    "frame_idx": int(idx),
                    "psnr": float(psnr_value),
                    "ssim": float(ssim_value),
                    "lpips": float(lpips_value)
                }
                face_frame_metrics[face_key].append(frame_metric)
    
    # 计算每个面的平均指标
    output = dict()
    for face_key in faces:
        if len(face_metrics[face_key]["psnr"]) > 0:
            output[f"{face_key}_psnr"] = float(np.mean(face_metrics[face_key]["psnr"]))
            output[f"{face_key}_ssim"] = float(np.mean(face_metrics[face_key]["ssim"]))
            output[f"{face_key}_lpips"] = float(np.mean(face_metrics[face_key]["lpips"]))
    
    # 计算所有面的总体平均指标
    all_psnr = []
    all_ssim = []
    all_lpips = []
    for face_key in faces:
        all_psnr.extend(face_metrics[face_key]["psnr"])
        all_ssim.extend(face_metrics[face_key]["ssim"])
        all_lpips.extend(face_metrics[face_key]["lpips"])
    
    output["mean_psnr"] = float(np.mean(all_psnr)) if all_psnr else 0.0
    output["mean_ssim"] = float(np.mean(all_ssim)) if all_ssim else 0.0
    output["mean_lpips"] = float(np.mean(all_lpips)) if all_lpips else 0.0
    
    # 记录日志
    log_msg = f'Overall mean - psnr: {output["mean_psnr"]:.4f}, ssim: {output["mean_ssim"]:.4f}, lpips: {output["mean_lpips"]:.4f}\n'
    for face_key in faces:
        if f"{face_key}_psnr" in output:
            log_msg += f'  {face_key}: psnr={output[f"{face_key}_psnr"]:.4f}, ssim={output[f"{face_key}_ssim"]:.4f}, lpips={output[f"{face_key}_lpips"]:.4f}\n'
    Log(log_msg, tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    
    # 如果是after_opt，保存每个面的每帧指标到render文件夹下
    if save_images:
        for face_key in faces:
            if len(face_frame_metrics[face_key]) > 0:
                # 按frame_uid排序
                face_frame_metrics[face_key].sort(key=lambda x: x["frame_uid"])
                # 保存到JSON文件（放在render文件夹下）
                render_face_dir = os.path.join(render_save_dir, face_key)
                metrics_file = os.path.join(render_face_dir, "frame_metrics.json")
                json.dump(
                    face_frame_metrics[face_key],
                    open(metrics_file, "w", encoding="utf-8"),
                    indent=4,
                )
                Log(f"Saved {len(face_frame_metrics[face_key])} frame metrics for {face_key} face to {metrics_file}", tag="Eval")
    
    return output


def render_video_demo(
    frames,
    gaussians,
    save_dir,
    pipe,
    background,
    cube,
    fovx_deg: float = 150.0,
    fovy_deg: float = 90.0,
):
    """
    渲染用于视频demo的序列，每帧一个视场为 150°(宽) × 90°(高) 的图像。
    相机位姿使用front主视角的位姿。
    
    输出:
        在 save_dir/video/ 下保存连续帧:
        frame_000000.png, frame_000001.png, ...
    """
    if save_dir is None:
        Log("render_video_demo called with save_dir=None, skip.", tag="Video")
        return

    video_dir = os.path.join(save_dir, "video")
    mkdir_p(video_dir)

    # 处理 frames 可能是字典或列表的情况
    if isinstance(frames, dict):
        all_frame_indices = sorted(frames.keys())
    else:
        all_frame_indices = list(range(len(frames)))

    if not all_frame_indices:
        Log("No frames available for video rendering.", tag="Video")
        return

    # 目标视场（弧度）
    fovx = math.radians(fovx_deg)
    fovy = math.radians(fovy_deg)

    for idx in all_frame_indices:
        # 获取帧
        if isinstance(frames, dict):
            frame = frames[idx]
        else:
            frame = frames[idx]

        frame_uid = frame.uid if hasattr(frame, "uid") else idx

        # 记录原始相机参数，后面恢复
        orig_FoVx = frame.FoVx
        orig_FoVy = frame.FoVy
        orig_proj = frame.projection_matrix
        orig_H = frame.image_height
        orig_W = frame.image_width

        try:
            # 使用与训练相同的分辨率，只调整视场
            frame.FoVx = fovx
            frame.FoVy = fovy

            # 使用基于视场的透视投影矩阵（znear/zfar 与渲染保持一致）
            proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)
            frame.projection_matrix = proj.to(device=orig_proj.device, dtype=orig_proj.dtype)

            # 渲染front主视角
            out = render(frame, gaussians, pipe, background, face_key="front")
            image = out["render"]  # [3, H, W], 0-1

            image = torch.clamp(image, 0.0, 1.0)
            image_np = image.detach().cpu().permute(1, 2, 0).numpy()  # HWC, RGB
            image_uint8 = (image_np * 255.0).clip(0, 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(video_dir, f"frame_{frame_uid:06d}.png")
            cv2.imwrite(out_path, image_bgr)
        except Exception as e:
            Log(f"Failed to render video frame {frame_uid}: {e}", tag="Video")
        finally:
            # 恢复原始相机参数，避免影响其他流程
            frame.FoVx = orig_FoVx
            frame.FoVy = orig_FoVy
            frame.projection_matrix = orig_proj
            frame.image_height = orig_H
            frame.image_width = orig_W

    Log(f"Video frames saved to {video_dir}", tag="Video")


def save_video_frame_realtime(
    viewpoint,
    gaussians,
    pipe,
    background,
    save_dir,
    frame_idx: int,
):
    """
    在SLAM运行过程中，实时保存当前帧已经渲染好的front图像。
    直接使用 tracking/mapping 中的渲染结果（不重新渲染、不改相机参数）。

    Args:
        viewpoint: 当前帧相机（其 R/T 将在渲染期间被暂时替换为固定位姿）
        gaussians: 当前高斯点云
        pipe, background: 渲染管线与背景颜色
        save_dir: 结果根目录
        frame_idx: 帧索引（用于命名）
    """
    if save_dir is None or viewpoint is None or gaussians is None:
        return

    video_root = os.path.join(save_dir, "video")
    fixed_dir = os.path.join(video_root, "fixed")
    mkdir_p(fixed_dir)

    try:
        # 固定相机位姿（用户提供的4x4矩阵，假定为C2W）
        c2w_np = np.array([
            [0.3069085971971861, 0.2049068307152814, -0.9294232010461136, -0.33499632751416314],
            [-0.9517396651834062, 0.06412520765696421, -0.3001387974818148, -0.5950420965423048],
            [-0.0019013041333737755, 0.9766815199906779, 0.21469787690248776, 1.435755458785994],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        device = viewpoint.R.device
        c2w = torch.from_numpy(c2w_np).to(device=device, dtype=viewpoint.R.dtype)
        # 转为 world->camera (W2C)
        w2c = torch.linalg.inv(c2w)
        R_fixed = w2c[:3, :3]
        T_fixed = w2c[:3, 3]

        # 备份原始相机参数（位姿+视场+投影矩阵+图像尺寸）
        orig_R = viewpoint.R.clone()
        orig_T = viewpoint.T.clone()
        orig_FoVx = viewpoint.FoVx
        orig_FoVy = viewpoint.FoVy
        orig_proj = viewpoint.projection_matrix
        orig_H = viewpoint.image_height
        orig_W = viewpoint.image_width

        # 使用固定位姿渲染当前高斯点云
        viewpoint.R = R_fixed
        viewpoint.T = T_fixed

        # 调整视场：水平方向 120°，垂直方向保持原来的 FoVy（一般为90°）
        new_FoVx = math.radians(130.0)
        new_FoVy = orig_FoVy
        viewpoint.FoVx = new_FoVx
        viewpoint.FoVy = new_FoVy

        # 根据视场比调整图像宽度，保持像素角分辨率尽量一致
        try:
            aspect_fov = math.tan(new_FoVx / 2.0) / max(1e-6, math.tan(new_FoVy / 2.0))
            new_W = int(round(orig_H * aspect_fov))
            if new_W < 1:
                new_W = orig_W
        except Exception:
            new_W = orig_W
        viewpoint.image_height = orig_H
        viewpoint.image_width = new_W

        # 使用基于FOV的投影矩阵（与渲染器中的tan(FoV)一致）
        # 注意：Camera 中保存的 projection_matrix 是转置后的形式（参见 Camera.init_from_gui）
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=new_FoVx, fovY=new_FoVy)
        proj = proj.transpose(0, 1)  # 与现有代码保持一致的矩阵布局
        viewpoint.projection_matrix = proj.to(device=device, dtype=orig_proj.dtype)

        out = render(viewpoint, gaussians, pipe, background, face_key="front")
        image = out["render"]
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image

        # 确保在[0,1]范围
        image_np = np.clip(image_np, 0.0, 1.0)
        image_uint8 = (image_np * 255.0).clip(0, 255).astype(np.uint8)

        # 如果是灰度，扩展到三通道
        if image_uint8.ndim == 2:
            image_uint8 = np.stack([image_uint8] * 3, axis=-1)

        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(fixed_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, image_bgr)
    except Exception as e:
        Log(f"Failed to save realtime video frame {frame_idx}: {e}", tag="Video")
    finally:
        # 恢复原始相机参数
        viewpoint.R = orig_R
        viewpoint.T = orig_T
        viewpoint.FoVx = orig_FoVx
        viewpoint.FoVy = orig_FoVy
        viewpoint.projection_matrix = orig_proj
        viewpoint.image_height = orig_H
        viewpoint.image_width = orig_W


def save_face_render_realtime(
    render_tensor,
    save_dir,
    frame_idx: int,
    face_key: str,
):
    """
    保存tracking过程中某个面的实时渲染结果到 video/<face_key>/frame_xxxxxx.png。

    Args:
        render_tensor: 形状为 [C, H, W] 或 [1, C, H, W] 的torch张量，值范围约[0,1]
        save_dir: 结果根目录
        frame_idx: 帧索引
        face_key: 面名称（front/left/right/back）
    """
    if save_dir is None or render_tensor is None:
        return

    video_root = os.path.join(save_dir, "video")
    face_dir = os.path.join(video_root, face_key)
    mkdir_p(face_dir)

    try:
        image = render_tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image

        image_np = np.clip(image_np, 0.0, 1.0)
        image_uint8 = (image_np * 255.0).clip(0, 255).astype(np.uint8)
        if image_uint8.ndim == 2:
            image_uint8 = np.stack([image_uint8] * 3, axis=-1)

        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(face_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, image_bgr)
    except Exception as e:
        Log(f"Failed to save face render frame {frame_idx} ({face_key}): {e}", tag="Video")


def save_gaussians(gaussians, name, iteration, final=False):
    """保存高斯模型
    
    如果量化的参数为空，就使用原始的保存3d模型方式。
    如果有量化参数，那就保存码本和没有被量化的参数。
    
    Args:
        gaussians: GaussianModel对象
        name: 保存目录
        iteration: 迭代次数
        final: 是否为最终保存
    """
    if name is None:
        return
    if final:
        # 颜色优化后的点云使用独立目录 final_after_opt，与优化前的 final 分开保存
        if str(iteration) == "final_after_opt":
            point_cloud_path = os.path.join(name, "point_cloud/final_after_opt")
            Log("Saving point cloud after color refinement to point_cloud/final_after_opt", tag="Save")
        else:
            point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    
    # 检查是否有量化参数（检查quantization_centers是否有非None的值）
    has_quantization = False
    quantized_params = []
    
    if gaussians.quantization_enabled and gaussians.quantization_centers:
        # 获取配置中指定的量化参数列表
        if hasattr(gaussians, 'kmeans_config') and gaussians.kmeans_config:
            quantized_params = gaussians.kmeans_config.get('quantized_params', [])
        
        # 检查哪些参数真的被量化了（centers不为None）
        actual_quantized = []
        for param in quantized_params:
            if (gaussians.quantization_centers.get(param) is not None and
                gaussians.quantization_indices.get(param) is not None):
                actual_quantized.append(param)
                has_quantization = True
        
        quantized_params = actual_quantized
    
    if has_quantization and len(quantized_params) > 0:
        # 有量化参数：只保存未被量化的参数到PLY文件
        # 定义所有可能的属性及其对应的量化参数名称
        all_attributes = {
            'xyz': 'pos',
            'f_dc': 'dc', 
            'f_rest': 'sh',
            'opacities': None,  # opacities不会被量化
            'scale': 'scale',
            'rotation': 'rot'
        }
        
        # 确定哪些属性应该被保存（未被量化的参数）
        save_attributes = []
        for attr_name, quant_param in all_attributes.items():
            if quant_param is None:
                # opacities总是被保存
                save_attributes.append(attr_name)
            elif quant_param not in quantized_params:
                # 该参数未被量化，需要保存
                save_attributes.append(attr_name)
        
        # 保存量化版本的PLY文件（只包含未被量化的参数）
        gaussians.save_ply(
            os.path.join(point_cloud_path, "point_cloud.ply"),
            save_q=quantized_params,
            save_attributes=save_attributes
        )
        
        # 同时保存一份完整的非量化版本的PLY文件
        gaussians.save_ply(
            os.path.join(point_cloud_path, "point_cloud_original.ply"),
            save_q=[]  # 不使用量化参数，保存完整的原始版本
        )
        
        # 保存量化参数的码本和索引
        save_kmeans(gaussians, quantized_params, point_cloud_path)
        
        Log(f"Saved quantized model: quantized_params={quantized_params}, saved_attributes={save_attributes}", tag="Save")
        Log(f"Saved original (non-quantized) model: point_cloud_original.ply", tag="Save")
    else:
        # 没有量化参数：使用原始方式保存所有参数
        gaussians.save_ply(
            os.path.join(point_cloud_path, "point_cloud.ply"),
            save_q=[]
        )
        
        Log("Saved original (non-quantized) model", tag="Save")
