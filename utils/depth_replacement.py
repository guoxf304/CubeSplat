import numpy as np


def _default_cfg():
    return {
        "min_overlap_pixels": 2048,
        "scale_clip": (0.3, 3.0),
        "error_threshold": 0.25,
        "opacity_threshold": 0.02,
        "min_depth": 1e-3,
        "default_scale": 1.0,
        "blend_alpha": 0.5,
    }


def _tensor_to_numpy(tensor):
    if tensor is None:
        return None
    array = tensor.detach().float().cpu().squeeze()
    return array.numpy()


def pointmap_replacement(
    render_depth,
    valid_rgb_mask,
    opacity=None,
    prev_depth=None,
    prev_scale=1.0,
    cfg=None,
):
    """
    Render depth replacement inspired by S3PO-GS Pointmap Replacement.

    Args:
        render_depth (torch.Tensor): shape (1, H, W)
        valid_rgb_mask (torch.Tensor): shape (1, H, W) boolean mask
        opacity (torch.Tensor, optional): shape (1, H, W), higher value = more reliable
        prev_depth (np.ndarray, optional): previous refined depth map
        prev_scale (float): previously used scale factor
        cfg (dict): configuration overrides

    Returns:
        tuple(np.ndarray, float): refined depth map, updated scale factor
    """
    params = _default_cfg()
    if cfg:
        params.update(cfg)

    render_np = _tensor_to_numpy(render_depth)
    valid_rgb = _tensor_to_numpy(valid_rgb_mask).astype(bool)
    opacity_np = _tensor_to_numpy(opacity)
    current_scale = params["default_scale"]

    if prev_scale is not None:
        current_scale = prev_scale

    scaled_depth = render_np.copy()
    base_mask = (scaled_depth > params["min_depth"]) & valid_rgb
    if opacity_np is not None:
        base_mask &= opacity_np > params["opacity_threshold"]

    if prev_depth is not None:
        prev_mask = (prev_depth > params["min_depth"])
        overlap_mask = base_mask & prev_mask
        if overlap_mask.sum() >= params["min_overlap_pixels"]:
            prev_median = np.median(prev_depth[overlap_mask])
            curr_median = np.median(scaled_depth[overlap_mask])
            if curr_median > params["min_depth"]:
                scale = prev_median / (curr_median + 1e-6)
                min_scale, max_scale = params["scale_clip"]
                scale = float(np.clip(scale, min_scale, max_scale))
                current_scale = params["blend_alpha"] * scale + (1 - params["blend_alpha"]) * current_scale
        scaled_depth *= current_scale

        denom = np.maximum(np.abs(prev_depth), 1e-6)
        rel_error = np.abs(scaled_depth - prev_depth) / denom
        replace_mask = (rel_error > params["error_threshold"]) & prev_mask
        scaled_depth[replace_mask] = prev_depth[replace_mask]

        hole_mask = (~base_mask) & prev_mask
        scaled_depth[hole_mask] = prev_depth[hole_mask]
    else:
        scaled_depth *= current_scale

    scaled_depth[~valid_rgb] = 0.0
    return scaled_depth, current_scale


def patch_based_scale_alignment(
    render_depth,
    mono_depth,
    valid_rgb_mask,
    patch_size=10,
    mean_threshold=0.25,
    std_threshold=0.3,
    error_threshold=0.1,
    final_error_threshold=0.15,
    max_iter=4,
    epsilon=0.01,
    min_accurate_pixels_ratio=0.01,
):
    """
    基于补丁的深度尺度对齐算法（用于RGBD模式，参考S3PO-GS的process_depth）
    
    该函数通过迭代优化，对齐渲染深度图和预测深度（加载的深度）的尺度，并融合两者生成最终的深度图。
    主要步骤包括：
    1. 基于补丁的初始过滤（通过均值和标准差筛选）
    2. 补丁归一化
    3. 精确像素过滤
    4. 迭代计算尺度因子
    5. 深度融合（Pointmap Replacement）：用预测深度填充渲染深度的无效区域
    
    Args:
        render_depth (np.ndarray): 渲染得到的深度图（从高斯点云渲染），形状为 (H, W) 或 (1, H, W)
        mono_depth (np.ndarray): 预测深度图（从数据集加载），形状为 (H, W)
        valid_rgb_mask (np.ndarray): RGB有效像素掩码，形状为 (H, W)，bool类型
        patch_size (int): 补丁大小，默认10
        mean_threshold (float): 均值差异阈值（相对值），默认0.25
        std_threshold (float): 标准差差异阈值（相对值），默认0.3
        error_threshold (float): 归一化后的误差阈值，用于精确像素过滤，默认0.1
        final_error_threshold (float): 最终深度融合时的相对误差阈值，默认0.15
        max_iter (int): 最大迭代次数，默认4
        epsilon (float): 尺度因子收敛阈值，默认0.01
        min_accurate_pixels_ratio (float): 最小精确像素比例，默认0.01
    
    Returns:
        tuple: (final_depth, scale_factor, error_mask, num_accurate_pixels)
            - final_depth (np.ndarray): 融合后的最终深度图，形状为 (H, W)
            - scale_factor (float): 计算得到的尺度因子（用于校正预测深度的尺度）
            - error_mask (np.ndarray): 误差掩码，True表示该像素使用预测深度填充
            - num_accurate_pixels (int): 精确像素的数量
    """
    # 确保深度图是(H, W)格式
    if render_depth.ndim == 3:
        render_depth = render_depth[0]
    if mono_depth.ndim == 3:
        mono_depth = mono_depth[0]
    
    H, W = render_depth.shape
    scale_factor = 1.0  # 初始尺度因子
    prev_scale_factor = 0.0  # 上一次迭代的尺度因子，用于判断收敛
    final_mask = np.zeros((H, W), dtype=bool)  # 最终精确像素掩码
    
    total_pixels = H * W
    # 计算最小精确像素数量（用于判断是否有足够的可靠像素进行尺度估计）
    min_accurate_pixels = int(min_accurate_pixels_ratio * total_pixels)
    
    num_accurate_pixels = 0
    
    # 创建有效深度掩码（深度大于0且在RGB有效区域内）
    valid_render = (render_depth > 0) & valid_rgb_mask
    valid_mono = (mono_depth > 0) & valid_rgb_mask
    
    # 调试信息：初始统计
    valid_render_count = np.sum(valid_render)
    valid_mono_count = np.sum(valid_mono)
    overlap_count = np.sum(valid_render & valid_mono)
    render_depth_mean = np.mean(render_depth[valid_render]) if valid_render_count > 0 else 0.0
    mono_depth_mean = np.mean(mono_depth[valid_mono]) if valid_mono_count > 0 else 0.0
    render_depth_range = (np.min(render_depth[valid_render]), np.max(render_depth[valid_render])) if valid_render_count > 0 else (0, 0)
    mono_depth_range = (np.min(mono_depth[valid_mono]), np.max(mono_depth[valid_mono])) if valid_mono_count > 0 else (0, 0)
    
    print(f"[Depth Scale Alignment] Initial stats:")
    print(f"  Render depth: mean={render_depth_mean:.4f}, range=({render_depth_range[0]:.4f}, {render_depth_range[1]:.4f}), valid_pixels={valid_render_count}")
    print(f"  Loaded depth (predicted): mean={mono_depth_mean:.4f}, range=({mono_depth_range[0]:.4f}, {mono_depth_range[1]:.4f}), valid_pixels={valid_mono_count}")
    print(f"  Overlap pixels: {overlap_count}, min_required: {min_accurate_pixels}")
    
    # 迭代优化尺度因子
    for k in range(max_iter):
        # 如果尺度因子已经收敛（变化小于epsilon且不为初始值1.0），则提前退出
        if (abs(scale_factor - prev_scale_factor) < epsilon) and (scale_factor != 1.0):
            print(f"  Iteration {k+1}: Scale factor converged (scale={scale_factor:.6f}, change={abs(scale_factor - prev_scale_factor):.6f} < {epsilon})")
            break
        prev_scale_factor = scale_factor
        print(f"  Iteration {k+1}: Starting with scale_factor={scale_factor:.6f}")
        
        patch_num = 0  # 通过第一阶段过滤的补丁数量
        
        # 步骤2：初始补丁过滤
        # 通过比较补丁的均值和标准差，筛选出可能一致的补丁
        accurate_pixels = np.zeros((H, W), dtype=bool)  # 精确像素掩码
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                # 提取当前补丁
                render_patch = render_depth[i:i+patch_size, j:j+patch_size]
                # 使用当前尺度因子缩放预测深度补丁
                mono_patch = mono_depth[i:i+patch_size, j:j+patch_size] * scale_factor
                
                # 检查补丁大小，避免边界溢出
                if render_patch.size == 0 or mono_patch.size == 0:
                    continue
                
                # 检查补丁内是否有有效像素
                patch_valid_render = valid_render[i:i+patch_size, j:j+patch_size]
                patch_valid_mono = valid_mono[i:i+patch_size, j:j+patch_size]
                patch_valid = patch_valid_render & patch_valid_mono
                
                if patch_valid.sum() < patch_size:  # 至少需要一些有效像素
                    continue
                
                # 只使用有效像素计算统计量
                render_patch_valid = render_patch[patch_valid]
                mono_patch_valid = mono_patch[patch_valid]
                
                if len(render_patch_valid) == 0 or len(mono_patch_valid) == 0:
                    continue
                
                # 通过均值和标准差过滤补丁
                # 均值条件：两个补丁的均值差异小于阈值（相对值）
                mean_render = np.mean(render_patch_valid)
                mean_mono = np.mean(mono_patch_valid)
                mean_condition = abs(mean_render - mean_mono) < mean_threshold * mean_mono if mean_mono > 0 else False
                
                # 标准差条件：两个补丁的标准差差异小于阈值（相对值）
                std_render = np.std(render_patch_valid)
                std_mono = np.std(mono_patch_valid)
                std_condition = abs(std_render - std_mono) < std_threshold * std_mono if std_mono > 0 else False
                
                # 如果补丁通过了均值和标准差过滤
                if mean_condition and std_condition:
                    patch_num = patch_num + 1
                    # 步骤3：补丁归一化
                    # 将补丁归一化为零均值、单位方差，以便进行像素级别的比较
                    render_norm = (render_patch - mean_render) / (std_render + 1e-6)
                    mono_norm = (mono_patch - mean_mono) / (std_mono + 1e-6)
                    
                    # 步骤4：精确像素过滤
                    # 在归一化后的补丁中，找出归一化值差异小于阈值的像素
                    patch_mask = np.abs(render_norm - mono_norm) < error_threshold
                    # 只标记有效像素
                    patch_mask = patch_mask & patch_valid
                    # 将精确像素标记到掩码中
                    accurate_pixels[i:i+patch_size, j:j+patch_size] = patch_mask
        
        # 如果有足够的精确像素，基于这些像素计算尺度因子
        num_accurate_pixels = 0
        accurate_count = np.sum(accurate_pixels)
        print(f"    Patches passed filtering: {patch_num}, Accurate pixels: {accurate_count} ({accurate_count/total_pixels*100:.2f}%)")
        
        # 条件：存在精确像素，且（前两次迭代或精确像素数量足够）
        if np.any(accurate_pixels) and ((k < 2) or (accurate_count >= min_accurate_pixels)):
            # 计算尺度因子：渲染深度与预测深度的均值比
            render_mean = np.mean(render_depth[accurate_pixels])
            mono_mean = np.mean(mono_depth[accurate_pixels])
            if mono_mean > 0:
                new_scale = render_mean / mono_mean
                scale_change = abs(new_scale - scale_factor)
                scale_factor = new_scale
                print(f"    Updated scale_factor: {scale_factor:.6f} (change={scale_change:.6f}, render_mean={render_mean:.4f}, mono_mean={mono_mean:.4f})")
            # 保存当前精确像素掩码
            final_mask = accurate_pixels.copy()
            num_accurate_pixels = np.sum(final_mask)
        elif accurate_count < min_accurate_pixels and k >= 2:
            # 如果精确像素太少，使用简单的均值比作为回退
            overlap_mask = valid_render & valid_mono
            if overlap_mask.sum() > min_accurate_pixels:
                render_mean = np.mean(render_depth[overlap_mask])
                mono_mean = np.mean(mono_depth[overlap_mask])
                if mono_mean > 0:
                    scale_factor = render_mean / mono_mean
                    # 限制尺度因子范围
                    scale_factor = np.clip(scale_factor, 0.5, 2.0)
                    print(f"    Fallback: Using overlap mask, scale_factor={scale_factor:.6f} (clipped to [0.5, 2.0])")
            else:
                print(f"    Warning: Not enough overlap pixels ({overlap_mask.sum()}) for fallback")
            break
    
    # 步骤5：填充无效（误差）像素（Pointmap Replacement策略）
    # 使用计算得到的尺度因子缩放预测深度
    mono_depth_scaled = mono_depth * scale_factor
    # 计算相对误差：|渲染深度 - 缩放后的预测深度| / 缩放后的预测深度
    relative_error = np.abs(render_depth - mono_depth_scaled) / (mono_depth_scaled + 1e-8)
    # 创建误差掩码：相对误差超过阈值的像素将被预测深度替换
    error_mask = relative_error > final_error_threshold
    
    # 同时填充渲染深度为零的像素（这些像素没有有效的渲染深度值）
    error_mask[render_depth == 0] = True
    # 只替换有效区域内的像素
    error_mask = error_mask & valid_mono
    
    # 深度融合：对于误差掩码为True的像素，使用缩放后的预测深度；否则使用渲染深度
    final_depth = np.where(error_mask, mono_depth_scaled, render_depth)
    
    # 确保无效RGB区域的深度为0
    final_depth[~valid_rgb_mask] = 0.0
    
    # 调试信息：最终统计
    replaced_count = np.sum(error_mask)
    replaced_ratio = replaced_count / total_pixels * 100 if total_pixels > 0 else 0
    final_valid = (final_depth > 0) & valid_rgb_mask
    final_depth_mean = np.mean(final_depth[final_valid]) if np.sum(final_valid) > 0 else 0.0
    final_depth_range = (np.min(final_depth[final_valid]), np.max(final_depth[final_valid])) if np.sum(final_valid) > 0 else (0, 0)
    
    print(f"[Depth Scale Alignment] Final results:")
    print(f"  Final scale_factor: {scale_factor:.6f}")
    print(f"  Accurate pixels used: {num_accurate_pixels} ({num_accurate_pixels/total_pixels*100:.2f}%)")
    print(f"  Pixels replaced: {replaced_count} ({replaced_ratio:.2f}%)")
    print(f"  Final depth: mean={final_depth_mean:.4f}, range=({final_depth_range[0]:.4f}, {final_depth_range[1]:.4f})")
    print(f"  Final valid pixels: {np.sum(final_valid)}")
    print("-" * 60)
    
    return final_depth, scale_factor, error_mask, num_accurate_pixels

