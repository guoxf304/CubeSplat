import torch


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    '''计算梯度掩码'''
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

# 未使用
def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False, face_key=None):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint, face_key)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, face_key)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, face_key):
    gt_image = viewpoint.Cubemap_image[face_key].cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask[face_key]
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, face_key=None):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint, face_key)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, face_key=face_key)


def get_loss_mapping_rgb(config, image, depth, viewpoint, face_key=None):
    assert face_key is not None, "face_key must be provided for get_loss_mapping_rgb in cubemap mode."

    gt_image = viewpoint.Cubemap_image[face_key].cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    
    # 如果有depth_mask（从预测深度生成的mask），应用到rgb_pixel_mask
    if hasattr(viewpoint, 'depth_mask') and viewpoint.depth_mask is not None:
        if face_key is not None and face_key in viewpoint.depth_mask and viewpoint.depth_mask[face_key] is not None:
            depth_mask = viewpoint.depth_mask[face_key]
            # 确保mask是正确形状
            if depth_mask.dim() == 2:
                depth_mask = depth_mask.unsqueeze(0)
            depth_mask = depth_mask.to(device=rgb_pixel_mask.device, dtype=rgb_pixel_mask.dtype)
            # 确保mask形状匹配
            if depth_mask.shape != mask_shape:
                depth_mask = torch.nn.functional.interpolate(
                    depth_mask.unsqueeze(0), size=(h, w), mode='nearest'
                ).squeeze(0)
            # 将depth_mask应用到rgb_pixel_mask（只在mask内计算loss）
            rgb_pixel_mask = rgb_pixel_mask * depth_mask
    
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False, face_key=None):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    # 支持cubemap模式
    if face_key is not None:
        gt_image = viewpoint.Cubemap_image[face_key].cuda()
        if isinstance(viewpoint.depth, dict) and face_key in viewpoint.depth:
            gt_depth = viewpoint.depth[face_key]
            if isinstance(gt_depth, torch.Tensor):
                gt_depth = gt_depth.to(device=image.device, dtype=torch.float32)
            else:
                gt_depth = torch.from_numpy(gt_depth).to(
                    dtype=torch.float32, device=image.device
                )
            if gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(0)  # (1, H, W)
        else:
            gt_depth = None
    else:
        gt_image = viewpoint.Cubemap_image.cuda()
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=image.device
        )[None]
    
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    # 计算深度loss（如果有深度）
    if gt_depth is not None:
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
        
        # 如果有depth_mask（从预测深度生成的mask），应用到depth_pixel_mask
        if hasattr(viewpoint, 'depth_mask') and viewpoint.depth_mask is not None:
            if face_key is not None and face_key in viewpoint.depth_mask and viewpoint.depth_mask[face_key] is not None:
                depth_mask = viewpoint.depth_mask[face_key]
                # 确保mask是正确形状
                if depth_mask.dim() == 2:
                    depth_mask = depth_mask.unsqueeze(0)
                depth_mask = depth_mask.to(device=depth_pixel_mask.device, dtype=depth_pixel_mask.dtype)
                # 确保mask形状匹配
                if depth_mask.shape != mask_shape:
                    depth_mask = torch.nn.functional.interpolate(
                        depth_mask.unsqueeze(0), size=(h, w), mode='nearest'
                    ).squeeze(0)
                # 将depth_mask应用到depth_pixel_mask（只在mask内计算loss）
                depth_pixel_mask = depth_pixel_mask * depth_mask
        
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
    else:
        return l1_rgb.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    '''计算中值深度'''
    depth = depth.detach().clone()
    # 确保depth是2D的（H, W）
    while depth.dim() > 2:
        depth = depth.squeeze()
    if depth.dim() != 2:
        raise ValueError(f"depth must be 2D (H, W), got shape {depth.shape}")
    
    valid = depth > 0
    
    if opacity is not None:
        opacity = opacity.detach()
        # 确保opacity是2D的
        while opacity.dim() > 2:
            opacity = opacity.squeeze()
        if opacity.dim() == 1:
            # 如果是1D，尝试reshape为2D
            if opacity.numel() == depth.numel():
                opacity = opacity.view(depth.shape)
            else:
                opacity = None
        if opacity is not None and opacity.dim() == 2:
            # 确保opacity和depth的形状匹配
            if opacity.shape == depth.shape:
                valid = torch.logical_and(valid, opacity > 0.95)
            else:
                # 如果形状不匹配，尝试broadcast或取第一个通道
                if opacity.shape[0] == 1:
                    opacity = opacity[0]
                if opacity.shape == depth.shape:
                    valid = torch.logical_and(valid, opacity > 0.95)
    
    if mask is not None:
        # 确保mask是2D的
        while mask.dim() > 2:
            mask = mask.squeeze()
        if mask.dim() == 1:
            # 如果是1D，尝试reshape为2D
            if mask.numel() == depth.numel():
                mask = mask.view(depth.shape)
            else:
                mask = None
        if mask is not None and mask.dim() == 2:
            # 确保mask和depth的形状匹配
            if mask.shape == depth.shape:
                valid = torch.logical_and(valid, mask)
            else:
                # 如果形状不匹配，尝试broadcast
                if mask.shape[0] == 1 and mask.shape[1:] == depth.shape:
                    mask = mask[0]
                if mask.shape == depth.shape:
                    valid = torch.logical_and(valid, mask)
    
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
