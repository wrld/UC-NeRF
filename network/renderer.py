import torch
import torch.nn.functional as F
from utils.utils import normal_vect, index_point_feature, build_color_volume


def depth2dist(z_vals, cos_angle):
    # z_vals: [N_ray N_sample]
    device = z_vals.device
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists,
         torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)],
        -1)  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists


def ndc2dist(ndc_pts, cos_angle):
    dists = torch.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], dim=-1)
    dists = torch.cat([dists, 1e10 * cos_angle.unsqueeze(-1)],
                      -1)  # [N_rays, N_samples]
    return dists


def raw2alpha(sigma, dist, net_type):

    alpha_softmax = F.softmax(sigma, 1)

    alpha = 1. - torch.exp(-sigma)

    T = torch.cumprod(
        torch.cat([
            torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10
        ], -1), -1)[:, :-1]
    weights = alpha * T  # [N_rays, N_samples]
    return alpha, weights, alpha_softmax


def raw2alpha_fuse(sigma_1, sigma_2, dist, net_type):

    alpha_joint = 1. - torch.exp(-(sigma_1 + sigma_2))
    alpha_1 = 1. - torch.exp(-sigma_1)

    alpha_2 = 1. - torch.exp(-sigma_2)

    T = torch.cumprod(
        torch.cat([
            torch.ones(alpha_joint.shape[0], 1).to(alpha_joint.device),
            1. - alpha_joint + 1e-10
        ], -1), -1)[:, :-1]
    weights_1 = alpha_1 * T  # [N_rays, N_samples]
    weights_2 = alpha_2 * T  # [N_rays, N_samples]

    return weights_1, weights_2


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, alpha_only):
        if alpha_only:
            return torch.cat([
                fn.forward_alpha(inputs[i:i + chunk])
                for i in range(0, inputs.shape[0], chunk)
            ], 0)
        else:
            return torch.cat([
                fn(inputs[i:i + chunk])
                for i in range(0, inputs.shape[0], chunk)
            ], 0)

    return ret


def run_network_mvs(pts,
                    viewdirs,
                    alpha_feat,
                    fn,
                    embed_fn,
                    embeddirs_fn,
                    netchunk=1024):
    """Prepares inputs and applies network 'fn'.
    """

    if embed_fn is not None:
        pts = embed_fn(pts)

    if alpha_feat is not None:
        pts = torch.cat((pts, alpha_feat), dim=-1)

    if viewdirs is not None:
        if viewdirs.dim() != 3:
            viewdirs = viewdirs[:, None].expand(-1, pts.shape[1], -1)

        if embeddirs_fn is not None:
            viewdirs = embeddirs_fn(viewdirs)
        pts = torch.cat([pts, viewdirs], -1)

    alpha_only = viewdirs is None
    outputs_flat = batchify(fn, netchunk)(pts, alpha_only)
    outputs = torch.reshape(outputs_flat,
                            list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def raw2outputs(raw, z_vals, dists, white_bkgd=False, net_type='v2'):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    device = z_vals.device

    rgb_1 = raw[..., :3]  # [N_rays, N_samples, 3]
    sigma_1 = raw[..., 3]
    alpha_1, weights_1, alpha_softmax_1 = raw2alpha(sigma_1, dists, net_type)
    weights = weights_1
    var, mean = torch.var_mean(weights, dim=1)
    rgb_map = torch.sum(weights_1[..., None] * rgb_1, -2)
    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1. / torch.max(
        1e-10 * torch.ones_like(depth_map, device=device),
        depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map, disp_map, acc_map, weights, depth_map, weights, var


def gen_angle_feature(c2ws, rays_pts, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]

    Returns:

    """
    N_rays, N_samples = rays_pts.shape[:2]
    dirs = normal_vect(
        rays_pts.unsqueeze(2) -
        c2ws[:3, :3, 3][None, None])  # [N_rays, N_samples, v, 3]
    angle = torch.sum(dirs[:, :, :3] * rays_dir.reshape(N_rays, 1, 1, 3),
                      dim=-1,
                      keepdim=True).reshape(N_rays, N_samples, -1)
    return angle


def gen_dir_feature(w2c_ref, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]

    Returns:

    """
    dirs = rays_dir @ w2c_ref[:3, :3].t()  # [N_rays, 3]
    return dirs


def gen_pts_feats(imgs,
                  volume_feature,
                  rays_pts,
                  pose_ref,
                  rays_ndc,
                  feat_dim,
                  img_feat=None,
                  near_fars=None,
                  img_downscale=1.0,
                  use_color_volume=False,
                  net_type='v0',
                  confidence=None):
    N_rays, N_samples = rays_pts.shape[:2]
    if not use_color_volume:
        input_feat = torch.empty((N_rays, N_samples, feat_dim),
                                 device=imgs.device,
                                 dtype=torch.float)
        N, V, C, H, W = imgs.shape
        confidence = confidence.reshape(1, 1, 1, H, W)
        ray_feats, confidence_sampled = index_point_feature(
            volume_feature, confidence, rays_ndc)
        input_feat[..., :24] = ray_feats
        input_feat[..., 24:-1] = build_color_volume(rays_pts,
                                                    pose_ref,
                                                    imgs,
                                                    near_fars,
                                                    img_feat,
                                                    confidence=confidence,
                                                    with_mask=True,
                                                    downscale=img_downscale)
        input_feat[..., -1:] = confidence_sampled.unsqueeze(-1)
    else:
        input_feat = index_point_feature(
            volume_feature, rays_ndc) if torch.is_tensor(
                volume_feature) else volume_feature(rays_ndc)
    return input_feat


def rendering(args,
              pose_ref,
              rays_pts,
              rays_ndc,
              depth_candidates,
              rays_dir,
              volume_feature=None,
              imgs=None,
              network_fn=None,
              near_fars=None,
              img_feat=None,
              network_query_fn=None,
              white_bkgd=False,
              confidence=None,
              **kwargs):

    # rays angle
    cos_angle = torch.norm(rays_dir, dim=-1)

    # using direction
    if pose_ref is not None:
        angle = gen_dir_feature(pose_ref['w2cs'][0], rays_dir /
                                cos_angle.unsqueeze(-1))  # view dir feature
    else:
        angle = rays_dir / cos_angle.unsqueeze(-1)

    if pose_ref['w2cs'].shape[0] == args.view_num:
        pose_ref['w2cs'] = pose_ref['w2cs'][1:]
        pose_ref['intrinsics'] = pose_ref['intrinsics'][1:]
    input_feat = gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, args.feat_dim, \
                               img_feat, near_fars, args.img_downscale, args.use_color_volume, args.net_type, confidence=confidence)
    raw = network_query_fn(rays_ndc['ndc'], angle, input_feat, network_fn)
    if raw.shape[-1] > 4:
        input_feat = torch.cat((input_feat[..., :8], raw[..., 4:]), dim=-1)

    dists = depth2dist(depth_candidates, cos_angle)
    rgb_map, disp_map, acc_map, weights, depth_map, alpha, var = raw2outputs(
        raw, depth_candidates, dists, white_bkgd, args.net_type)
    ret = {}

    return rgb_map, depth_map


def render_density(network_fn,
                   rays_pts,
                   density_feature,
                   network_query_fn,
                   chunk=1024 * 5):
    densities = []
    device = density_feature.device
    for i in range(0, rays_pts.shape[0], chunk):

        input_feat = rays_pts[i:i + chunk].to(device)

        density = network_query_fn(input_feat, None,
                                   density_feature[i:i + chunk], network_fn)
        densities.append(density)

    return torch.cat(densities)