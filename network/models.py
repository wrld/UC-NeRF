from tkinter import X
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from utils.utils import homo_warp
from inplace_abn import InPlaceABN
from network.renderer import run_network_mvs
from network.mvs_models import depth_regression, mvs_depth_regression, RefineNet, CascadeMVSNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        self.freq_bands = freq_bands.reshape(1,-1,1).cuda()

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        repeat = inputs.dim()-1
        inputs_scaled = (inputs.unsqueeze(-2) * self.freq_bands.view(*[1]*repeat,-1,1)).reshape(*inputs.shape[:-1],-1)
        inputs_scaled = torch.cat((inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)),dim=-1)
        return inputs_scaled

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class BaseAdapt_Renderer(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False, fine=False, view_num=4):
        """
        """
        super(BaseAdapt_Renderer, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.view_num = view_num - 1
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in range(D-1)])
        self.pts_bias_depth_fine = nn.Linear(24+4*self.view_num, W)
        
        self.pts_bias_confidence = nn.Linear(8*self.view_num, W)
        self.pts_bias_confidence_1 = nn.Linear(1, 1)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.view_confi_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.feature_linear_1 = nn.Linear(W, W)
            self.confi_linear = nn.Linear(W, W)
            
            self.alpha_linear = nn.Linear(W//2, 1)
            self.alpha_linear_1 = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
            self.confi_rgb_linear = nn.Linear(W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.pts_bias_depth_fine.apply(weights_init)
        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.view_confi_linears.apply(weights_init)
        
        self.confi_linear.apply(weights_init)
        self.pts_bias_confidence_1.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.feature_linear_1.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)
        self.confi_rgb_linear.apply(weights_init)

    def forward_alpha(self, x):

        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = torch.relu(self.alpha_linear(h))
        return alpha


    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)
        h = input_pts
        mvs_feats = input_feats[..., :24]
        mvs_feats_color = input_feats[..., 24:24+4*self.view_num]
        mvs_feats_cat = torch.cat([mvs_feats, mvs_feats_color], dim=-1)
        
        img_feats = input_feats[..., 24+4*self.view_num:24+4*self.view_num + 8*self.view_num]
        # convert from confidence to uncertainty
        uncertainty = 1-input_feats[..., -1:]
        depth_bias_fine = self.pts_bias_depth_fine(mvs_feats_cat)
        feats_bias = self.pts_bias_confidence(img_feats)
        uncertainty = uncertainty
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)*depth_bias_fine
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        if self.use_viewdirs:
            base_color = self.confi_rgb_linear(h)
            base_alpha = self.alpha_linear_1(h)

            feature = self.feature_linear(h*feats_bias)#
            
            h_1 = torch.cat([feature, input_views], -1)
            
            h_1_ = self.views_linears[0](h_1)
            h_1_= F.relu(h_1_)
            adapt_color = self.rgb_linear(h_1_)

            h_2 = self.view_confi_linears[0](h_1)
            h_2 = F.relu(h_2)
            adapt_alpha = self.alpha_linear(h_2)

            # uncertainty-aware adaptation
            rgb = torch.sigmoid(base_color*(1-uncertainty) + adapt_color*uncertainty)
            alpha = torch.relu(adapt_alpha* (1-uncertainty) + base_alpha* uncertainty)

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class UCNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch_pts=3, input_ch_views=3, input_ch_feat=8, skips=[4], net_type='v2', fine=False, view_num=4):
        """
        """
        super(UCNeRF, self).__init__()

        self.in_ch_pts, self.in_ch_views,self.in_ch_feat = input_ch_pts, input_ch_views, input_ch_feat

        self.nerf = BaseAdapt_Renderer(D=D, W=W,input_ch_feat=input_ch_feat,
                    input_ch=input_ch_pts, output_ch=4, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=True, fine=fine, view_num=view_num)

    def forward_alpha(self, x):
        return self.nerf.forward_alpha(x)

    def forward_uncertainty(self, x):
        return 1 - x

    def forward(self, x):
        RGBA = self.nerf(x)
        return RGBA

def create_ucnerf(args, pts_embedder=True, dir_embedder=True):
    """Instantiate mvs NeRF's MLP model.
    """

    if pts_embedder:
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)#, input_dims=args.pts_dim)
    else:
        embed_fn, input_ch = None, args.pts_dim

    embeddirs_fn = None
    if dir_embedder:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)#, input_dims=args.dir_dim)
    else:
        embeddirs_fn, input_ch_views = None, args.dir_dim


    skips = [4]
    model = UCNeRF(D=args.netdepth, W=args.netwidth,
                 input_ch_pts=input_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_feat=args.feat_dim, net_type=args.net_type, view_num=args.view_num).to(device)

    grad_vars = []
    grad_vars += list(model.parameters())

    network_query_fn = lambda pts, viewdirs, rays_feats, network_fn: run_network_mvs(pts, viewdirs, rays_feats, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)
    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a)
        grad_vars += list(embedding_a.parameters())
    EncodingNet = CascadeMVSNet(view_num=args.view_num)
    torch.utils.model_zoo.load_url(
            "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/48_32_8-4-2-1_dlossw-0.5-1.0-2.0/casmvsnet.ckpt",
            model_dir="pretrained_weights",
    )
    ckpt_file = "pretrained_weights/casmvsnet.ckpt"
    state_dict = torch.load(ckpt_file, map_location=torch.device("cpu"))
    EncodingNet.load_state_dict(state_dict['model'], strict=True)
    EncodingNet = EncodingNet.to(device)
    if args.finetune is None:
        grad_vars += list(EncodingNet.parameters())   

    start = 0
    ckpts = []
    if args.ckpt is not None and args.ckpt != 'None':
        ckpts = [args.ckpt]    
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 :
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load model
        state_dict = ckpt['network_mvs_state_dict']
        EncodingNet.load_state_dict(state_dict)

        model.load_state_dict(ckpt['network_fn_state_dict'])
        
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'network_mvs': EncodingNet,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std
    }
    if args.encode_a: render_kwargs_train['embedding_a'] = embedding_a

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars





#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # print("after x", x.shape, x[0, 0, 0, 0])
        x = self.bn(x)
        # print("after bn", x.shape, x[0, 0, 0, 0])
        return x

###################################  feature net  ######################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)
        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))
        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

        
    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        cost = conv0 + self.conv11(x)
        del conv0
        x = self.prob(cost) 
        return cost, x

class MVSNet(nn.Module):
    def __init__(self,
                 view_num=11,
                 num_groups=1,
                 norm_act=InPlaceABN,
                 levels=1):
        super(MVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [48,32,8]
        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = FeatureNet()
        self.refine_network = RefineNet()
        self.chunk = 1024
        self.cost_reg_2 = CostRegNet(32 + (view_num-1)*3, norm_act)
        self.view_num = view_num

    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0, novel_view=False):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        img_feat = torch.empty((B, 32+3*(self.view_num-1), D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        
        ori_ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)
        ref_volume = torch.zeros_like(ori_ref_volume)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs[1:], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:, (i ) * 3:(i +1) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)
            
            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i:i+1] = in_mask.float()
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2
        img_mean = volume_sum * count
        # img_feat = torch.cat([img_feat, img_mean], dim=0)
        del volume_sq_sum, volume_sum, count

        return img_feat, img_mean

    def forward(self, imgs, proj_mats, near_far, pad=0,  return_depth=False, lindisp=False, novel_view=False):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)

        B, V, _, H, W = imgs.shape
        imgs = imgs.reshape(B * V, 3, H, W)
        feats = self.feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        imgs = imgs.view(B, V, 3, H, W)
        feats_l = feats  # (B*V, C, h, w)
        feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)
        D = 128
        t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        near, far = near_far  # assume batch size==1
        depth_values = near * (1.-t_vals) + far * (t_vals)
        depth_values = depth_values.unsqueeze(0)#.expand(B, -1)
        volume_feat, img_mean = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad, novel_view=novel_view)
        volume_feat, prob = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])
        prob = F.upsample(prob, [128, 256, 320], mode='trilinear')
        prob_volume_pre = prob.squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = mvs_depth_regression(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = mvs_depth_regression(prob_volume, depth_values=torch.arange(depth_values.shape[1], device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.long()
            depth_index = depth_index.clamp(min=0, max=D - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            # photometric_confidence = torch.sigmoid(photometric_confidence)
            
        if pad > 0:
            depth = depth[:, pad:H+pad, pad:W+pad]
            photometric_confidence = photometric_confidence[:, pad:H+pad, pad:W+pad]
        return volume_feat, photometric_confidence, depth, feats_l

class RefVolume(nn.Module):
    def __init__(self, volume):
        super(RefVolume, self).__init__()

        self.feat_volume = nn.Parameter(volume)

    def forward(self, ray_coordinate_ref):
        '''coordinate: [N, 3]
            z,x,y
        '''

        device = self.feat_volume.device
        H, W = ray_coordinate_ref.shape[-3:-1]
        grid = ray_coordinate_ref.view(-1, 1, H, W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
        features = F.grid_sample(self.feat_volume, grid, align_corners=True, mode='bilinear')[:, :, 0].permute(2, 3, 0,1).squeeze()
        return features


