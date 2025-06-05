
from opt import config_parser
from torch.utils.data import DataLoader
import imageio
from data import dataset_dict
from network.models import create_ucnerf
from network.renderer import *
from utils.utils import *
import time
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from network.mvs_models import cas_mvsnet_loss, EdgePreservingSmoothnessLoss
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from utils.evaluation import depth_evaluation, rgb_evaluation
from utils.loss import SL1Loss, GradientLoss
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UCNeRFSystem(LightningModule):

    def __init__(self, args):
        super(UCNeRFSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 24 + (args.view_num - 1) * (4 + 8) + 1
        self.learning_rate = args.lrate
        self.idx = 0
        self.validation_step_outputs = []
        # Create ucnerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_ucnerf(
            args, dir_embedder=True, pts_embedder=True)
        # Create Consistency Learner
        self.Consist_Learner = self.render_kwargs_train['network_mvs']
        filter_keys(self.render_kwargs_train)
        self.loss = SL1Loss()
        self.smooth_loss = EdgePreservingSmoothnessLoss()
        self.edge_loss = GradientLoss()

        self.render_kwargs_train.pop('network_mvs')
        self.render_kwargs_train['NDC_local'] = False
        self.eval_metric = [0.01, 0.05, 0.1]

    def decode_batch(self, batch, idx=list(torch.arange(4))):

        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
        pose_ref = {
            'w2cs': data_mvs['w2cs'].squeeze(),
            'intrinsics': data_mvs['intrinsics'].squeeze(),
            'c2ws': data_mvs['c2ws'].squeeze(),
            'near_fars': data_mvs['near_fars'].squeeze()
        }

        return data_mvs, pose_ref

    def unpreprocess(self, data, shape=(1, 1, 3, 1, 1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224,
                             -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224,
                            1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def forward(self):
        return

    def prepare_data(self):
        dataset = dataset_dict[self.args.dataset_name]
        self.train_dataset = dataset(args,
                                     split='train',
                                     n_views=self.args.view_num)
        self.train_sampler = None
        self.val_dataset = dataset(args,
                                   split='val',
                                   n_views=self.args.view_num)

    def configure_optimizers(self):
        eps = 1e-7
        variable_dict = [{"params": self.grad_vars, "lr": self.learning_rate}]
        self.optimizer = torch.optim.Adam(variable_dict, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.args.num_epochs,
                                      eta_min=eps)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            num_workers=8,
            batch_size=1,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        depth_sparse_ms = batch['sparse_depths_ms']
        weight_ms = batch['weight_ms']

        if 'scan' in batch.keys():
            batch.pop('scan')
            batch.pop('sparse_depths_ms')
            batch.pop('weight_ms')
        data_mvs, pose_ref = self.decode_batch(batch)

        imgs = data_mvs['images']
        near_fars = pose_ref['near_fars']
        dpt = data_mvs['dpt']
        sparse_depths = data_mvs['sparse_depths']
        sparse_weights = data_mvs['sparse_depths_weight']
        batch_depth = data_mvs['rays_depth']

        batch_depth = batch_depth.squeeze(0)
        batch_depth = torch.transpose(batch_depth, 0, 1)
        target_depths = batch_depth[0, :, 0]
        target_weights = batch_depth[1, :, 0]
        depth_coord = batch_depth[2, :, 0:2]

        affine_mat, affine_mat_inv = data_mvs['affine_mat'][0], data_mvs[
            'affine_mat_inv'][0]
        imgs_input = imgs[:, 1:]
        volume_feature, uncertainty_map, mvs_depth, outputs = self.Consist_Learner(
            imgs_input,
            affine_mat,
            affine_mat_inv,
            near_fars[0],
            pad=args.pad)
        imgs = self.unpreprocess(imgs)

        N_rays, N_samples = args.batch_size, args.N_samples
        c2ws, w2cs, intrinsics = pose_ref['c2ws'], pose_ref['w2cs'], pose_ref[
            'intrinsics']
        rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters, pixel_coordinates = \
            build_rays(args, imgs, uncertainty_map, sparse_depths, depth_coord, pose_ref, w2cs, c2ws, intrinsics,\
                 N_rays, N_samples, pad=args.pad, with_depth=True, outputs=outputs)

        rgb, depth_pred = rendering(args,
                                    pose_ref,
                                    rays_pts,
                                    rays_NDC,
                                    depth_candidates,
                                    rays_dir,
                                    outputs,
                                    imgs[:, 1:],
                                    near_fars=near_fars[0],
                                    img_feat=outputs['stage3']['img_feats'],
                                    confidence=uncertainty_map,
                                    ndc_parameters=ndc_parameters,
                                    **self.render_kwargs_train)
        patch_pts = args.patch_num * args.patch_size * args.patch_size

        target_depths = sparse_depths[:, pixel_coordinates[0, N_rays:],
                                      pixel_coordinates[1, N_rays:]]
        target_weights = sparse_weights[:, pixel_coordinates[0, N_rays:],
                                        pixel_coordinates[1, N_rays:]]

        loss_mvs, _ = cas_mvsnet_loss(outputs, depth_sparse_ms, weight_ms)
        patch_depth = depth_pred[:patch_pts].reshape(-1, args.patch_size,
                                                     args.patch_size)
        patch_dpt = dpt[:, pixel_coordinates[0, :patch_pts],
                        pixel_coordinates[1, :patch_pts]].reshape(
                            -1, args.patch_size, args.patch_size, 1)
        smooth_loss = self.smooth_loss(patch_depth[:args.patch_num // 2, ...],
                                       patch_dpt[:args.patch_num // 2, ...])
        loss_nerf_depth = torch.mean(
            ((depth_pred[N_rays:] - target_depths)**2) * target_weights)
        loss_scaleinvariant = self.edge_loss(
            patch_depth[args.patch_num // 2:, ...],
            patch_dpt[args.patch_num // 2:, ...].squeeze(-1),
            torch.ones_like(patch_depth[args.patch_num // 2:, ...]).to(device))
        depth_loss = loss_nerf_depth * 0.05 + loss_mvs * 0.05 + smooth_loss * 0.05 + loss_scaleinvariant * 0.008
        
        img_loss = img2mse(rgb, target_s)
        loss = depth_loss + img_loss * 5.0
        psnr = mse2psnr2(img_loss.item())

        with torch.no_grad():
            if self.global_step % 5000 == 4999:
                self.save_ckpt(f'{self.global_step}')
            if self.args.log:
                self.log('train/loss', loss, prog_bar=True)
                self.log('train/img_mse_loss', img_loss.item(), prog_bar=False)
                self.log('train/PSNR', psnr.item(), prog_bar=True)
                self.log('train/depth loss', loss.item(), prog_bar=True)
                self.log('train/smooth loss',
                         smooth_loss.item(),
                         prog_bar=True)
                self.log('train/loss_scaleinvariant',
                         loss_scaleinvariant.item(),
                         prog_bar=True)
                self.log('train/depth loss_mvs',
                         loss_mvs.item(),
                         prog_bar=False)
                self.log('train/depth loss_nerf_depth',
                         loss_nerf_depth.item(),
                         prog_bar=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        scene = batch['scan']

        if 'scan' in batch.keys():
            batch.pop('scan')
            batch.pop('sparse_depths_ms')
            batch.pop('weight_ms')
        log = {}
        data_mvs, pose_ref = self.decode_batch(batch)
        imgs, proj_mats = data_mvs['images'], data_mvs['proj_mats']
        near_fars = pose_ref['near_fars']
        depths_h = data_mvs['depths_h']
        dpt = data_mvs['dpt']
        self.Consist_Learner.train()
        H, W = imgs.shape[-2:]
        H, W = int(H), int(W)

        ##################  rendering #####################
        keys = ['gt_depth', 'pred_depth', 'gt_rgb', 'pred_rgb', 'mask']
        log = init_log(log, keys)
        with torch.no_grad():
            world_to_ref = pose_ref['w2cs'][0]
            imgs_input = imgs[:, 1:]
            affine_mat, affine_mat_inv = data_mvs['affine_mat'][0], data_mvs[
                'affine_mat_inv'][0]
            volume_feature, photo_confidence, mvs_depth, outputs = self.Consist_Learner(
                imgs_input,
                affine_mat,
                affine_mat_inv,
                near_fars[0],
                pad=args.pad)

            imgs = self.unpreprocess(imgs)
            imgs_input = imgs[:, 1:]
            tgt_to_world, intrinsic = pose_ref['c2ws'][0], pose_ref[
                'intrinsics'][0]
            rgbs, depth_preds = [], []

            for chunk_idx in range(H * W // args.chunk +
                                   int(H * W % args.chunk > 0)):

                rays_pts, rays_dir, rays_NDC, depth_candidates, rays_o, ndc_parameters = \
                    build_rays_test(H, W, tgt_to_world, world_to_ref, intrinsic, near_fars, \
                            near_fars[-1], args.N_samples, pad=args.pad, chunk=args.chunk, idx=chunk_idx, outputs=outputs)

                # rendering
                rgb, depth_pred = rendering(
                    args,
                    pose_ref,
                    rays_pts,
                    rays_NDC,
                    depth_candidates,
                    rays_dir,
                    outputs,
                    imgs_input,
                    near_fars=near_fars[0],
                    img_feat=outputs["stage3"]['img_feats'],
                    confidence=photo_confidence,
                    ndc_parameters=ndc_parameters,
                    **self.render_kwargs_train)

                rgbs.append(rgb.cpu())
                depth_preds.append(depth_pred.cpu())

            render_rgb, render_depth = torch.clamp(
                torch.cat(rgbs).reshape(H, W, 3).permute(2, 0, 1), 0,
                1), torch.cat(depth_preds).reshape(H, W)
            depth_gt = depths_h[0].cpu()
            gt_rgb = imgs.cpu()[0, 0]
            log['pred_depth'] = render_depth
            log['gt_depth'] = depth_gt
            log['pred_rgb'] = render_rgb
            log['gt_rgb'] = gt_rgb
            log['mask'] = depth_gt > 0
            depth_gt_render_vis = visualize_depth(depth_gt)
            depth_pred_r_ = visualize_depth(render_depth)
            photo_confidence = self.render_kwargs_train['network_fn'].forward_uncertainty(photo_confidence.reshape(1, -1, 1)).reshape(H, W)

            save_path = f'{self.args.basedir}/{self.args.expname}/test_results'
            os.makedirs(save_path, exist_ok=True)

            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(gt_rgb.permute(1, 2, 0))  
            axs[0, 0].set_title('Ground Truth RGB')
            axs[0, 0].axis('off') 

            axs[0, 1].imshow(render_rgb.permute(1, 2, 0).cpu().numpy())  
            axs[0, 1].set_title('Rendered RGB')
            axs[0, 1].axis('off')

            axs[1, 0].imshow(depth_gt_render_vis.permute(1, 2, 0).cpu().numpy())  
            axs[1, 0].set_title('Ground Truth Depth')
            axs[1, 0].axis('off')

            axs[1, 1].imshow(depth_pred_r_.permute(1, 2, 0).cpu().numpy())
            axs[1, 1].set_title('Rendered Depth')
            axs[1, 1].axis('off')

           
            plt.tight_layout()
            # Save the figure
            plt.savefig(
                f'{save_path}/{self.global_step:08d}_{self.idx:02d}.png')
            plt.close(fig)


            self.idx += 1

        del rays_NDC, rays_dir, rays_pts, volume_feature
        self.validation_step_outputs.append(log)

        return log

    def on_validation_epoch_end(self):

        pred_rgb = torch.stack(
            [x['pred_rgb'] for x in self.validation_step_outputs])
        pred_depth = torch.stack(
            [x['pred_depth'] for x in self.validation_step_outputs])

        gt_rgb = torch.stack(
            [x['gt_rgb'] for x in self.validation_step_outputs])
        gt_depth = torch.stack(
            [x['gt_depth'] for x in self.validation_step_outputs])
        mask = torch.stack([x['mask'] for x in self.validation_step_outputs])
        test_num = 10
        all_rgb_errors = []
        all_depth_errors = []
        all_mvs_depth_errors = []
        if pred_rgb.shape[0] > 1:
            for i in range(len(self.val_dataset.scans)):
                print(self.val_dataset.scans[i])
                psnr, ssim, lpips = rgb_evaluation(
                    gt_rgb[i * test_num:(i + 1) * test_num, ...].cpu().numpy(),
                    pred_rgb[i * test_num:(i + 1) * test_num,
                             ...].cpu().numpy(),
                    savedir=
                    f'{self.args.basedir}/{self.args.expname}/test_results/')
                depth_errors = depth_evaluation(
                    gt_depth[i * test_num:(i + 1) * test_num,
                             ...].cpu().numpy(),
                    pred_depth[i * test_num:(i + 1) * test_num,
                               ...].cpu().numpy(),
                    pred_masks=mask[i * test_num:(i + 1) * test_num,
                                    ...].cpu().numpy(),
                    savedir=
                    f'{self.args.basedir}/{self.args.expname}/test_results/')

                all_rgb_errors.append([psnr, ssim, lpips])
                all_depth_errors.append(depth_errors)

        else:
            psnr, ssim, lpips = rgb_evaluation(
                gt_rgb.cpu().numpy(),
                pred_rgb.cpu().numpy(),
                savedir=f'{self.args.basedir}/{self.args.expname}/test_results/'
            )
            depth_errors = depth_evaluation(
                gt_depth.cpu().numpy(),
                pred_depth.cpu().numpy(),
                pred_masks=mask.cpu().numpy(),
                savedir=f'{self.args.basedir}/{self.args.expname}/test_results/'
            )

            all_rgb_errors.append([psnr, ssim, lpips])
            all_depth_errors.append(depth_errors)
        all_rgb_errors = np.stack(all_rgb_errors).mean(axis=0)
        all_depth_errors = np.stack(all_depth_errors).mean(axis=0)
        if self.args.log:
            self.log('val/PSNR', all_rgb_errors[0], prog_bar=False)
            self.log('val/SSIM', all_rgb_errors[1], prog_bar=False)
            self.log('val/LPIPS', all_rgb_errors[2], prog_bar=False)
            self.log('val/abs_rel', all_depth_errors[0], prog_bar=False)
            self.log('val/sq_rel', all_depth_errors[1], prog_bar=False)
            self.log('val/rmse', all_depth_errors[2], prog_bar=False)
            self.log('val/rmse_log', all_depth_errors[3], prog_bar=False)
            self.log('val/a1', all_depth_errors[4], prog_bar=False)
            self.log('val/a2', all_depth_errors[5], prog_bar=False)
            self.log('val/a3', all_depth_errors[6], prog_bar=False)
        result_RGB = 'psnr: {0}, ssim: {1}, lpips: {2}'.format(
            all_rgb_errors[0], all_rgb_errors[1], all_rgb_errors[2])
        result_depth = 'abs_rel: {0}, sq_rel: {1}, rmse: {2}, rmse_log: {3}, a1: {4}, a2: {5}, a3: {6}'.format(
            all_depth_errors[0], all_depth_errors[1], all_depth_errors[2],
            all_depth_errors[3], all_depth_errors[4], all_depth_errors[5],
            all_depth_errors[6])

        print(result_RGB)
        print(result_depth)
        self.validation_step_outputs = []
        return

    def save_ckpt(self, name='latest'):
        save_dir = f'{self.args.basedir}/{self.args.expname}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {'network_fn_state_dict':
            self.render_kwargs_train['network_fn'].state_dict(),
            'network_mvs_state_dict':
            self.Consist_Learner.state_dict()
        }
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    system = UCNeRFSystem(args)
    checkpoint_callback = ModelCheckpoint(os.path.join(
        f'{args.basedir}/{args.expname}/ckpts/', '{epoch:02d}'),
                                          monitor='val/PSNR',
                                          mode='max',
                                          save_top_k=0)
    if args.log:
        logger = WandbLogger()
    else:
        logger = False
    args.num_gpus, args.use_amp = 1, False
    trainer = Trainer(max_epochs=args.num_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      enable_progress_bar=True,
                      devices=args.num_gpus,
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch=2,
                      benchmark=True,
                      precision=16 if args.use_amp else 32)

    if not args.eval:
        trainer.fit(system)
    else:
        trainer.validate(system)
    system.save_ckpt()
