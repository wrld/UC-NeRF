import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os, imageio
import PIL.Image
from torchvision import transforms as T
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
from .ray_utils import *
from utils.run_nerf_helpers import *
import random
from pathlib import Path
import cv2
import tqdm
from configparser import ConfigParser
from utils.common import interp_poses
import operator
from utils.utils import get_nearest_pose_ids
# config_object = ConfigParser()
TINY_NUMBER = 1e-5
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

import numpy as np
def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
    )
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    assert (
        R1.shape[-1] == 3
        and R2.shape[-1] == 3
        and R1.shape[-2] == 3
        and R2.shape[-2] == 3
    )
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1)
            / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )
def center_poses(poses, blender2opencv):
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:
                  3] = pose_avg  # convert to homogeneous coordinate for faster computation
    last_row = np.tile(np.array([0, 0, 0, 1]),
                       (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(
        pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    
    return poses_centered, (np.linalg.inv(pose_avg_homo) @ blender2opencv)[:, :3]


def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    num_select,
    tar_id=-1,
    angular_dist_method="dist",
    scene_center=(0, 0, 0),
):
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == "matrix":
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3]
        )
    elif angular_dist_method == "vector":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == "dist":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception("unknown angular distance calculation method!")

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]

    return selected_ids

def average_poses(poses):
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses, c2w

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                        [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5,
                                       radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class HamlynDataset(Dataset):

    def __init__(self,
                 args,
                 split='train',
                 n_views=3):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = os.path.join(args.datadir)
        self.downsample = 1.0
        self.sample_rate = 2
        
        self.img_wh = (int(320 * self.downsample),
                    int(256 * self.downsample))
        self.split = split
        self.finetune = args.finetune
        print("set downsample: ", self.downsample, self.img_wh)
        assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
            'image width must be divisible by 32, you may need to modify the imgScale'
        
        self.nviews = n_views
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                        [0, 0, -1, 0], [0, 0, 0, 1]])
        self.build_metas()

        self.define_transforms()
        self.white_back = False
        
    

    def build_metas(self):
        self.metas = []
        self.scans = os.listdir(self.root_dir)
        self.scans.sort()
        if self.finetune is not None:
            self.scans = [self.finetune]
        self.data_dict = {
            k: {
                'bounds': None,
                'root_dir': None,
                'img_paths': None,
                'focal': None,
                'poses': None,
                'pose_avg': None,
                'directions': None,
                'near_original': None,
                'scale_factor': None,
                'depth_gts': None
            }
            for k in self.scans
        }
        print("all scans", self.scans)

        self.id_list = []
        self.near_far = [10, 0]
        for scan in tqdm.tqdm(self.scans):
            self.data_dict[scan]['root_dir'] = os.path.join(
                self.root_dir, scan)
            
            
            self.data_dict[scan]['img_paths'] = sorted(
                glob.glob(
                    os.path.join(self.data_dict[scan]['root_dir'],
                                 'images/*')))

            self.data_dict[scan]['depth_root'] = os.path.join(self.data_dict[scan]['root_dir'],
                                 'depth/')
            
            
            poses_bounds = np.load(
                os.path.join(self.data_dict[scan]['root_dir'],
                             'poses_bounds.npy'))  # (N_images, 17)
            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
            self.data_dict[scan]['bounds'] = poses_bounds[:, -2:]  # (N_images, 2)
            self.data_dict[scan]['raw_bounds'] = poses_bounds[:,-2:].transpose([1, 0])
            self.data_dict[scan]['depth_gts'] = self.load_colmap_depth(
                os.path.join(self.root_dir, scan),
                factor=1/self.downsample,
                bd_factor=.75,
                bds_raw=poses_bounds[:,-2:].transpose([1, 0]))
          
            H, W, focal = poses[0, :,-1]  # original intrinsics, same for all images
            self.data_dict[scan]['focal'] = [
                focal * self.img_wh[0] / W, focal * self.img_wh[1] / H
            ]

            poses = np.concatenate(
                [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            self.data_dict[scan]['scale_factor'] = 1./(self.data_dict[scan]['bounds'].min() * 0.75)  # 0.75 is the default parameter

            self.data_dict[scan]['bounds'] *= self.data_dict[scan]['scale_factor']
            poses[..., 3] *= self.data_dict[scan]['scale_factor']
            poses, pose_avg = center_poses(poses, self.blender2opencv)
            self.data_dict[scan]['poses'], self.data_dict[scan][
                'pose_avg'] = poses, pose_avg
            ids = np.arange(len(self.data_dict[scan]['img_paths']))

            self.data_dict[scan]['train_index'] = ids[int(self.sample_rate/2)::self.sample_rate]
            self.data_dict[scan]['test_index'] = np.array([i for i in ids if i not in self.data_dict[scan]['train_index']])
            
            self.test_num_perscene = len(self.data_dict[scan]['test_index']) 
            if self.split=='train': 
                num_samples = 200
            elif self.split == "val":
                num_samples = self.test_num_perscene
            
            for k in range(num_samples):
                if self.split == "train":
                    random_select = self.data_dict[scan]['train_index']
                    random.shuffle(random_select)
                    ref_view = random_select[0]
                    src_views = np.array(random_select[1:self.nviews]).tolist()
                    
                elif self.split == "val":
                    ref_view =  self.data_dict[scan]['test_index'][k]
                    src_views = get_nearest_pose_ids(poses[ref_view], poses[self.data_dict[scan]['train_index']], self.nviews-1)
                    src_views = np.array(self.data_dict[scan]['train_index'])[src_views]
                    src_views = src_views.tolist()
                    
                self.metas += [(scan, ref_view, src_views)]
                self.id_list.append([ref_view] + src_views)
                        
    def get_poses(self, images):
        poses = []
        for i in images:
            R = images[i].qvec2rotmat()
            t = images[i].tvec.reshape([3, 1])
            bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
            w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(w2c)
            poses.append(c2w)
        return np.array(poses)

    def load_colmap_depth(self,
                          basedir,
                          ids=None,
                          bds_raw=None,
                          factor=8,
                          bd_factor=.75):
        data_file = Path(basedir) / 'colmap_depth.npy'
        images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
        points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')
        Errs = np.array([point3D.error for point3D in points.values()])
        Err_mean = np.mean(Errs)
        
        poses = self.get_poses(images)
        bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
        sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
        data_list = []
        all_images = []
        for id_im in range(1, len(images)+1):
            all_images.append(images[id_im].name)
        perm = np.argsort(all_images)
        for id_im in range(1, len(images)+1):
            depth_list = []
            coord_list = []
            weight_list = []
            depth_img = np.zeros((self.img_wh[1], self.img_wh[0]))
            weight_img = np.zeros((self.img_wh[1], self.img_wh[0]))
            
            # print(images[id_im].name)
            for i in range(len(images[id_im].xys)):
                point2D = images[id_im].xys[i]
                point2D = np.array([point2D[1], point2D[0]])
                id_3D = images[id_im].point3D_ids[i]
                if id_3D == -1:
                    continue
                point3D = points[id_3D].xyz
                depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
                if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                    continue
                err = points[id_3D].error
                weight = 2 * np.exp(-(err/Err_mean)**2)
                w, h = int((point2D/factor)[1]), int((point2D/factor)[0])
                if w>=self.img_wh[0] or h >= self.img_wh[1]:
                    continue
                
                depth_list.append(depth)
                coord_list.append(np.array([h, w]))
                weight_img[h, w] = weight
                depth_img[h, w] = depth
                weight_list.append(weight)
            if len(depth_list) > 0:
                data_list.append({"name":images[id_im].name, "depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list), "depth_img": depth_img, "weight_img": weight_img})
            else:
                print(id_im, len(depth_list))
        save_lists = []
        for i in perm:
            save_lists.append(data_list[i])
        np.save(data_file, save_lists)
        return save_lists
    
    def read_depth(self, filename, dpt=False):
        # print(filename)
        if dpt == False:
            depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) 
            depth_h = cv2.resize(depth, None, fx=self.downsample, fy=self.downsample,
                                interpolation=cv2.INTER_NEAREST)  # (600, 800)
        else:
            depth = np.load(filename)['pred']
            if depth.shape[0] == 1:
                depth = depth[0]
            depth_h = cv2.resize(depth, self.img_wh)  # (600, 800)
        return depth_h                                                                                                      

    def load_colmap_llff(self, basedir):
        basedir = Path(basedir)

        train_imgs = np.load(basedir / 'train_images.npy')
        test_imgs = np.load(basedir / 'test_images.npy')
        train_poses = np.load(basedir / 'train_poses.npy')
        test_poses = np.load(basedir / 'test_poses.npy')
        video_poses = np.load(basedir / 'video_poses.npy')
        depth_data = np.load(basedir / 'train_depths.npy', allow_pickle=True)
        bds = np.load(basedir / 'bds.npy')

        return train_imgs, test_imgs, train_poses, test_poses, video_poses, depth_data, bds

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i

    def define_transforms(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, target_view, src_views = self.metas[idx]
        
        view_ids = [target_view] + src_views


        rays_depth_list = []

        affine_mat, affine_mat_inv = [], []
        imgs = []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i, vid in enumerate(view_ids):
            near_far = [self.data_dict[scan]["bounds"].min()*0.9, self.data_dict[scan]["bounds"].max()*1.1]

            near_fars.append(near_far)
            if i == 0:
                index = vid
                sparse_depth_img = self.data_dict[scan]["depth_gts"][index]['depth_img']
                weight_img = self.data_dict[scan]["depth_gts"][index]['weight_img']
                weight_img -= weight_img.min()
                weight_img /= weight_img.max()
                h, w = sparse_depth_img.shape
        
                sparse_depths_ms = {
                    "stage1": cv2.resize(sparse_depth_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                    "stage2": cv2.resize(sparse_depth_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
                    "stage3": sparse_depth_img,
                }
                weight_ms = {
                    "stage1": cv2.resize(weight_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                    "stage2": cv2.resize(weight_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
                    "stage3": weight_img,
                }
                depth_value = np.repeat(
                    self.data_dict[scan]["depth_gts"][index]['depth'][:, None, None],
                    3,
                    axis=2)  # N x 1 x 3
                depth_coord =  self.data_dict[scan]["depth_gts"][index]['coord']
                
                depth_coord = np.concatenate((depth_coord.reshape(-1, 1, 2), np.ones((depth_coord.shape[0], 1, 1))), axis=2)
                
                weights = np.repeat(
                    self.data_dict[scan]["depth_gts"][index]['weight'][:, None,
                                                                    None],
                    3,
                    axis=2)  # N x 1 x 3
                weights -= weights.min()
                weights /= weights.max()
                depth_value = np.concatenate([depth_value, weights, depth_coord],
                                            axis=1)  # N x 4 x 3
                
                rays_depth_list.append(depth_value)
            
                img_filename = self.data_dict[scan]["img_paths"][index]
                depth_filename = img_filename.replace('images', 'depths').replace('jpg', 'png')
                dpt_filename = img_filename.replace('images', 'dpt').replace('jpg', 'npz')
                depth_h = self.read_depth(depth_filename)
                dpt = self.read_depth(dpt_filename, True)
                
            
        
            img_filename = self.data_dict[scan]["img_paths"][vid]
            img_ori = PIL.Image.open(img_filename)
            img = img_ori.resize(self.img_wh, PIL.Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]
        
            c2w = torch.eye(4).float()
            c2w[:3] = torch.FloatTensor(self.data_dict[scan]["poses"][vid])
            w2c = torch.inverse(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            intrinsic = torch.tensor(
                [[self.data_dict[scan]["focal"][0], 0, self.img_wh[0] / 2],
                [0, self.data_dict[scan]["focal"][1], self.img_wh[1] / 2],
                [0, 0, 1]]).float()

            intrinsics.append(intrinsic.clone())

            aff = []
            aff_inv = []
            for scale in range(3):
                proj_mat_l = torch.eye(4)
                stage_intrinsic = intrinsic.clone()
                stage_intrinsic[:2] = intrinsic[:2] / (2**(2-scale))
                proj_mat_l[:3, :4] = stage_intrinsic @ w2c[:3, :4]
                aff.append(proj_mat_l)
                aff_inv.append(np.linalg.inv(proj_mat_l))
            aff = np.stack(aff)
            aff_inv = np.stack(aff_inv)
            affine_mat.append(aff)
            affine_mat_inv.append(aff_inv)
            proj_mat_l = torch.tensor(aff[2])
            if i == 0:  # reference view
                ref_proj_inv = torch.inverse(proj_mat_l)
                proj_mats += [torch.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]

            
        imgs = torch.stack(imgs).float()
        proj_mats = np.stack(proj_mats)[:, :3]
            
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(
            affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(
            w2cs), np.stack(c2ws), np.stack([near_fars])
        
        rays_depth = np.concatenate(rays_depth_list, axis=0)

        
        sample['images'] = imgs  # (V, H, W, 3)
        np.random.shuffle(rays_depth)
        rays_depth = rays_depth[:1024]# (1024, 4, 3)
        sample['depths_h'] = depth_h  # (V, H, W)
        sample['dpt'] = dpt  # (V, H, W)
        sample['sparse_depths_ms'] = sparse_depths_ms
        sample['sparse_depths'] = sparse_depth_img
        sample['sparse_depths_weight'] = weight_img
        sample['weight_ms'] = weight_ms
        sample['rays_depth'] = rays_depth.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)

        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)

        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        sample['scan'] = self.scans.index(scan)
        return sample