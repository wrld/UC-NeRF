import os
import cv2
import numpy as np
import torch
import lpips
import skimage

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def depth_evaluation(gt_depths, pred_depths, savedir=None, pred_masks=None, min_depth=0.0001, max_depth=100):
    assert gt_depths.shape[0] == pred_depths.shape[0]

    gt_depths_valid = []
    pred_depths_valid = []
    errors = []
    num = gt_depths.shape[0]
    for i in range(num):
        gt_depth = gt_depths[i]
        mask = (gt_depth > min_depth) * (gt_depth < max_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = pred_depths[i]
        if pred_masks is not None:
            pred_mask = pred_masks[i]
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_width, gt_height)) > 0.5
            mask = mask * pred_mask

        if mask.sum() == 0:
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        pred_depths_valid.append(pred_depth)
        gt_depths_valid.append(gt_depth)

    ratio = np.median(np.concatenate(gt_depths_valid)) / \
                np.median(np.concatenate(pred_depths_valid))

    for i in range(len(pred_depths_valid)):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        pred_depth *= ratio
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    return mean_errors
    
def rgb_evaluation(gts, predicts, savedir):
    assert gts.max() <= 1
    gts = gts.astype(np.float32)
    predicts = predicts.astype(np.float32)
    ssim_list = []
    lpips_list = []
    mse = ((gts - predicts)**2).mean(-1).mean(-1).mean(-1)
    psnr = (-10*np.log10(mse)).mean()
    lpips_metric = lpips.LPIPS(net='alex', version='0.1')
    gts_torch = torch.from_numpy((2*gts - 1)).type(torch.FloatTensor)
    predicts_torch = torch.from_numpy((2*predicts - 1)).type(torch.FloatTensor)
    lpips_ = lpips_metric(gts_torch, predicts_torch).mean()

    for i in range(gts.shape[0]):
        gt = gts[i]
        predict = predicts[i]
        gt = np.moveaxis(gt, 0, -1)
        predict = np.moveaxis(predict, 0, -1)
        ssim_list.append(skimage.metrics.structural_similarity(gt, predict, data_range=1,multichannel=True, channel_axis=2))

    ssim = np.array(ssim_list).mean()
    
    with open(os.path.join(savedir, 'rgb_evaluation.txt'), 'w') as f:
        result = 'psnr: {0}, ssim: {1}, lpips: {2}'.format(psnr, ssim, lpips_)
        print(result)
    return psnr, ssim, lpips_
