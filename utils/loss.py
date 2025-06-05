import torch
import torch.nn as nn
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mse_weight = lambda x, y, w : torch.mean((x - y) ** 2 * w)
minmax = lambda x: (x-x.min())/(x.max()-x.min())
# con_cal = lambda x, y: torch.mean((x - y) ** 2, dim=1)
def con_cal(pred, gt):
    confidence = torch.mean((pred - gt) ** 2, dim=1)
    return minmax(-confidence)

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward_loss(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss
    
    def forward(self, depth_pred, depth_gt):
        # if None == mask:
        #     mask = depth_gt > 0
        loss = self.loss(depth_pred, depth_gt) * 2 ** (1 - 2)
        return loss
    
def compute_depth_loss(pred_depth, gt_depth):   
    # pred_depth_e = NDC2Euclidean(pred_depth_ndc)
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor
    
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    # M = torch.sum(mask, (1, 2))

    diff = prediction - target
    # diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    # mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    # grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    # mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    # grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    image_loss = torch.sum(image_loss)/prediction.shape[0]
    return image_loss

class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        self.__reduction = reduction_batch_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        # total = 0

        # for scale in range(self.__scales):
        #     step = pow(2, scale)
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        total = gradient_loss(prediction_ssi, target, mask, reduction=self.__reduction)

        return total

class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        # if reduction == 'batch-based':
        self.__reduction = reduction_batch_based
        # else:
        #     self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)
    
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total