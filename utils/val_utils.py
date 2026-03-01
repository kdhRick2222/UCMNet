import torch
import time
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math


# ---- 사용 함수 (네 코드 대체) ----
def tensor2im(image_tensor, imtype=np.uint8, is_scale=False, scale=(1024, 512)):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    if is_scale:
        image_numpy_scale = cv2.resize(image_numpy, dsize=scale, interpolation=cv2.INTER_NEAREST)
        out = image_numpy_scale
    else:
        out = image_numpy
    return out.astype(imtype)

def SSIM_bnudc(img1, img2):
    # ssim_val = structural_similarity(img1, img2, gaussian_weights=True, use_sample_covariance=False, channel_axis=-1, data_range=1.0)
    ssim_val = structural_similarity(img1, img2, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
    return ssim_val

def PSNR_bnudc(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX / mse)

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += PSNR_bnudc(clean[i], recoverd[i])
        ssim += SSIM_bnudc(clean[i], recoverd[i])
        # ssim += SSIM(clean[i], recoverd[i])

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0



