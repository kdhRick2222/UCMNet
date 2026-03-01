import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import pytorch_msssim
import numpy as np

from losses.loss_utils import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def log_mse_loss(pred, target):
    return torch.log(F.mse_loss(pred, target, reduction='none'))


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

# @weighted_loss
def psnr_loss(pred, target): # NCHW
    mseloss = F.mse_loss(pred, target, reduction='none').mean((1,2,3))
    psnr_val = 10 * torch.log10(1 / mseloss).mean()
    return psnr_val

    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        x = x.squeeze()
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h = self._tensor_size(x[:, 1:])
        count_w = self._tensor_size(x[1:, :])
        w_tv = torch.pow((x[:, 1:]-x[:, :w_x-1]), 2).sum()
        h_tv = torch.pow((x[1:, :]-x[:h_x-1, :]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)

    def _tensor_size(self,t):
        return t.size()[0]*t.size()[1]


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # return self.loss_weight * psnr_loss(pred, target, weight) * -1.0 
        return self.loss_weight * psnr_loss(pred, target) * -1.0 

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class FrequencyLoss(nn.Module):
    '''
    Calculates the amplitude of frequencies loss.
    '''
    def __init__(self, loss_weight = 0.01, criterion ='l2', reduction = 'mean'):
        super(FrequencyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')       
        self.loss_weight = loss_weight
        self.reduction = reduction

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        pred_freq = self.get_fft_amplitude(pred)
        target_freq = self.get_fft_amplitude(target)
        
        return self.loss_weight * self.criterion(pred_freq, target_freq)

    def get_fft_amplitude(self, inp):
        
        inp_freq = torch.fft.rfft2(inp, norm='backward')
        amp = torch.abs(inp_freq)
        return amp

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction)


#-----------------------------------------------------------------------------
# define the perceptual loss
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion = 'l1', reduction='mean'):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')
        
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weight = loss_weight

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss


class PerceptualLoss2(nn.Module):
    def __init__(self):
        super(PerceptualLoss2,self).__init__()
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        vgg = torchvision.models.vgg19(pretrained=True).eval()
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval()
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval()
    def forward(self,x,y):
        loss1 = self.L1(self.loss_net1(x),self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x),self.loss_net3(y))
        loss = 0.5*loss1+0.5*loss3
        return loss

#---------------------------------------------------------------
#define the edge loss to enhance the deblurring task
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion = 'l2',reduction='mean'):
        super(EdgeLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')        

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()
        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y, var):
        # loss = self.criterion(self.laplacian_kernel(x), self.laplacian_kernel(y))
        x = self.laplacian_kernel(x)
        y = self.laplacian_kernel(y)
        sr = torch.mul(x ,var)
        hr = torch.mul(y, var)
        loss = l1_loss(sr, hr) + 2*torch.mean(var)
        return loss*self.weight


def SSIM_loss(pred_img, real_img, data_range):
    SSIM_loss = pytorch_msssim.ssim(pred_img, real_img, data_range = data_range)
    return SSIM_loss

class SSIM(nn.Module):
    def __init__(self, loss_weight=1.0, data_range = 1.):
        super(SSIM, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * SSIM_loss(pred, target, self.data_range)

class SSIMloss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range = 1.):
        super(SSIMloss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * (1 - SSIM_loss(pred, target, self.data_range))

class LPIPSloss(nn.Module):
    def __init__(self, loss_weight=1.0, net_type='alex'):  # 'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        super(LPIPSloss, self).__init__()
        self.loss_weight = loss_weight
        self.lpips_loss = LPIPS(net_type=net_type).eval()

    def forward(self, pred, target, **kwargs):
        self.lpips_loss = self.lpips_loss.to(pred.device)
        return self.loss_weight * self.lpips_loss(pred, target)

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
        
class L_deblur(nn.Module):
    """L_deblur."""
    def __init__(self, loss_weight=1.0, gamma1 = 0.4, gamma2 = 0.2, gamma3 = 0.2, gamma4 = 0.2):
        super(L_deblur, self).__init__()
        self.loss_weight = loss_weight
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4

    def forward(self, X, Y):
        loss = self.gamma1 * l1_loss(X, Y) + self.gamma2 * mse_loss(X, Y) + self.gamma4 * SSIM_loss(X, Y)
        return self.loss_weight * loss

class L_enhance(nn.Module):
    """L_enhance."""
    def __init__(self, loss_weight=1.0, gamma1 = 0.5, gamma2 = 0.3, gamma3 = 0.2):
        super(L_enhance, self).__init__()
        self.loss_weight = loss_weight
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

    def forward(self, X, Y):
        loss = self.gamma1 * l1_loss(X, Y) + self.gamma2 * mse_loss(X, Y)
        return self.loss_weight * loss

class L_reblur(nn.Module):
    """L_enhance."""
    def __init__(self, loss_weight=1.0, gamma1 = 1.0):
        super(L_reblur, self).__init__()
        self.loss_weight = loss_weight
        self.gamma1 = gamma1

    def forward(self, X, Y):
        loss = self.gamma1 * l1_loss(X, Y)
        return self.loss_weight * loss
    
class EnhanceLoss(nn.Module):
    '''
    Applies the enhanceLoss. This loss is the l1 loss of the image downsampled at the middle of the
    encoder-decoder plus the l1 of the features of this downsample image given by the vgg19 (a perceptual
    element).
    '''
    def __init__(self, loss_weight=1.0, criterion = 'l1', reduction='mean'):
        super(EnhanceLoss, self).__init__()
        self.loss_weight = loss_weight
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss') 
        
        self.vgg19 = VGGLoss(loss_weight = 0.01,
                              criterion = criterion,
                              reduction = 'mean')
        
    def forward(self, gt, enhanced, scale_factor = 16):
        gt_low_res = F.interpolate(gt, scale_factor=scale_factor, mode = 'nearest')
        return self.vgg19(gt_low_res, enhanced) + self.loss_weight * self.criterion(gt_low_res, enhanced)
    
def UDL_loss(out_list, clean_list, var_list, consider_blur=True):
    
    loss_uncertarinty = 0

    for i in range(len(var_list)):            
        s = torch.exp(-var_list[i])
        sr = torch.mul(out_list[i] ,s)
        hr = torch.mul(clean_list[i], s)
        loss_uncertarinty += l1_loss(sr, hr) + 2*torch.mean(var_list[i])

    if consider_blur:
        gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=2)
        for j in range(4):            
            s = torch.exp(-var_list[j])
            sr = torch.mul(gaussian(out_list[j]) ,s)
            hr = torch.mul(gaussian(clean_list[j]), s)
            loss_uncertarinty += l1_loss(sr, hr) + 2*torch.mean(var_list[j])

    return loss_uncertarinty*0.25

def laplacian_hf(x):
    """Extract high-frequency component via Laplacian."""
    kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                          dtype=x.dtype, device=x.device).view(1,1,3,3)
    kernel = kernel.repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


def HF_UDL_loss(out_list, clean_list, var_list):
    
    loss_uncertarinty = 0
    N = 1

    for i in range(len(var_list)):
        H_sr = laplacian_hf(out_list[i])
        H_hr  = laplacian_hf(clean_list[i])
        s = torch.exp(-var_list[i])
        sr = torch.mul(H_sr ,s)
        hr = torch.mul(H_hr, s)
        loss_uncertarinty += l1_loss(sr, hr) / N #+ 2*torch.mean(var_list[i])
        N *= 2

    return loss_uncertarinty


def HF_UDL_loss_normalized(out_list, clean_list, var_list, eps=1e-6):
    """High-frequency uncertainty-driven loss with Laplacian normalization."""
    loss_uncertainty = 0.0
    N = 1.0

    for i in range(len(var_list)):
        # (1) Extract high-frequency components
        H_sr = laplacian_hf(out_list[i])
        H_hr = laplacian_hf(clean_list[i])

        # (2) Normalize Laplacian maps by their L2 norm (stabilize scale)
        norm_sr = torch.sqrt(torch.mean(H_sr ** 2, dim=[1, 2, 3], keepdim=True) + eps)
        norm_hr = torch.sqrt(torch.mean(H_hr ** 2, dim=[1, 2, 3], keepdim=True) + eps)
        H_sr_norm = H_sr / norm_sr
        H_hr_norm = H_hr / norm_hr

        # (3) Apply uncertainty weighting (exp(-var))
        s = torch.exp(-var_list[i])
        sr = H_sr_norm * s
        hr = H_hr_norm * s

        # (4) Compute normalized high-frequency UDL loss
        # loss_uncertainty += F.l1_loss(sr, hr) + 2*torch.mean(var_list[i]) / N
        loss_uncertainty += F.l1_loss(sr, hr) / N
        N *= 2

    return loss_uncertainty


