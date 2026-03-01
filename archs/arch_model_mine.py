import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

try:
    from .arch_util_mine import LayerNorm2d
except:
    from arch_util_mine import LayerNorm2d


from torch.nn.modules.utils import _pair, _quadruple


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
# import basicsr.archs.blocks as blocks
from einops import rearrange
from torchvision.utils import make_grid, save_image


def standardize_per_channel(x, eps=1e-8):
    """
    x: [N, H, W] or [N,1,H,W] 형태
    채널별로 평균 0, 표준편차 1로 정규화
    """
    if x.dim() == 3:  # [C,H,W]
        x = x.unsqueeze(1)  # [C,1,H,W]
    B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    flat = x.view(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(eps)
    normed = ((flat - mean) / std).view_as(x)
    return normed

def save_feature_maps(feat, filename="features.png", max_ch=16):
    feat = feat.detach().cpu()
    if feat.dim() == 4:
        feat = feat[0]          # [C,H,W]
    csel = min(feat.shape[0], max_ch)
    normed = standardize_per_channel(feat[:csel])  # [C,1,H,W]
    grid = make_grid(normed, nrow=int(csel**0.5), padding=2)
    save_image(grid, filename)
    print(f"Feature maps saved to {filename}")

def save_kernels(weights, filename="kernels.png", max_out=16):
    weights = weights.detach().cpu()
    weights_norm = (weights-weights.min()) / (weights.max() - weights.min())
    save_image(weights_norm.repeat(1, 3, 1, 1), filename)
    print(f"Kernels saved to {filename}")


class DynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x))))))
        weight = weight.view(b * self.channels, 1, self.kernel_size, self.kernel_size)
        # save_kernels(weight.detach().cpu(), "/home/daehyun/Lowlight_models/UDC_folder/Kernel_vis/kernels.png")
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


# net = DynamicDWConv(channels=64, kernel_size=3, stride=1, groups=64)#.cuda()
# x1 = torch.randn(1, 64, 256, 256)#.cuda()
# y = net(x1)
# print(y.shape)

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
'''
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
'''

# --------------------------------
# --------------------------------
def get_uperleft_denominator(img, kernel):
    ker_f = convert_psf2otf(kernel, img.size())  # complex FFT of kernel
    nsr = wiener_filter_para(img)
    inv_kernel = inv_fft_kernel_est(ker_f, nsr.cuda())  # complex tensor

    # Numerator: FFT of blurred input
    numerator = torch.fft.fft2(img)
    # Element-wise deconvolution in frequency domain
    deblur_f = numerator.cuda() * inv_kernel
    # Inverse FFT and keep real part
    deblur = torch.fft.ifft2(deblur_f).real
    return deblur

# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[2])
    mean_n = torch.sum(diff, (2,3)).view(-1,diff.shape[1],1,1)/num
    # print(diff.shape, num, mean_n.shape) #torch.Size([8, 64, 70, 70]) 4900 torch.Size([512, 1, 1, 1])
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2,3))/(num-1)
    mean_input = torch.sum(_input_blur, (2,3)).view(-1,diff.shape[1],1,1)/num
    var_s2 = (torch.sum((_input_blur-mean_input)*(_input_blur-mean_input), (2,3))/(num-1))**(0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1,diff.shape[1],1,1)
    return NSR

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    inv_denominator = (ker_f.real ** 2 + ker_f.imag ** 2) + NSR
    inv_ker_f = ker_f.conj() / inv_denominator  # element-wise complex division
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # 복소수 텐서 곱
    deblur_f = inv_ker_f * fft_input_blur  # element-wise complex multiply
    deblur = torch.fft.ifft2(deblur_f).real  # inverse FFT 후 실수값만 추출
    return deblur

# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size):
    psf = torch.zeros(size).cuda()
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    otf = torch.fft.fftn(psf, dim=(-3, -2, -1))
    return otf

# --------------------------------
# --------------------------------


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
        # return x1 * torch.sigmoid(x2)

class Adapter(nn.Module):
    
    def __init__(self, c, ffn_channel = None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2
        else:
            ffn_channel = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1)

    def forward(self, input):
        
        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)
        
        return x

class FreMLP(nn.Module):
    
    def __init__(self, nc, expand = 2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 3, 1, 1),
            )
        # self.fft = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        _, _, H_fft, W_fft = x_freq.shape
        x_freq = x_freq #+ x_freq*self.fft
        
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out



class FreMLP_new(nn.Module):
    def __init__(self, nc, expand=2, patch_size=8):
        super().__init__()
        self.nc = nc
        self.patch_size = patch_size

        # magnitude 조정용 conv 블록 (spatial domain)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 3, 1, 1),
        )

        # 패치 주파수 마스크 (실수 스케일): (1, C, 1, 1, P, P//2+1)
        self.freq_mask = nn.Parameter(
            torch.ones(1, nc, 1, 1, patch_size, patch_size // 2 + 1),
            requires_grad=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # 1) 패딩해서 P 배수 맞추기
        pad_h = (P - (H % P)) % P
        pad_w = (P - (W % P)) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[2], x.shape[3]  # padded size
        Hb, Wb = Hp // P, Wp // P

        # 2) 패치로 분해: (B,C,Hb,Wb,P,P)
        x_patch = rearrange(x, "b c (hb p1) (wb p2) -> b c hb wb p1 p2", p1=P, p2=P)

        # 3) 패치 rFFT: (B,C,Hb,Wb,P,P//2+1) - complex
        x_patch_fft = torch.fft.rfft2(x_patch.float(), dim=(-2, -1), norm="backward")

        # 4) 주파수 마스크 곱 (실수 스케일을 복소수에 브로드캐스트)
        x_patch_fft = x_patch_fft * self.freq_mask.to(x_patch_fft.dtype)

        # 5) mag/pha 분리
        mag = torch.abs(x_patch_fft)            # (B,C,Hb,Wb,P,P//2+1) real
        pha = torch.angle(x_patch_fft)          # same shape, real

        # 6) mag를 패치-공간으로 내려서 conv 처리
        #    (B,C,Hb,Wb,P,P) ← iFFT(mag)
        mag_spatial = torch.fft.irfft2(mag, s=(P, P), dim=(-2, -1), norm="backward")
        #    (B,C,Hp,Wp) ← depatchify
        mag_spatial = rearrange(mag_spatial, "b c hb wb p1 p2 -> b c (hb p1) (wb p2)")
        #    conv 처리 (B,C,Hp,Wp) → (B,C,Hp,Wp)
        mag_proc_spatial = self.process1(mag_spatial)
        #    다시 패치화 (B,C,Hb,Wb,P,P)
        mag_proc_patch = rearrange(mag_proc_spatial, "b c (hb p1) (wb p2) -> b c hb wb p1 p2", p1=P, p2=P)
        #    다시 rFFT로 주파수 도메인 (B,C,Hb,Wb,P,P//2+1), real
        mag_proc_fft = torch.fft.rfft2(mag_proc_patch.float(), dim=(-2, -1), norm="backward")

        # 7) mag_proc + pha로 복소수 스펙트럼 재구성
        # mag_proc_fft: complex
        magitude = torch.abs(mag_proc_fft)    # real Tensor
        real = magitude * torch.cos(pha)      # real Tensor
        imag = magitude * torch.sin(pha)      # real Tensor

        real_4d = rearrange(real, "b c hb wb p1 p2 -> b c (hb p1) (wb p2)")
        imag_4d = rearrange(imag, "b c hb wb p1 p2 -> b c (hb p1) (wb p2)")

        x_patch_fft_out = torch.complex(real_4d, imag_4d)  # OK

        # 8) 패치 iFFT → (B,C,Hb,Wb,P,P)
        x_patch_out = torch.fft.irfft2(x_patch_fft_out, s=(H, W), norm='backward')

        # 10) crop to (H,W)
        if pad_h or pad_w:
            x_patch_out = x_patch_out[:, :, :H, :W]

        return x_patch_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)
    
class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.act = nn.GELU()
        
    def forward(self, inp, adapter = None):

        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        # z = self.act(z)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        # x = self.conv4(y)
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]
        x = y + x * self.gamma
        
        return x, x


class DBlock_freq(nn.Module):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        ## This part is for CCM
        self.k_pred = nn.Linear(512, 3)  # 예시 MLP 레이어
        # Optional: Color Correction Matrix (CCM)
        self.ccm = nn.Parameter(torch.eye(3))
        
    def forward(self, inp, adapter = None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # y = self.beta * inp + x

        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        # x = self.conv4(y)
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]
        x = y + x * self.gamma
        
        return x 
  
class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=1)
        # self.freq = FreMLP_new(nc=c, expand=1, patch_size=8)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.act = nn.GELU()
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)

    def forward(self, inp):

        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        z = self.act(z)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        # without
        y_low = y
        # blur version
        # y_low = self.gaussian(y)

        #second step
        x_step2 = self.norm2(y_low) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2) # size [B, C, H, W]

        # x_high = x_freq * self.gamma
        # x_high = y_high + x_high

        x_high = y + x_freq * self.gamma
        return x_high, x_high



class EBlock_dynamic(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        # self.sca = nn.Sequential(
        #                nn.AdaptiveAvgPool2d(1),
        #                nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #                groups=1, bias=True, dilation = 1),  
        # )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=1)
        self.dynamic_conv = DynamicDWConv(channels=c, kernel_size=3, stride=1, groups=c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.act = nn.GELU()

    def forward(self, inp):

        x = self.norm1(inp)
        x = self.conv1(self.act(self.extra_conv(x)))
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        z = self.act(z)
        z = self.sg1(z)
        # x = self.sca(z) * z
        x = self.dynamic_conv(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        #second step
        x_step2 = self.norm2(y) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2) # size [B, C, H, W]

        x_high = y + x_freq * self.gamma

        return x_high, x_high


class Dynamic_Affine_block(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        # self.dynamic_conv = DynamicDWConv(channels=c, kernel_size=3, stride=1, groups=c)
        self.dynamic_conv = DynamicDWConv(channels=c, kernel_size=7, stride=1, groups=c)
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=1)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.act = nn.GELU()
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)

    def forward(self, inp):

        x_freq = self.norm1(inp) # size [B, 2*C, H, W]
        x_freq = self.freq(x_freq) # size [B, C, H, W]
        x = inp + inp * x_freq * self.gamma

        x_blur = self.gaussian(x)
        x_hf0 = x - x_blur

        x_hf = self.norm2(x_hf0)
        x_hf = self.conv1(self.extra_conv(x_hf))
        z = 0
        for branch in self.branches:
            z += branch(x_hf)
        
        z = self.sg1(z)
        z = self.sca(z) * z
        A = self.dynamic_conv(z)
        # save_feature_maps(z[0].detach().cpu(), "/home/daehyun/Lowlight_models/UDC_folder/Kernel_vis/featmap_before.png")
        # save_feature_maps(A[0].detach().cpu(), "/home/daehyun/Lowlight_models/UDC_folder/Kernel_vis/featmap_after.png")
        # save_feature_maps(A[0].detach().cpu(), "/home/daehyun/Lowlight_models/UDC_folder/Kernel_vis1/featmap_after.png")
        A = self.conv3(A)

        x_high = A * x_hf0 * self.beta
        x_low = x #x_blur
        y = x_high + x_low

        return y, x_high


class Gaussian_block(nn.Module):
    '''
    Change this block using Branch
    '''
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True, dilation = 1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=1)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.act = nn.GELU()

    def forward(self, inp):
        # step1
        x_step1 = self.norm1(inp) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step1) # size [B, C, H, W]
        x = inp + x_freq * self.gamma
        x_low = x
        x_hf = x

        x_hf = self.norm2(x_hf)
        x_hf = self.conv1(self.extra_conv(x_hf))
        z = 0
        for branch in self.branches:
            z += branch(x_hf)
        
        z = self.sg1(z)
        x_hf = self.sca(z) * z
        x_high = self.conv3(x_hf)
        y = x_low + x_high * self.beta
        
        return y, y


#----------------------------------------------------------------------------------------------


if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    # net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net  = EBlock(c = img_channel, dilations = dilations, extra_depth_wise=extra_depth_wise)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    output, _ = net(torch.randn((4, 3, 256, 256)))
    # print('Values of EBlock:')
    print(macs, params)

    channels = 128
    resol = 32
    ksize = 5

