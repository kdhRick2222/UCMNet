import os
import random
import copy
from PIL import Image
import numpy as np
import glob

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize
import torchvision.transforms as T
import torch

from utils.image_utils import random_augmentation, crop_patch_pair
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F  # 함수형 API

# ---------- Paired transforms ----------
class PairedToTensor:
    def __call__(self, x, y):
        return F.to_tensor(x), F.to_tensor(y)  # [0,1]

class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, x, y):
        if random.random() < self.p:
            x = F.hflip(x)
            y = F.hflip(y)
        return x, y

class PairedRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, x, y):
        if random.random() < self.p:
            x = F.vflip(x)
            y = F.vflip(y)
        return x, y

class PairedCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y

# ---------- Dataset ----------
class TrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = self.args.dataset
        self._init_clean_ids(data_type=self.dataset)

        # torchvision for paired lq and gt
        self.transforms = PairedCompose([
            PairedToTensor(),                 # PIL -> Tensor (both)
            PairedRandomVerticalFlip(p=0.5), 
            PairedRandomHorizontalFlip(p=0.5)
        ])
        self.transforms_synth = PairedCompose([
            PairedToTensor(),                 # PIL -> Tensor (both)
        ])

        self.noise_min, self.noise_max = 0.0, 1e-3
        self.normalize = T.Compose([T.Normalize(mean=[0.0], std=[1.0]),])

    def _init_clean_ids(self, data_type):
        if data_type == 'TOLED':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/UDC_dataset/TOLED/train/HQ/*")
        elif data_type == 'POLED':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/UDC_dataset/POLED/train/HQ/*")
        elif data_type == 'LOLv2_real':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/LOLv2/Real_captured/Train/Normal/*")
        elif data_type == 'LOLv2_syn':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/LOLv2/Synthetic/Train/Normal/*")
        elif data_type == 'Align_before':
            name_list = glob.glob("/home/daehyun/Lowlight_models/AlignFormer/datasets/iphone_dataset/ref/train/*")
        elif data_type == 'Align_after':
            name_list = glob.glob("/home/daehyun/Lowlight_models/AlignFormer/datasets/iphone_dataset/AlignFormer/train/*")
        else:
            name_list = glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_1/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_2/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_3/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_4/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_5/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_6/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_7/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_8/train/*") +\
                        glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_9/train/*")
        self.clean_ids = name_list
        self.num_clean = len(self.clean_ids)

    def _tonemap(self, x, type='simple'):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x

    def __len__(self):
        return self.num_clean

    def __getitem__(self, idx):
        if self.dataset != "SYNTH":
            hq_path = self.clean_ids[idx]
            if self.dataset == "POLED" or self.dataset == "TOLED":
                lq_path = hq_path.replace('HQ', 'LQ')
            elif self.dataset == "Align_before":
                lq_path = hq_path.replace('ref', 'lq')
            elif self.dataset == "Align_after":
                lq_path = hq_path.replace('AlignFormer/train', 'lq/train')
            else:
                lq_path = hq_path.replace('Normal', 'Low')

            hq_img = Image.open(hq_path).convert('RGB')
            lq_img = Image.open(lq_path).convert('RGB')

            hq, lq = self.transforms(hq_img, lq_img)  # Tensor: [C,H,W], [0,1]
            lq = self.normalize(lq)

            sigma = torch.empty(1).uniform_(self.noise_min, self.noise_max).item()
            if sigma > 0:
                lq = lq + torch.randn_like(lq) * sigma
                lq = torch.clamp(lq, 0.0, 1.0)

            clean_name = hq_path.split("/")[-1].split('.')[0]
        
        else:
            lq_path = self.clean_ids[idx]
            lq_name = lq_path.split('/')[-1]
            hq_path = "/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/GT/train/" + lq_name
            
            lq_img = np.load(lq_path)
            hq_img = np.load(hq_path)
            lq_img = self._tonemap(lq_img) 
            hq_img = self._tonemap(hq_img) 

            hq, lq = self.transforms_synth(hq_img, lq_img)  # Tensor: [C,H,W], [0,1]
            # hq, lq = self.crop_patch_pair(hq, lq, 1, 1024)
            clean_name = hq_path.split("/")[-1].split('.')[0]

        return [clean_name], hq, lq

    def crop_patch_pair(self, im1, im2, ratio, pch_size):
        H = im1.shape[1]
        W = im1.shape[2]
        ind_H = random.randint(0, H - pch_size)
        ind_W = random.randint(0, W - pch_size*ratio)
        pch1 = im1[:, ind_H:ind_H + pch_size, ind_W:ind_W + pch_size*ratio]
        pch2 = im2[:, ind_H:ind_H + pch_size, ind_W:ind_W + pch_size*ratio]
        return pch1, pch2


class TestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = self.args.dataset
        self._init_clean_ids(data_type=self.dataset)
        self.normalize = T.Compose([T.Normalize(mean=[0.0], std=[1.0]),])

    def _init_clean_ids(self, data_type):
        if data_type == 'TOLED':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/UDC_dataset/TOLED/test/HQ/*")
        elif data_type == 'POLED':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/UDC_dataset/POLED/test/HQ/*")
        elif data_type == 'LOLv2_real':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/LOLv2/Real_captured/Test/Normal/*")
        elif data_type == 'LOLv2_syn':
            name_list = glob.glob("/home/daehyun/Lowlight_models/data/LOLv2/Synthetic/Test/Normal/*")
        elif data_type == 'Align_before':
            name_list = glob.glob("/home/daehyun/Lowlight_models/AlignFormer/datasets/iphone_dataset/ref/test_sub/*")
        elif data_type == 'Align_after':
            name_list = glob.glob("/home/daehyun/Lowlight_models/AlignFormer/datasets/iphone_dataset/AlignFormer/test_sub/*")
        else:
            name_list = glob.glob("/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/input/ZTE_new_5/test/*")
        self.clean_ids = sorted(name_list)
        self.num_clean = len(self.clean_ids)

    def _tonemap(self, x, type='simple'):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x
        
    def __len__(self):
        return self.num_clean

    def __getitem__(self, idx):
        if self.dataset != "SYNTH":
            hq_path = self.clean_ids[idx]
            if self.dataset == "POLED" or self.dataset == "TOLED":
                lq_path = hq_path.replace('HQ', 'LQ')
            elif self.dataset == "Align_before":
                lq_path = hq_path.replace('ref', 'lq')
            elif self.dataset == "Align_after":
                lq_path = hq_path.replace('AlignFormer/test_sub', 'lq/test_sub')
            else:
                lq_path = hq_path.replace('Normal', 'Low')
            clean_name = hq_path.split("/")[-1].split('.')[0]

            # --- PIL -> Tensor [C,H,W], [0,1] ---
            hq = F.to_tensor(Image.open(hq_path).convert('RGB'))  # (3,H,W)
            lq = F.to_tensor(Image.open(lq_path).convert('RGB'))  # (3,H,W)
            # lq = self.normalize(lq)

            _, H, W = hq.shape

            # torch.linspace & meshgrid
            y_coords = torch.linspace(-(H - 1) / 2, (H - 1) / 2, steps=H) / ((H - 1) / 2)  # (H,)
            x_coords = torch.linspace(-(W - 1) / 2, (W - 1) / 2, steps=W) / ((W - 1) / 2)  # (W,)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H,W)

            # (H,W) -> (1,H,W)
            y_channel = yy.unsqueeze(0)
            x_channel = xx.unsqueeze(0)

            # --- RGB+XY (5 channels) ---
            hq_5 = torch.cat([hq, x_channel, y_channel], dim=0)  # (5,H,W)
            lq_5 = torch.cat([lq, x_channel, y_channel], dim=0)  # (5,H,W)

        else:
            lq_path = self.clean_ids[idx]
            lq_name = lq_path.split('/')[-1]
            hq_path = "/home/daehyun/mnt/nas12/DISCNet/datasets/synthetic_data/GT/test/" + lq_name
            
            lq_img = np.load(lq_path)
            hq_img = np.load(hq_path)
            lq_img = self._tonemap(lq_img)
            hq_img = self._tonemap(hq_img)

            lq = F.to_tensor(lq_img)  # (3,H,W)
            hq = F.to_tensor(hq_img)  # (3,H,W)
            clean_name = hq_path.split("/")[-1].split('.')[0]

        return [clean_name], hq, lq
        