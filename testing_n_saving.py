import subprocess
from tqdm import tqdm
import os, sys, cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.dataset_utils import TrainDataset, TestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim

from option import options as opt
from archs.inference_UCMNet import UCMNet


def test_UDC(net, save_path, dataset):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], clean_patch, degrad_patch) in tqdm(testloader):
            print(clean_name)
            B, C, H, W = degrad_patch.shape
            out_list, var_list, uncertainty_list, score_list = net(degrad_patch[:, :3, :, :].cuda())
            restored = out_list[0]
            temp_psnr, temp_ssim, N = compute_psnr_ssim(torch.clip(restored.detach(), 0, 1), clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            # save prediction
            save_image(torch.clip(restored, 0, 1), save_path + clean_name[0] + f"_{temp_psnr:.2f}_{temp_ssim:.4f}.png")

    return psnr.avg, ssim.avg


if __name__ == '__main__':

    torch.cuda.set_device(opt.cuda)

    testset = TestDataset(opt)

    img_channel = 3
    width = 64
    block_list = [1, 1, 1, 1]
    enc_blks = block_list
    middle_blk_num_enc = 1
    middle_blk_num_dec = 1
    dec_blks = block_list
    extra_depth_wise = True
    net = UCMNet(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  extra_depth_wise = extra_depth_wise)

    ckpt_path = './checkpoints/' + opt.dataset + '.pth'

    net.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    net = net.cuda()
    net.eval()
    print("Model is loaded!!!!")

    psnr_list = []
    ssim_list = []

    # Start training
    print('Start training...')

    # #### This part is for evaluation ####
    save_folder = 'SaveVis/Final_model/'
    base_path = '/home/daehyun/Lowlight_models/UCMNet/' + save_folder

    subprocess.check_output(['mkdir', '-p', base_path + opt.dataset + '_Predictions/'])
    save_path = base_path + opt.dataset + '_Predictions/'

    test_psnr, test_ssim = test_UDC(net, save_path, testset)
    eval_result_str = f'Result >> {test_psnr:.2f}/{test_ssim:.4f}'
    print(opt.dataset, ' | ', eval_result_str)


