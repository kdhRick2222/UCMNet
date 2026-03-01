import subprocess
from tqdm import tqdm
import os, sys, cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from utils.dataset_utils import TrainDataset, TestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim

from option import options as opt

from archs.arch_UCMNet import UCMNet
from losses.loss import SSIMloss, PSNRLoss, UDL_loss, HF_UDL_loss, HF_UDL_loss_normalized, VGGLoss
import lpips



def test_UDC(net, save_path, dataset):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], clean_patch, degrad_patch) in tqdm(testloader, dynamic_ncols=True):
            B, C, H, W = degrad_patch.shape
            B0, C0, H0, W0 = clean_patch.shape
            dH = H0%16
            dW = W0%16
            if dH!=0 or dW!=0:
                clean_patch = F.interpolate(clean_patch, (H0 - dH, W0 - dW),mode="bilinear")
                degrad_patch = F.interpolate(degrad_patch, (H0 - dH, W0 - dW),mode="bilinear")
            out_list, _ = net(degrad_patch[:, :3, :, :].cuda())
            restored = out_list[0]
            save_image(torch.cat([degrad_patch[:, :3, :, :].cuda(), torch.clip(restored, 0, 1), clean_patch.cuda()], dim=0), save_path + "predictions/" + clean_name[0] + ".png")
            
            if dH!=0 or dW!=0:
                restored = F.interpolate(restored, (H0 - dH, W0 - dW),mode="bilinear")
    
            save_image(torch.clip(restored, 0, 1), save_path + "predictions/" + clean_name[0] + ".png")
            temp_psnr, temp_ssim, N = compute_psnr_ssim(torch.clip(restored.detach(), 0, 1), clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

    return psnr.avg, ssim.avg


def log_configs(save_path, log_file='log.txt'):
    if os.path.exists(f'{save_path}/{log_file}'):
        log_file = open(f'{save_path}/{log_file}', 'a')
    else:
        log_file = open(f'{save_path}/{log_file}', 'w')
    return log_file


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    # print(out_str)


if __name__ == '__main__':

    torch.cuda.set_device(opt.cuda)
    
    save_path = 'result/' + opt.dataset + '/'
    subprocess.check_output(['mkdir', '-p', save_path + 'plots/'])
    subprocess.check_output(['mkdir', '-p', save_path + 'ckpt/'])
    subprocess.check_output(['mkdir', '-p', save_path + 'examples/'])
    subprocess.check_output(['mkdir', '-p', save_path + 'predictions/'])

    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=1, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)
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
    net = net.cuda()
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.99))

    opt.logfile = log_configs(save_path, log_file='train_log.txt')

    l1_loss = nn.L1Loss().cuda()
    l2_loss = nn.MSELoss().cuda()
    vgg_loss = VGGLoss().cuda() # this is for SYNTH only
    ssim_loss = SSIMloss()
    psnr_loss = PSNRLoss()
    # psnr_loss = nn.L1Loss().cuda() # this is for SYNTH only

    loss_total_list = []
    loss_PSNR_list = []
    loss_uncertainty_list = []
    loss_vgg_list = []
    
    if opt.dataset == "POLED":
        psnr_mean = [20]
        ssim_mean = [0.8]
    else:
        psnr_mean = [30]
        ssim_mean = [0.9]


    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        
        loss_total_sub = []
        loss_psnr_sub = []
        loss_uncertainty_sub = []
        loss_vgg_sub = []
        
        for step, ([clean_name], clean_patch, degrad_patch) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{opt.epochs}", dynamic_ncols=True), start=1):
            clean_patch = clean_patch.cuda()
            degrad_patch = degrad_patch.cuda()

            B0, C0, H0, W0 = clean_patch.shape
            dH = H0%16
            dW = W0%16
            if dH!=0 or dW!=0:
                clean_patch = F.interpolate(clean_patch, (H0 - dH, W0 - dW),mode="bilinear")
                degrad_patch = F.interpolate(degrad_patch, (H0 - dH, W0 - dW),mode="bilinear")

            optimizer.zero_grad()
            # torch.cuda.empty_cache()

            B, C, H, W = degrad_patch.shape
            
            out_list, var_list = net(degrad_patch[:, :3, :, :], side_loss = True)
            
            clean_list = [clean_patch]
            x = clean_patch
            clean_list += [F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)]
            clean_list += [F.interpolate(clean_list[-1], scale_factor=0.5, mode='bilinear', align_corners=False)]
            clean_list += [F.interpolate(clean_list[-1], scale_factor=0.5, mode='bilinear', align_corners=False)]
            clean_list += [F.interpolate(clean_list[-1], scale_factor=0.5, mode='bilinear', align_corners=False)]
            clean_list += [F.interpolate(clean_list[-1], scale_factor=0.5, mode='bilinear', align_corners=False)]

            # PSNR_loss
            loss_psnr_ = 0
            N = 1
            for i in range(len(out_list)-1):
                N *= 2
                loss_psnr_ += psnr_loss(clean_list[i+1], out_list[i+1])/N
            loss_psnr = loss_psnr_ + psnr_loss(clean_list[0], out_list[0])

            loss_uncertainty = HF_UDL_loss(out_list, clean_list, var_list) * 100

            if opt.dataset == 'SYNTH':
                loss_vgg = vgg_loss(clean_patch, out_list[0])
                loss = loss_psnr + loss_uncertainty + 0.1*loss_vgg 
            else:
                loss_vgg = loss_psnr
                loss = loss_psnr + loss_uncertainty

            loss = loss_psnr + loss_uncertainty #+ 0.1*loss_vgg 

            loss_total_sub.append(float(loss.item()))
            loss_psnr_sub.append(float(loss_psnr.item()))
            loss_uncertainty_sub.append(float(loss_uncertainty.item()))
            loss_vgg_sub.append(float(loss_vgg.item()))
            
            for i in range(len(out_list)):
                out_list[i].detach()
            for i in range(len(var_list)):
                var_list[i].detach()
            clean_patch.detach()

            # backward
            loss.backward()
            optimizer.step()

        length = len(loss_total_sub)
        loss_result_str = f'Epoch {epoch}  Loss: {sum(loss_total_sub)/length:5f}, PSNR loss: {sum(loss_psnr_sub)/length:5f}, Uncertainty loss: {sum(loss_uncertainty_sub)/length:5f}, VGG loss: {sum(loss_vgg_sub)/length:5f},'
        write_log(opt.logfile, loss_result_str)

        loss_total_list.append(sum(loss_total_sub)/length)
        loss_PSNR_list.append(sum(loss_psnr_sub)/length)
        loss_uncertainty_list.append(sum(loss_uncertainty_sub)/length)
        loss_vgg_list.append(sum(loss_vgg_sub)/length)

        GPUS = 1
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(net.state_dict(), save_path + "ckpt/" + 'epoch_' + str(epoch + 1) + '.pth')
            save_image(torch.cat([degrad_patch[:, :3, :, :], clean_patch, out_list[0]], dim=0), save_path + "examples/" + str(epoch+1) + "_examples.png")
        
        if (epoch + 1) % 10 == 0:
            X = np.linspace(0, epoch, epoch + 1)
            plt.plot(X, np.array(loss_total_list))
            plt.plot(X, np.array(loss_PSNR_list))
            plt.plot(X, np.array(loss_uncertainty_list))
            plt.plot(X, np.array(loss_vgg_list))
            plt.legend(('Total','PSNR loss', 'Uncertainty loss', 'VGG loss'))
            plt.savefig(save_path + 'plots/loss_plot.png')
            plt.cla()


        #### This part is for evaluation ####
        if (epoch + 1) % 10 == 0:
            net.eval()
            test_psnr, test_ssim = test_UDC(net, save_path, testset)
            psnr_mean.append(test_psnr)
            ssim_mean.append(test_ssim)
            eval_result_str = f'Epoch {epoch+1} >> {test_psnr:2f}/{test_ssim:4f}'
            print(opt.dataset, ' | ', eval_result_str)
            write_log(opt.logfile, eval_result_str)

            if (epoch + 1) % 100 == 0:
                # # PSNR
                X = np.linspace(0, epoch + 1, (epoch + 1)//10 + 1)
                plt.plot(X, np.array(psnr_mean))
                plt.legend(('PSNR'))
                plt.savefig(save_path + 'plots/PSNR.png')
                plt.cla()
                # SSIM
                X = np.linspace(0, epoch + 1, (epoch + 1)//10 + 1)
                plt.plot(X, np.array(ssim_mean))
                plt.legend(('SSIM'))
                plt.savefig(save_path + 'plots/SSIM.png')
                plt.cla()
        
    #### Return the best value!! ####
    psnr_max = np.argmax(psnr_mean)
    ssim_max = np.argmax(ssim_mean)
    print("For ", save_path.split('/')[1], " the best result is at", psnr_max*10, ssim_max*10)
    print("PSNR best - PSNR: ", psnr_mean[psnr_max], " SSIM: ", ssim_mean[psnr_max])
    print("SSIM best - PSNR: ", psnr_mean[ssim_max], " SSIM: ", ssim_mean[ssim_max])
    bestPSNR_eval_str = f'For PSNR: best epoch - {psnr_max*10}, PSNR/SSIM - {psnr_mean[psnr_max]:2f}/{ssim_mean[psnr_max]:4f}'
    write_log(opt.logfile, bestPSNR_eval_str)
    bestSSIM_eval_str = f'For SSIM: best epoch - {ssim_max*10}, PSNR/SSIM - {psnr_mean[ssim_max]:2f}/{ssim_mean[ssim_max]:4f}'
    write_log(opt.logfile, bestSSIM_eval_str)


