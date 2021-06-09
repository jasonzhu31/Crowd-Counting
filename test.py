import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import cv2
import torchvision

from models.sobel import CSRNet as sobel_csrnet
from models.sobelrgb import CSRNet as sobelrgb_csrnet
from models.ssim_csrnet import CSRNet as ssim_csrnet

from dataloader.sobel_dataloader import get_train_shanghaitechpartA_dataloader as sobel_train_dataloader 
from dataloader.sobel_dataloader import get_test_shanghaitechpartA_dataloader as sobel_test_dataloader
from dataloader.sobelrgb_dataloader import get_train_shanghaitechpartA_dataloader as sobelrgb_train_dataloader 
from dataloader.sobelrgb_dataloader import get_test_shanghaitechpartA_dataloader as sobelrgb_test_dataloader
from dataloader.ssim_csrnet_dataloader import get_train_shanghaitechpartA_dataloader as ssim_train_dataloader 
from dataloader.ssim_csrnet_dataloader import get_test_shanghaitechpartA_dataloader as ssim_test_dataloader

# from nwpudataloader import create_test_nwpu_dataloader
import numpy as np
import time
import os
import sys
import errno
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test crowdcounting model')

parser.add_argument('--test-files', type=str, default='')
parser.add_argument('--best-model', type=str, default='')
parser.add_argument('--use-avai-gpus', action='store_true')
parser.add_argument('--gpu-devices', type=str, default='0')
parser.add_argument('--model', type=str, default='CSRNet')
parser.add_argument('--test-batch', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--result', type=str, default='')

# parser.add_argument('--checkpoints', type=str, default='./checkpoints')

args = parser.parse_args()

criterion = nn.MSELoss(reduction='sum')

if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()
use_gpu = False

if use_gpu:
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
else:
    print("Currently using CPU (GPU is highly recommended)")

global model, total_criterion, test_loader

if args.model == 'sobel':
    model = sobel_csrnet()
    total_criterion = nn.MSELoss(reduction='sum')
    test_loader = sobel_test_dataloader(file_list=args.test_files)
elif args.model == 'sobelrgb':
    model = sobelrgb_csrnet()   
    total_criterion = nn.MSELoss(reduction='sum')
    test_loader = sobelrgb_test_dataloader(file_list=args.test_files)
elif args.model == 'ssim_csrnet':
    model = ssim_csrnet()
    total_criterion = nn.MSELoss(reduction='sum')
    test_loader = ssim_test_dataloader(file_list=args.test_files)

if os.path.isfile(args.best_model):
    pkl = torch.load(args.best_model)
    state_dict = pkl['state_dict']
    # print("Currently epoch {}".format(pkl['epoch']))
    # model.load_state_dict(state_dict)
    model.load_state_dict({k.replace("module.",""):v for k,v in state_dict.items()})


model.eval()

with torch.no_grad():
    epoch_mae = 0.0
    epoch_rmse_loss = 0.0
    if __name__ == '__main__':
        for i, data in enumerate(tqdm(test_loader)):
            image = data['image']
            gt_densitymap = data['densitymap']
            if use_gpu:
                image = data['image'].cuda()
                gt_densitymap = data['densitymap'].cuda()
                
            et_densitymap = model(image).detach()

            result_path = args.result + '\\IMG_{}.jpg'.format(i)
            img_np = et_densitymap.numpy()
            img_np = img_np[0].transpose([1,2,0])
            img_np = (img_np - np.min(img_np))/(np.max(img_np) - np.min(img_np)) * 255.0
            img_np = img_np.astype(np.uint8)
            img_cv = cv2.applyColorMap(img_np, cv2.COLORMAP_JET)
            cv2.imwrite(result_path, img_cv)

            mae = abs(et_densitymap.data.sum() - gt_densitymap.sum())
            rmse = mae * mae

            epoch_mae += mae.item()
            epoch_rmse_loss += rmse.item()

        epoch_mae /= len(test_loader.dataset)
        epoch_rmse_loss = math.sqrt(epoch_rmse_loss / len(test_loader.dataset))
print("bestmae: ", epoch_mae)
print("rmse: ", epoch_rmse_loss)



