import torch
import torch.nn as nn
import torch.nn.functional as cudnn
import torchvision

from torch.optim.lr_scheduler import StepLR  # optional

import os
import sys
import errno
import argparse
import math

from tqdm import tqdm
from tensorboardX import SummaryWriter  # optional

from optimizers.optimizers import init_optim

from models.sobel import CSRNet as sobel_csrnet
from models.sobelrgb import CSRNet as sobelrgb_csrnet
from models.ssim_csrnet import CSRNet as ssim_csrnet

from dataloader.sobel_dataloader import get_train_shanghaitechpartA_dataloader as sobel_train_dataloader 
from dataloader.sobel_dataloader import get_test_shanghaitechpartA_dataloader as sobel_test_dataloader
from dataloader.sobelrgb_dataloader import get_train_shanghaitechpartA_dataloader as sobelrgb_train_dataloader 
from dataloader.sobelrgb_dataloader import get_test_shanghaitechpartA_dataloader as sobelrgb_test_dataloader
from dataloader.ssim_csrnet_dataloader import get_train_shanghaitechpartA_dataloader as ssim_train_dataloader 
from dataloader.ssim_csrnet_dataloader import get_test_shanghaitechpartA_dataloader as ssim_test_dataloader

#from loss.MixLoss import MixLoss
from losses.pytorch_msssim import MSSSIM



parser = argparse.ArgumentParser(description='Train crowdcounting model')

parser.add_argument('--model', type=str, default='CSRNet')
parser.add_argument('--train-files', type=str)
parser.add_argument('--val-files', type=str)

parser.add_argument('--resume', type=str, default='')
parser.add_argument('--use-avai-gpus', action='store_true', default=True)
parser.add_argument('--gpu-devices', type=str, default='0')

parser.add_argument('--max-epoch', type=int, default=2000)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-05)
parser.add_argument('--weight-decay', default=5e-04, type=float)
parser.add_argument('--train-batch', type=int, default=1)
parser.add_argument('--val-batch', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--checkpoints', type=str, default='./checkpoints')
parser.add_argument('--summary-writer', type=str, default='./runs')
parser.add_argument('--save-txt', type=str, default='train_log.txt')

args = parser.parse_args()

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
         
        try: 
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def main():
    mkdir_if_missing(args.checkpoints)
    mkdir_if_missing(args.summary_writer)
    print("model name: ", args.model)
    print("trainset filename: ", args.train_files)
    print("valset filename: ", args.val_files)
    print("training batch: ", args.train_batch)
    print("val batch: ", args.val_batch)
    print("optimizer: ", args.optim)
    print("learning rate: ", args.lr)
    print("weight decay: ", args.weight_decay)
    #print(torch.cuda.is_available())

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    use_gpu = False

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    global model, total_criterion, train_loader, val_loader

    if args.model == 'sobel':
        model = sobel_csrnet()
        total_criterion = nn.MSELoss(reduction='sum')
        train_loader = sobel_train_dataloader(labeled_file_list=args.train_files, use_flip=True, batch_size=args.train_batch)
        val_loader = sobel_test_dataloader(file_list=args.val_files)
    elif args.model == 'sobelrgb':
        model = sobelrgb_csrnet()   
        total_criterion = nn.MSELoss(reduction='sum')
        train_loader = sobelrgb_train_dataloader(labeled_file_list=args.train_files, use_flip=True, batch_size=args.train_batch)
        val_loader = sobelrgb_test_dataloader(file_list=args.val_files)
    elif args.model == 'ssim_csrnet':
        model = ssim_csrnet()
        total_criterion = MSSSIM()
        train_loader = ssim_train_dataloader(labeled_file_list=args.train_files, use_flip=True, batch_size=args.train_batch)
        val_loader = ssim_test_dataloader(file_list=args.val_files)

    print("Currently using {} model".format(args.model))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    min_mae = sys.maxsize

    start_epoch = 0
    if not args.resume == '':
        if os.path.isfile(args.resume):
            pkl = torch.load(args.resume)
            state_dict = pkl['state_dict']
            print("Currently epoch {}".format(pkl['epoch']))
            model.load_state_dict(state_dict)
            start_epoch = pkl['epoch']
            if ('mae' in pkl.keys()):
                min_mae = pkl['mae']
    print("Currently epoch {}".format(start_epoch))

    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    min_mae_epoch = -1

    writer = SummaryWriter(args.summary_writer)

    with open(os.path.join(args.checkpoints, args.save_txt), 'a') as f:
        for epoch in range(start_epoch, start_epoch + args.max_epoch):
            model.train()
            epoch_loss = 0.0

            for i, label_data in enumerate(tqdm(train_loader)):
                label_image = label_data['image']
                gt_densitymap = label_data['densitymap']
                if use_gpu:
                    label_image = label_data['image'].cuda()
                    gt_densitymap = label_data['densitymap'].cuda()

                et_densitymap = model(label_image)

                if args.model == 'sobel' or args.model == 'sobelrgb':
                    loss = total_criterion(et_densitymap, gt_densitymap)
                elif args.model == 'ssim_csrnet':
                    loss = 1 - total_criterion(gt_densitymap, et_densitymap)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # (optional)
            # print(optimizer.param_groups[0]['lr'])
            # scheduler.step()


            writer.add_scalar('Train_Loss', epoch_loss / len(train_loader.dataset), epoch)
            f.write("epoch: " + str(epoch) + " training_loss: " + str(epoch_loss / len(train_loader.dataset)) + "\n")
            print("epoch: ", epoch, " training_loss: ", epoch_loss / len(train_loader.dataset))

            model.eval()
            with torch.no_grad():
                epoch_mae = 0.0
                epoch_rmse_loss = 0.0
                for i, data in enumerate(tqdm(val_loader)):
                    image = data['image']
                    gt_densitymap = data['densitymap']
                    
                    if use_gpu:
                        image = data['image'].cuda()
                        gt_densitymap = data['densitymap'].cuda()
                    
                    et_densitymap = model(image).detach()

                    mae = abs(et_densitymap.data.sum() - gt_densitymap.sum())
                    rmse = mae * mae

                    epoch_mae += mae.item()
                    epoch_rmse_loss += rmse.item()

                epoch_mae /= len(val_loader.dataset)
                epoch_rmse_loss = math.sqrt(epoch_rmse_loss / len(val_loader.dataset))

                f.write("epoch: " + str(epoch) + " bestmae: " + str(min_mae) + " testmae: " + str(epoch_mae) + " testrmse: " + str(epoch_rmse_loss) + "\n")

                if epoch_mae <= min_mae:
                    min_mae, min_mae_epoch = epoch_mae, epoch
                    torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'mae': epoch_mae,
                    }, os.path.join(args.checkpoints, "bestvalmodel.pth"))

                writer.add_scalar('Val_MAE', epoch_mae, epoch)
                print("epoch: ", epoch, " bestmae: ", min_mae, "testmae: ", epoch_mae, " testrmse: ", epoch_rmse_loss)

    f.close()
    return


if __name__ == '__main__':
    main()