python main.py --model ssim_csrnet ^
--train-files .\\data\\train\\train_list.txt ^
--val-files .\\data\\train\\val_list.txt ^
--gpu-devices 0 --lr 1e-4 --optim adam ^
--checkpoints .\\checkpoints\\ssim_csrnet ^
--summary-writer .\\runs\\ssim_csrnet