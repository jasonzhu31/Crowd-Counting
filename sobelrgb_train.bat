python main.py --model sobelrgb ^
--train-files .\\data\\train\\train_list.txt ^
--val-files .\\data\\train\\val_list.txt ^
--gpu-devices 0 --lr 1e-4 --optim adam ^
--checkpoints .\\checkpoints\\sobelrgb ^
--summary-writer .\\runs\\sobelrgb