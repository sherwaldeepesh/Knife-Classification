#!/bin/bash
cd /mnt/fast/nobackup/users/ds01502/MLKnifeProject/EEEM066_Knife_Classification_code

/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname wideResNet16Classhead8ArcFaceLoss --modelname wide_resnet50_2 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 16 --wtdecay 0.00000001 --classhead True --dropout 0.8

/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname wideResNet16Classhead8ArcFaceLoss --modelname wide_resnet50_2 --pretrain True --classhead True --dropout 0.8

