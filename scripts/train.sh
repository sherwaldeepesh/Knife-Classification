#!/bin/bash
cd /mnt/fast/nobackup/users/ds01502/MLKnifeProject/EEEM066_Knife_Classification_code

/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --modelname resnet50 --pretrain True

/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --modelname resnet50 --pretrain True

