#!/bin/bash
cd /mnt/fast/nobackup/users/ds01502/MLKnifeProject/EEEM066_Knife_Classification_code

#case - 5 batch size

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7batch64 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 64
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7batch64 --modelname tf_efficientnet_b7 --pretrain True


# #case - 6 weight decay
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7wtdc8 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 32 --wtdecay 0.00000001
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7wtdc8 --modelname tf_efficientnet_b7 --pretrain True

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7wtdc6 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 32 --wtdecay 0.000001
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7wtdc6 --modelname tf_efficientnet_b7 --pretrain True

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7wtdc12 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 32 --wtdecay 0.000000000001
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7wtdc12 --modelname tf_efficientnet_b7 --pretrain True

# #case - 7 class head
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7head32_2 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 32 --wtdecay 0.00000001 --classhead True --dropout 0.2
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7head32_2 --modelname tf_efficientnet_b7 --pretrain True --classhead True --dropout 0.8

# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7head32_4 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 32 --wtdecay 0.00000001 --classhead True --dropout 0.4
# /mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7head32_4 --modelname tf_efficientnet_b7 --pretrain True --classhead True --dropout 0.8

/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7head16_6 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 16 --wtdecay 0.00000001 --classhead True --dropout 0.6
/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7head16_6 --modelname tf_efficientnet_b7 --pretrain True --classhead True --dropout 0.8

/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python train.py --exname tf_efficientnet_b7head16_8 --modelname tf_efficientnet_b7 --pretrain True  --lr 0.00005 --optim adam --trainbatchsize 16 --wtdecay 0.00000001 --classhead True --dropout 0.8
/mnt/fast/nobackup/users/ds01502/miniconda3/envs/mlproject/bin/python test.py --exname tf_efficientnet_b7head16_8 --modelname tf_efficientnet_b7 --pretrain True --classhead True --dropout 0.8

