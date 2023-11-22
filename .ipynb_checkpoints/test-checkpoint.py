## import libraries for training
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


def save_topk_images(images, fnames, preds, k, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(images)):
        topk_pred = preds[i][:k]
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
        for j in range(k):
            img = Image.open(images[i])
            axes[j].imshow(img)
            axes[j].set_title(f"Prediction: {topk_pred[j]}")
            axes[j].axis("off")

        plt.savefig(os.path.join(save_dir, fnames[i].split('.')[0] + f"_top{k}_predictions.png"))


# Validating the model
def evaluate(val_loader,model, save_dir):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    accuracy_one = AverageMeter()
    accuracy_five = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
                # indx1 = torch.argmax(preds, dim=1)
                predsn = torch.argsort(logits, dim=1, descending=True)
            
            # print(indx1)
            # print(indx2)
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            accuracy_one.update(valid_acc1,img.size(0))
            accuracy_five.update(valid_acc5, img.size(0))

    save_topk_images([os.path.join(config.folder_path, fname) for fname in fnames], fnames, predsn, 5, os.path.join(save_dir, "top5_predictions"))
    # save_topk_images([os.path.join(config.folder_path, fname) for fname in fnames], fnames, predsn, 1, os.path.join(save_dir, "top1_predictions"))

            
    return map.avg, accuracy_one.avg, accuracy_five.avg

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)

        correct = top.eq(truth.view(-1, 1).expand_as(top))

        
        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

######################## load file and get splits #############################
print('reading test file')
test_files = pd.read_csv("test.csv")
print('Creating test dataloader')

test_gen = knifeDataset(config.folder_path,test_files,mode="val")
test_loader = DataLoader(test_gen,batch_size=64,shuffle=False,pin_memory=True,num_workers=8)

print('loading trained model')
model = timm.create_model('resnet50d', pretrained=False,num_classes=0)

model.head = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Linear(in_features=2048, out_features=768, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(768),
    nn.Dropout(0.4),
    nn.Linear(in_features=768, out_features=192, bias=False)
)

model.load_state_dict(torch.load('/mnt/fast/nobackup/scratch4weeks/ds01502/MLDataset-Knife/ModelFiles/resnet50d/50.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Training #################################

# Run the evaluate function and specify the directory to save the images
print('Evaluating trained model')
save_directory = "prediction_images/"
mAP, accuracy1, accuracy5 = evaluate(test_loader, model, save_directory)
print("mAP =",mAP)
print("Top-1 Accuracy =",accuracy1)
print("Top-5 Accuracy =",accuracy5)


   
