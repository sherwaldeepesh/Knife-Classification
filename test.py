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

print(f"\n Model : {config.model_name}")

read_f = pd.read_csv('Label_id_list.csv')

def save_topk_image(read_f,images, fnames, probs, preds, k, save_dirs, index,label):
    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)
    
    # Increase figure size or adjust the aspect ratio as needed
    plt.figure(figsize=(4*k, 4*k))
    
    # Shorten the titles or reduce font size
    title_fontsize = 10  # Or another number that fits your titles well

    _,axes = plt.subplots(1,k+1, figsize = (4*k, 4))

    img = Image.open(images[index])
    axes[0].imshow(img)
    axes[0].set_title(f'Input Image: {label[index]}', fontsize=title_fontsize)
    axes[0].axis('off')
    for j in range(1,k+1):
        # print(int(preds[index][j-1]))
        img = Image.open("".join([config.folder_path,str(read_f[read_f['Label']==int(preds[index][j-1])]['IdImg'].values[0])[2:]]))
        name_img = str(read_f[read_f['Label']==int(preds[index][j-1])]['IdName'].values[0])
        # print(len(images))
        axes[j].imshow(img)
        # Wrap the title or manually insert newline characters as needed
        axes[j].set_title(f'Prediction {j}: {preds[index][j-1]} {name_img}', fontsize=title_fontsize, wrap=True)    #(Prob : {probs[index][j-1]*100:.4f})
        axes[j].axis('off')
    # print(fnames)
    # Adjust subplot parameters to fit titles
    plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dirs, fnames[index].split('.')[0] + f'_top_{k}_predictions.png'))

    
# Validating the model
def evaluate(read_f,val_loader,model, save_dirs):
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
                preds = torch.softmax(logits, dim=1)    #.softmax(1)

                _,predsn = torch.topk(preds, 5, dim=1)
            # print(preds)
            for index in range(min(50, len(images))):
                save_topk_image(read_f,[os.path.join(config.folder_path, fname) for fname in fnames], fnames, preds, predsn, 5, os.path.join(save_dirs, f'sample_{index}_pred'), index,label)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            accuracy_one.update(valid_acc1,img.size(0))
            accuracy_five.update(valid_acc5, img.size(0))



            
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
if config.head == False:
    model = timm.create_model(config.model_name, pretrained=config.inputs.pretrain, num_classes = 192)
else:
    model = timm.create_model(config.model_name, pretrained=config.inputs.pretrain, num_classes = 0)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(2560),
        nn.Linear(in_features=2560, out_features=768, bias=False),
        nn.ReLU(),
        nn.BatchNorm1d(768),
        nn.Dropout(config.dropout),
        nn.Linear(in_features=768, out_features=192, bias=False)
    )

model.load_state_dict(torch.load(f'/mnt/fast/nobackup/scratch4weeks/ds01502/MLDataset-Knife/ModelFiles/{config.name}/{config.model_name}/50.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Training #################################

# Run the evaluate function and specify the directory to save the images
print('Evaluating trained model')
save_dir = "/".join(["Pred_img",config.model_name,f'{config.name}/'])
mAP, accuracy1, accuracy5 = evaluate(read_f,test_loader, model,save_dir)
print("mAP =",mAP)
print("Top-1 Accuracy =",accuracy1)
print("Top-5 Accuracy =",accuracy5)


   
