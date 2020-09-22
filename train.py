import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import DataProcess.download_data as dataprocess
import net_model.dpn as dpn
import net_model.densenet as densenet
import net_model.resnet as resnet
import net_model.gcnet as gcnet
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5), #依据概率p对PIL图片进行水平翻转
    transforms.RandomVerticalFlip(0.5),   #依据概率p对PIL图片进行垂直翻转
    torchvision.transforms.RandomAffine(degrees=(-30,30), translate=(0.1,0.1), scale=(0.9,1.1)),#仿射变换
    transforms.ToTensor(),
    normalize
])

test_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])


#****************训练过程****************#
def train(model,optimizer, epoch, train_loader):
    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):
        if opt.use_gpu:
            data, target = batch_samples['img'].cuda(), batch_samples['label'].cuda()
        else:
            data, target = batch_samples['img'], batch_samples['label']

        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        loss.backward()
        optimizer.step()


        train_loss += loss
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

    print('\nTrain set{}/{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, opt.n_epochs,train_loss / len(train_loader.dataset), train_correct, 
        len(train_loader.dataset),100.0 * train_correct / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset) , train_correct / len(train_loader.dataset)

#**************评估代码************
def prediction(model,val_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for _, batch_samples in enumerate(val_loader):
            if opt.use_gpu:
                data, target = batch_samples['img'].cuda(), batch_samples['label'].cuda()
            else:
                data, target = batch_samples['img'], batch_samples['label']

            output = model(data)
            test_loss += criteria(output, target.long())
            score = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist


def evaluate(targetlist, scorelist, predlist):
    precision = precision_score(targetlist, predlist, average='micro')
    print('precision', precision)

    return precision

if __name__ == '__main__':
    print('start')

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--save_frequency", type=int, default=20, help="How often to save the model")
    parser.add_argument("--print_frequency", type=int, default=20, help="How often to print information")
    parser.add_argument("--net", type=str, default='dpn68', help="net model")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether to load the pre-trained model")
    parser.add_argument("--use_gpu", type=bool, default=True, help="whether to load the pre-trained model")
    parser.add_argument("--gpu_id", type=str, default='0,1,2,3', help="whether to load the pre-trained model")

    opt = parser.parse_args()
    

    path_dir = "model_result/{}_train_{}_{}".format(opt.net, opt.n_epochs,opt.batch_size)

    if not os.path.exists(path_dir):  # 如果路径不存在,则创建该路径
        os.makedirs(path_dir)


    trainset = dataprocess.CovidCTDataset(root_dir='./covid19_dataset',
                                        txt_COVID='./covid19_dataset/COVID.txt',
                                        txt_Non='./covid19_dataset/Normal.txt',
                                        txt_CP='./covid19_dataset/CP.txt',
                                        transform=train_transformer)

    train_loader = DataLoader(trainset, batch_size=opt.batch_size, drop_last=False, shuffle=True)


    if opt.net == 'dpn68':
        model = dpn.dpn68(pretrained=opt.pretrained)
    else:
        print("you print net model %s is error", opt.net )

    print('create ' + opt.net + ' model')

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    if opt.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_id
        model = torch.nn.DataParallel(model).cuda()


    
    loss_history = []
    accuracy_history = []

    for epoch in range(1, opt.n_epochs + 1):
        average_loss, average_accuracy = train(model, optimizer, epoch, train_loader)
        loss_history.append(average_loss)
        accuracy_history.append(average_accuracy)

        

    print("***************train end!**************************")
    targetlist, scorelist, predlist = prediction(model, train_loader)
    precision= evaluate(targetlist, scorelist, predlist)


    if opt.use_gpu:
        torch.save(model.module.state_dict(), path_dir + "/{}_{}_finish.pt".format(opt.net,precision))
    else:
        torch.save(model.state_dict(), path_dir + "/{}_{}_finish.pt".format(opt.net,precision))

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(211)
    plt.plot(loss_history, color='r', linestyle='-')
    plt.xlabel('Training epoch')  # fill the meaning of X axis
    plt.ylabel('average loss')  # fill the meaning of X axis
    plt.title('loss change')  # add the title of the figure

    plt.subplot(212)
    plt.plot(accuracy_history, color='g', linestyle='--')
    plt.xlabel('Training epoch')
    plt.ylabel('accury')
    plt.title('Recognition')
    plt.savefig(path_dir+'/los_s{}.jpg'.format(opt.net))    
