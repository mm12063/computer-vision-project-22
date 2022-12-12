import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader
import natsort
from PIL import Image
import PIL
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import pandas as pd
import ssl

# Disable SSL Certificate
ssl._create_default_https_context = ssl._create_unverified_context

# Reproducibility
torch.manual_seed(42)

print_vals = False
batch_size = 32
lr = 0.001
epochs = 10
log_interval = 100
NUM_CLASSES = 45
MAX_TRA_SZ = 1920
MAX_VAL_SZ = 640
TRA_SZ = MAX_TRA_SZ
VAL_SZ = MAX_VAL_SZ

# lr_reducer = 1

num_workers = 0
best_acc = 0

# RESIZE THE LARGE IMAGES
# from os import listdir
# from os.path import isfile, join
# files = [f for f in listdir(TRAIN_DS) if isfile(join(TRAIN_DS, f))]
#
# TRAIN_DS_PROC = f"{TRAIN_DS}processed/"
# if not os.path.exists(TRAIN_DS_PROC):
#     os.makedirs(TRAIN_DS_PROC)
#
# large_img_size = (4288, 2848)
#
# for i in files:
#     full_loc = TRAIN_DS+i
#     filename = full_loc[full_loc.rfind('/')+1:]
#     if (filename == ".DS_Store"):
#         continue
#     im = Image.open(full_loc)
#     if im.size == large_img_size:
#         resizedImage = im.resize((int(large_img_size[0] * .5), int(large_img_size[1] * .5)), PIL.Image.ANTIALIAS)
#         resizedImage.save(f"{TRAIN_DS_PROC}{filename}", 'png')

def plot(metric, train_vals, test_vals, xtick_interval=2):
    figure(figsize=(5, 3))
    plt.plot(train_vals, color="red")
    plt.plot(test_vals, color="blue")
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend(['Train','Test'])
    plt.title(f'Train vs Test {metric.capitalize()}')
    plt.xticks(np.arange(0, len(train_vals)+1, xtick_interval))
    plt.show()

transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((1424, 1424)),
        transforms.Resize((256,256))
    ])
#
## --- RESIZE THE LARGE IMAGES
##ORIGINAL#TRAIN_DS = "rmfid/Training_Set/Training_Set/Training/resized/"
#TRAIN_DS = "rmfid/Training_Set/Training_Set/Training/"
#TRAIN_CSV = "rmfid/Training_Set/Training_Set/RFMiD_Training_Labels.csv"
##ORIGINAL #VAL_DS = "rmfid/Evaluation_Set/Evaluation_Set/Validation/resized/"
#VAL_DS = "rmfid/Evaluation_Set/Evaluation_Set/Validation/"
#VAL_CSV = "rmfid/Training_Set/Training_Set/RFMiD_Training_Labels.csv"
#
#from os import listdir
#from os.path import isfile, join
#
#
#DIR_TO_UPDATE = TRAIN_DS # UPDATE THIS FOR TRAIN / VAL
#
#files = [f for f in listdir(DIR_TO_UPDATE) if isfile(join(DIR_TO_UPDATE, f))]
#
#TRAIN_DS_PROC = f"{DIR_TO_UPDATE}resized/"
#if not os.path.exists(TRAIN_DS_PROC):
#    os.makedirs(TRAIN_DS_PROC)
#
#large_img_size = (4288, 2848)
#
#for i in files:
#     full_loc = DIR_TO_UPDATE+i
#     filename = full_loc[full_loc.rfind('/')+1:]
#     if (filename == ".DS_Store"):
#         continue
#     im = Image.open(full_loc)
#     if im.size == large_img_size:
#         resizedImage = im.resize((int(large_img_size[0] * .5), int(large_img_size[1] * .5)), PIL.Image.ANTIALIAS)
#         resizedImage.save(f"{TRAIN_DS_PROC}{filename}", 'png')
#
### --- END RESIZING

TRAIN_DS = "rmfid/Training_Set/Training_Set/Training/resized/"
TRAIN_CSV = "rmfid/Training_Set/Training_Set/RFMiD_Training_Labels.csv"
VAL_DS = "rmfid/Evaluation_Set/Evaluation_Set/Validation/resized/"
VAL_CSV = "rmfid/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"

class CustomDataSet(Dataset):
    def __init__(self, main_dir, y_values, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.y = y_values

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        try:
            img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image, self.y[idx]
        except ValueError as ve:
            print("IDX: "+str(idx)+", ve: "+str(ve));
        except Exception as e:
            print("IDX: "+str(idx)+", e: "+str(e));
        
y_train_vals = pd.read_csv(TRAIN_CSV)
y_train_vals = np.array(y_train_vals.drop(['ID','Disease_Risk'], axis=1))

y_val_vals = pd.read_csv(VAL_CSV)
y_val_vals = np.array(y_val_vals.drop(['ID','Disease_Risk'], axis=1))

train_dataset = CustomDataSet(TRAIN_DS, y_values=y_train_vals, transform=transforms)
val_dataset = CustomDataSet(VAL_DS, y_values=y_val_vals, transform=transforms)

trainsubset_ind = torch.randperm(len(train_dataset))[:TRA_SZ]
train_dataset = torch.utils.data.Subset(train_dataset, trainsubset_ind)
valsubset_ind = torch.randperm(len(val_dataset))[:VAL_SZ]
val_dataset = torch.utils.data.Subset(val_dataset, valsubset_ind)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def imshow(img, title=None):
    # img = img / 2 + 0.5
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

    inp = img.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

images, labels = next(iter(train_loader))
# imshow(utils.make_grid(images))
#pretrained => weights
#

class McResnet18(nn.Module):
    def __init__(self):
        super(McResnet18, self).__init__()
        #self.model_resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        #num_ftrs = self.model_resnet.fc.in_features
        #self.model_resnet.fc = nn.Identity()
        #self.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        self.network = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.network.fc = nn.Linear(self.network.fc.in_features, NUM_CLASSES)
        
    def forward(self, x):
        #x = self.network(x)
        #output = self.network.fc(x)
        
        output = self.network(x)
        m = nn.Sigmoid()
        return m(output)

model = McResnet18()
#model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
#num_ftrs = resnet18.fc.in_features
#model.fc = nn.Linear(num_ftrs, 128)


criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        if print_vals:
            print("output.shape",output.shape)
            print("target.shape", targets.shape)

        targets = targets.to(torch.float32)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        predicted = output
        predicted = (predicted > 0.5).float()
        total += targets.size(0)

        if predicted.equal(targets):
            correct += 1
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_acc = 100. * correct/total
    train_loss /= len(train_loader.dataset)
    print("train_loss",train_loss)

    return train_loss, train_acc

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    partial_cor = 0;
    class_cor = torch.zeros(NUM_CLASSES)
    for data, targets in val_loader:
        output = model(data)
        targets = targets.to(torch.float32)
        validation_loss+= criterion(output, targets)
        
        predicted = output
        predicted = (predicted > 0.5).float()
        
        for i in range(len(targets)):
            t = targets[i];
            p = predicted[i];
            res_by_class = p.eq(t);
            
            if p.equal(t):
                correct += 1
            if torch.any(res_by_class):
                partial_cor +=1
            
            for j in range(len(class_cor)):
                if res_by_class[j]:
                    class_cor[j]+=1;
            
    validation_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
    pacc = 100 * partial_cor / len(val_loader.dataset)
    num_items = len(val_loader.dataset)
    print('\nValidation set:\n Average loss: {:.4f},\n Partial Accuracy: {}/{} ({:.2f}%)\n Full Accuracy: {}/{} ({:.2f}%)'.format(
        validation_loss,
        partial_cor,num_items, pacc,
        correct,num_items,acc))
    print(' class_cor: '+str(class_cor));
    print(' class_acc: '+str(class_cor/num_items*100));


    return validation_loss, acc

best_model_name = ""
run_training = True

losses_train = []
losses_test = []
acc_train = []
acc_test = []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(epoch)
    losses_train.append(train_loss)
    acc_train.append(train_acc)

    test_loss, test_acc = validation()
    losses_test.append(test_loss)
    acc_test.append(test_acc)

    # model_file = 'models/model_' + str(epoch) + '.pth'
    # torch.save(model.state_dict(), model_file)
    # print('Saved model to ' + model_file + '.\n')

xticks = 5
#plot('loss', losses_train, losses_test, xtick_interval=xticks)
#plot('accuracy', acc_train, acc_test, xtick_interval=xticks)













#
#
#
# net = models.resnet18(pretrained=True)
# # net = net.cuda() if device else net
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
#
# def accuracy(out, labels):
#     _,pred = torch.max(out, dim=1)
#     return torch.sum(pred == labels).item()
#
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 128)
# # net.fc = net.fc.cuda() if use_cuda else net.fc
#
# n_epochs = 1
# print_every = 10
# valid_loss_min = np.Inf
# val_loss = []
# val_acc = []
# train_loss = []
# train_acc = []
# total_step = len(train_loader)
# for epoch in range(1, n_epochs + 1):
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     print(f'Epoch {epoch}\n')
#     for batch_idx, (data_, target_) in enumerate(train_loader):
#         data_, target_ = data_.to(device), target_.to(device)
#         optimizer.zero_grad()
#
#         outputs = net(data_)
#         loss = criterion(outputs, target_)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         _, pred = torch.max(outputs, dim=1)
#         correct += torch.sum(pred == target_).item()
#         total += target_.size(0)
#         if (batch_idx) % 20 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
#     train_acc.append(100 * correct / total)
#     train_loss.append(running_loss / total_step)
#     print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
#     batch_loss = 0
#     total_t = 0
#     correct_t = 0
#     with torch.no_grad():
#         net.eval()
#         for data_t, target_t in (val_loader):
#             data_t, target_t = data_t.to(device), target_t.to(device)
#             outputs_t = net(data_t)
#             loss_t = criterion(outputs_t, target_t)
#             batch_loss += loss_t.item()
#             _, pred_t = torch.max(outputs_t, dim=1)
#             correct_t += torch.sum(pred_t == target_t).item()
#             total_t += target_t.size(0)
#         val_acc.append(100 * correct_t / total_t)
#         val_loss.append(batch_loss / len(val_loader))
#         network_learned = batch_loss < valid_loss_min
#         print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
#
#         if network_learned:
#             valid_loss_min = batch_loss
#             torch.save(net.state_dict(), 'resnet.pt')
#             print('Improvement-Detected, save-model')
#     net.train()
#
# fig = plt.figure(figsize=(20,10))
# plt.title("Train-Validation Accuracy")
# plt.plot(train_acc, label='train')
# plt.plot(val_acc, label='validation')
# plt.xlabel('num_epochs', fontsize=12)
# plt.ylabel('accuracy', fontsize=12)
# plt.legend(loc='best')
#
#
# def visualize_model(net, num_images=4):
#     images_so_far = 0
#     fig = plt.figure(figsize=(15, 10))
#
#     for i, data in enumerate(val_loader):
#         inputs, labels = data
#         if use_cuda:
#             inputs, labels = inputs.cuda(), labels.cuda()
#         outputs = net(inputs)
#         _, preds = torch.max(outputs.data, 1)
#         preds = preds.cpu().numpy() if use_cuda else preds.numpy()
#         for j in range(inputs.size()[0]):
#             images_so_far += 1
#             ax = plt.subplot(2, num_images // 2, images_so_far)
#             ax.axis('off')
#             ax.set_title('predictes: {}'.format(val_loader.classes[preds[j]]))
#             imshow(inputs[j])
#
#             if images_so_far == num_images:
#                 return
#
#
# plt.ion()
# visualize_model(net)
# plt.ioff()
#
#
