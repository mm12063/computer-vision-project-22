import argparse
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
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import pandas as pd
import file_locs

#TODO
# Balance

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

def get_mean_and_std(train_dataset):
    imgs = torch.stack([img_t for img_t, _ in train_dataset], dim=3)
    print(imgs.view(3, -1).mean(dim=1))
    print(imgs.view(3, -1).std(dim=1))

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
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.y[idx]





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




def train(args, epoch, model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        if args.print_vals:
            print("output.shape",output.shape)
            print("target.shape", targets.shape)

        targets = targets.to(torch.float32)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        predicted = output
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_acc = 100. * correct/total
    train_loss /= len(train_loader.dataset)
    print("train_loss",train_loss)

    return train_loss, train_acc





def validation(model, val_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, targets in val_loader:
        output = model(data)
        targets = targets.to(torch.float32)
        validation_loss += F.cross_entropy(output, targets, reduction="sum").item() # sum up batch loss

        # _, pred_t = torch.max(output, dim=1)
        correct += torch.sum(output == targets).item()

        # pred = output.data
        # print(output.data)
        # correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    print('\nValidation set:\n Average loss: {:.4f},\n Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        acc))

    return validation_loss, acc




def main():
    parser = argparse.ArgumentParser(description="DESCRIPTION")
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of intervals to log (default: 10)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='number of workers to use (default: 0)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--model-name', type=str, default="resent18", metavar='model',
                        help='Resnet Size: resent18, resnet50, resnet152')
    parser.add_argument('--print-vals', action='store_true', default=False,
                        help='Print the values during run')
    parser.add_argument('--use-cutout', action='store_true', default=False,
                        help='Adds cutout transformation')
    parser.add_argument('--cutout-prob', type=float, default=0.7, metavar='CP',
                        help='probability that cutout will be used (default: 0.7)')
    parser.add_argument('--image-size', type=int, default=256, metavar='N',
                        help='resize image to this (square) size (default: 256)')
    args = parser.parse_args()


    # Create the transformations
    trans = []
    trans.append(torchvision.transforms.ToTensor())
    trans.append(torchvision.transforms.Resize((args.image_size, args.image_size)))
    trans.append(torchvision.transforms.Normalize(
        (0.5130, 0.3182, 0.1666), # Correct vals
        (0.2551, 0.1793, 0.1329)) # Correct vals
    )

    if args.use_cutout:
        trans.append(torchvision.transforms.RandomErasing(p=0.6,scale=(0.04, 0.12)))

    transforms = torchvision.transforms.Compose(trans)


    y_train_vals = pd.read_csv(file_locs.TRAIN_CSV)
    y_train_vals = np.array(y_train_vals.drop(['ID'], axis=1))
    y_val_vals = pd.read_csv(file_locs.VAL_CSV)
    y_val_vals = np.array(y_val_vals.drop(['ID'], axis=1))

    train_dataset = CustomDataSet(file_locs.TRAIN_DS + "resized/auto_crop/same_size/", y_values=y_train_vals, transform=transforms)
    val_dataset = CustomDataSet(file_locs.VAL_DS + "resized/auto_crop/same_size/", y_values=y_val_vals, transform=transforms)
    # get_mean_and_std(train_dataset)

    trainsubset_ind = torch.randperm(len(train_dataset))[:10]
    train_dataset = torch.utils.data.Subset(train_dataset, trainsubset_ind)
    valsubset_ind = torch.randperm(len(val_dataset))[:3]
    val_dataset = torch.utils.data.Subset(val_dataset, valsubset_ind)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    # Show some images
    images, labels = next(iter(train_loader))
    imshow(utils.make_grid(images))


    # Set the model
    if args.model_name == "resnet50":
        model = models.resnet18(pretrained=True)
    elif args.model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)

    # Update the output layer
    num_classes = 46
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Set the Loss/Optimizers
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    best_model_name = ""
    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, epoch, model, optimizer, criterion, train_loader)
        losses_train.append(train_loss)
        acc_train.append(train_acc)

        test_loss, test_acc = validation(model, val_loader)
        losses_test.append(test_loss)
        acc_test.append(test_acc)

        # model_file = 'models/model_' + str(epoch) + '.pth'
        # torch.save(model.state_dict(), model_file)
        # print('Saved model to ' + model_file + '.\n')

    # Show the metric plots
    xticks = 5
    plot('loss', losses_train, losses_test, xtick_interval=xticks)
    plot('accuracy', acc_train, acc_test, xtick_interval=xticks)
















































if __name__ == '__main__':
    main()












































