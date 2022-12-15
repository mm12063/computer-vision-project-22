import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import torchvision.transforms as T
from torch.utils.data import Dataset
import natsort
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import ssl

NUM_CLASSES = 45
# Disable SSL Certificate
ssl._create_default_https_context = ssl._create_unverified_context
# Reproducibility
torch.manual_seed(42)
higest_class_acc = 0.0

print(f"Using cuda: {torch.cuda.is_available()}")

class McModel(nn.Module):
    def __init__(self, model_name):
        super(McModel, self).__init__()

        if model_name == "resnet50":
            self.network = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif model_name == "resnet152":
            self.network = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        elif model_name == "alexnet":
            self.network = models.alexnet(weights='AlexNet_Weights.DEFAULT')
        elif model_name == "vgg19":
            self.network = models.vgg19(weights='VGG19_Weights.DEFAULT')
            in_feats = self.network.classifier[6].in_features
            self.network.classifier[6] = nn.Linear(in_feats, NUM_CLASSES)
        else:
            self.network = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        if not model_name == "vgg19":
            self.network.fc = nn.Linear(self.network.fc.in_features, NUM_CLASSES)

    def forward(self, x):
        output = self.network(x)
        m = nn.Sigmoid()
        return m(output)

def plot(metric, train_vals, test_vals, xtick_interval=2, save=False, save_loc=None, plot_name="Default_Name"):
    plt.figure(figsize=(5, 3))
    plt.plot(train_vals, color="red")
    plt.plot(test_vals, color="blue")
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend(['Train', 'Test'])
    plt.title(f'Train vs Test {metric.capitalize()}')
    plt.xticks(np.arange(0, len(train_vals) + 1, xtick_interval))
    if save:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        full_loc = f"{save_loc}{plot_name}"
        print(f"Plot saved to: {full_loc}")
        plt.savefig(full_loc)
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
    remove_norm = T.Normalize(
        mean=[-0.5130 / 0.2551, -0.3182 / 0.1793, -0.1666 / 0.1329],
        std=[1 / 0.2551, 1 / 0.1793, 1 / 0.1329]
    )

    img = remove_norm(img)
    inp = img.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def train(args, device, epoch, model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if args.print_vals:
            print("output.shape", output.shape)
            print("target.shape", target.shape)

        target = target.to(torch.float32)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        predicted = output
        predicted = (predicted > 0.5).float()
        total += target.size(0)

        if predicted.equal(target):
            correct += 1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_acc = 100. * correct / total
    train_loss /= len(train_loader.dataset)

    return train_loss, train_acc

def validation(device, model, val_loader, criterion):
    model.eval()
    validation_loss = 0
    correct = 0
    partial_cor = 0
    class_cor = torch.zeros(NUM_CLASSES)
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = target.to(torch.float32)
        validation_loss += criterion(output, target)

        predicted = output
        predicted = (predicted > 0.5).float()

        for i in range(len(target)):
            t = target[i]
            p = predicted[i]
            res_by_class = p.eq(t)

            if p.equal(t):
                correct += 1
            if torch.any(res_by_class):
                partial_cor += 1

            for j in range(len(class_cor)):
                if res_by_class[j]:
                    class_cor[j] += 1

    validation_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
    pacc = 100 * partial_cor / len(val_loader.dataset)
    num_items = len(val_loader.dataset)
    print(
        '\nValidation set:\n Average loss: {:.4f},\n Partial Accuracy: {}/{} ({:.2f}%)\n Full Accuracy: {}/{} ({:.2f}%)'.format(
            validation_loss,
            partial_cor, num_items, pacc,
            correct, num_items, acc))

    class_acc = class_cor / num_items * 100

    global higest_class_acc
    if acc > higest_class_acc:
        higest_class_acc = acc

    print(' class_cor: ' + str(class_cor))
    print(' class_acc: ' + str(class_acc))

    return validation_loss.item(), acc


def main():
    parser = argparse.ArgumentParser(description="Main Arguments")
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable cuda (default: False)')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--sgd', action='store_true', default=False,
                        help='use sgd or not (default: Adam)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of intervals to log (default: 10)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='number of workers to use (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=1.0, metavar='decay',
                        help='decay (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                        help='Change the Momentum of Optimizer (default: 0.9)')
    parser.add_argument('--model-name', type=str, default="resnet18", metavar='model',
                        help='Resnet Size: resent18, resnet50, resnet152')
    parser.add_argument('--print-vals', action='store_true', default=False,
                        help='Print the values during run')
    parser.add_argument('--use-cutout', action='store_true', default=False,
                        help='Adds cutout transformation')
    parser.add_argument('--cutout-prob', type=float, default=0.7, metavar='CP',
                        help='probability that cutout will be used (default: 0.7)')
    parser.add_argument('--image-size', type=int, default=128, metavar='N',
                        help='resize image to this (square) size (default: 256)')
    parser.add_argument('--train-ds-sz', type=int, default=1920, metavar='N',
                        help='Training Dataset Size (default: 1920)')
    parser.add_argument('--val-ds-sz', type=int, default=640, metavar='N',
                        help='Validation Dataset Size (default: 640)')
    parser.add_argument('--save-model', action="store_true", default=False,
                        help='Saving Model')
    parser.add_argument('--save-model-dir', type=str, default="./", metavar='model_dir',
                        help='Directory to Save Model')
    parser.add_argument('--save-plots', action="store_true", default=False,
                        help='Saving Plots')
    parser.add_argument('--save-plot-dir', type=str, default="./", metavar='plot_dir',
                        help='Directory to Save Plots')
    parser.add_argument('--root-dir', type=str, default="./", metavar='root_dir',
                        help='root dir of the project')
    parser.add_argument('--train-imgs-dir', type=str, default="./", metavar='train_imgs',
                        help='directory of training images')
    parser.add_argument('--train-csv', type=str, default="./no-train-csv.csv", metavar='train_csv',
                        help='training csv file')
    parser.add_argument('--val-imgs-dir', type=str, default="./", metavar='val_imgs',
                        help='directory of training images')
    parser.add_argument('--val-csv', type=str, default="./no-val-csv.csv", metavar='val_csv',
                        help='validation csv file')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set the Loss/Optimizers
    model = McModel(args.model_name).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.sgd:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)


    # Create the transformations
    trans = []
    trans.append(T.ToTensor())
    trans.append(T.Normalize(
        (0.5130, 0.3182, 0.1666),
        (0.2551, 0.1793, 0.1329))
    )

    is_cutout_used = "False"
    if args.use_cutout:
        trans.append(T.RandomErasing(p=0.6, scale=(0.04, 0.12)))
        is_cutout_used = "True"

    transforms = T.Compose(trans)

    DROP_VALS = ['ID', 'Disease_Risk']
    y_train_vals = pd.read_csv(args.train_csv)
    y_train_vals = np.array(y_train_vals.drop(DROP_VALS, axis=1))
    y_val_vals = pd.read_csv(args.val_csv)
    y_val_vals = np.array(y_val_vals.drop(DROP_VALS, axis=1))

    train_dataset = CustomDataSet(args.train_imgs_dir, y_values=y_train_vals, transform=transforms)
    val_dataset = CustomDataSet(args.val_imgs_dir, y_values=y_val_vals, transform=transforms)
    # get_mean_and_std(train_dataset)

    trainsubset_ind = torch.randperm(len(train_dataset))[:args.train_ds_sz]
    train_dataset = torch.utils.data.Subset(train_dataset, trainsubset_ind)
    valsubset_ind = torch.randperm(len(val_dataset))[:args.val_ds_sz]
    val_dataset = torch.utils.data.Subset(val_dataset, valsubset_ind)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Show some images
    # images, labels = next(iter(train_loader))
    # imshow(utils.make_grid(images))

    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, device, epoch, model, optimizer, criterion, train_loader)
        losses_train.append(train_loss)
        acc_train.append(train_acc)

        test_loss, test_acc = validation(device, model, val_loader, criterion)
        losses_test.append(test_loss)
        acc_test.append(test_acc)

    # Create full model name str
    lr_str = str(args.lr)
    lr_str = lr_str[lr_str.rfind(".") + 1:]
    full_model_name = f'{args.model_name}'
    full_model_name += f'__l_{lr_str}__b_{args.batch_size}__i_{args.image_size}'
    full_model_name += f'__e_{args.epochs}__c_{is_cutout_used}'
    if args.sgd:
        full_model_name += f'__sgd__d_{args.decay}__m_{args.momentum}'

    if args.save_model:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        full_path = args.save_model_dir + full_model_name + ".pth"
        print(f"model: {model}")
        torch.save(model.state_dict(), full_path)
        print(f'Saved model to {full_path}')

    # Show the metric plots
    xticks = 5
    plot('loss',
         losses_train,
         losses_test,
         xtick_interval=xticks,
         save=args.save_plots,
         save_loc=args.save_plot_dir,
         plot_name=f'loss__{full_model_name}.png'
         )

    plot('accuracy',
         acc_train,
         acc_test,
         xtick_interval=xticks,
         save=args.save_plots,
         save_loc=args.save_plot_dir,
         plot_name=f'acc__{full_model_name}.png'
         )

    print(f"Highest Acc Overall: {round(higest_class_acc, 2)}%")

if __name__ == '__main__':
    main()
