from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import MLP, NAC, NALU
import os
import scipy.io
import numpy as np
import h5py

from torch.utils.data.dataset import Dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 20, kernel_size=4)
        self.conv4 = nn.Conv2d(1, 3, kernel_size=28, stride=(1,28))
        self.fc1 = nn.Linear(300, 50)
        self.fc2 = nn.Linear(50, 1)
        self.nalu = NALU(num_layers=1, in_dim=50,hidden_dim=0, out_dim=1)
        self.nac = NAC(num_layers = 1, in_dim=50, hidden_dim=0, out_dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 300)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #x = self.nac(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log_string('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    N = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target)
            pred = output.max(1, keepdim=True)[1]
            pred = output*18.0
            target = target*18.0
            correct += (torch.abs(target - pred)).mean().item()
            N += 1

    test_loss=test_loss/N
    log_string('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f})\n'.format(
        test_loss, correct/N))

def main():

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = prepareMnistDataSets()

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    LOG_FOUT.close()

class MyDataset(Dataset):
    def __init__(self, images,labels):
        self.images = [torch.transpose(torch.FloatTensor(images[index, :, :]),1,0) for index in range(labels.shape[1])]
        self.labels =  torch.squeeze(torch.FloatTensor(labels-2.0))

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = image * 2 -1
        new_image= image.view( 1, 28, 84)
        new_label = label.view(1)
        return (new_image,new_label/18.0)

    def __len__(self):
        return len(self.images)

def prepareMnistDataSets():

    file_name = r'C:\Users\mor1joseph\Nextcloud\deepNeuralNetworkCourse\project_nalu\mnistData2.mat'

    mnist_new =h5py.File(file_name,'r')
    imgDataTestArith = mnist_new.get('imgDataTestArith')
    imgDataTestArith = np.array(imgDataTestArith)

    labelTes = mnist_new.get('labelTes')
    labelTes = np.array(labelTes)
    imgDataTrainArith = mnist_new.get('imgDataTrainArith')
    imgDataTrainArith = np.array(imgDataTrainArith)
    labelTr = mnist_new.get('labelTr')
    labelTr = np.array(labelTr)
    dataset = MyDataset(imgDataTrainArith, labelTr)
    testDataset = MyDataset(imgDataTestArith, labelTes)
    
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 0} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,**kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=1000,
        shuffle=False,**kwargs)
    return train_loader,test_loader


if __name__ == '__main__':
    main()