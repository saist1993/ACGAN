from __future__ import print_function
import os
import json
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms


'''
    Split the dataset in such a manner, that it follows the structure of imagenet. Run this once.
'''
SPLIT = False
ngpu = 0
nc = 3
ndf = 64


def split_dataset():
    train_paths = json.load(open('./food101/meta/train.json'))
    test_paths = json.load(open('./food101/meta/test.json'))
    foldernames = open('./food101/meta/classes.txt')
    for folder in os.listdir('./food101/images/'):
        for filepath in os.listdir('./food101/images/'+folder):

            compatible_filepath = folder+'/'+filepath.replace('.jpg', '')
            complete_filepath = './food101/images/'+folder+'/'+filepath
            train_filepath = './food101/train/'+folder+'/'+filepath
            test_filepath = './food101/test/'+folder+'/'+filepath

            # Check if it is in test
            if compatible_filepath in test_paths[folder]:

                # Move it to the train thing
                os.rename(complete_filepath, test_filepath)

            else:
                os.rename(complete_filepath, train_filepath)


if SPLIT:
    split_dataset()

#transforming it in exactly the same manner as done for imagenet
# Traindir, Testdir
traindir = "./food101/train/"
testdir = "./food101/test/"
train_data = dset.ImageFolder(root=traindir,
                               transform=transforms.Compose([
   transforms.Resize(64),
   transforms.CenterCrop(64),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]),
                                 target_transform=None)

test_data = dset.ImageFolder(root=testdir,
                               transform=transforms.Compose([
   transforms.Resize(64),
   transforms.CenterCrop(64),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]),
                                 target_transform=None)



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),#0
            nn.LeakyReLU(0.2, inplace=True),#1
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),#2
            nn.BatchNorm2d(ndf * 2),#3
            nn.LeakyReLU(0.2, inplace=True),#4
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),#5
            nn.BatchNorm2d(ndf * 4),#6
            nn.LeakyReLU(0.2, inplace=True),#7
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),#8
            nn.BatchNorm2d(ndf * 8),#9
            nn.LeakyReLU(0.2, inplace=True),#10
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),#11
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # output = self.main(input)
            op1 = self._modules['main'][0](input)
            max_pool_1 = nn.MaxPool2d(8)(op1)

            op2 = self._modules['main'][2](self._modules['main'][1](op1))
            max_pool_2 = nn.MaxPool2d(4)(op2)

            op3 = self._modules['main'][5](self._modules['main'][4](self._modules['main'][3](op2)))
            max_pool_3 = nn.MaxPool2d(2)(op3)

            op4 = self._modules['main'][8](self._modules['main'][7](self._modules['main'][6](op3)))
            max_pool_4 = nn.MaxPool2d(1)(op4)

            # op5 = self._modules['main'][11](self._modules['main'][10](self._modules['main'][9](op2)))
            # max_pool_5 = nn.MaxPool2d(8)(op5)t
            return max_pool_1, max_pool_2, max_pool_3, max_pool_4


netD = _netD(ngpu)
netD.apply(weights_init)
print(netD)
netD.load_state_dict(torch.load('netD_epoch_2.pth', map_location=lambda storage, loc: storage))



def transformer(image):
    # Fool the network to think you're doing this in batches
    image = image.resize_([1, image.shape[0], image.shape[1], image.shape[2]])
    max_pool_1, max_pool_2, max_pool_3, max_pool_4 = netD(Variable(image))
    op = torch.cat((
        max_pool_1.resize(np.product(max_pool_1.shape)),
        max_pool_2.resize(np.product(max_pool_2.shape)),
        max_pool_3.resize(np.product(max_pool_3.shape)),
        max_pool_4.resize(np.product(max_pool_4.shape))
    )).data.numpy()
    return image

# Create Train data
train_Y = []
train_X = []

for i, data in enumerate(train_data, 0):
    image, label = data
    op = transformer(image)
    train_X.append(op)
    train_Y.append(label)

print("Done collecting traindata features")


# Create Test data
test_X = []
test_Y = []

for i, data in enumerate(test_data, 0):
    image, label = data

    # Fool the network to think you're doing this in batches
    op = transformer(image)
    test_X.append(op)
    test_Y.append(label)

print("Done collecting testdata features")


pickle.dump({'trainX':train_X, 'trainY':train_Y, 'testX':test_X, 'testY':test_Y}, open("matrices.dat","w+"))
