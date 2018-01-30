from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import PIL
from PIL import Image

ngpu = 0
nz = 100
ngf = 64
ndf = 64
nc = 3
lr = 0.0002
beta1 = 0.5
netD_loc = 'netD_epoch_24.pth'
netG_loc = 'netG_epoch_24.pth'
batchSize = 1
imageSize = 64


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if netG != '':
    netG.load_state_dict(torch.load(netG_loc, map_location=lambda storage, loc: storage))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if netD != '':
    netD.load_state_dict(torch.load(netD_loc, map_location=lambda storage, loc: storage))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
noise = torch.FloatTensor(batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0

# if opt.cuda:
#     netD.cuda()
#     netG.cuda()
#     criterion.cuda()
#     input, label = input.cuda(), label.cuda()
#     noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))




def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im



def Vector_arith():
	'''
		Create a Z vector
	'''
	z_a = [Variable(torch.FloatTensor(batchSize, nz, 1, 1).resize_(1, nz, 1, 1).normal_(0, 1)) for i in xrange(3)]
	z_b = [Variable(torch.FloatTensor(batchSize, nz, 1, 1).resize_(1, nz, 1, 1).normal_(0, 1)) for i in xrange(3)]
	z_c = [Variable(torch.FloatTensor(batchSize, nz, 1, 1).resize_(1, nz, 1, 1).normal_(0, 1)) for i in xrange(3)]
	image_a = [netG(noisev) for noisev in z_a]
	image_b = [netG(noisev) for noisev in z_b]
	image_c = [netG(noisev) for noisev in z_c]
	z_avg_a = (z_a[0]+z_a[1]+z_a[2])/3
	z_avg_b = (z_b[0]+z_b[1]+z_b[2])/3
	z_avg_c = (z_c[0] + z_c[1] + z_c[2])/3
	z_avg_new = (z_avg_a - z_avg_b + z_avg_c)
	image_final = netG(z_avg_new)
	for i,image in enumerate(image_a):
		vutils.save_image(image.data,
					  'image_a_%d.png' %(i),
					  normalize=True)
	for i,image in enumerate(image_b):
		vutils.save_image(image.data,
					  'image_b_%d.png' %(i),
					  normalize=True)
	for i,image in enumerate(image_c):
		vutils.save_image(image.data,
					  'image_c_%d.png' %(i),
					  normalize=True)
	vutils.save_image(image_final.data,
					  'image_final.png',
					  normalize=True)

def save_large_image():
	list_im = ['image_a_0.png','image_a_1.png','image_a_2.png']
	imgs = [PIL.Image.open(i) for i in list_im]
	min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
	imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
	imgs_comb = PIL.Image.fromarray(imgs_comb)
	imgs_comb.save('a_horz.png')

	list_im = ['image_b_0.png', 'image_b_1.png', 'image_b_2.png']
	imgs = [PIL.Image.open(i) for i in list_im]
	min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
	imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
	imgs_comb = PIL.Image.fromarray(imgs_comb)
	imgs_comb.save('b_horz.png')

	list_im = ['image_c_0.png', 'image_c_1.png', 'image_c_2.png']
	imgs = [PIL.Image.open(i) for i in list_im]
	min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
	imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
	imgs_comb = PIL.Image.fromarray(imgs_comb)
	imgs_comb.save('c_horz.png')

	list_im = ['a_horz.png','b_horz.png','c_horz.png']
	imgs = [PIL.Image.open(i) for i in list_im]
	min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
	imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
	imgs_comb = PIL.Image.fromarray(imgs_comb)
	imgs_comb.save('combined.png')

	images = map(PIL.Image.open, ['combined.png', 'image_final.png'])
	combo_1 = append_images(images, direction='horizontal')
	combo_1.save('final_conc.png')

Vector_arith()
save_large_image()