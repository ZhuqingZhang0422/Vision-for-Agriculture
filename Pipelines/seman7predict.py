import os
import torch
import numpy as np
import random
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#from mxtorch import transforms as tfs
import torchvision.transforms as tfs
from datetime import datetime
import six
import matplotlib.pyplot as plt

voc_root = '/work/07034/byz/maverick2/Term/Agriculture'
classes = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128]]
def read_images(root=voc_root, train=True):
    if train:
        rgbdir = os.path.join(root, 'train','images','rgb')
        labeldir = os.path.join(root,'train','processedlabels')
        rgb_fig_names = os.listdir(rgbdir)
        fig_ids = [fname[:-4] for fname in rgb_fig_names]
        rgb_img = [os.path.join(rgbdir, fig_id + '.jpg') for fig_id in fig_ids]
        label_img=[os.path.join(labeldir, fig_id + '.png') for fig_id in fig_ids]
        return rgb_img, label_img
    else:
        rgbdir = os.path.join(root, 'test', 'images', 'rgb')
        labeldir = os.path.join(root,'test','processedlabels')
        rgb_fig_names = os.listdir(rgbdir)
        fig_ids = [fname[:-4] for fname in rgb_fig_names]
        rgb_img = [os.path.join(rgbdir, fig_id + '.jpg') for fig_id in fig_ids]
        label_img=[os.path.join(labeldir, fig_id + '.png') for fig_id in fig_ids]
        return rgb_img, label_img

def img_transforms(img):
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = img_tfs(img)
    #labels= np.array(label, dtype='int64')
    #labels = torch.from_numpy(labels)
    return img

class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''
    def __init__(self, train, transforms):
        #self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = data_list
        self.label_list = label_list
        print('Read ' + str(len(self.data_list)) + ' images')

    def __getitem__(self, idx):
        img = self.data_list[idx]
        #label = self.label_list[idx]
        img = Image.open(img)
        #labels = Image.open(label)
        #img, labels = self.transforms(img, labels, self.crop_size)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.data_list)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

import torchvision.models as models
pretrained_net = models.resnet34(pretrained=True)

num_classes = 7

class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) # 第一段
        self.stage2 = list(pretrained_net.children())[-4] # 第二段
        self.stage3 = list(pretrained_net.children())[-3] # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel


    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8

        x = self.stage2(x)
        s2 = x # 1/16

        x = self.stage3(x)
        s3 = x # 1/32

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s

net = fcn(num_classes)
net.load_state_dict(torch.load('/work/07034/byz/maverick2/Term/7class1.pth'))
net = net.eval()
net = net.cuda()
cm = np.array(colormap).astype('uint8')

def predict(img): # 预测结果
    img = Variable(img.unsqueeze(0)).cuda()
    out = net(img)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    #pred = cm[pred]
    return pred
processdir="/work/07034/byz/maverick2/Term/Agriculture/test/results"
data_list, label_list = read_images(voc_root,train=False)
for i in range(len(data_list)):
    img = data_list[i]
    img = Image.open(img)
    img = img_transforms(img)
    pred = predict(img)
    pred = np.array(pred, dtype='uint8')
    Im = Image.fromarray(pred)
    outf = os.path.join(processdir, data_list[i][59:-4] + '.png')
    Im.save(outf, "PNG", quality=100)
