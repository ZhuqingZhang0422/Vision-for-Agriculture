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
import glob
import matplotlib.pyplot as plt

def mklabel(myfig,label_paths):
    label_imgs=[os.path.join(mydir,myfig+'.png') for mydir in label_paths]
    return label_imgs

voc_root = '/work/07034/byz/maverick2/Term/Agriculture'
def read_images(root=voc_root, train=True):
    label_names = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
    if train:
        rgbdir = os.path.join(root, 'train','images','rgb')
        labeldir = [os.path.join(root,'train','labels', label_name) for label_name in label_names]
        rgb_fig_names = os.listdir(rgbdir)
        fig_ids = [fname[:-4] for fname in rgb_fig_names]
        rgb_img = [os.path.join(rgbdir, fig_id + '.jpg') for fig_id in fig_ids]
        label_img=[mklabel(fig_id,labeldir) for fig_id in fig_ids]
        return rgb_img, label_img
    else:
        rgbdir = os.path.join(root, 'val', 'images', 'rgb')
        labeldir = [os.path.join(root, 'val', 'labels', label_name) for label_name in label_names]
        rgb_fig_names = os.listdir(rgbdir)
        fig_ids = [fname[:-4] for fname in rgb_fig_names]
        rgb_img = [os.path.join(rgbdir, fig_id + '.jpg') for fig_id in fig_ids]
        label_img = [mklabel(fig_id, labeldir) for fig_id in fig_ids]
        return rgb_img, label_img

def img_transforms(img, label):
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = img_tfs(img)
    for i in range(6):
        tmp = label[i]
        tmp=np.array(tmp)
        tmp[tmp > 0] = i+1
        label[i] = tmp
    tmplabel = label[0]
    size0 = np.sum(label[0] > 0)
    size1 = np.sum(label[1] > 0)
    size2 = np.sum(label[2] > 0)
    size3 = np.sum(label[3] > 0)
    size4 = np.sum(label[4] > 0)
    size5 = np.sum(label[5] > 0)
    se = np.argmax([size0, size1, size2, size3, size4, size5])
    for i in range(label[0].shape[0]):
        for j in range(label[0].shape[1]):
            if np.sum([label[0][i][j] > 0, label[1][i][j] > 0, label[2][i][j] > 0,
                       label[3][i][j] > 0, label[4][i][j] > 0, label[5][i][j] > 0]) > 1:
                tmplabel[i][j] = label[se][i][j]
            else:
                tmplabel[i][j] = np.sum([label[0][i][j], label[1][i][j], label[2][i][j],
                                         label[3][i][j], label[4][i][j], label[5][i][j]])
    labels = tmplabel
    labels= np.array(labels, dtype='int8')
    #labels = torch.from_numpy(labels)
    return img, labels

x=[]
y=[]
data_list, label_list = read_images(train=True)
processdir="/work/07034/byz/maverick2/Term/Agriculture/train/processedlabels"
for i in range(len(data_list)):
    img=data_list[i]
    label=label_list[i]
    img = Image.open(img)
    labels = [Image.open(label[i]) for i in range(6)]
    img, labels = img_transforms(img, labels)
    Im = Image.fromarray(labels)
    outf = os.path.join(processdir,data_list[i][60:-4]+'.png')
    Im.save(outf, "PNG", quality=100)
    print(i)

###test
#s = Image.open(outf)
#s= np.array(s, dtype='int64')
#x=np.array(labels, dtype='int64')