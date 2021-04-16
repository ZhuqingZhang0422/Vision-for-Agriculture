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

voc_root = '/work/07034/byz/maverick2/Term/Agriculture'
label_names = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
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
        rgbdir = os.path.join(root, 'val', 'images', 'rgb')
        labeldir = os.path.join(root,'val','processedlabels')
        rgb_fig_names = os.listdir(rgbdir)
        fig_ids = [fname[:-4] for fname in rgb_fig_names]
        rgb_img = [os.path.join(rgbdir, fig_id + '.jpg') for fig_id in fig_ids]
        label_img=[os.path.join(labeldir, fig_id + '.png') for fig_id in fig_ids]
        return rgb_img, label_img

def random_crop(data, label, crop_size):
    height, width = crop_size
    #data, rect = tfs.RandomCrop((height, width))(data)
    #label = tfs.FixedCrop(*rect)(label)
    w, h = data.size
    x1, y1 = random.randint(0, w - width), random.randint(0, h - height)
    input=data.crop((x1, y1, x1 + width, y1 + height))
    label=label.crop((x1, y1, x1 + width, y1 + height))
    return input, label

def img_transforms(img, label, crop_size):
    img, label = random_crop(img, label, crop_size)
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = img_tfs(img)
    labels= np.array(label, dtype='int64')
    labels = torch.from_numpy(labels)
    return img, labels

class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''
    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = data_list
        self.label_list = label_list
        print('Read ' + str(len(self.data_list)) + ' images')

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        labels = Image.open(label)
        img, labels = self.transforms(img, labels, self.crop_size)
        return img, labels

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
net = net.cuda()
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, img_transforms)
voc_test = VOCSegDataset(False, input_shape, img_transforms)

train_data = DataLoader(voc_train, 4, shuffle=True, num_workers=4)
valid_data = DataLoader(voc_test, 16, num_workers=4)

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    Args:
        pred_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
        gt_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.
    """
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.
    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.
    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.
    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) -
                       np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou


def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.
    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.
    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.
    Args:
        pred_labels (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`pred_labels`, ":math:`[(H, W)]`", :obj:`int32`, \
        ":math:`[0, \#class - 1]`"
        :obj:`gt_labels`, ":math:`[(H, W)]`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    Returns:
        dict:
        The keys, value-types and the description of the values are listed
        below.
        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.
    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}
criterion = nn.NLLLoss2d()
basic_optim = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = basic_optim

if __name__ == "__main__":
    for e in range(80):
        train_loss = 0
        train_acc = 0
        train_mean_iu = 0

        prev_time = datetime.now()
        net = net.train()
        for data in train_data:
            im = Variable(data[0].cuda())
            labels = Variable(data[1].cuda())
            # forward
            out = net(im)
            out = F.log_softmax(out, dim=1)  # (b, n, h, w)
            loss = criterion(out, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data

            pred_labels = out.max(dim=1)[1].data.cpu().numpy()
            pred_labels = [i for i in pred_labels]

            true_labels = labels.data.cpu().numpy()
            true_labels = [i for i in true_labels]

            eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)

            train_acc += eval_metrics['mean_class_accuracy']
            train_mean_iu += eval_metrics['miou']

        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_mean_iu = 0
        with torch.no_grad():
            for data in valid_data:
                im = Variable(data[0].cuda())
                labels = Variable(data[1].cuda())
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, labels)
                eval_loss += loss.data

                pred_labels = out.max(dim=1)[1].data.cpu().numpy()
                pred_labels = [i for i in pred_labels]

                true_labels = labels.data.cpu().numpy()
                true_labels = [i for i in true_labels]

                eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)


        eval_acc += eval_metrics['mean_class_accuracy']
        eval_mean_iu += eval_metrics['miou']

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
    Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            e, train_loss / len(train_data), train_acc / len(train_data), train_mean_iu / len(train_data),
               eval_loss / len(valid_data), eval_acc, eval_mean_iu))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        torch.save(net.state_dict(), '/work/07034/byz/maverick2/Term/7class1.pth')
        print(epoch_str + time_str)