
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
from openslide import open_slide
from matplotlib import pyplot as plt
from GPUtil import showUtilization as gpu_usage

from loss import NCutLoss2D, OpeningLoss2D


from pkg_resources import resource_stream
from PIL import Image
import skimage.color
from skimage.segmentation import mark_boundaries
from itertools import chain
from pysnic.algorithms.snic import snic, compute_grid
from pysnic.ndim.operations_collections import nd_computations
from pysnic.metric.snic import create_augmented_snic_distance

from torch import einsum

use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
print("Initial GPU Usage")
gpu_usage()

parser = argparse.ArgumentParser(description='PyTorch Group Affinity Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=60, type=int,
                    help='number of channels')
parser.add_argument('--nGroup', metavar='G', default=60, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=5000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=50, type=float,
                    help='compactness of superpixels')
parser.add_argument('--slide', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--level', metavar='level', default=8, type=int,
                    help='level of WSI')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name')
args = parser.parse_args()


def loadWSI():
    img = open_slide(args.slide)
    print(img.level_count)
    for i in range(img.level_count):
        print("Level", i, "demension", img.level_dimensions[i], "down factor", img.level_downsamples[i])
    imgLevel = img.read_region((0, 0), args.level, img.level_dimensions[args.level])
    imgLevel = cv2.cvtColor(np.array(imgLevel), cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(imgLevel, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    ret3, thresh = cv2.threshold(gray_blur, 8, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = ~thresh

    cont_img = thresh.copy()

    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        areaSize = cv2.contourArea(contour)

        if areaSize> imgLevel.shape[0]*imgLevel.shape[1]/10:
            x, y, w, h = cv2.boundingRect(contour)
            colorAVG = np.array(cv2.mean(gray_blur[y:y + h, x:x + w])).astype(np.uint8)
            cv2.drawContours(imgLevel, [contour], -1, (255,255,255), -1)

    filename = ".\\output\\test.png"
    plt.imsave(filename, imgLevel)
    return imgLevel, filename

class CNNModel(nn.Module):
    def __init__(self,input_dim ):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.groupNorm1 = nn.GroupNorm(args.nGroup, args.nChannel)
        self.conv2 = nn.ModuleList()
        self.groupNorm2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.groupNorm2.append( nn.GroupNorm(args.nGroup,args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.groupNorm3 = nn.GroupNorm(args.nGroup,args.nChannel)

    def forward(self, x ):
        x = self.conv1(x)
        x = self.groupNorm1(x)
        x = F.relu( x )
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = self.groupNorm2[i](x)
            x = F.relu( x )
        x = self.conv3(x)
        x = self.groupNorm3(x)
        return x


def SNIC(filename,compactness, numSegments):
    # load image
    color_image = np.array(Image.open(resource_stream(__name__, filename)))
    color_image = cv2.imread(filename)
    number_of_pixels = color_image.shape[0] * color_image.shape[1]
    # compute grid
    grid = compute_grid(color_image.shape, numSegments)
    seeds = list(chain.from_iterable(grid))
    seed_len = len(seeds)
    # choose a distance metric
    distance_metric = create_augmented_snic_distance(color_image.shape, seed_len, compactness)
    segmentation, distances, numSegments = snic(
        skimage.color.rgb2lab(color_image).tolist(),
        seeds,
        compactness, nd_computations["3"], distance_metric)
    return np.array(segmentation)

def my_cross_entropy(x, y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def Lovasz_Softmax (inputs,targets):
    num_classes = inputs.size(1)
    losses = []
    for c in range(num_classes):
        target_c = (targets == c).float()
        if num_classes == 1:
            input_c = inputs[:, 0]
        else:
            input_c = inputs[:, c]
        loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
        loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
        target_c_sorted = target_c[loss_index]
        losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
    losses = torch.stack(losses)
    loss = losses.mean()
    return loss


im, fN = loadWSI()

#im = cv2.imread(args.input)

data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
if use_cuda:
    data = data.cuda()
data = Variable(data)



labels = SNIC(fN,args.compactness,args.num_superpixels)
labels = labels.reshape(im.shape[0]*im.shape[1])

u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# train
model = CNNModel(data.size(1))

if use_cuda:
    model.cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))
for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite(".\\output\\output"+str(batch_idx)+".jpg",im_target_rgb)
    cv2.imshow("output", im_target_rgb)
    cv2.waitKey(10)

    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
    target = torch.from_numpy( im_target )
    if use_cuda:
        target = target.cuda()
    target = Variable( target )
    loss = Lovasz_Softmax(output, target)

    #loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

cv2.imwrite( ".\\output\\output.png", im_target_rgb )
