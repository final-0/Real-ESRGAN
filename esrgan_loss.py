import math
import os
import torch
from torch import autograd as autograd
from collections import OrderedDict
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import vgg19
import math


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.feature_extractor(img)

class VGGFeatureExtractor(nn.Module):
    
    def __init__(self,layer_list):
        super(VGGFeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)
        self.layer_list = layer_list
        self.names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
        ]
        self.names1 = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
        ]
        # only borrow layers that will be used to avoid unused params
        features = vgg19_model.features[:18]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                 # in some cases, we may want to change the default stride
                modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        
        self.vgg_net.eval()
        for param in self.parameters():
            param.requires_grad = False
        
        # the mean is for image with range [0, 1]
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
            
        x = (x - self.mean) / self.std
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_list:
                output[key] = x.clone()
        return output

class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):
        loss = self.criterion(pred,target)
        return  loss

class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        #self.perceptual_weight = 1.0
       
        self.layer_weights = {'conv1_2': 0.1, 
                              'conv2_2': 0.1,
                              'conv3_4': 1,
                              'conv4_4': 1,
                              'conv5_4': 1}
        self.vgg = VGGFeatureExtractor(layer_list = ['conv1_2','conv2_2','conv3_4','conv4_4','conv5_4'])
        self.criterion = torch.nn.L1Loss()
        
    def forward(self, x, gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        # calculate perceptual loss
        percep_loss = 0
        for k in x_features.keys():    
            percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        return percep_loss
"""
    def _gram_mat(self, x):
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
"""

from torch.autograd import Variable

class GANLoss(nn.Module):

    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss_weight = 0.1
        self.loss = torch.nn.BCEWithLogitsLoss()

    def get_label(self, input, tf):
        if tf == True:
            return Variable(input.new_ones(input.size()), requires_grad=False) * 1.0
            #return input.new_ones(input.size()) * 1.0
        else:
            return Variable(input.new_ones(input.size()), requires_grad=False) * 0.0
            #return input.new_ones(input.size()) * 0.0

    def forward(self, input, tf, is_disc):
        
        target_label = self.get_label(input, tf)
        loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight

