import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import glob
import time
import torch
import torchvision.transforms as tr
from PIL import Image
#from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
#from basicsr.data.transforms import augment
#from basicsr.utils import  get_root_logger, imfrombytes, img2tensor
from torch.utils.data import Dataset
from esrgan_2d import *
from esrgan_noise import *

class RealESRGANDataset(Dataset):

    def __init__(self, root):
        super(RealESRGANDataset, self).__init__()
        """
        #self.file_client = None
        #self.io_backend_opt = opt['io_backend']
        self.gt_folder = 'データセットの名前'

        # disk backend with meta_info
        # Each line in the meta_info describes the relative path to an image
        with open(self.opt['meta_info']) as fin:
            paths = [line.strip().split(' ')[0] for line in fin]
            self.paths = [os.path.join(self.gt_folder, v) for v in paths]
        """
        # blur settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03] # a list for each kernel probability
        self.blur_sigma = [0.2,3]
        self.betag_range = [0.5, 4]# betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2]  # betap used in plateau blur kernels
        self.sinc_prob = 0.1  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1

        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        self.File = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        
        img_gt = Image.open(self.File[index % len(self.File)])
        img_gt = np.array(img_gt)
        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        #img_gt = augment(img_gt, True, False)

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 1600
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < 0.1:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < 0.1:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < 0.8:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_usm = usm_sharpener(img_gt)
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0]
        #img_usm = img2tensor([img_usm], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        return return_d

    def __len__(self):
        return len(self.File)