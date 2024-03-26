import torch.utils.data as data
import os.path
import cv2
import numpy as np
from data import common

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = ['.png', '.npy']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)  #Return True or False

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):                    #type(sorted(os.walk(dir))) is a list inside list file
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images                                                   #['./Dataset/DIV2K//DIV2K_train/DIV2K_train_HR/0670.png',...]


class div2k(data.Dataset):
    def __init__(self, opt):                                        #n_train=800(number of input images) #batch_size=16/64/128 #test_every=1000(dont know)
        self.opt = opt
        self.scale = self.opt.scale                                 #scale=4
        self.root = self.opt.root                                   #root="./Dataset/DIV2K/"
        self.ext = self.opt.ext                                     #ext=extension= '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = 10                                            #self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)                             #Update the root, and define directory for hr and lr
        self.images_hr, self.images_lr = self._scan()               #

    def _set_filesystem(self, dir_data):
        self.dir_hr = "./Datasets/DIV2K/DIV2K_train/DIV2K_train_HR/"
        self.dir_lr = "./Datasets/DIV2K/DIV2K_train/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4/"
        
    def __getitem__(self, idx):             #idx=2826, ...
        lr, hr = self._load_file(idx)       #Load file of index = 2826%800 = 426
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size  #patch_size=192
        scale = self.scale                #scale=4
        if self.train:
            img_in, img_tar = common.get_patch(img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        #print(type(make_dataset(self.dir_hr)))   #<class list>
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        #print(idx)  2826, 977, ...
        idx = self._get_index(idx)
        #print(idx)  426, 177, ...
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr
