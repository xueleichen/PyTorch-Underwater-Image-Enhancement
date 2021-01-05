'''
Authot:Xuelei Chen(chenxuelei@hotmail.com)
'''
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

def get_imgs_list(ori_dirs,ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        img_name = (ori_imgdir.split('/')[-1]).split('.')[0]
        ucc_imgdir = os.path.dirname(ucc_dirs[0])+'/'+img_name+'.png'

        if ucc_imgdir in ucc_dirs:
            img_list.append(tuple([ori_imgdir,ucc_imgdir]))

    return img_list

class uwcc(data.Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(uwcc, self).__init__()

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.train = train
        self.loader = loader

        if self.train == True:
            print('Found {} pairs of training images'.format(len(self.img_list)))
        else:
            print('Found {} pairs of testing images'.format(len(self.img_list)))
            
    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        if self.train == True:
            oritransform = transforms.Compose([
                # transforms.RandomResizedCrop(256,scale=(0.5,1.0)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
            ucctransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            sample[0] = oritransform(sample[0])
            sample[1] = ucctransform(sample[1])
        else:
            oritransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ucctransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            sample[0] = oritransform(sample[0])
            sample[1] = ucctransform(sample[1])

        return sample

    def __len__(self):
        return len(self.img_list)