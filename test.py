'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
Usgae:
python test.py CHECKPOINTS_PATH TEST_RAW_IMAGE_FOLDER
'''
import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import sys
from torchvision import transforms
import datetime
import math

def main():

    modelpath = sys.argv[1]

    test_ori_fd = sys.argv[2]
    ori_dirs = []
    for f in os.listdir(test_ori_fd):
        ori_dirs.append(os.path.join(test_ori_fd,f))

    #load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).cuda()
    print("=> loading trained model")
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = (imgdir.split('/')[-1]).split('.')[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.cuda()
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = './results/results_{}'.format(checkpoint['epoch'])
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(dir+'/{}corrected.png'.format(img_name))
    endtime = datetime.datetime.now()
    print(endtime-starttime)
if __name__ == '__main__':
    main()