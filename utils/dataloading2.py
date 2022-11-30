from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2


# from utils.dataGenerator import sample_random_patch_png, sample_label_with_coor_png


# 将文件夹里的数据读成dataset
class BasicDataset(Dataset):
    # 构造函数
    def __init__(self, imgs_dir, masks_dir,size, augumentation,probability_dir = None):
        self.size = size  # 图像统一裁剪尺寸
        self.imgs_dir = imgs_dir  # 图像文件
        self.masks_dir = masks_dir  # 标签文件夹
        self.probability_dir = probability_dir
        self.augumentation = augumentation

        # 获得所有图像文件名
        self.ids = [file for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        # logging.info(f'Creating dataset with {len(self.ids)} examples')

    # dataset长度函数
    def __len__(self):
        return len(self.ids)

    # 数据预处理
    @classmethod
    def preprocess(cls, pil_img, is_img, size):
        # 如果是图像对象，将它转成numpy数组
        if is_img:
            img_nd = np.array(pil_img)
        else:
            img_nd = pil_img

        # 如果是一通道图像，将它转成3通道
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd
        # 将图像的像素值归一化到0到1之间
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # 将图像裁剪为统一大小
        if len(size) > 0:
            x = size[1]  # 获取统一裁剪x轴方向上的长度
            y = size[0]  # 获取统一裁剪y轴方向上的长度
            jiany = int((img_trans.shape[1] - y + 10) / 2)  # 计算y轴方向两侧需要裁减掉的距离
            jianx = int((img_trans.shape[0] - x + 10) / 2)  # 计算x轴方向两侧需要裁减掉的距离

            # 因为裁减掉的长度不一定被2整除，所以判断各种情况来进行左右填充
            if (img_trans.shape[0] - x) % 2 == 1 and (img_trans.shape[1] - y) % 2 == 1:
                img_trans = img_trans[jianx + 1:-1 * jianx, jiany + 1:-1 * jiany, :]
            elif (img_trans.shape[0] - x) % 2 == 1:
                img_trans = img_trans[jianx + 1:-1 * jianx, jiany:-1 * jiany, :]
            elif (img_trans.shape[1] - y) % 2 == 1:
                img_trans = img_trans[jianx:-1 * jianx, jiany + 1:-1 * jiany, :]
            else:
                img_trans = img_trans[jianx:-1 * jianx, jiany:-1 * jiany, :]

        return img_trans  # 返回处理后的图像numpy数组

    # dataset返回对应位置数据函数
    def __getitem__(self, i):
        idx = self.ids[i]  # 获取第i个图像的文件名
        mask_file = self.masks_dir + idx[:-4] + '.png'  # 获取第i个标签的文件路径
        img_file = self.imgs_dir + idx  # 获取第i个图像的文件路径


        mask_original = np.asarray(Image.open(mask_file))  # 读取标签
        mask = mask_original.copy()  # 有时候没有修改权限，需要复制一份
        mask = mask / 255  # 将mask转为0、1标签

        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     

        if self.augumentation:
            sample_img = self.augumentation(image = image, mask = mask)
            image = sample_img['image']
        img = Image.fromarray(image)
       
        image = self.preprocess(image, 1, self.size)  # 图像预处理
        mask = self.preprocess(mask, 0, self.size)  # 标签预处理

        idx_un = list(np.unique(mask))
        for i in range(len(idx_un)):
            mask[mask == idx_un[i]] = i

        if self.probability_dir:
            npy_file = np.load(self.probability_dir + idx[:-4] + '.npy')
            probability_map = npy_file.copy()
            probability_map = probability_map / 10 #npy file保存的概率，值在0-10之间
            probability_map = self.preprocess(probability_map, 0, self.size)

            return {
                'image': torch.from_numpy(image.transpose((2, 0, 1))).type(torch.FloatTensor),  # 返回tensor类型的图像
                'mask': torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.FloatTensor),  # 返回tensor类型的标签
                'probability_map':torch.from_numpy(probability_map.transpose((2, 0, 1))).type(torch.FloatTensor)#返回概率图谱
            }

        return {
            'image': torch.from_numpy(image.transpose((2, 0, 1))).type(torch.FloatTensor),  # 返回tensor类型的图像
            'mask': torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.FloatTensor)  # 返回tensor类型的标签
        }