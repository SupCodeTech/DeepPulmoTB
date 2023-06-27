# 从nill文件中，对 Train 数据集的 Mask 图片的提取
import numpy as np
import os
import random
import warnings
import cv2
import gdown
from functools import partial

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pandas as pd
from matplotlib import cm
from numpy.random import rand
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)
    keras.mixed_precision.set_global_policy("mixed_float16")
except:
    pass

seed = 1337
tf.random.set_seed(seed)

from PIL import Image

import SimpleITK as sitk


def main():
  import cv2 as cv

  parser = ArgumentParser()

  parser.add_argument('Caverns_Detection_Train_CT', help='Caverns detection train CT files path')
  parser.add_argument('Original_Image_Dataset', help='Original image dataset path')

  args = parser.parse_args()

  def getarrayFromslice(file_path):
    image = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(image)
    shape = img_array.shape
    return shape[0]

  # 一组 nill 数据的获取
  def getImFromNill_(nill_file):

    num_slices = getarrayFromslice(nill_file)
    img = sitk.ReadImage(nill_file)
    img_array = sitk.GetArrayFromImage(img)
    s = 0
    print("The name of files ： " + str(nill_file[-14:-7]))
    count = 0
    count = count + 1
    for i in range(num_slices):
      img_select = img_array[i,:,:]
     
      img_pic = Image.fromarray(img_select)
      img_prefiex = args.Original_Image_Dataset

      img_save_dir_ = nill_file[-14:-7] # TRN_000_regsegm_py.nii.gz
      img_savename_ = '/{}.png'.format(i)
      img_savename = img_prefiex + img_save_dir_ + img_savename_
      img_savename_dir = img_prefiex + img_save_dir_
      img_path_folder = os.path.exists(img_savename_dir)

      if not img_path_folder:
        os.makedirs(img_savename_dir)  

      plt.imsave(img_savename, img_pic, cmap = 'gray')


  for i in range(558):
    if i < 10:
      getImFromNill_(args.Caverns_Detection_Train_CT + '/TRN_00{}.nii.gz'.format(i))
    elif i >= 10 and i <100:
      getImFromNill_(args.Caverns_Detection_Train_CT + '/TRN_0{}.nii.gz'.format(i))
    elif i >=100 and i <1000:
      getImFromNill_(args.Caverns_Detection_Train_CT + '/TRN_{}.nii.gz'.format(i))


if __name__ == '__main__':
    main()