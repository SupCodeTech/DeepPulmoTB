def getarrayFromslice(file_path):
  image = sitk.ReadImage(file_path)
  img_array = sitk.GetArrayFromImage(image)
  shape = img_array.shape
  return shape[0]
  
import nibabel as nib
from nibabel.testing import data_path
import os
import cv2 as cv
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import math
import pathlib
from skimage.io import imread
from skimage.feature import canny
import pandas as pd
import numpy as np
from scipy import signal
from PIL import Image
import SimpleITK as sitk
from PIL import Image, ImageDraw


def main():
  
  parser = ArgumentParser()

  parser.add_argument('Test_masks1', help=' ')
  parser.add_argument('Test_masks2', help=' ')
  parser.add_argument('Ouput_Dirs', help=' ')
  args = parser.parse_args()
    
  nill_file = None
  threshold_value = 0
  extract_threshold_value = 0
  
  for files_num in range(1,422):

    if files_num < 10:
      data_dir = "TST_000{}".format(files_num)
    elif files_num >= 10 and files_num < 100:
      data_dir = "CTR_TST_00{}".format(files_num)
    elif files_num >= 100 and files_num < 1000:
      data_dir = "CTR_TST_0{}".format(files_num)
  
    nill_file = data_dir
  
    print("The name of the file to process: " + str(nill_file))
  
    imgs = nib.load(args.Test_masks1 + "/" + nill_file + '.nii.gz')
  
    newimg = imgs.get_fdata()
  
    newimg = newimg.transpose(2,1,0)
  
    src_mask1_ = nib.load(args.Test_masks1 + "/" + nill_file + '.nii.gz')
  
    src_mask1 = src_mask1_.get_fdata()
  
    src_mask1 = src_mask1.transpose(2,1,0)
  
    src_mask2_ = nib.load(args.Test_masks2 + "/" + nill_file + '.nii.gz')
  
    src_mask2 = src_mask2_.get_fdata()
  
    src_mask2 = src_mask2.transpose(2,1,0)
  
    image_count = getarrayFromslice(args.Test_masks2 +  "/" + nill_file + '.nii.gz')
  
    image_type = []
  
    count_ = 0
    s = 0
    count = 0
    slices = 0
    img_read_list = []
    img_read_lists = []
    img_read_list_ = []
    img_read_list_piexls_threholds = []
  
    select_flag = True
  
    if select_flag:
      for i in range(image_count):
        img_read_list_piexls_count = []
        for f1 in range(512):
          for f2 in range(512):
            if src_mask1[i][f1][f2] > 0 or src_mask2[i][f1][f2] > 0:
              newimg[i][f1][f2] = 1
            else:
              newimg[i][f1][f2] = 0
  
      img_dirs =  args.Ouput_Dirs + '/' + nill_file
      
      newimg = newimg.transpose(2,1,0)
      
      final_img = nib.Nifti1Image(newimg, src_mask1_.affine)
      
      print("unique" + str(np.unique(newimg)))

      nib.save(final_img, img_dirs + '.nii.gz')

      print("Flie " + img_dirs + '.nii.gz' + " saved！！")
