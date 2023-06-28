import cv2 as cv
import matplotlib.pyplot as plt
import math
import pathlib
from skimage.io import imread
from skimage.feature import canny
import pandas as pd
import numpy as np
from scipy import signal
from PIL import Image
import SimpleITK as sitk
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import os

def calculate_histogram(image):
    
    histogram = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def sort_histogram_frequencies(histogram):
    
    frequencies = histogram[0]
    pixel_values = histogram[1][:-1]  
    valid_indices = np.where(pixel_values >= 5)  
    sorted_indices = np.argsort(frequencies[valid_indices])[::-1]  
    sorted_frequencies = frequencies[valid_indices][sorted_indices]  
    sorted_pixel_values = pixel_values[valid_indices][sorted_indices]  
    return sorted_pixel_values, sorted_frequencies

def find_pixel_differences(sorted_pixel_values):
    
    for i in range(1, len(sorted_pixel_values)):
        difference = abs(sorted_pixel_values[0] - sorted_pixel_values[i])
        if difference > 30:
            return sorted_pixel_values[0], sorted_pixel_values[i]
    return None

def plot_histogram(histogram, pixel_1, pixel_2):
    
    plt.figure()
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(histogram[1][:-1], histogram[0])
    plt.axvline(x=pixel_1, color='r', linestyle='--', label='Pixel 1')
    plt.axvline(x=pixel_2, color='g', linestyle='--', label='Pixel 2')
    plt.legend()
    plt.show()

def main():
  
  parser = ArgumentParser()

  parser.add_argument('Cavern_detection_train_bboxes', help='Caverns detection train CT CVS files path')
  parser.add_argument('Cavern_Detection_Train_CT', help=' Cavern Detection Train CT nii.gz files ')
  parser.add_argument('Cavern_Detection_Train_CT_PNG', help=' Cavern Detection Train CT PNG files ')
    
  args = parser.parse_args()

  data_pd_ = pd.read_csv(args.Cavern_detection_train_bboxes)

  id = list(data_pd_['id'])
  X1 = list(data_pd_['bbox_X1'])
  X2 = list(data_pd_['bbox_X2'])
  Y1 = list(data_pd_['bbox_Y1'])
  Y2 = list(data_pd_['bbox_Y2'])
  Z1 = list(data_pd_['bbox_Z1'])
  Z2 = list(data_pd_['bbox_Z2'])

  nill_file = None
  threshold_value = 0
  extract_threshold_value = 0
  for files_num in range(558):

    if files_num < 10:
      data_dir = "TRN_00{}".format(files_num)
    elif files_num >= 10 and files_num < 100:
      data_dir = "TRN_0{}".format(files_num)
    elif files_num >= 100 and files_num < 1000:
      data_dir = "TRN_{}".format(files_num)

    nill_file = data_dir

    print("File name for processing: " + str(nill_file))  

    image_type = []

    count_ = 0
    a = [] 
    for p in range(len(data_pd_)):
      if id[p] == nill_file:
        count_ = count_ + 1
        a.append(p)

    imgs = None
    newimg = None

    if len(a) > 0:

      imgs = nib.load(args.Cavern_Detection_Train_CT + '/' + nill_file + '.nii.gz')

      image_count = newimg.shape[2]

      newimg = imgs.get_fdata()

      newimg = newimg.transpose(2,1,0)

      
      print('Total Samples: ', image_count)

      print(" The number of bounding boxes contained in the " + nill_file + " file： " + str(len(a)))

      s = 0
      count = 0
      slices = 0

      k = np.zeros((image_count,512,512))
      ksc = np.zeros((image_count,512,512))
      newing_ = np.zeros((image_count,512,512))
      k_lungcavity = np.zeros((image_count,512,512))
      k_lesion = np.zeros((image_count,512,512))
      k_closed = np.zeros((image_count,512,512))

      img_read_list = []
      img_read_lists = []
      img_read_list_ = []
      img_read_list_piexls_threholds = []

      select_flag = False

      if len(a) > 0:
        select_flag = True

      if select_flag:
        for i in range(image_count):
            
          img_read_list_piexls_count = []
        
          srcs = cv2.imread(args.Cavern_Detection_Train_CT_PNG + '/' + nill_file + '/{}.png'.format(i))
          srcs = cv2.cvtColor(srcs, cv2.COLOR_BGR2GRAY)

          img_read_lists.append(srcs)

        abc_stacks = np.stack(img_read_lists, axis = 0)

        print("abc_stacks.shape: " + str(abc_stacks.shape))

      while slices < len(a) and select_flag:
        for i in range(512):

          z1 = data_pd_['bbox_Z1'][a[slices]]
          z2 = data_pd_['bbox_Z2'][a[slices]]
          bbz_len = z2 - z1

          y1 = data_pd_['bbox_Y1'][a[slices]]
          y2 = data_pd_['bbox_Y2'][a[slices]]
          bby_len = y2 - y1

          if i >= data_pd_['bbox_X1'][a[slices]] and i <= data_pd_['bbox_X2'][a[slices]]:

            patch_ee = None
              
            ass = (y1 - bby_len)
            bss = (y2 + bby_len)
            css = (z1 - bbz_len)
            dss = (z2 + bbz_len)

            if (y1 - bby_len) <= 0:
              ass = 0
            if (z1 - bbz_len) <= 0:
              css = 0
            if (y2 + bby_len) >= 512:
              bss = 512
            if (z2 + bbz_len) >= image_count:
              dss = image_count

            patch_ee = abc_stacks[css:dss, ass : bss, i]
            patch_ee_ = abc_stacks[z1:z2, y1 : y2, i]

            histogram = calculate_histogram(patch_ee)
              
            sorted_pixel_values, sorted_frequencies = sort_histogram_frequencies(histogram)

            pixel_1, pixel_2 = find_pixel_differences(sorted_pixel_values)

            threshold_value = (pixel_1 + pixel_2) // 2

            threshold, binary = cv2.threshold(patch_ee, threshold_value, 255, cv2.THRESH_BINARY)

            for pz in range(binary.shape[0]):
              for py in range(binary.shape[1]):
                if newimg[pz + css, py + ass, i] == 1 or newimg[pz + css, py + ass, i] == 3:
                  k_lesion[pz + css, py + ass, i] = 255
                  binary[pz][py] = 255

                elif binary[pz][py] > 0:
                  k_lesion[pz + css, py + ass, i] = 255

            binary = binary[bbz_len:2*bbz_len, bby_len:2*bby_len]

            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            binary = np.dstack((binary, binary, binary))

            mask = np.zeros_like(binary)

            for sr in range(len(contours)):
              
                if hierarchy[0][sr][3] != -1:
                   
                    cv2.drawContours(mask, contours, sr, (255, 255, 255), cv2.FILLED)

            left_up_corner = False
            right_up_corner = False
            left_down_corner = False
            right_down_corner = False

            for pz in range(mask.shape[0]):
              for py in range(mask.shape[1]):
                k_lungcavity[pz + z1, py + y1, i] = mask[pz][py][0]

            cross_move = 0
            up_move = True

            caluc_up = 0
            delete_detect = 0

            z1_, y1_ = z1, y1
            z2_, y2_ = z2, y2

            abc_stack = k_lesion

        slices = slices + 1
        img_dirs = args.Cavern_Detection_Train_CT + '/' + nill_file

        if slices == len(a):

          if newimg.shape != k.shape:
            print("C k: " + str(k.shape))
            
            os.system(pause)

          for rt1 in range(image_count):
            for rt2 in range(512):
              for rt3 in range(512):

                if newimg[rt1][rt2][rt3] > 0:
                  if k_lungcavity[rt1][rt2][rt3] > 0 and newimg[rt1][rt2][rt3] != 1:
                    newimg[rt1][rt2][rt3] = 3
                      
          newimg = newimg.transpose(2,1,0)
          final_img = nib.Nifti1Image(newimg, imgs.affine)
          print("unique" + str(np.unique(newimg)))

          nib.save(final_img, img_dirs + '.nii.gz')

          print("File " + img_dirs + '.nii.gz' + " saved！！")


if __name__ == '__main__':
    main()
