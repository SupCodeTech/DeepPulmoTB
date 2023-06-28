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
from argparse import ArgumentParser

adaptive_adjustment_Flag = False

def count_pixels_in_circle(image, x0, y0, radius):
    height, width = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                if image[y, x] > 0:
                    count += 1

    return count

def calculate_histogram(image):

  histogram = np.histogram(image.flatten(), bins=256, range=[0, 256])
  return histogram

def sort_histogram_frequencies(histogram):

  frequencies = histogram[0]
  pixel_values = histogram[1][:-1]  
  valid_indices = np.where(pixel_values > 1)  
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
    plt.grid(linestyle='-.')

    plt.legend()
    plt.show()


def main():
  import cv2 as cv

  parser = ArgumentParser()

  parser.add_argument('Cavern_Detection_Train_Bboxes_cvs', help='Caverns detection or report train bboxes cvs')
  parser.add_argument('Cavern_Detection_Train_CT', help='Original image dataset path')
  parser.add_argument('Cavern_Detection_Train_Masks1', help='Caverns detection train masks1 path')
  parser.add_argument('Cavern_Detection_Train_Masks2', help='Caverns detection train masks2 path')
  parser.add_argument('Cavern_Detection_Train_CT_PNG', help='Original PNG Image Dataset Path')
  parser.add_argument('Training_Mask_Dataset', help='Original PNG Image Dataset Path')

  args = parser.parse_args()

  data_pd_ = pd.read_csv(args.Cavern_Detection_Train_Bboxes_cvs)

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

    print("The name of the file to process: " + str(nill_file))  

    imgs = nib.load(args.Cavern_Detection_Train_CT + "/" + nill_file + '.nii.gz')

    newimg = imgs.get_fdata()

    image_count = newimg.shape[2]

    newimg = newimg.transpose(2,1,0)

    src_mask1_ = nib.load(args.Cavern_Detection_Train_Masks1 + "/" + nill_file + '_mask.nii.gz')

    src_mask1 = src_mask1_.get_fdata()

    src_mask1 = src_mask1.transpose(2,1,0)

    src_mask2_ = nib.load(args.Cavern_Detection_Train_Masks2 + "/" + nill_file + '_regsegm_py.nii.gz')

    src_mask2 = src_mask2_.get_fdata()

    src_mask2 = src_mask2.transpose(2,1,0)

    print('Total Samples: ', image_count)

    image_type = [] 
            
    count_ = 0
    a = [] 
    for p in range(len(data_pd_)):
      if id[p] == nill_file:
        count_ = count_ + 1
        a.append(p)

    print("Number of bounding boxes in file" + nill_file +": " + str(len(a)))

    s = 0
    count = 0
    slices = 0

    k = np.zeros((image_count,512,512))
    ksc = np.zeros((image_count,512,512))
    newing_ = np.zeros((image_count,512,512))
    k_lungcavity = np.zeros((image_count,512,512))
    k_lesion = np.zeros((image_count,512,512))
    k_closed = np.zeros((image_count,512,512))
    abc_stack = None

    img_read_list = []
    img_read_lists = []
    img_read_list_ = []
    img_read_list_piexls_threholds = []

    select_flag = False  

    if len(a) > 0:
      select_flag = True


    k = np.zeros((image_count,512,512))
    ksc = np.zeros((image_count,512,512))
    newing_ = np.zeros((image_count,512,512))


    if len(a) > 0:
      for i in range(image_count):

        img_read_list_piexls_count = []

        srcs = cv2.imread(args.Cavern_Detection_Train_CT_PNG + '/' + nill_file + '/{}.png'.format(i))
        srcs = cv2.cvtColor(srcs, cv2.COLOR_BGR2GRAY)      

        k1=np.zeros((512,512))

        for f1 in range(512):
          for f2 in range(512):

            if src_mask1[i][f1][f2] > 0 or src_mask2[i][f1][f2] > 0:
              newimg[i][f1][f2] = 2
            else:
              newimg[i][f1][f2] = 0

        img_read_lists.append(srcs)

      abc_stacks = np.stack(img_read_lists, axis = 0)

    if len(a) == 0:
      img_dirs = args.Training_Mask_Dataset + '/' + nill_file

      newimg=newimg.transpose(2,1,0)

      final_img = nib.Nifti1Image(newimg, imgs.affine)

      nib.save(final_img, img_dirs + '.nii.gz')
        
      print("File: " + img_dirs + '.nii.gz' + " saved！！")


    while slices < len(a) and select_flag:
      for i in range(image_count):
        
        x1 = data_pd_['bbox_X1'][a[slices]]
        x2 = data_pd_['bbox_X2'][a[slices]]
        bbx_len = x2 - x1

        bbx_len_over_150 = False

        if bbx_len >= 150:
          bbx_len_over_150 = True

        y1 = data_pd_['bbox_Y1'][a[slices]] 
        y2 = data_pd_['bbox_Y2'][a[slices]]
        bby_len = y2 - y1

        bby_len_over_150 = False

        if bby_len >= 150:
          bby_len_over_150 = True

        if i >= data_pd_['bbox_Z1'][a[slices]] and i <= data_pd_['bbox_Z2'][a[slices]]:

          lesion_area_piexls = 1
          
          b1 = 0
          b2 = 0
          b3 = 0
          b4 = 0

          bb1_piexl = 0
          bb1_piexl_flag = False
          bb1_piexl_len = 0
          b1_bounding_flag = False
          b1_temp_bounding_flag = False

          bb2_piexl = 0
          bb2_piexl_flag = False
          bb2_piexl_len = 0
          b2_bounding_flag = False
          b2_temp_bounding_flag = False

          bb3_piexl = 0
          bb3_piexl_flag = False
          bb3_piexl_len = 0
          b3_bounding_flag = False
          b3_temp_bounding_flag = False

          bb4_piexl = 0
          bb4_piexl_flag = False
          bb4_piexl_len = 0
          b4_bounding_flag = False
          b4_temp_bounding_flag = False

          b1_flag = 0
          b1_flag_ = False
          b2_flag = 0
          b3_flag = 0
          b4_flag = 0

          b1_flag_ = 0
          b2_flag_ = 0
          b3_flag_ = 0
          b4_flag_ = 0

          b1_flag_end = 0
          b2_flag_end = 0
          b3_flag_end = 0
          b4_flag_end = 0

          b1_minus = 0
          b2_minus = 0
          b3_minus = 0
          b4_minus = 0

          patch_ee = None
          
          ass = (y1 - bby_len)
          bss = (y2 + bby_len)
          css = (x1 - bbx_len)
          dss = (x2 + bbx_len)


          if bbx_len_over_150 == True:
            css = x1
            dss = x2

          if bby_len_over_150 == True:
            ass = y1
            bss = y2

          if (y1 - bby_len) <= 0:
            ass = 0
          if (x1 - bbx_len) <= 0:
            css = 0
          if (y2 + bby_len) >= 512:
            bss = 512
          if (x2 + bbx_len) >= 512:
            dss = 512

          patch_ee = abc_stacks[i, ass : bss, css:dss]
          gray = patch_ee
      
          patch_ee_ = abc_stacks[i, y1 : y2, x1:x2]

          histogram = calculate_histogram(patch_ee)

          sorted_pixel_values, sorted_frequencies = sort_histogram_frequencies(histogram)

          pixel_1, pixel_2 = find_pixel_differences(sorted_pixel_values)

          threshold_value = (pixel_1 + pixel_2) // 2

          ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

          binary = thresh

          if pixel_2 >= pixel_1:
            pixel_2 = pixel_1 

          ret1, thresh1 = cv2.threshold(gray, pixel_2 + 10, 255, cv2.THRESH_BINARY)

          for px in range(binary.shape[1]):
            for py in range(binary.shape[0]):
              k_lesion[i][py + ass][px + css] = binary[py][px]

          binary = thresh1

          if bbx_len_over_150 == True and bby_len_over_150 == False:
            binary = binary[bby_len:2*bby_len, :]
          elif bbx_len_over_150 == False and bby_len_over_150 == True:
            binary = binary[:, bbx_len:2*bbx_len]
          elif bbx_len_over_150 == True and bby_len_over_150 == True:
            binary = binary[:, :]
          else:
            binary = binary[bby_len:2*bby_len, bbx_len:2*bbx_len]

          contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          patch_ee_ = np.dstack((patch_ee_, patch_ee_, patch_ee_))

          mask = np.zeros_like(patch_ee_)

          for sr in range(len(contours)):
              if hierarchy[0][sr][3] != -1:
                  cv2.drawContours(mask, contours, sr, (255, 255, 255), cv2.FILLED)

          whe_ex_no_parent = False

          left_up_corner = False
          right_up_corner = False
          left_down_corner = False
          right_down_corner = False

          for px in range(mask.shape[1]):
            for py in range(mask.shape[0]):
              
              k_lungcavity[i][py + y1][px + x1] = mask[py][px][0]
    
          cross_move = 0
          up_move = True

          caluc_up = 0
          delete_detect = 0
          
          abc_stack = k_lesion
          x_len = bbx_len

          extract_threshold_value = 250
          lesion_area_piexls = 255
            

          if bbx_len and x_len > 90:
            for x in range(512):
              for y in range(512):
             
                if x >= x1 and x <= x2:
                  if y >= y1 and y <= y2:
                    if abc_stack[i][y][x] >= extract_threshold_value:
                     
                      k[i][y][x] = lesion_area_piexls

          if x_len<=90:
            
            for x in range(512):
              for y in range(512):
                
                if x >= x1 and x <= x2:
                  if y >= y1 and y <= y2:
                    if abc_stack[i][y][x] >= extract_threshold_value:
                      k[i][y][x] = lesion_area_piexls
                      ksc[i][y][x] = lesion_area_piexls

                if x == x1 and b1_flag == 0:
                  if y >= y1 and y <= y2:
                    
                    b1_flag_ = False
                    
                    if abc_stack[i][y][x-1] > threshold_value and abc_stack[i][y][x+1] > threshold_value and abc_stack[i][y][x+2] > threshold_value and abc_stack[i][y][x+3] > threshold_value and abc_stack[i][y][x-2] > threshold_value:
                      b1_flag_ = True
                  
                                  
                    if b1_flag_:
                      bb1_piexl = bb1_piexl + 1
                      if y == y2:
                        if bb1_piexl > bb1_piexl_len:
                          bb1_piexl_len = bb1_piexl

                    elif b1_flag_ == False and bb1_piexl > 0:
                      
                      if bb1_piexl > bb1_piexl_len:
                        bb1_piexl_len = bb1_piexl

                      bb1_piexl = 0

                    if y == y2:
                      if bb1_piexl_len/(y2-y1) > 0.1:

                        b1 = b1 + 1
                        b1_flag = 1

                      bb1_piexl_len = 0
                      bb1_piexl = 0       
                  
                
                if b1_flag == 1 and (x == x1 + 1) and b1_flag_end == 0:
                  if y == y2:  
                    
                    b1_aixs = True
                    b1_aixs_minus = 2
                    bb1_piexl_len = 0
                    bb1_piexl = 0

                  
                    while b1_aixs:
                      
                      b1_minus = y2 - y1

                      b1_move_jug = 0
                      while b1_minus > 0:

                        b1_flag_ = False


                        pixel_count = count_pixels_in_circle(abc_stack[i], x-1-b1_aixs_minus, y - b1_minus, 5)

                        
                        b1_minus = b1_minus - 1

                        if pixel_count >= 31:
                          bb1_piexl = bb1_piexl + 1

                          b1 = b1 + 1
                          break
                        elif pixel_count < 31:
                          b1_move_jug = b1_move_jug + 1
                        if b1_move_jug >= y2 - y1 - 2:
                          b1_aixs = False
                        if b1 >= (y2 - y1) - 1:
                          b1_aixs = False

                      b1_aixs_minus = b1_aixs_minus + 1
                      if b1_aixs_minus >= (y2 - y1):
                        b1_aixs = False

                    if adaptive_adjustment_Flag:
                      b1 = x2 - x1
                    
                    if b1 > 0:
                      
                      for t1 in range(x1 - b1 - 2, x1):
                        for t2 in range(y1, y2):
                          if abc_stack[i][t2][t1] >= extract_threshold_value:
                            
                            k[i][t2][t1] = lesion_area_piexls
                            
                    b1_flag_end = 1
                      
                    
                if x == x2 and b2_flag == 0:
                  if y >= y1 and y <= y2:

                    b2_flag_ = False
                    
                    if abc_stack[i][y][x-1] > threshold_value and abc_stack[i][y][x+1] > threshold_value and abc_stack[i][y][x-2] > threshold_value and abc_stack[i][y][x-3] > threshold_value and abc_stack[i][y][x+2] > threshold_value:
                      b2_flag_ = True
                            
                                  
                    if b2_flag_:
                      bb2_piexl = bb2_piexl + 1
                      if y == y2:
                        if bb2_piexl > bb2_piexl_len:
                          bb2_piexl_len = bb2_piexl

                    elif b2_flag_ and bb2_piexl > 0:
                      
                      if bb2_piexl > bb2_piexl_len:
                        bb2_piexl_len = bb2_piexl
                      bb2_piexl = 0

                    if y == y2:
                      if bb2_piexl_len/(y2-y1) > 0.1:

                        b2 = b2 + 1
                        b2_flag = 1

                      bb2_piexl_len = 0
                      bb2_piexl = 0

                if b2_flag == 1 and (x == x2 + 1) and b2_flag_end == 0:
                  if y == y2:  
                    
                    b2_aixs = True
                    b2_aixs_minus = 0
                    bb2_piexl_len = 0
                    bb2_piexl = 0

              
                    while b2_aixs:
          
                      b2_minus = y2 - y1

              
                      b2_move_jug = 0

                      while b2_minus > 0:

                        b2_flag_ = False


                        pixel_count = count_pixels_in_circle(abc_stack[i], x-1+b2_aixs_minus, y - b2_minus, 5)

                        if pixel_count >= 31:
                          bb2_piexl = bb2_piexl + 1
                          b2 = b2 + 1
                          break
                        elif pixel_count < 31:
                          
                          b2_move_jug = b2_move_jug + 1
                        if b2_move_jug >= y2 - y1 - 2:
                          b2_aixs = False

                        if b2 >= (y2 - y1) - 1:
                          b2_aixs = False

                        b2_minus = b2_minus - 1


                      b2_aixs_minus = b2_aixs_minus + 1
                      
                      if b2_aixs_minus >= (y2 - y1):
                        b2_aixs = False

                    if adaptive_adjustment_Flag:
                      b2 = x2 - x1

                    if b2 > 0:
                      for t1 in range(x2 - 1, x2 + b2 + 1):
                        for t2 in range(y1, y2):
                          if abc_stack[i][t2][t1] >= extract_threshold_value:
                            k[i][t2][t1] = lesion_area_piexls
                            
                    b2_flag_end = 1
              
          
          for y in range(512):
            for x in range(512):
                
              if y == y1 and b3_flag == 0:
                if x >= x1 and x <= x2:

                  b3_flag_ = False
                  
                  if abc_stack[i][y - 2][x] > threshold_value and abc_stack[i][y - 1][x] > threshold_value and abc_stack[i][y + 1][x] > threshold_value and abc_stack[i][y + 2][x] > threshold_value and abc_stack[i][y + 3][x] > threshold_value:
                    b3_flag_ = True
                                                        

                  if b3_flag_:
                    bb3_piexl = bb3_piexl + 1
                    if x == x2:
                      if bb3_piexl > bb3_piexl_len:
                        bb3_piexl_len = bb3_piexl

                  elif b3_flag_ == False and bb3_piexl > 0:
                    
                    if bb3_piexl > bb3_piexl_len:
                      bb3_piexl_len = bb3_piexl
                      
                    bb3_piexl = 0               

                  if x == x2:
                    if bb3_piexl_len/(x2-x1) >= 0.1:
                      b3 = b3 + 1
                      b3_flag = 1

                    bb3_piexl_len = 0
                    bb3_piexl = 0      
                
              if True:

                if b3_flag == 1 and (y == y1 + 1) and b3_flag_end == 0:
                  if x == x2:
                    
                    b3_aixs = True
                    b3_aixs_minus = 2
                    bb3_piexl_len = 0

                   
                    while b3_aixs and b3_aixs_minus <= (x2 - x1):

                      b3_move_jug = 0

                      b3_minus = x2 - x1
                      
                      while b3_minus >= 0:

                        b3_flag_ = False

                        pixel_count = count_pixels_in_circle(abc_stack[i], x - b3_minus, y -1-b3_aixs_minus, 5)

                        b3_minus = b3_minus - 1

                        if pixel_count >= 31:
                          bb3_piexl = bb3_piexl + 1

                          b3 = b3 + 1
                          break
                        elif pixel_count < 31:
                          b3_move_jug = b3_move_jug + 1
                        if b3_move_jug >= x2 - x1 - 2:
                          b3_aixs = False
                        if b3 >= (x2 - x1) - 1:
                          b3_aixs = False

                      b3_aixs_minus = b3_aixs_minus + 1

                      if b3_aixs_minus >= (x2 - x1):
                        b3_aixs = False


                    if adaptive_adjustment_Flag:
                      b3 = y2 - y1
                    if b3 > 0:
                     
                      for t2 in range(y1-b3-1, y1):
                        for t1 in range(x1 - b1, x2 + b2):
                          if abc_stack[i][t2][t1] >= extract_threshold_value:
                            # k[i][t2][t1] = abc_stack[i][t2][t1]
                            k[i][t2][t1] = lesion_area_piexls
                            
                    b3_flag_end = 1
                    
                    

              if y == y2 and b4_flag == 0:
                if x >= x1 and x <= x2:

                  b4_flag_ = False
                  
                  if abc_stack[i][y - 1][x] > threshold_value and abc_stack[i][y + 1][x] > threshold_value and abc_stack[i][y + 2][x] > threshold_value and abc_stack[i][y - 2][x] > threshold_value and abc_stack[i][y - 3][x] > threshold_value:
                    b4_flag_ = True
                                                          
                  if b4_flag_:
                    bb4_piexl = bb4_piexl + 1
                    if x == x2:
                      if bb4_piexl > bb4_piexl_len:
                        bb4_piexl_len = bb4_piexl

                  elif b4_flag_ == False and bb4_piexl > 0:
                    
                    if bb4_piexl > bb4_piexl_len:
                      bb4_piexl_len = bb4_piexl

                    bb4_piexl = 0

                  if x == x2:
                    if bb4_piexl_len/(x2-x1) > 0.1:
                      b4 = b4 + 1
                      b4_flag = 1

                    bb4_piexl_len = 0
                    bb4_piexl = 0     
                
              if True:
                if b4_flag == 1 and (y == y2 + 1) and b4_flag_end == 0:
                  if x == x2:
                    
                    b4_aixs = True
                    b4_aixs_minus = 0
                    bb4_piexl_len = 0

                   
                    while b4_aixs and b4_aixs_minus <= (x2 - x1):
                      b4_minus = x2 - x1

                      b4_move_jug = 0
                      
                      while b4_minus > 0:

                        b4_flag_ = False

                        pixel_count = count_pixels_in_circle(abc_stack[i], x - b4_minus, y -1+b4_aixs_minus, 5)
                        
                        
                        b4_minus = b4_minus - 1

                        if pixel_count >= 31:
                          bb4_piexl = bb4_piexl + 1

                          b4 = b4 + 1
                          break
                        elif pixel_count < 31:
                          b4_move_jug = b4_move_jug + 1
                        if b4_move_jug >= x2 - x1 - 2:
                          b4_aixs = False
                        if b4 >= (x2 - x1) - 1:
                          b4_aixs = False

                      b4_aixs_minus = b4_aixs_minus + 1
                      
                      if b4_aixs_minus >= (x2 - x1):
                        b4_aixs = False

                    if adaptive_adjustment_Flag:
                      b4 = y2 - y1

                    if b4 > 0:
                                          
                      for t2 in range(y2, y2 + b4 + 1):
                        for t1 in range(x1 - b1, x2 + b2):
                          
                          if abc_stack[i][t2][t1] >= extract_threshold_value:
                            k[i][t2][t1] = lesion_area_piexls

                            
                    b4_flag_end = 1


                      

      slices = slices + 1
      img_dirs = args.Training_Mask_Dataset + '/' + nill_file

      if slices == len(a):

        if newimg.shape != k.shape:
          print("C k: " + str(k.shape))
          print("save error ！！")
          os.system(pause)

        for rt1 in range(image_count):
          for rt2 in range(512):
            for rt3 in range(512):

              if newimg[rt1][rt2][rt3] > 0:
                if k_lungcavity[rt1][rt2][rt3] > 0:
                  newimg[rt1][rt2][rt3] = 3
                if k[rt1][rt2][rt3] > 0:
                  newimg[rt1][rt2][rt3] = 1
             
        newimg = newimg.transpose(2,1,0)
        final_img = nib.Nifti1Image(newimg, imgs.affine)
        print("unique" + str(np.unique(newimg)))

        nib.save(final_img, img_dirs + '.nii.gz')

        print("File: " + img_dirs + '.nii.gz' + " saved！！")

        

if __name__ == '__main__':
    main()
 
