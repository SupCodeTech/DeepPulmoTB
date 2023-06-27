
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
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import numpy as np


def calculate_histogram(image):
    # 计算直方图
    histogram = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def sort_histogram_frequencies(histogram):
    # 对直方图频率进行排序
    frequencies = histogram[0]
    pixel_values = histogram[1][:-1]  # 像素值数组
    valid_indices = np.where(pixel_values >= 5)  # 有效的像素值索引
    sorted_indices = np.argsort(frequencies[valid_indices])[::-1]  # 按频率降序排序的索引
    sorted_frequencies = frequencies[valid_indices][sorted_indices]  # 排序后的频率
    sorted_pixel_values = pixel_values[valid_indices][sorted_indices]  # 排序后的像素值
    return sorted_pixel_values, sorted_frequencies

def find_pixel_differences(sorted_pixel_values):
    # 寻找像素值之差大于40的两个像素
    for i in range(1, len(sorted_pixel_values)):
        difference = abs(sorted_pixel_values[0] - sorted_pixel_values[i])
        if difference > 30:
            return sorted_pixel_values[0], sorted_pixel_values[i]
    return None

def plot_histogram(histogram, pixel_1, pixel_2):
    # 绘制直方图
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

  parser.add_argument('Caverns_detection_train_bboxes', help='Caverns detection train CT CVS files path')
  parser.add_argument('Cavern_Detection_Train_CT', help=' ')
  parser.add_argument('Cavern_Detection_Train_CT_PNG', help=' ')

  args = parser.parse_args()

  data_pd_ = pd.read_csv(args.Caverns_detection_train_bboxes)

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

  import nibabel as nib
  from nibabel.testing import data_path
  import os

  for files_num in range(60):

    if files_num < 10:
      data_dir = "TRN_0{}".format(files_num)
    elif files_num >= 10 and files_num < 100:
      data_dir = "TRN_{}".format(files_num)

    nill_file = data_dir

    print("File name for processing: " + str(nill_file))  

    image_type = [] 
            
    count_ = 0
    a = [] # 属于这个文件的所有id
    for p in range(len(data_pd_)):
      if id[p] == nill_file:
        count_ = count_ + 1
        a.append(p)

    print("The number of bounding boxes contained in the " + nill_file + " file： " + str(len(a)))

    # ttt = os.path.exists(args.Training_Mask_Dataset + '/' + nill_file + '.nii.gz')
    imgs = None
    newimg = None

    if len(a) > 0:
      imgs = nib.load(args.Training_Mask_Dataset + '/' + nill_file + '.nii.gz')

      newimg = imgs.get_fdata()

      image_count = newimg.shape[2]

      newimg = newimg.transpose(2,1,0)

    print('Total Samples: ', image_count)

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

    if select_flag:
      for i in range(image_count):

        # print("第 " + str(i) +  " 张图片")

        img_read_list_piexls_count = []

        # 读取已经被提取Mask部分的图像
        
        srcs = cv2.imread(args.Cavern_Detection_Train_CT_PNG + '/' + nill_file + '/{}.png'.format(i))
        srcs = cv2.cvtColor(srcs, cv2.COLOR_BGR2GRAY)

        
        img_read_lists.append(srcs)
        
      
      abc_stacks = np.stack(img_read_lists, axis = 0)

      # cv2_imshow(abc_stacks[:,50,:])

      print("abc_stacks.shape: " + str(abc_stacks.shape))
      # abc_stack_ = np.stack(img_read_list_, axis = 0)


    # 正面切片 - Front

    while slices < len(a) and select_flag:
      for i in range(image_count):
        
        x1 = data_pd_['bbox_X1'][a[slices]]
        x2 = data_pd_['bbox_X2'][a[slices]]
        bbx_len = x2 - x1

        y1 = data_pd_['bbox_Y1'][a[slices]] 
        y2 = data_pd_['bbox_Y2'][a[slices]]
        bby_len = y2 - y1
        
        if i >= data_pd_['bbox_Z1'][a[slices]] and i <= data_pd_['bbox_Z2'][a[slices]]:

          # patch_ee = np.zeros((bbx_len*2,bby_len*2))

          # print("Z1: " + str(data_pd_['Z1'][a[slices]]) + "Z2: " + str(data_pd_['Z2'][a[slices]]) + " 阈值： " + str(abc_stack_piexls[i] ))
          
          threshold_value = 254
          extract_threshold_value = 254

          lesion_area_piexls = 1

          patch_ee = None

          ass = (y1 - bby_len)
          bss = (y2 + bby_len)
          css = (x1 - bbx_len)
          dss = (x2 + bbx_len)
          
          if (y1 - bby_len) <= 0:
            ass = 0
          if (x1 - bbx_len) <= 0:
            css = 0
          if (y2 + bby_len) >= 512:
            bss = 512
          if (x2 + bbx_len) >= 512:
            dss = 512

          patch_ee = abc_stacks[i, ass : bss, css:dss]
          patch_ee_ = abc_stacks[i, y1 : y2, x1:x2]

          # '''

          # print("修复前： *****************")

          # cv2_imshow(patch_ee[bby_len:2*bby_len, bbx_len:2*bbx_len])

          # '''
          
          # patch_ees_ = abc_stacks[i, y1 : y2, x1:x2]
          # patch_ees_ = np.dstack((patch_ee, patch_ees_, patch_ees_))

          ###################### CONSOLIDATION 阈值提取 ##########################

          # 计算灰度图像的平均值

          # ksc = cv2.cvtColor(patch_ee, cv2.COLOR_BGR2GRAY)

          # 输入图片
          # image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

          # 计算直方图
          histogram = calculate_histogram(patch_ee)

          # 对直方图频率进行排序
          sorted_pixel_values, sorted_frequencies = sort_histogram_frequencies(histogram)

          # 寻找像素值之差大于40的两个像素
          pixel_1, pixel_2 = find_pixel_differences(sorted_pixel_values)

          # 打印结果
          # print('Sorted Frequencies:', sorted_frequencies)
          # print('Pixel 1:', pixel_1)
          # print('Pixel 2:', pixel_2)

          # patch_ees_ = abc_stacks[i, y1 : y2, x1:x2]
          # patch_ees_ = np.dstack((patch_ees_, patch_ees_, patch_ees_))


          ###################### CONSOLIDATION 阈值提取 ##########################

          # binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

          threshold_value = (pixel_1 + pixel_2) // 2

          print("threshold_value： " + str(threshold_value))

          # gray = patch_ee

          # # cv2_imshow(gray)

          # # 使用平均灰度值作为阈值进行二值化
          threshold, binary = cv2.threshold(patch_ee, threshold_value, 255, cv2.THRESH_BINARY)

          for px in range(binary.shape[1]):
            for py in range(binary.shape[0]):
              # abc_stacks[i][py + (y1)][px + (x1)] = binary[py][px]
              if newimg[i][py + ass][px + css] == 1 or newimg[i][py + ass][px + css] == 3:
                k_lesion[i][py + ass][px + css] = 255
                binary[py][px] = 255
                
              elif binary[py][px] > 0:
                k_lesion[i][py + ass][px + css] = 255
                

          # cv2_imshow(binary)

          # print("binary.shape" + str(binary.shape) + " bbx_len: " + str(bbx_len) + " bby_len: " + str(bby_len))

          # cv2_imshow(k_lesion[i])

          # print("thresh.shape: " + str(thresh.shape))

          binary = binary[bby_len:2*bby_len, bbx_len:2*bbx_len]


          # print("修复前： *****************")

          # '''

          # cv2_imshow(binary)

          # '''

          # 轮廓发现
          contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


          ###################### Lung cavity 阈值提取 ##########################

          patch_ee_ = np.dstack((patch_ee_, patch_ee_, patch_ee_))

          # print("contours.shape: " + str(contours.shape))

          # 创建与原始图像大小相同的全黑图像
          mask = np.zeros_like(patch_ee_)

          # 循环所有轮廓，并绘制它们
          for sr in range(len(contours)):
              # 如果轮廓具有父轮廓，则表示它是空洞轮廓
              if hierarchy[0][sr][3] != -1:
                  # 绘制轮廓
                  cv2.drawContours(mask, contours, sr, (255, 255, 255), cv2.FILLED)

          # '''

          # cv2_imshow(mask)

          # print("修复后： *****************")

          # '''

          left_up_corner = False
          right_up_corner = False
          left_down_corner = False
          right_down_corner = False

          for px in range(mask.shape[1]):
            for py in range(mask.shape[0]):
              # abc_stacks[i][py + (y1)][px + (x1)] = binary[py][px]
              k_lungcavity[i][py + y1][px + x1] = mask[py][px][0]
        

          cross_move = 0
          up_move = True

          caluc_up = 0
          delete_detect = 0

          abc_stack = k_lesion

          # for i in range(image_count):
          #   cv2_imshow(abc_stack[i])

          x_len = bbx_len
          
      slices = slices + 1
      img_dirs = args.Cavern_Detection_Train_CT + '/' + nill_file

      # img_1024_path_folder = os.path.exists(img_dirs)
      # if not img_1024_path_folder:
      #   os.makedirs(img_dirs)

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
