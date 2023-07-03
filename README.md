# DeepPulmoTB


DeepPulmoTB is a comprehensive Tuberculosis lesion tissue segmentation dataset. In DeepPulmoTB, a series of images for each patient consists of about 125 slices in the axial projection. For the consolidation and lung cavity segmentation categories, the data are sourced from ImageCLEF2022 TB Cavern detection and cavern report, totaling 618 patient CT images (approximately 77,250 CT slice images). For the lung area mask (lung mask version III) segmentation category, there are 2498 patient CT images (approximately 312,250 CT slice images), combining Mask versions I and II from the ImageCLEF 2020-2022 TB challenge.

Examples of DeepPulmoTB are shown in Figure 1.

![Figure35-1](https://github.com/SupCodeTech/DeepPulmoTB/assets/111235455/fa734bff-7299-4d2f-8416-58a9cab589c5)


Figure 1: Examples of DeepPulmoTB.


# DeepPulmoTB Training Mask Download

DeepPulmoTB is divided into two parts, Part1 and Part2, in which part1 is a multi-category semantic segmentation task, and part2 is a lung segmentation task. Note: Part 2 only contains the lung region segmentation recognition mask, which does not mean that there is no TB lesion tissue.

DeepPulmoTB dataset is available in the [DeepPulmoTB dataset](https://drive.google.com/drive/folders/1QPinffQ59BufdGapQyLykfaqOiDhtrnX?usp=sharing). As the paper is still under review, the dataset has been encrypted. The decryption password is TMI Manuscript ID.

Please take a look at the Data Description below for detailed information about the dataset.

After decompressing DeepPulmoTB, you can get the following directory:

```none
├── DeepPulmoTB
│ ├── Training_Mask_Dataset
│ │ ├── Part 1
│ │ │ ├── TRN_00.nii.gz
│ │ │ ├── TRN_000.nii.gz
│ │ │ ├──  …
│ │ ├── Part 2
│ │ │ ├── CTR_TRN_001
│ │ │ ├── ...
```

In Part 1, pixel values 1 to 3 represent consolidation, Lung Mask version III, and Lung cavity.
In Part 2, pixel value 1 represents Lung Mask version III.

# DeepPulmoTB Training Data (ImageCLEF training data) Preparation

For part1, we need to download the following datasets:

ImageCLEF 2022 Tuberculosis - [Caverns Report](https://www.aicrowd.com/challenges/imageclef-2022-tuberculosis-caverns-report) and [Caverns Detection](https://www.aicrowd.com/challenges/imageclef-2022-tuberculosis-caverns-detection)

and unzip the Cavern Detection Train CT files (1 to 7) and the Cavern Report Train CT files (1 to 2):

```none
-------------------------------------------------------------------------------------
|                     Zip files                                    Unzip files      |
-------------------------------------------------------------------------------------
| de0e8772-594d-41ce-9e85-578c4b59e9f3_detection_train_CT_1    (TRN_000 - TRN_099)  |
| 29cc320d-1c9d-4c7c-8d12-a006499c2f2f_detection_train_CT_2    (TRN_100 - TRN_179)  |
| 3b9da027-6015-4806-bbe9-04efb760ee53_detection_train_CT_3    (TRN_180 - TRN_269)  |
| c45b6a20-e3dc-49b5-bd93-1a86dc10721c_detection_train_CT_4    (TRN_270 - TRN_339)  |
| 447557ed-45dd-4468-8381-dbf0642b4312_detection_train_CT_5    (TRN_340 - TRN_419)  |
| f9da57c9-dbb5-4a3d-bebb-0618a8aef99f_detection_train_CT_6    (TRN_420 - TRN_499)  |
| eac9b86a-7673-4b24-924a-29529de6f130_detection_train_CT_7    (TRN_500 - TRN_558)  |
------------------------------------------------------------------------------------
| 45037ba5-e1e7-4011-98c6-35ed190e204a_cavern_report_train_CT_1   (TRN_00 - TRN_29) |
| e222bfea-28db-4d3a-b7cf-b68a7e18992b_cavern_report_train_CT_2   (TRN_30 - TRN_59) |
-------------------------------------------------------------------------------------
```
they will be placed in the following directory:

```none
├── DeepPulmoTB
│ ├── Original_Image_Dataset
│ │ ├── Part 1
│ │ │ ├── TRN_00.nii.gz
│ │ │ ├── TRN_000.nii.gz
│ │ │ ├──  …
```

For part2, we need to download the following datasets:

ImageCLEF 2021 Tuberculosis - [TBT classification](https://www.aicrowd.com/challenges/imageclef-2021-tuberculosis-tbt-classification) \
ImageCLEF 2020 Tuberculosis - [CT report](https://www.aicrowd.com/challenges/imageclef-2020-tuberculosis-ct-report)

and unzip CTR_TST_masks (1 to 2), CTR_TRN_masks (1 to 2), test-masks (1 to 2), train-masks (1 to 2)

```none
--------------------------------------------------------------------------------------
|                     Zip files                                    Unzip files       |
--------------------------------------------------------------------------------------
| 4e36978f-e771-4974-a0fb-6ae8a08ba3b2_CTR_TST_masks1    (CTR_TST_001 - CTR_TST_120) |
| 4f37522e-e5f0-4811-97e5-a5a9c6edc699_CTR_TRN_masks2    (CTR_TRN_001 - CTR_TRN_120) |
| 60ea64a3-11ff-4be9-9d81-f138dc6f3812_CTR_TRN_masks1    (CTR_TRN_001 - CTR_TRN_120) |
| cdbd8feb-ca25-4eb4-ac7d-8fbebb3e23f7_CTR_TST_masks2    (CTR_TST_001 - CTR_TST_120) |
--------------------------------------------------------------------------------------
| 9cf78de9-d287-4d15-a8a7-fed01da0bf11_test-masks1       (TST_0001 - TST_0421)       |
| 266ffd05-d03d-47e7-9ca9-d5130db1f7ac_train-masks1      (TRN_0001 - TRN_0917)       |
| ccafd927-61c8-4554-b340-fb747c1310ea_train-masks2      (TRN_0001 - TRN_0917)       |
| fe8530a3-94dd-41d0-8ec2-c5edfed59cbb_test-masks2       (TST_0001 - TST_0421)       |
--------------------------------------------------------------------------------------
```

, and place them in the following directory:

```none
├── DeepPulmoTB
│ ├── Original_Image_Dataset
│ │ ├── Part 2
│ │ │ ├── TST_0001.nii.gz
│ │ │ ├──  …
```


# Construction method of DeepPulmoTB

If you want to build DeepPulmoTB yourself, you can follow the instruction below:

## Environment configuration

```shell
pip install SimpleITK
pip install Pillow
pip install opencv-python
pip install gdown
pip install matplotlib
pip install tensorflow
```

In this part, we introduce the construction method of DeepPulmoTB.

## Original Data Preparation

For part1, we need to download the following datasets:

ImageCLEF 2022 Tuberculosis - [Caverns Report](https://www.aicrowd.com/challenges/imageclef-2022-tuberculosis-caverns-report) and [Caverns Detection](https://www.aicrowd.com/challenges/imageclef-2022-tuberculosis-caverns-detection)

and unzip it, it will be placed in the following directory:

```none
├── DeepPulmoTB
│ ├── CVS
│ │ ├── cavern_report_train_bounding_boxes
│ │ ├── cavern_report_train_labels
│ │ ├── cavern_detection_train_bboxes
│ │ ├── cavern_test_bounding_boxes
│ ├── Original_Image_Dataset
│ │ ├── Part_1
│ │ │ ├── Cavern_Detection_Train_CT
│ │ │ │ ├── TRN_000.nii.gz
│ │ │ │ ├──  …
│ │ │ ├── Cavern_Detection_Train_Masks
│ │ │ │ ├── Cavern_Detection_Train_Masks1
│ │ │ │ │ ├── train_masks1
│ │ │ │ │ │ ├── TRN_000_mask.nii.gz
│ │ │ │ │ │ ├──  …
│ │ │ │ ├── Cavern_Detection_Train_Masks2
│ │ │ │ │ ├── train_masks2
│ │ │ │ │ │ ├── TRN_000_regsegm_py.nii.gz
│ │ │ │ │ │ ├──  …
│ │ │ ├── Cavern_Report_Train_CT
│ │ │ │ ├── TRN_00.nii.gz
│ │ │ │ ├──  …
│ │ │ ├── Cavern_Report_Train_Masks
│ │ │ │ ├── Cavern_Report_Train_Masks1
│ │ │ │ │ ├── TRN_00_mask.nii.gz
│ │ │ │ │ ├──  …
│ │ │ │ ├── Cavern_Report_Train_Masks2
│ │ │ │ │ ├── TRN_00_regsegm_py.nii.gz
│ │ │ │ │ ├──  …
```

In addition, we need to unzip the `nii.gz` files under the Directories `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Detection_Train_CT` and `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Report_Train_CT` with the following two commands:

```shell
python Cavern_Detection_Train_CT_Nii_to_PNG.py ${Cavern_Detection_Train_CT} ${Cavern_Detection_Train_CT_PNG}
```

`Cavern_Detection_Train_CT`: This path is the path to store the Caverns Detection Train CT files.

`Cavern_Detection_Train_CT_PNG `: This path is the path to decompress the Caverns Detection Train CT files into PNG files.

This command can decompress all the `nii.gz` format files under the directory `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Detection_Train_CT` into `PNG` format to the directory: `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Detection_Train_CT_PNG`.

```shell
python Cavern_Report_Train_CT_Nii_to_PNG.py ${Cavern_Report_Train_CT} ${Cavern_Report_Train_CT_PNG}
```
`Cavern_Report_Train_CT`: This path is the path to store the Caverns Report Train CT files.
`Cavern_Report_Train_CT_PNG`: This path is the path to decompress the Caverns Report Train CT files into PNG files.

This command can decompress all the `nii.gz` format files under the directory `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Report_Train_CT` into `PNG` format to the directory: `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Report_Train_CT_PNG`.

The final directory is as follows:

```shell
├── DeepPulmoTB
│   ├── CVS
│   │   ├── cavern_report_train_bboxes
│   │   ├── cavern_report_train_labels
│   │   ├── cavern_detection_train_bboxes
│   │   ├── cavern_test_bounding_boxes
│   ├── Original_Image_Dataset
│   │   ├── Part_1
│   │   │   ├── Cavern_Detection_Train_CT
│   │   │   │   ├── TRN_000.nii.gz
│   │   │   │   ├── …
│   │   │   ├── Cavern_Detection_Train_CT_PNG
│   │   │   │   ├── TRN_000
│   │   │   │   │   ├── 0.png
│   │   │   │   │   ├── …
│   │   │   │   ├── …
│   │   │   ├── Cavern_Detection_Train_Masks
│   │   │   │   ├── Cavern_Detection_Train_Masks1
│   │   │   │   │   ├── train_masks1
│   │   │   │   │   │   ├── TRN_000_mask.nii.gz
│   │   │   │   │   │   ├── …
│   │   │   │   ├── Cavern_Detection_Train_Masks2
│   │   │   │   │   ├── train_masks2
│   │   │   │   │   │   ├── TRN_000_regsegm_py.nii.gz
│   │   │   │   │   │   ├── …
│   │   │   ├── Cavern_Report_Train_CT
│   │   │   │   ├── TRN_00.nii.gz
│   │   │   │   ├── …
│   │   │   ├── Cavern_Detection_Train_CT_PNG
│   │   │   │   ├── TRN_00
│   │   │   │   │   ├── 0.png
│   │   │   │   │   ├── …
│   │   │   │   ├── …
│   │   │   ├── Cavern_Report_Train_Masks
│   │   │   │   ├── Cavern_Report_Train_Masks1
│   │   │   │   │   ├── TRN_00_mask.nii.gz
│   │   │   │   │   ├── …
│   │   │   │   ├── Cavern_Report_Train_Masks2
│   │   │   │   │   ├── TRN_00_regsegm_py.nii.gz
│   │   │   │   │   ├── …
```

## Construction of DeepPulmoTB training mask dataset Part 1

### Caverns Detection task data source

First, execute the following statements to complete the construction of the consolidation and lung area, as well as the lung cavity (first stage) in the Z-axis direction.

```shell
python Cavern_Detection_training_mask_dataset_Part_1.py ${Cavern_detection_train_bboxes} ${Cavern_Detection_Train_CT} ${Cavern_Detection_Train_Masks1} ${Cavern_Detection_Train_Masks2} ${Cavern_Detection_Train_CT_PNG} ${Training_Mask_Dataset}
```
`Caverns_detection_train_bboxes`: The path input is the path where the `CVS` file of cavern detection bounding boxes is located. \
`Cavern_Detection_Train_CT`: This path refers to the storage path of the original CT data files of the cavern detection task. \
`Cavern_Detection_Train_Masks1`: This path refers to the storage path of the train mask version 1 data files of the lung area of the cavern detection task. \
`Cavern_Detection_Train_Masks2`: This path refers to the storage path of the train mask version 2 data files of the lung area of the cavern detection task. \
`Cavern_Detection_Train_CT_PNG`: This path refers to the storage path of the PNG image file decompressed from the original CT data files of the cavern detection task. \
`Training_Mask_Dataset`: This path refers to the storage path of the generated DeepPulmoTB data.

Secondly, execute the following statement repeated three times to complete the three-stage construction of the lung cavity:

Recognition of Lung cavity in the Z-axis direction:

```shell
python Cavern_Detection_Lung_Cavity_Z_Aixs.py ${Cavern_detection_train_bboxes} ${Cavern_Detection_Train_CT} ${Cavern_Detection_Train_CT_PNG}
```

Identification of Lung cavity in the Y-axis direction:

```shell
python Cavern_Detection_Lung_Cavity_Y_Aixs.py ${Cavern_detection_train_bboxes} ${Cavern_Detection_Train_CT} ${Cavern_Detection_Train_CT_PNG}
```
Recognition of Lung cavity in the X-axis direction:

```shell
python Cavern_Detection_Lung_Cavity_X_Aixs.py ${Cavern_detection_train_bboxes} ${Cavern_Detection_Train_CT} ${Cavern_Detection_Train_CT_PNG}
```

### Caverns_Report task data source

First, execute the following statements to complete the construction of the consolidation and lung area, as well as the lung cavity (first stage) in the Z-axis direction.

```shell
python Cavern_Report_training_mask_dataset_Part_1.py ${Cavern_Report_Train_Bboxes_cvs} ${Cavern_Report_Train_CT} ${Cavern_Report_Train_Masks1} ${Cavern_Report_Train_Masks2} ${Cavern_Report_Train_CT_PNG} ${Training_Mask_Dataset}
```

`Cavern_report_train_bboxes`: The path input is the file path of cavern report bounding boxes cvs. \
`Cavern_Report_Train_CT`: This path refers to the original CT data files of the two tasks of cavern detection and cavern report. \
`Cavern_Report_Train_Masks1`: This path refers to the train mask version 1 data files of the lung area of the cavern report task. \
`Cavern_Report_Train_Masks2`: This path refers to the train mask version 2 data files of the lung area of the cavern report task. \
`Cavern_Report_Train_CT_PNG`: This path refers to the decompressed images of the original CT data files of the two tasks of cavern detection and cavern report. \
`Training_Mask_Dataset`: This path refers to the storage path of the generated DeepPulmoTB data.

Secondly, execute the following statement repeated three times to complete the three-stage construction of the lung cavity

Recognition of Lung cavity in the Z-axis direction

```shell
python Cavern_Report_Lung_Cavity_Z_Aixs.py ${Cavern_Report_train_bboxes} ${Cavern_Report_Train_CT} ${Cavern_Report_Train_CT_PNG}
```

Identification of Lung cavity in the Y-axis direction

```shell
python Cavern_Report_Lung_Cavity_Y_Aixs.py ${Cavern_Report_train_bboxes} ${Cavern_Report_Train_CT} ${Cavern_Report_Train_CT_PNG}
```
Recognition of Lung cavity in the X-axis direction

```shell
python Cavern_Report_Lung_Cavity_X_Aixs.py ${Cavern_Report_train_bboxes} ${Cavern_Report_Train_CT} ${Cavern_Report_Train_CT_PNG}
```

## Construction of DeepPulmoTB training mask dataset Part 2

For part2, we need to download the following datasets:

ImageCLEF 2021 Tuberculosis - [TBT classification](https://www.aicrowd.com/challenges/imageclef-2021-tuberculosis-tbt-classification) \
ImageCLEF 2020 Tuberculosis - [CT report](https://www.aicrowd.com/challenges/imageclef-2020-tuberculosis-ct-report)

and unzip it,and place it in the following directory:

```none

│ ├── Training Dataset
│ │ ├── Part 2
│ │ │ ├── 2021_ImageCLEF
│ │ │ │ ├── test_mask1
│ │ │ │ │ ├── TST_0001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ │ ├── test_mask2
│ │ │ │ │ ├── TST_0001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ │ ├── train_mask1
│ │ │ │ │ ├── TRN_0001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ │ ├── train_mask2
│ │ │ │ │ ├── TRN_0001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ ├── 2020_ImageCLEF
│ │ │ │ ├── CTR_TST_masks1
│ │ │ │ │ ├── CTR_TST_001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ │ ├── CTR_TST_masks2
│ │ │ │ │ ├── CTR_TST_001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ │ ├── CTR_TRN_masks1
│ │ │ │ │ ├── CTR_TRN_001.nii.gz
│ │ │ │ │ ├── ...
│ │ │ │ ├── CTR_TRN_masks2
│ │ │ │ │ ├── CTR_TRN_001.nii.gz
│ │ │ │ │ ├── ...
```

For the 2020 ImageCLEF test masks, we use the following command:

```shell
python 2020_ImageCLEF_CTR_TST_mask_dataset_Part_2.py ${CTR_TST_masks1} ${CTR_TST_masks2} ${Ouput_Dirs}
```

For the 2020 ImageCLEF train masks, we use the following command:

```shell
python 2020_ImageCLEF_CTR_TRN_mask_dataset_Part_2.py ${CTR_TRN_masks1} ${CTR_TRN_masks2} ${Ouput_Dirs}
```
For the 2021 ImageCLEF test masks, we use the following command:

```shell
python 2021_ImageCLEF_TST_mask_dataset_Part_2.py ${test_mask1} ${test_mask2} ${Ouput_Dirs}
```

For the 2021 ImageCLEF train masks, we use the following command:

```shell
python 2021_ImageCLEF_TRN_mask_dataset_Part_2.py ${train_mask1} ${train_mask2} ${Ouput_Dirs}
```


## Contact
If you have any questions, please feel free to contact me via tan.joey@student.upm.edu.my

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
