# DeepPulmoTB


DeepPulmoTB is a comprehensive Tuberculosis lesion tissue segmentation dataset. In DeepPulmoTB, a series of images for each patient consists of about 125 slices in the axial projection. For the consolidation and lung cavity segmentation categories, the data are sourced from ImageCLEF2022 TB Cavern detection and cavern report, totaling 759 patient CT images (approximately 95k CT slice images). For the Lung Area Mask (Mask version III) segmentation category, there are 2516 patient CT images (approximately 314k CT slice images), combining Mask versions I and II from the ImageCLEF 2020-2022 TB challenge.

Examples of DeepPulmoTB are shown in Figure 1.

![Figure35-1](https://github.com/SupCodeTech/DeepPulmoTB/assets/111235455/6dc42454-7387-4618-b4f8-ad7ae46fa682)

Figure 1: Examples of DeepPulmoTB.


# DeepPulmoTB download

DeepPulmoTB is divided into two parts, Part1 and Part2, in which part1 is a multi-category semantic segmentation task, and part2 is a lung segmentation task.

DeepPulmoTB dataset is available in the [DeepFashion2 dataset](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing). However, because our paper is still being submitted, the DeepPulmoTB dataset has not been made public and is encrypted. To access the dataset, please email gs63891@student.upm.edu.my to get the password for unzipping files. Please take a look at the Data Description below for detailed information about the dataset.

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

# Construction method of DeepPulmoTB

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
│ │ │ ├── Caverns_Detection_Train_Masks
│ │ │ │ ├── Caverns_Detection_Train_Masks1
│ │ │ │ │ ├── train_masks1
│ │ │ │ │ │ ├── TRN_000_mask.nii.gz
│ │ │ │ │ │ ├──  …
│ │ │ │ ├── Caverns_Detection_Train_Masks2
│ │ │ │ │ ├── train_masks2
│ │ │ │ │ │ ├── TRN_000_regsegm_py.nii.gz
│ │ │ │ │ │ ├──  …
│ │ │ ├── Cavern_Report_Train_CT
│ │ │ │ ├── TRN_00.nii.gz
│ │ │ │ ├──  …
│ │ │ ├── Caverns_Report_Train_Masks
│ │ │ │ ├── Caverns_Report_Train_Masks1
│ │ │ │ │ ├── TRN_00_mask.nii.gz
│ │ │ │ │ ├──  …
│ │ │ │ ├── Caverns_Report_Train_Masks2
│ │ │ │ │ ├── TRN_00_regsegm_py.nii.gz
│ │ │ │ │ ├──  …
```

In addition, we need to unzip the `nii.gz` files under the Directories `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Detection_Train_CT` and `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Report_Train_CT` with the following two commands:

```shell
python Cavern_Detection_Train_CT_Nii_to_PNG.py ${Caverns_Detection_Train_CT} ${Cavern_Detection_Train_CT_PNG}
```

`Caverns_Detection_Train_CT`: This path is the path to store the Caverns Detection Train CT file.

`Cavern_Detection_Train_CT_PNG `: This path is the path to decompress the Caverns Detection Train CT file into a PNG file.

This command can decompress all the `nii.gz` format files under the directory `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Detection_Train_CT` into `PNG` format to the directory: `DeepPulmoTB/Original_Image_Dataset/Part_1/ Cavern_Detection_Train_CT_PNG`.

```shell
python Cavern_Report_Train_CT_Nii_to_PNG.py ${Caverns_Report_Train_CT} ${Cavern_Report_Train_CT_PNG}
```
`Caverns_Report_Train_CT`: This path is the path to store the Caverns Report Train CT file.
`Cavern_Report_Train_CT_PNG`: This path is the path to decompress the Caverns Report Train CT file into a PNG file.

This command can decompress all the `nii.gz` format files under the directory `DeepPulmoTB/Original_Image_Dataset/Part_1/Cavern_Report_Train_CT` into `PNG` format to the directory: `DeepPulmoTB/Original_Image_Dataset/Part_1/ Cavern_Report_Train_CT_PNG`.

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
│   │   │   ├── Caverns_Detection_Train_Masks
│   │   │   │   ├── Caverns_Detection_Train_Masks1
│   │   │   │   │   ├── train_masks1
│   │   │   │   │   │   ├── TRN_000_mask.nii.gz
│   │   │   │   │   │   ├── …
│   │   │   │   ├── Caverns_Detection_Train_Masks2
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
│   │   │   ├── Caverns_Report_Train_Masks
│   │   │   │   ├── Caverns_Report_Train_Masks1
│   │   │   │   │   ├── TRN_00_mask.nii.gz
│   │   │   │   │   ├── …
│   │   │   │   ├── Caverns_Report_Train_Masks2
│   │   │   │   │   ├── TRN_00_regsegm_py.nii.gz
│   │   │   │   │   ├── …
```

## Construction of DeepPulmoTB training mask dataset Part 1

### Caverns Detection task data source

First, execute the following statements to complete the construction of the consolidation and lung area, as well as the lung cavity (first stage) in the Z-axis direction.

```shell
python Cavern_Detection_training_mask_dataset_Part_1.py ${Cavern_detection_train_bboxes} ${Cavern_Detection_Train_CT} ${Cavern_Detection_Train_Masks1} ${Caverns_Detection_Train_Masks2} ${Cavern_Detection_Train_CT_PNG} ${Training_Mask _Dataset}
```
`Caverns_detection_train_bboxes`: The path input is the path where the `CVS` file of cavern detection bounding boxes is located. \
`Cavern_Detection_Train_CT`: This path refers to the storage path of the original CT data file of the cavern detection task. \
`Caverns_Detection_Train_Masks1`: This path refers to the storage path of the train mask version 1 data file of the lung area of the cavern detection task. \
`Caverns_Detection_Train_Masks2`: This path refers to the storage path of the train mask version 2 data file of the lung area of the cavern detection task. \
`Cavern_Detection_Train_CT_PNG`: This path refers to the storage path of the PNG image file decompressed from the original CT data file of the cavern detection task. \
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
python Cavern_Report_training_mask_dataset_Part_1.py ${Caverns_Report_Train_Bboxes_cvs} ${Original_Image_Dataset} ${Caverns_Report_Train_Masks1} ${Caverns_Report_Train_Masks2} ${Original_Image_PNG_Dataset} ${Training_Mask_Dataset}
```

`Caverns_report_train_bboxes`: The path input is the file path of cavern report bounding boxes cvs. \
`Cavern_Report_Train_CT`: This path refers to the original CT data files of the two tasks of cavern detection and cavern report. \
`Caverns_Report_Train_Masks1`: This path refers to the train mask version 1 data file of the lung area of the cavern report task. \
`Caverns_Report_Train_Masks2`: This path refers to the train mask version 2 data file of the lung area of the cavern report task. \
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

ImageCLEF 2021 Tuberculosis - [TBT classification](https://www.aicrowd.com/challenges/imageclef-2021-tuberculosis-tbt-classification) and [CT report](https://www.aicrowd.com/challenges/imageclef-2020-tuberculosis-ct-report)

and unzip it,and place it in the following directory:

```none

│ ├── Training Dataset
│ │ ├── Part 2
│ │ │ ├── CTR_TRN_001
│ │ │ ├── ...
```

The final overall directory is as follows:

```none
├── DeepPulmoTB
│ ├── CVS
│ │ ├── cavern_report_train_bounding_boxes
│ │ ├── cavern_report_train_labels
│ │ ├── cavern_detection_train_bboxes
│ │ ├── cavern_detection_test_bounding_boxes
│ ├── Training Dataset
│ │ ├── Part 1
│ │ │ ├── TRN_001
│ │ │ ├── ...
│ │ ├── Part 2
│ │ │ ├── CTR_TRN_001
│ │ │ ├── ...
```

The code for this part will be published soon


## Contact
If you have any question, please feel free to contact me via tan.joey@student.upm.edu.my

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
