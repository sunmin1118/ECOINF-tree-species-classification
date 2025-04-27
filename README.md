# ECOINF-D-25-00627
Datasets and code for ECOINF-D-25-00627<br>
The full set of supporting files for UNet 3+ can be downloaded from the GitHub repository: https://github.com/bubbliiiing/unet-pytorch/tree/main.
The DeepLabv3+ code were downloaded from the open-source semantic segmentation toolbox: https://github.com/open-mmlab/mmsegmentation.
The Mask R-CNN code were downloaded from the open-source object detection toolbox: https://github.com/open-mmlab/mmdetection.
The SNIC code were downloaded from the GitHub repository: https://github.com/ertugrulqayibov/GEE-SNIC-and-Canny.

1. UNet 3+ Model
U_train.py: the training script for the UNet 3+ model;<br>
U_predict.py: the prediction script for the UNet 3+ model;<br>
U_val.py: the validation script for the UNet 3+ model;<br>
model_def.py: the model definition file for UNet 3+;<br>
UnetPlus3.py: the PyTorch implementation of the UNet 3+ model architecture.<br>
3. DeepLabv3+ Model<br>
D_train.py: the training script for the DeepLabv3+ model;<br>
D_val.py: the validation script for the DeepLabv3+ model;<br>
deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py: the implementation file of the DeepLabv3+ model architecture;<br>
deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth: the pretrained weights for the DeepLabv3+ model;<br>
schedule_20k.py: the training schedule configuration file for DeepLabv3+.<br>
3. Mask R-CNN Model<br>
M_train.py: the training script for the Mask R-CNN model;<br>
M_test_P_R_F1.py: the testing script for the Mask R-CNN model;<br>
mask-rcnn_r50_fpn_1x_coco.py: the configuration file for the Mask R-CNN model;<br>
mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth: the pretrained weights for the Mask R-CNN model;<br>
schedule_1x.py: the training schedule configuration file for Mask R-CNN.<br>
4. Ensemble Model<br>
S_base_train.py: the training script for the four base models;<br>
S_feature_prepare.py: the script for weighted feature preparation;<br>
S_gbdt_train.py: the training script for the ensemble model;<br>
significance test.py: the script for McNemar test.<br>

For access to the confidential data, please contact us with a reasonable statement for your request. We are willing to assist you in submitting an application for data access approval.
