_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
load_from = 'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'