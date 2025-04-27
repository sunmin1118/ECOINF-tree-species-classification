_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8))

load_from = 'deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth'