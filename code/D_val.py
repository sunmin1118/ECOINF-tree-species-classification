from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
with_labels = False
title = 'result'
opacity = 0.5
imagepath = r''
savepath = r''
config_file = r''
checkpoint_file = r''
device = 'cpu'
model = init_model(config_file, checkpoint_file, device=device)
if  device == 'cpu':
    model = revert_sync_batchnorm(model)
for filename in os.listdir(imagepath):
    print("正在测试图片：", filename)
    img = os.path.join(imagepath, filename)
    result = inference_model(model, img)
    out_file = os.path.join(savepath, filename)
    show_result_pyplot(
        model,
        img,
        result,
        title=title,
        opacity=opacity,
        with_labels=with_labels,
        draw_gt=False,
        show=False if out_file is not None else True,
        out_file=out_file)
