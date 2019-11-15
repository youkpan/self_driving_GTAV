from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np

def predict_init():
        # 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
    config_file = 'D:\\self-driving\\mmdetection\\configs/faster_rcnn_r50_fpn_1x.py'
    checkpoint_file = 'D:\\self-driving\\mmdetection\\checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
    #checkpoint_file = 'D:\\self-driving\\mmdetection\\checkpoints/rpn_x101_32x4d_fpn_2x_20181218-0510af40.pth'

    # 初始化模型
    model = init_detector(config_file, checkpoint_file)

    return model


def getpredict(image,model):
    
    result = inference_detector(model, image)
    #show_result(image, result, model.CLASSES,  wait_time=1)

    return result
