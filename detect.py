# Simple pickle reading program; Displays image from dataset
import pickle
import gzip
import cv2
from deepgtav.messages import frame2numpy
from mmdet.apis import init_detector, inference_detector, show_result


def crop_bottom_half(image):
    ''' Crops to bottom half of image '''
    return image[int(image.shape[0] / 2):image.shape[0]]

file = gzip.open('dataset_test.pz')
count=0

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'D:\\self-driving\\mmdetection\\configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'D:\\self-driving\\mmdetection\\checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
#checkpoint_file = 'D:\\self-driving\\mmdetection\\checkpoints/rpn_x101_32x4d_fpn_2x_20181218-0510af40.pth'
 
# 初始化模型
model = init_detector(config_file, checkpoint_file)
 
# 测试一张图片
img = 'demo\\coco_test_12510.jpg'


while True:
    try:
        data_dict = pickle.load(file) # Iterates through pickle generator
        count += 1
        # Every 10000 frames prints steering and displays frame 
        if (count%100)==0:
            for k in data_dict.keys():
                if k!='frame':
                    print(k,data_dict[k])
            print(str(data_dict['steering']) + '    On Count: ' + str(count))
            frame = data_dict['frame']
            # Show full image
            image = frame2numpy(frame, (320,160))

            result = inference_detector(model, image)
            show_result(image, result, model.CLASSES)
            '''
            cv2.imshow('img',image)
            cv2.waitKey(-1) # Must press q on keyboard to continue to next frame
            # Show cropped image
            image = crop_bottom_half(image)
            cv2.imshow('img',image)
            cv2.waitKey(-1)
            '''

    except EOFError:
        break
print(count)
