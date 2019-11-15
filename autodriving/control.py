from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
import autodriving.line_detect

def in_safe_zone(bboxes,labels,imgwidth,imgheight):
    safe_region_x = [0,0,0,0]
    safe_region_y = [0,0,0,0]

    safe_region_x[0] = imgwidth*210/1280
    safe_region_y[0] = imgheight

    safe_region_x[1] = imgwidth*505/1280
    safe_region_y[1] = 370*imgheight/640

    safe_region_x[2] = imgwidth*1070/1280
    safe_region_y[2] = imgheight

    safe_region_x[3] = imgwidth*783/1280
    safe_region_y[3] = 370*imgheight/640

    print("len(bboxes)",len(bboxes))
    # x = 350 y=0
    # x = 1080 y=640
    # ax+b = y
    #b = -350 a
    #1080 a  + b = 640  
    #a = 0.877
    #b = -306.8
    #line 1: y = 0.877 x - 306.8

    #x= 210 y=640
    #x=930 y = 0
    # 210 a + b = 640
    # 930 a + b = 0
    # - 720 a = 640
    #a = -0.89
    #b = 826.7
    #line left: y = -0.89 x + 826.7


    for i in range(len(bboxes)):
        if bboxes[i][-1] > 0.4:# and labels[i][0:5]!='train':
            for j in range(2):
                for k in range(2):
                    x = bboxes[i][2*j]
                    y = bboxes[i][2*j+k]
                    for s in range(4):
                        if y > safe_region_y[1] :
                            ys = -0.89 * x + 826.7
                            if y > ys:
                                 ys_right = 0.877 * x - 306.8
                                 if y > ys_right:
                                    return True


    return False

def solve_data(image,bboxes,labels,imginfo,message):
    
    try:
        speed = message['speed']
    except Exception as e:
        speed = 0

    imgwidth = imginfo['imgwidth']
    imgheight = imginfo['imgheight']
    throttle = 0.5
    breaker = 0
    if speed >10:
        throttle = 0.5- (speed-10)*0.1


    if(in_safe_zone(bboxes,labels,imgwidth,imgheight)):
        print("box in_safe_zone !!")
        throttle = message['throttle'] - 0.1
        breaker = 0.2+(speed)*0.05
    else:
        breaker = 0

    if breaker>1:
        breaker = 1
    if throttle <0:
        throttle = 0
        breaker = (speed)*0.1

    out_image,steering_prediction = line_detect.detect(image,imgwidth,imgheight)

    return bboxes,labels,throttle,breaker,steering_prediction,out_image

def process_result(img,
                result,
                class_names,
                imginfo,
                message,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    bboxes,labels,throttle,breaker,steering_prediction,out_image = solve_data(img,bboxes,labels,imginfo,message)
    if message['count'] %5 == 0:
        mmcv.imshow_det_bboxes(
            out_image,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    
    return throttle,breaker,steering_prediction

def getcontrol(image,result,model,imginfo,message):
    
    throttle = 0.5
    breaker = 0
    steering_prediction = 0

    throttle,breaker,steering_prediction =process_result(image, result, model.CLASSES,imginfo, message, wait_time=1)
 

    return throttle,breaker,steering_prediction