from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
import autodriving.line_detect
import cv2
from datetime import datetime as dt
import datetime
import time
import sys


def get_safe_zone_lane(imgwidth,imgheight):
    safe_region_x,safe_region_y = get_safe_zone_detect(imgwidth,imgheight)

    safe_region_x[0] = imgwidth*210/1280
    safe_region_y[0] = imgheight

    safe_region_x[1] = imgwidth*615/1280
    safe_region_y[1] = 338*imgheight/640

    safe_region_x[2] = imgwidth*665 /1280
    safe_region_y[2] = 338*imgheight/640

    safe_region_x[3] = imgwidth*1070/1280
    safe_region_y[3] = imgheight
    return safe_region_x,safe_region_y

def lane_in_safe_zone(bboxes,labels,img,imgwidth,imgheight):
    safe_region_x,safe_region_y = get_safe_zone_lane(imgwidth,imgheight)
    slope1 = (safe_region_y[1] - safe_region_y[0]) / (safe_region_x[1] - safe_region_x[0] + np.finfo(float).eps)
    bias1 = safe_region_y[0] - slope1 * safe_region_x[0]
    slope2 = (safe_region_y[3] - safe_region_y[2]) / (safe_region_x[3] - safe_region_x[2] + np.finfo(float).eps)
    bias2 = safe_region_y[2] - slope2 * safe_region_x[2]

    has_returen = 0
    count_left=0
    count_right=0
    return1 = 0
    return2 = 0
    for i in range(len(bboxes)):
        if bboxes[i][-1] > 0.4 :# and labels[i][0:5]!='train':
            for j in range(2):
                for k in range(1):
                    x = bboxes[i][2*j]
                    y = bboxes[i][2*k+1]

                    if y > safe_region_y[1] :
                        ys =  slope1 * x + bias1
                        if y > ys:
                             ys_right = slope2 * x + bias2
                             if y > ys_right:
                                if x<imgwidth/2:
                                    #print(bboxes[i],j,k)
                                    color=[0,0,255]
                                    cv2.line(img, (int(x), int(y)), (int(x+1), int(y+1)),color,5)
                                    has_returen = 1
                                    count_left +=1
                                    return1 =(1,x,y)
                                else:
                                    #print(bboxes[i],j,k)
                                    has_returen = 2
                                    color=[255,0,255]
                                    cv2.line(img, (int(x), int(y)), (int(x+1), int(y+1)),color,5)
                                    count_right +=1
                                    return2 = (2,x,y)

    if count_right > count_left + 5 :
        return return2

    if has_returen ==1:
        return return1
    if has_returen ==2:
        return return2

    return 0,0,0

def draw_safe_zone_line(img,imgwidth,imgheight,safe_region_x,safe_region_y,color=[0, 0, 200],thickness=2):
 
    for i in range(3):
        cv2.line(img, (int(safe_region_x[i]), int(safe_region_y[i])), (int(safe_region_x[i+1]), int(safe_region_y[i+1])), color, thickness)

    return  img

def get_safe_zone_detect(imgwidth,imgheight):
    safe_region_x = [0,0,0,0]
    safe_region_y = [0,0,0,0]

    safe_region_x[0] = imgwidth*210/1280
    safe_region_y[0] = imgheight

    safe_region_x[1] = imgwidth*615/1280
    safe_region_y[1] = 370*imgheight/640

    safe_region_x[2] = imgwidth*665/1280
    safe_region_y[2] = 370*imgheight/640

    safe_region_x[3] = imgwidth*1070/1280
    safe_region_y[3] = imgheight

    return safe_region_x,safe_region_y

def in_safe_zone(bboxes,labels,imgwidth,imgheight,check_lane=False):

    safe_region_x,safe_region_y = get_safe_zone_detect(imgwidth,imgheight)
    slope1 = (safe_region_y[1] - safe_region_y[0]) / (safe_region_x[1] - safe_region_x[0] + np.finfo(float).eps)
    bias1 = safe_region_y[0] - slope1 * safe_region_x[0]
    slope2 = (safe_region_y[3] - safe_region_y[2]) / (safe_region_x[3] - safe_region_x[2] + np.finfo(float).eps)
    bias2 = safe_region_y[2] - slope2 * safe_region_x[2]


    for i in range(len(bboxes)):
        if bboxes[i][-1] > 0.4 or check_lane:# and labels[i][0:5]!='train':
            for j in range(2):
                for k in range(2):
                    x = bboxes[i][2*j]
                    y = bboxes[i][2*k+1]

                    if y > safe_region_y[1] :
                        #ys = -0.89 * x + 826.7
                        ys = slope1 * x + bias1
                        if y > ys:
                             #ys_right = 0.877 * x - 306.8
                             ys_right = slope2 * x + bias2
                             if y > ys_right:
                                if x<imgwidth/2:
                                    print(bboxes[i],j,k)
                                    return 1,x,y
                                else:
                                    print(bboxes[i],j,k)
                                    return 2,x,y


    return 0,0,0

steering = 0
last_contol =0
control_timer = 0
wait_timer1 = 0
wait_timer2 = 0
throttle =0.5
last_contol = 0
lane_in_safe_zone_i_1 = 0
lane_in_safe_zone_i_2 = 0

def solve_data(image,bboxes,labels,imginfo,message):
    global steering,control_timer,wait_timer1,wait_timer2,throttle,last_contol
    global lane_in_safe_zone_i_1,lane_in_safe_zone_i_2

    try:
        speed = message['speed']
    except Exception as e:
        speed = 0

    control_param = get_control_param()

    imgwidth = imginfo['imgwidth']
    imgheight = imginfo['imgheight']
    breaker = 0
    #steering = message['steering']
    if speed >10:
        throttle = 0.5- (speed-10)*0.1

    if len(bboxes)>0:
        in_safe_zone_i,x,y = in_safe_zone(bboxes,labels,imgwidth,imgheight)
        if(in_safe_zone_i >0):
            print("box in_safe_zone !!")
            throttle = message['throttle'] - control_param['throttle_in_safe_zone_fix']
            breaker = control_param['breaker_in_safe_zone']+(speed)*control_param['breaker_in_safe_zone_speed']
        else:
            breaker = 0


    '''
    out_image,lane_lines = autodriving.line_detect.detect(image,imgwidth,imgheight)
    coords = []
    for lane in lane_lines:
        coord = lane.get_coords_line()
        #lane.draw(out_image, color=[0, 255, 0], thickness=5)
        for co in coord:
            if co[1]>40 and len(coord)>imgheight/15:
                #print(co)
 
                coords.append(co)

    

    lane_in_safe_zone_i,x,y = lane_in_safe_zone(coords,labels,out_image,imgwidth,imgheight)
    if lane_in_safe_zone_i == 1:
        lane_in_safe_zone_i_1 +=1
    if lane_in_safe_zone_i == 2:
        lane_in_safe_zone_i_2 +=1


    if (wait_timer1==0 or wait_timer1<time.time()) and (  lane_in_safe_zone_i_1 >lane_in_safe_zone_i_2):
        if steering <0:
            steering = 0
        wait_timer1 = 0
        lane_in_safe_zone_i_1 = 0
        lane_in_safe_zone_i_2 = 0

        if last_contol !=1:
            steering = control_param['steering_in_safe_zone_acc_base0']
        ' ''
        steering += control_param['steering_in_safe_zone_acc']+speed*control_param['steering_in_safe_zone_acc_speed']
        if steering>control_param['steering_in_safe_zone_acc_limit1']:
            steering += control_param['steering_in_safe_zone_acc_1']
        if steering>control_param['steering_in_safe_zone_acc_limit2']:
            steering = control_param['steering_in_safe_zone_acc_limit2']
        ' ''

        if steering <control_param['steering_in_safe_zone_acc_base']:
            steering = control_param['steering_in_safe_zone_acc_base']

        print("lane ! zone:",lane_in_safe_zone_i,"steering",steering,x,y)
        throttle = throttle - control_param['steering_in_safe_zone_acc_throttle']
        last_contol = 1
        control_timer = time.time() + control_param['steering_in_safe_zone_acc_timer']
        wait_timer1 = time.time() + control_param['steering_in_safe_zone_acc_wait_timer']
        wait_timer2 = wait_timer1
        cv2.line(out_image, (320,320), (350, 225),[0,255,0],2)

    elif (wait_timer2==0 or wait_timer2<time.time()) and (   lane_in_safe_zone_i_1 <lane_in_safe_zone_i_2): 
        if steering >0:
            steering= 0
        wait_timer2 = 0
        lane_in_safe_zone_i_1 = 0
        lane_in_safe_zone_i_2 = 0

        if last_contol !=2:
            steering = control_param['steering_in_safe_zone_dsc_base0']
        ' ''
        steering -= (control_param['steering_in_safe_zone_dsc']+speed*control_param['steering_in_safe_zone_dsc_speed'])
        if steering<control_param['steering_in_safe_zone_dsc_limit1']:
            steering -= control_param['steering_in_safe_zone_dsc_1']
        if steering<control_param['steering_in_safe_zone_dsc_limit2']:
            steering = control_param['steering_in_safe_zone_dsc_limit2']
        ' ''

        if steering >  control_param['steering_in_safe_zone_dsc_base']:
            steering = control_param['steering_in_safe_zone_dsc_base']

        print("lane ! zone:",lane_in_safe_zone_i,"steering",steering,x,y)
        throttle = throttle - control_param['steering_in_safe_zone_dsc_throttle']
        last_contol = 2
        control_timer = time.time() + control_param['steering_in_safe_zone_dsc_timer']
        wait_timer2 = time.time() + control_param['steering_in_safe_zone_dsc_wait_timer']
        wait_timer1=wait_timer2
        cv2.line(out_image, (320,320), (290, 225),[0,255,0],2)
    else:
        #if last_contol == 1:
        if control_timer>0 and control_timer>time.time():
            steering = 0
            control_timer = 0

    

    safe_region_x,safe_region_y = get_safe_zone_lane(imgwidth,imgheight)
    thickness = 2
    if lane_in_safe_zone_i==1:
        thickness = 4
        color=[200,0,0]
    if lane_in_safe_zone_i==2:
        thickness = 4
        color=[0,0,200]
    else:
        color=[0,200,0]
    out_image=draw_safe_zone_line(out_image,imgwidth,imgheight,safe_region_x,safe_region_y,color,thickness=thickness)
    '''
    slope1 = (256 - message['lanet_center_y']) / (256 - message['lanet_center_x'] + np.finfo(float).eps)
    
    if slope1>100:
        slope1 = 100
    if slope1<1 and slope1>0:
        slope1 = 1
    if slope1>-1 and slope1<0:
        slope1 = -1
    if slope1 == 0:
        steering = 0
    else:
        steering = -1/slope1

    if len(bboxes)>0:
        thickness = 2
        if in_safe_zone_i==1:
            thickness = 4
            color=[200,0,0]
        if in_safe_zone_i==2:
            thickness = 4
            color=[0,0,200]
        else:
            color=[0,200,0]
    else:
        color=[0,200,0]

    thickness = 2
    out_image = image
    safe_region_x,safe_region_y = get_safe_zone_detect(imgwidth,imgheight)
    out_image=draw_safe_zone_line(out_image,imgwidth,imgheight,safe_region_x,safe_region_y,color,thickness=thickness)


    if throttle <0:
        throttle = 0
        breaker = (speed)*0.1

    if breaker>1:
        breaker = 1

    throttle = 0.4
    return bboxes,labels,throttle,breaker,steering,out_image

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
    bboxes = []
    labels = []
    if result!=None:
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

    bboxes,labels,throttle,breaker,steering,out_image = solve_data(img,bboxes,labels,imginfo,message)
    if result!=None:
        mmcv.imshow_det_bboxes(
            out_image,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    
    return throttle,breaker,steering

def getcontrol(image,result,model,imginfo,message):
    
    throttle = 0.5
    breaker = 0
    steering_prediction = 0

    throttle,breaker,steering =process_result(image, result, model.CLASSES,imginfo, message, wait_time=1)
 

    return throttle,breaker,steering


def get_control_param():
  params = {}
  try:
    with open("./autodriving/control_param.txt",'r', encoding='UTF-8') as f:
      for line in f:
        lined = line.split("=")
        if len(lined)>0:
          params[lined[0]] = lined[1]
          try:
            params[lined[0]] = float(lined[1])
          except Exception as e:
            pass

  except Exception as e:
    print("error get_control_param")
    pass

  return params