from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
import autodriving.line_detect
import cv2
from datetime import datetime as dt
import datetime
import time
import sys

def imgPutText(img,msg,bottomLeftCornerOfText,fontScale=1,fontColor=(255,255,255),font= cv2.FONT_HERSHEY_SIMPLEX):
    #bottomLeftCornerOfText = (10,500)
    #fontScale              = 1
    #fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img,msg, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

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

    safe_region_x[0] = imgwidth*420/1280
    safe_region_y[0] = imgheight

    safe_region_x[1] = imgwidth*615/1280
    safe_region_y[1] = 450*imgheight/720

    safe_region_x[2] = imgwidth*665/1280
    safe_region_y[2] = 450*imgheight/720

    safe_region_x[3] = imgwidth*860/1280
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

                        if y > ys :
                            ys_right = slope2 * x + bias2
                            if y > ys_right:
                                if x<imgwidth/2:
                                    print("in_safe_zone 1",bboxes[i],j,k)
                                    return 1,x,y
                                else:
                                    print("in_safe_zone 2",bboxes[i],j,k)
                                    return 2,x,y
                        elif j ==0  and k ==1 and y <=ys and x<imgwidth/2: #左下点，在安全区外
                            #框框包括安全区
                            #右下也在安全区外
                            x1 =bboxes[i][2]
                            y1 =bboxes[i][3]

                            if y1 > safe_region_y[1] and x1>imgwidth/2:
                                ys_right1 = slope2 * x1 + bias2
                                if  y1 < ys_right1:
                                    print("in_safe_zone full",bboxes[i],j,k)
                                    return 3,x,y




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
breaker =0
decress_speed_timer = 0
road_end_x_last = 256

def solve_data(image,bboxes,labels,imginfo,message):
    global steering,control_timer,wait_timer1,wait_timer2,throttle,last_contol,breaker,decress_speed_timer,road_end_x_last
    global lane_in_safe_zone_i_1,lane_in_safe_zone_i_2

    try:
        speed = message['speed']
    except Exception as e:
        speed = 0

    control_param = get_control_param()

    imgwidth = imginfo['imgwidth']
    imgheight = imginfo['imgheight']
    #breaker = 0
    #steering = message['steering']

    if speed > 1.2*control_param['target_speed']:
        throttle = control_param['base_throttle'] - (speed-10)*0.01

    if speed <0.8*control_param['target_speed'] and breaker==0 :
        throttle = throttle - (speed-control_param['target_speed'])*0.01

    if len(bboxes)>0:
        in_safe_zone_i,x,y = in_safe_zone(bboxes,labels,imgwidth,imgheight)
        if(in_safe_zone_i >0):
            print("box in_safe_zone !!")
            throttle = throttle - control_param['throttle_in_safe_zone_fix']
            breaker = control_param['breaker_in_safe_zone']+(speed)*control_param['breaker_in_safe_zone_speed']
        else:
            breaker = 0
            throttle = 0.4

    if speed<=0 and breaker>0:
        breaker = 0
 

    try:
        road_end_x = message['lane_data']["road_end"][0]

        road_end_y = message['lane_data']["road_end"][1]
        slope_road_end = (256 - road_end_y) / (256 - road_end_x + np.finfo(float).eps)
        if road_end_y > 410/720*256 and road_end_y<460/720*256 :
            road_end_x_last = road_end_x*control_param['road_end_x_last_smooth'] + road_end_x_last*(1-control_param['road_end_x_last_smooth'])
            if abs(slope_road_end)<control_param['slope_road_end_limit']:
                breaker = control_param['breaker_road_end_limit']
                throttle = throttle * control_param['throttle_road_end_limit']

                try:
                    cv2.line(image, (320,359), (int(road_end_x)  ,int(road_end_y)  ),[0,0,255],2)
                except Exception as e:
                    pass
            imgPutText(image,"[End limit]",(0,250),fontColor=(0,0,255),fontScale=0.5)
            try:
                cv2.circle(image,  (int(road_end_x)  ,int(road_end_y)  ), 5, [190,20,20], -1)
            except Exception as e:
                pass
    except Exception as e:
        pass

    slope1 = (256 - message['lanet_center_y']) / (256 - message['lanet_center_x'] + np.finfo(float).eps)
    steering1=0
    update_steering = 0
    if slope1 != 0 and abs(slope1)>0.1 and message['lanet_center_y'] < 540/720*256 and message['lanet_center_y']>510/720*256:

        steering1 = -1/slope1

        steering1 = steering1*control_param['steering_lanet_mut']

        if steering1 < control_param['steering_lanet_range'] and steering1> -control_param['steering_lanet_range']:
            if steering1 > 0.9:
                steering1 =0.9
            if steering1 <- 0.9:
                steering1 =-0.9

            steering1 =  (1+speed * control_param['steering_acc_speed'])*steering1

            if  abs(steering1-steering)< control_param['steering_lanet_diff_range']:
                steering = steering *(1-control_param['steering_lanet_smooth']) + steering1*control_param['steering_lanet_smooth']
                update_steering =1
            else:
                imgPutText(image,"[Turn limit]",(0,290),fontColor=(0,0,255),fontScale=0.5)


    if update_steering == 0:
        if message['lanet_center_y']>510/720*256:
            imgPutText(image,"[Unknow Road]",(0,280),fontColor=(0,0,255),fontScale=0.5)

        steering = control_param['steering_trun_back_rate']* steering
        throttle = throttle * control_param['uncerten_throttle_desc']
        breaker = control_param['uncerten_break']
    else:
        imgPutText(image,"[Road update]",(0,280),fontColor=(200,100,100),fontScale=0.5)

    yy = 160
    if steering!=0:
        #-1/steering * (320) + b = 359
        xx = -(yy- (359+1/steering * 320))*steering
    else:
        xx = 320

    if abs(steering) > 0.5:
        throttle = throttle * control_param['steering_overlimit_throttle_dsc']
        breaker = control_param['steering_overlimit_break']
        decress_speed_timer = time.time()+2

    elif decress_speed_timer>0 and time.time() >decress_speed_timer:
         decress_speed_timer = 0 
         breaker = 0

    try:
        cv2.line(image, (320,359), (int(xx)  ,int(yy)  ),[255,0,0],2)
    except Exception as e:
        pass
    
    print("slope1",slope1,"steering",steering,"steering1",steering1)

    if len(bboxes)>0:
        thickness = 2
        if in_safe_zone_i==1:
            thickness = 4
            color=[100,0,255]
        elif in_safe_zone_i==2:
            thickness = 4
            color=[0,100,200]
        elif in_safe_zone_i==3:
            thickness = 4
            color=[0,0,255]
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
        breaker += (speed)*0.02

    if breaker>1:
        breaker = 1
    if  throttle>1:
        throttle = 1

    #throttle = 0.4
    #throttle = 0
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
    
    #throttle = 0.5
    #breaker = 0
    #steering_prediction = 0

    throttle,breaker,steering =process_result(image, result, model.CLASSES,imginfo, message, wait_time=1)
 

    return throttle,breaker,steering


def get_control_param():
  params = {}
  try:
    with open("./autodriving/control_param.txt",'r', encoding='UTF-8') as f:
      for line in f:
        if line[0]=='#':
            continue
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