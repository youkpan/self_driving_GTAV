from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario,Commands
import glog as log
from deepgtav.client import Client
import cv2
import numpy as np
import time
import autodriving.predict
import autodriving.control
import gzip,pickle
import sys
sys.path.append("D:\\self-driving\\lanenet-lane-detection")
sys.path.append('D:\\self-driving\\lanenet-lane-detection\\tools')
from  tools import lanenet_detect
#import tools.lanenet_detect as lanenet_detect

def crop_bottom_half(image):
    ''' Crops to bottom half of image '''
    return image[int(image.shape[0] / 2):image.shape[0]]

#if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
#    print("Conintuing...")
image_path="D:\\self-driving\\lanenet-lane-detection\\data\\training_data_example\\image\\0001.png"
image_path="D:\\self-driving\\lanenet-lane-detection\\data\\training_data_example\\image\\road2.png "
weights_path="D:\\self-driving\\lanenet-lane-detection\\checkpoint\\tusimple_lanenet_vgg.ckpt"
#test_lanenet( image_path,  weights_path)
lanet = lanenet_detect.mlanenet(weights_path)
'''
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
a,b,c,d,e = lanet.inference(image)
cv2.imshow('lanet_img',c)
cv2.waitKey()
input()

exit()
'''

# Loads into a consistent starting setting 
print("Loading Scenario...")
USE_GTAV = 1
USE_DATASET = 2
USE_CAPTURE_CAM = 3
image_source = USE_GTAV
FPS = 10
show_src_img = False
show_imgwidth = 640
show_imgheight = 320

def reset(location):
    global FPS,show_imgwidth,show_imgheight
    ''' Resets position of car to a specific location '''
    # Same conditions as below | 
    dataset = Dataset(rate=FPS, frame=[show_imgwidth,show_imgheight],throttle=True, brake=True, steering=True,location=True, drivingMode=True,speed=True,yawRate=True,time=True,vehicles=True, peds=True, trafficSigns=True)
    #,yawRate=True,time=True,vehicles=True, peds=True, trafficSigns=True, direction=True, reward=True
    #[-1917.06640625, 4595.87255859375, 56.853]
    #-737.1954345703125, 1975.72265625, 133.54100036621094
    #[2723.626953125, 3224.170654296875, 54.402042388916016]
    #827.8175659179688, -1201.2620849609375, 45.51389694213867 街头竞速 没车
    scenario = Scenario(weather='EXTRASUNNY',vehicle='blista',time=[12,0],drivingMode=-1,location=location)
    client.sendMessage(Config(scenario=scenario,dataset=dataset))

if image_source == USE_GTAV:
    client = Client(ip='localhost', port=8000)#, datasetPath="self_driving.pz", compressionLevel=9) # Default interface
    '''
    try:
        #client.sendMessage(Stop()) # Stops DeepGTAV
        client.close()
    except Exception as e:
        pass

    client = Client(ip='localhost', port=8000, datasetPath="self_driving.pz", compressionLevel=9) # Default interface
    '''
    dataset = Dataset(rate=FPS, frame=[show_imgwidth,show_imgheight],throttle=True, brake=True, steering=True,location=True, drivingMode=True,speed=True,yawRate=True,time=True,vehicles=True, peds=True, trafficSigns=True )
    #dataset = None
     #blista { "blista", "voltic", "packer" };
     #隧道[-2573.13916015625, 3292.256103515625, 13.241103172302246]
     #[-3048.73486328125, 736.7617797851562, 21.694440841674805]
     #[1037.0552978515625, -2099.537353515625, 30.54058837890625]
     #{ "CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" };
    scenario = Scenario(weather='EXTRASUNNY',vehicle='voltic',time=[9,0],drivingMode=-1,location=[-3048.73486328125, 736.7617797851562, 21.694440841674805])

    client.sendMessage(Start(scenario=scenario,dataset=dataset))
    imgwidth0 = show_imgwidth
    imgheight0 = show_imgheight
elif image_source == USE_DATASET:
    file = gzip.open('dataset_test1.pz')
    imgwidth0 = 320
    imgheight0 = 160
elif image_source == USE_CAPTURE_CAM:
    cap = cv2.VideoCapture(0)
    imgwidth0 = 1280
    imgheight0 = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH ,imgwidth0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,imgheight0)
    cap.set(cv2.CAP_PROP_FPS,FPS)
    ret,frame = cap.read()
    print("cam image size",(frame.shape[1],frame.shape[0]))

model = autodriving.predict.predict_init()
steering = 0
count = 0
print("Starting Loop...")
t_start0 = time.time()
location_same_timer = 0
location_last = [0,0,0]

while True:
    try:   
        t_loop_start = time.time()

        if image_source == USE_GTAV:
            # Collect and preprocess image
            while True:
                t_start_recv = time.time()
                message = client.recvMessage()
                if time.time() - t_start_recv >0.05:#is new frame
                    break
            image = frame2numpy(message['frame'], (imgwidth0,imgheight0))
            image2 = image
        elif image_source == USE_DATASET:
            message = pickle.load(file) # Iterates through pickle generator
            image = frame2numpy(message['frame'], (imgwidth0,imgheight0))
            image2 =cv2.resize(image, (show_imgwidth, show_imgheight), interpolation=cv2.INTER_LINEAR)

        elif image_source == USE_CAPTURE_CAM:
            if not cap.isOpened():
                raise "not cap.isOpened()"
                exit()
            message = {}
            ret,frame = cap.read()
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image = frame2numpy(frame, (frame.shape[1],frame.shape[0]))
            image2 =cv2.resize(image, (show_imgwidth, show_imgheight), interpolation=cv2.INTER_LINEAR)
        #image = crop_bottom_half(image)
        #image = ((image/255) - .5) * 2

        count += 1
        #

        print("----------------------    getpredict    ----------------------")
        
        if count % 10 ==0 and show_src_img:
            cv2.imshow('image2',image2)
            cv2.waitKey(1)

        result = None
        if count % 6 ==0 and not show_src_img:
            t_start = time.time()
            result = autodriving.predict.getpredict(image2,model)
            print('getpredict time: {:.5f}s'.format(time.time() - t_start))

        imginfo = {"imgwidth":show_imgwidth,"imgheight":show_imgheight}
        message['count'] = count
        message['USE_GTAV'] = USE_GTAV
        try:
            speed = message['speed']
        except Exception as e:
            message['speed'] = 10
            speed = 10

        try:
            steering = message['steering']
        except Exception as e:
            message['steering'] = 10
            message['location']=[]

        #steering = message['steering']
        print("get steering",message['steering'],message['location'])

        message['lanet_center_x']=256
        message['lanet_center_y']=128

        #image2= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if  count % 6 ==3 and not show_src_img:
            t_start = time.time()
            #message['lanet_center_x'],message['lanet_center_y'],message['lanet_img'],message['lanet_out'],message['binary_image'] = lanet.inference(image)
            message['lanet_center_x'],message['lanet_center_y'] ,message['binary_image0'],message['binary_image'] = lanet.inference(image2)
            #message['lanet_img2'] = message['lanet_img'][:, :, (2, 1, 0)]
            print('lanet.inference time: {:.5f}s'.format(time.time() - t_start))

        if len(message['location'])>0:
            if location_same_timer>0 :
                print("------- not move, reset :",time.time() - location_same_timer)
                if time.time()>location_same_timer + 10 and len(location_last)>0:
                    location = location_last
                    reset(location)
                if time.time()>location_same_timer + 20:
                    location = [-1917.06640625, 4595.87255859375, 56.853]
                    reset(location)

            if abs(message['location'][0]-location_last[0])<1 :
                if location_same_timer == 0 :
                    location_same_timer = time.time()
            else:
                location_same_timer = 0
                location_last = message['location']

        throttle,breaker,steering = autodriving.control.getcontrol(image2,result,model,imginfo,message)
        print("count:",count,"throttle:",throttle,"breaker:",breaker,"steering:",steering,"speed:",speed)

        if  count % 6 ==3 and not show_src_img:
            try:

                #image3 =cv2.resize(message['lanet_img'], (640, 360), interpolation=cv2.INTER_LINEAR)
                #cv2.imshow('lanet_img',image3)
   
                img = message['binary_image']
                xx = message['lanet_center_x']/512*640
                yy = message['lanet_center_y']/256*320
                cv2.line(img, (320,319), (int(xx)  ,int(yy)  ),[0,200,0],2)

                yy = 160
                if steering!=0:
                    #-1/steering * (320) + b = 359
                    #ax + b = y
                    xx = -(yy- (359+1/steering * 320))*steering
                else:
                    xx = 320
                
                cv2.line(img, (320,319), (int(xx)  ,int(yy)  ),[255,0,0],2)

                cv2.imshow('binary_image',img)
                cv2.waitKey(1)

                pass
            except Exception as e:
                raise e
        if image_source == USE_GTAV:
            client.sendMessage(Commands(throttle,breaker,steering  )) # Mutiplication scales decimal prediction for harder turning
        print('loop time: {:.5f}s'.format(time.time() - t_loop_start))
        

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("Excepted as: " + str(e))
        #if image_source == USE_GTAV:
        #    client.sendMessage(Stop()) # Stops DeepGTAV
        #    client.close()
        raise e
        continue

if image_source == USE_GTAV:
    client.sendMessage(Stop()) # Stops DeepGTAV
    client.close()
