from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario,Commands
import glog as log
from deepgtav.client import Client
import cv2
import numpy as np
import time
import autodriving.predict
import autodriving.control
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
imgwidth = 640
imgheight = 320
# Loads into a consistent starting setting 
print("Loading Scenario...")
client = Client(ip='localhost', port=8000)#, datasetPath="self_driving.pz", compressionLevel=9) # Default interface
'''
try:
    #client.sendMessage(Stop()) # Stops DeepGTAV
    client.close()
except Exception as e:
    pass

client = Client(ip='localhost', port=8000, datasetPath="self_driving.pz", compressionLevel=9) # Default interface
'''
dataset = Dataset(rate=10, frame=[imgwidth,imgheight],throttle=True, brake=True, steering=True,location=True, drivingMode=True,speed=True,yawRate=True,time=True,vehicles=True, peds=True, trafficSigns=True )
#dataset = None
 #blista { "blista", "voltic", "packer" };
 #隧道[-2573.13916015625, 3292.256103515625, 13.241103172302246]
 #[-3048.73486328125, 736.7617797851562, 21.694440841674805]
 #{ "CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" };
scenario = Scenario(weather='OVERCAST',vehicle='voltic',time=[9,0],drivingMode=-1,location=[1037.0552978515625, -2099.537353515625, 30.54058837890625])
client.sendMessage(Start(scenario=scenario,dataset=dataset))

model = autodriving.predict.predict_init()
steering = 0
count = 0
print("Starting Loop...")
t_start0 = time.time()
while True:
    try:   
        t_loop_start = time.time()
        
        # Collect and preprocess image
        while True:
            t_start_recv = time.time()
            message = client.recvMessage()
            if time.time() - t_start_recv >0.05:#is new frame
                break
        image = frame2numpy(message['frame'], (imgwidth,imgheight))

        #image = crop_bottom_half(image)
        #image = ((image/255) - .5) * 2

        count += 1
        #image2 =cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
        image2 = image

        print("----------------------    getpredict    ----------------------")
        result = None
        if count % 6 ==0:
            t_start = time.time()
            result = autodriving.predict.getpredict(image2,model)
            print('getpredict time: {:.5f}s'.format(time.time() - t_start))

        imginfo = {"imgwidth":imgwidth,"imgheight":imgheight}
        message['count'] = count
        speed = message['speed']
        #steering = message['steering']
        print("get steering",message['steering'],message['location'])

        message['lanet_center_x']=256
        message['lanet_center_y']=128

        #image2= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if  count % 6 ==3 :
            t_start = time.time()
            #message['lanet_center_x'],message['lanet_center_y'],message['lanet_img'],message['lanet_out'],message['binary_image'] = lanet.inference(image)
            message['lanet_center_x'],message['lanet_center_y'] ,message['binary_image'] = lanet.inference(image)
            #message['lanet_img2'] = message['lanet_img'][:, :, (2, 1, 0)]
            print('lanet.inference time: {:.5f}s'.format(time.time() - t_start))


        throttle,breaker,steering = autodriving.control.getcontrol(image2,result,model,imginfo,message)
        print("count:",count,"throttle:",throttle,"breaker:",breaker,"steering:",steering,"speed:",speed)

        if  count % 6 ==3 :
            try:

                #image3 =cv2.resize(message['lanet_img'], (640, 360), interpolation=cv2.INTER_LINEAR)
                #cv2.imshow('lanet_img',image3)
                img = np.zeros([256, 512,3],np.uint8)       # 输出一张图片，属性为高400，宽400，通道3
                img[: ,: ,0] = message['binary_image']   
                img[: ,: ,1] = message['binary_image']   
                img[: ,: ,2] = message['binary_image']   

                xx = message['lanet_center_x']/512*640
                yy = message['lanet_center_y']/256*320
                img =cv2.resize(img, (640, 320), interpolation=cv2.INTER_LINEAR)
                cv2.line(img, (320,319), (int(xx)  ,int(yy)  ),[0,200,0],2)

                yy = 160
                if steering!=0:
                    #-1/steering * (320) + b = 359
                    #ax + b = y
                    xx = -(yy- (359+1/steering * 320))*steering
                else:
                    xx = 320
                
                cv2.line(img, (320,359), (int(xx)  ,int(yy)  ),[255,0,0],2)

                cv2.imshow('binary_image',img)

                pass
            except Exception as e:
                raise e

        client.sendMessage(Commands(throttle,breaker,steering  )) # Mutiplication scales decimal prediction for harder turning
        print('loop time: {:.5f}s'.format(time.time() - t_loop_start))
        

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("Excepted as: " + str(e))
        raise e
        continue

client.sendMessage(Stop()) # Stops DeepGTAV
client.close()
