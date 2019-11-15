from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario,Commands

from deepgtav.client import Client
import cv2
import numpy as np

import autodriving.predict
import autodriving.control


def crop_bottom_half(image):
    ''' Crops to bottom half of image '''
    return image[int(image.shape[0] / 2):image.shape[0]]

#if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
#    print("Conintuing...")


imgwidth = 640
imgheight = 320
# Loads into a consistent starting setting 
print("Loading Scenario...")
client = Client(ip='localhost', port=8000, datasetPath="self_driving.pz", compressionLevel=9) # Default interface
dataset = Dataset(rate=10, frame=[imgwidth,imgheight],throttle=True, brake=True, steering=True,location=True, drivingMode=True,speed=True,yawRate=True,time=True,vehicles=True, peds=True, trafficSigns=True )
#dataset = None
scenario = Scenario(weather='EXTRASUNNY',vehicle='blista',time=[12,0],drivingMode=-1,location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])
client.sendMessage(Start(scenario=scenario,dataset=dataset))

model = autodriving.predict.predict_init()

count = 0
print("Starting Loop...")
while True:
    try:    
        # Collect and preprocess image
        message = client.recvMessage()
        image = frame2numpy(message['frame'], (imgwidth,imgheight))

        #image = crop_bottom_half(image)
        #image = ((image/255) - .5) * 2
        

        count += 1

        if count %5 >0:
            continue

        print("getpredict   ")
        result = autodriving.predict.getpredict(image,model)
        imginfo = {"imgwidth":imgwidth,"imgheight":imgheight}
        message['count'] = count
        speed = message['speed']

        throttle,breaker,steering_prediction = autodriving.control.getcontrol(image,result,model,imginfo,message)
        print("count:",count,"throttle:",throttle,"breaker:",breaker,"steering:",steering_prediction,"speed:",speed)

        client.sendMessage(Commands(throttle,breaker,steering_prediction  )) # Mutiplication scales decimal prediction for harder turning
        

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("Excepted as: " + str(e))
        continue

client.sendMessage(Stop()) # Stops DeepGTAV
client.close()
