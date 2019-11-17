from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario,Commands
import glog as log
from deepgtav.client import Client
import cv2
import numpy as np
import time
import autodriving.predict
import autodriving.control
import sys
import multiprocessing
from multiprocessing import Pool 
from multiprocessing import Process, Manager

sys.path.append("D:\\self-driving\\lanenet-lane-detection")
sys.path.append('D:\\self-driving\\lanenet-lane-detection\\tools')
from  tools import lanenet_detect
#import tools.lanenet_detect as lanenet_detect

def crop_bottom_half(image):
    ''' Crops to bottom half of image '''
    return image[int(image.shape[0] / 2):image.shape[0]]

def recvMessageGTAV(multiProcess_sync_pack):
    message_count = 0

    while True:
        try:
            message = multiProcess_sync_pack['client'].recvMessage()
            message['time'] = time.time()
            message['count'] = message_count
            message_count +=1
            multiProcess_sync_pack['message'] = message
        except KeyboardInterrupt:
            break
        except Exception as e:
            break

    multiProcess_sync_pack['client'].sendMessage(Stop()) # Stops DeepGTAV
    multiProcess_sync_pack['client'].close()




def main_process(multiProcess_sync_pack,lanet,client,model,t_start0):
     

    while True:
        try:   
            t_loop_start = time.time()
            
            # Collect and preprocess image
            #message = client.recvMessage()
            try:
                message = multiProcess_sync_pack['message']
            except Exception as e:
                continue
            
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
            multiProcess_sync_pack['client'].sendMessage(Stop()) # Stops DeepGTAV
            multiProcess_sync_pack['client'].close()
            raise e
            continue

    multiProcess_sync_pack['client'].sendMessage(Stop()) # Stops DeepGTAV
    multiProcess_sync_pack['client'].close()


def main():
    with Manager() as manager:
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
        scenario = Scenario(weather='EXTRASUNNY',vehicle='voltic',time=[12,0],drivingMode=-1,location=[-3048.73486328125, 736.7617797851562, 21.694440841674805])
        client.sendMessage(Start(scenario=scenario,dataset=dataset))

        model = autodriving.predict.predict_init()
        steering = 0
        count = 0
        print("Starting Loop...")
        t_start0 = time.time()
        message = {}
        message_count = 0
        multiProcess_sync_pack = {}
        multiProcess_sync_pack = manager.dict()
        multiProcess_sync_pack['client'] = client
        print("The number of CPU is:" + str(multiprocessing.cpu_count()))
        pool = Pool(7)
        pool.apply_async(recvMessageGTAV,(multiProcess_sync_pack))
        pool.apply_async(main_process,(multiProcess_sync_pack,lanet,client,model,t_start0))
        pool.close()# close 必须在join之前
        pool.join()# join阻塞等待 
        '''
        process_lock = multiprocessing.Lock()
        for ii in range(0,12):
            #_thread.start_new_thread( get_same_like,("thread:"+str(ii),ii) )
            p=multiprocessing.Process(target=get_same_like, args=(ii,uid_vid_data_all ,vid_like_list,process_lock))
            p.start()
            p.join()
            #get_same_like("thread:"+str(ii),ii) 
        '''
        #for p in multiprocessing.active_children():
        #    print("child   p.name:" + p.name + "\tp.id" + str(p.pid))

if __name__ == '__main__':
    main()
