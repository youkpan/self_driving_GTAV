def recvMessageGTAV(multiProcess_sync_pack):
    message_count = 0
    print("start recvMessageGTAV")
    while True:
        try:
            message = multiProcess_sync_pack['client'].recvMessage()
            message['time'] = time.time()
            message['count'] = message_count
            print("recvmessage ",count)
            message_count +=1
            multiProcess_sync_pack['message'] = message
        except KeyboardInterrupt:
            break
        except Exception as e:
            break

    multiProcess_sync_pack['client'].sendMessage(Stop()) # Stops DeepGTAV
    multiProcess_sync_pack['client'].close()

def main_process(multiProcess_sync_pack,lanet,client,model,t_start0):
    print("start main_process")

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
