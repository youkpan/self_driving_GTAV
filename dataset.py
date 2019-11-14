from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario
from deepgtav.client import Client
import argparse
import time

def reset():
    ''' Resets position of car to a specific location '''
    # Same conditions as below | 
    dataset = Dataset(rate=10, frame=[320,160],throttle=True, brake=True, steering=True,location=True, drivingMode=True)
    scenario = Scenario(weather='EXTRASUNNY',vehicle='blista',time=[12,0],drivingMode=[786603,20.0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])
    client.sendMessage(Config(scenario=scenario,dataset=dataset))

# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-d', '--dataset_path', default='dataset_test.pz', help='Place to store the dataset')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port 
    client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9) 
    # Dataset options
    dataset = Dataset(rate=10, frame=[320,160],throttle=True, brake=True, steering=True,location=True, drivingMode=True)
    # Automatic driving scenario
    #
    scenario = Scenario(weather='EXTRASUNNY',vehicle='blista',time=[12,0],drivingMode=[786603,20.0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) 
    client.sendMessage(Start(scenario=scenario,dataset=dataset)) # Start request
    
    count = 0
    old_location = [0, 0, 0]
    
    while True: # Main loop
        try:
            # Message recieved as a Python dictionary
            message = client.recvMessage()
 
            if message!=None :
                '''
                if message['throttle']==int(message['throttle']) and message['throttle']!=0 and message['throttle']<200:
                    print("offset:",message['throttle'])
                    print("===================")
                    print("===================")
                #if message['throttle']<1000 and abs(message['throttle'])>0.01 and abs(message['steering']<100) and abs(message['steering'])>0.01 :
                #    print("+++++++++++++++++++")
                '''
                if (count % 20) == 0:#
                    print(count ,message['steering']*500+500)
                    for k in message.keys():
                        if k!='frame':
                            print(k,message[k])


            # Checks if car is suck, resets position if it is
            if (count % 250)==0:
                new_location = message['location']
                # Float position converted to ints so it doesn't have to be in the exact same position to be reset
                if int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(old_location[1]) and int(new_location[2]) == int(old_location[2]):
                    reset()
                old_location = message['location']
                print('At location: ' + str(old_location))
            count += 1

            if (count == 660000) :
                break
                
        except KeyboardInterrupt:
            i = input('Paused. Press p to continue and q to exit... ')
            if i == 'p':
                continue
            elif i == 'q':
                break
        except Exception as e:

            continue
            
    # DeepGTAV stop message
    client.sendMessage(Stop())
    client.close() 
