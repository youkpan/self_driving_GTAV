import os
import numpy as np
import random
import pickle
import gzip
from deepgtav.messages import frame2numpy
import math


def crop_bottom_half(image):
    ''' Crops to bottom half of image '''
    return image[int(image.shape[0] / 2):image.shape[0]]

def get_steering(steering):

    if steering >=0:
        steering1 =  int( 1.9* math.pow(0.2*float(steering),0.4) *20 )
    else:
        steering1 =  int( -1.9* math.pow(-0.2*float(steering),0.4) *20 )

    return steering1+19

frames = {}
frame_index = 0
frame_per_second = 10
max_frame = frame_per_second * 10
frames_inited= 0

def insert_image_fifo(frame):
    global frame_index,frames
    frames[frame_index] = frame
    frame_index +=1
    if( frame_index >= max_frame):
        frame_index = 0


def get_image_fifo(index_offset):
    global frame_index,frames
    idx = frame_index - index_offset 
    if idx<0:
        idx += max_frame
    return frames[idx]

def append_data(data1,data2):
    return np.append(data1,[data2],0)

def load_batches(verbose=1,samples_per_batch=128):
    ''' Generator for loading batches of frames'''
    global frame_index,frames,frames_inited
    dataset = gzip.open('dataset.pz')
    batch_count = 0
    while True:
        try:
            x_train = []
            x_train_0_5S = []
            x_train_2S = []
            x_train_5S = []

            y_train = []
            x_test = []
            x_test_0_5S = []
            x_test_2S = []
            x_test_5S = []

            y_test = []
            count = 0
            print('----------- On Batch: ' + str(batch_count) + ' -----------')
            while count < samples_per_batch or frames_inited==0:
                    data_dct = pickle.load(dataset)
                    frame = data_dct['frame']
                    image = frame2numpy(frame, (320,160))
                    image = crop_bottom_half(image)
                    image = ((image / 255) - .5) * 2 # Simple preprocessing
                    insert_image_fifo(image)
                    
                    if frames_inited == 0:
                        if len(frames)<50:
                            continue
                    count += 1
                    frames_inited = 1

                    image_0_5S = get_image_fifo(5)
                    image_2S = get_image_fifo(20)
                    #image_5S = get_image_fifo(50)

                    # Train test split
                    # TODO: Dynamic train test split | Test series at end of batch
                    if (count <samples_per_batch*0.9): # Train

                        x_train.append([image,image_0_5S,image_2S])
                        #x_train_0_5S.append(image_0_5S)
                        #x_train_2S.append(image_2S)
                        #x_train_5S.append(image_5S)

                        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                        #2*(0.2*x)^0.4
                        
                        steering1 = get_steering (float(data_dct['steering']))
                        y_train.append(steering1) 

                    else: # Test
                        x_test.append([image,image_0_5S,image_2S])
                        #x_test_0_5S.append(image_0_5S)
                        #x_test_2S.append(image_2S)
                        #x_test_5S.append(image_5S)
                        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                        steering1 = get_steering (float(data_dct['steering']))
                        y_test.append(steering1) 
                    
                    if (count % 250) == 0 and verbose == 1:
                        print('     ' + str(count) + ' data points loaded in batch.')

            print('Batch loaded.')
            #print("x_train.shape",len(x_train))
            #print("y_train.shape",len(y_train))
            batch_count += 1
            yield x_train, y_train, x_test, y_test
        except EOFError: # Breaks at end of file
            break
            