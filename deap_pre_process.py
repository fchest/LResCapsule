import scipy.io as sio
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
import pickle

np.random.seed(0)

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          data[0],    0,          data[16], 	0,  	    0, 	        0       )
    data_2D[1] = (0,  	   	0,          0,          data[1],    0,          data[17],   0,          0,          0       )
    data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
    data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
    data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
    data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
    data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
    data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
    # return shape:9*9
    return data_2D

def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized

def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],9,9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize( data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D

def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size

def segment_signal_without_transition(data,label,label_index,window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if((len(data[start:end]) == window_size)):
            if(start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index])) # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels

def apply_mixup(dataset_file,window_size,label,yes_or_not): # initial empty label arrays
    print("Processing",dataset_file,"..........")
    data_file_in = sio.loadmat(dataset_file)  # Load MATLAB file,import scipy.io as sio
    data_in = data_file_in["data"].transpose(0,2,1)  # reliably restored by inspect
    #0 valence, 1 arousal, 2 dominance, 3 liking
    if label=="arousal":
        label=1
    elif label=="valence":
        label=0
    elif label=="dominance":
        label=2
    label_in= data_file_in["labels"][:,label]>5
    label_inter	= np.empty([0]) # initial empty data arrays
    data_inter_cnn	= np.empty([0,window_size, 9, 9])
    data_inter_rnn	= np.empty([0, window_size, 32])
    trials = data_in.shape[0]   # Retrieve a scalar value from a `netcdf_variable` of length one.

    # Data pre-processing
    for trial in range(0,trials):
        if yes_or_not=="yes":
            base_signal = (data_in[trial,0:128,0:32]+data_in[trial,128:256,0:32]+data_in[trial,256:384,0:32])/3
        else:
            base_signal = 0
        data = data_in[trial,384:8064,0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0,60):
            data[i*128:(i+1)*128,0:32]=data[i*128:(i+1)*128,0:32]-base_signal  #experimental_signals - base_sinal
        label_index = trial
        #read data and label
        data = norm_dataset(data)
        data, label = segment_signal_without_transition(data, label_in,label_index,window_size)
        # cnn data process
        data_cnn    = dataset_1Dto2D(data)   #Call above function
        data_cnn    = data_cnn.reshape ( int(data_cnn.shape[0]/window_size), window_size, 9, 9)  #Returns an array containing the same data with a new shape.
        # rnn data process
        data_rnn    = data. reshape(int(data.shape[0]/window_size), window_size, 32)
        # append new data and label
        data_inter_cnn  = np.vstack([data_inter_cnn, data_cnn])
        data_inter_rnn  = np.vstack([data_inter_rnn, data_rnn])
        label_inter = np.append(label_inter, label)  #Append values to the end of an array
    '''
    print("total cnn size:", data_inter_cnn.shape)
    print("total rnn size:", data_inter_rnn.shape)
    print("total label size:", label_inter.shape)
    '''
    # shuffle data
    index = np.array(range(0, len(label_inter)))
    np.random.shuffle( index)  # This function only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed buttheir contents remains the same.
    shuffled_data_cnn	= data_inter_cnn[index]
    shuffled_data_rnn	= data_inter_rnn[index]  #call above function
    shuffled_label 	= label_inter[index]
    return shuffled_data_cnn ,shuffled_data_rnn,shuffled_label,record

if __name__ == '__main__' :
    begin = time.time()   #Return the current time in seconds since the Epoch.
    print("time begin:",time.localtime())  #Convert seconds since the Epoch to a time tuple expressing local time.
    dataset_dir		=   "/media/data/data_human/EEG/DEAP/data_preprocessed_matlab/"
    window_size		=	128
    output_dir		=   "./deap_shuffled_data/"
    label_class     =   'dominance'     # arousal/valence
    suffix          =   'yes'     # yes/no (using baseline signals or not)
    # get directory name for one subject
    record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir,task))]  #Test whether a path is a regular file
    output_dir = output_dir+suffix+"_"+label_class+"/"
    if os.path.isdir(output_dir)==False:
        os.makedirs(output_dir)   #Super-mkdir; create a leaf directory and all intermediate ones.
    # print(record_list)

    for record in record_list:     #call above function
        file = os.path.join(dataset_dir,record)   #Join two (or more) paths.
        shuffled_cnn_data,shuffled_rnn_data,shuffled_label,record = apply_mixup(file, window_size,label_class,suffix)  #call above function
        output_data_cnn = output_dir+record+"_win_"+str(window_size)+"_cnn_dataset.pkl"
        output_data_rnn = output_dir+record+"_win_"+str(window_size)+"_rnn_dataset.pkl"
        output_label= output_dir+record+"_win_"+str(window_size)+"_labels.pkl"

        with open(output_data_cnn, "wb") as fp:     #Open file and return a stream.  Raise IOError upon failure.
            pickle.dump( shuffled_cnn_data,fp, protocol=4)
        with open( output_data_rnn, "wb") as fp:
            pickle.dump(shuffled_rnn_data, fp, protocol=4)  #pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
        with open(output_label, "wb") as fp:
            pickle.dump(shuffled_label, fp)      #序列化对象，并将结果数据流写入到文件对象中。参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。
        end = time.time()
        print("end time:",time.localtime())
        print("time consuming:",(end-begin))
        
        # break
