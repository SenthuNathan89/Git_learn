import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import csv
import glob
import os

def conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))(input_tensor)
    x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding,data_format='channels_last')(x)
    x = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=2))(x)
    return x

def buildDeepEncoderDecoder(subSignalLength):
    knSize = 15
    filters = np.array([1,64,128,256,256,512,512,1024,2048]);
    inp = layers.Input(shape=(subSignalLength,1))
    # Encoder convolution layer 1
    enConvL1= layers.Conv1D(filters = filters[1], kernel_size = knSize,padding='same')(inp)
    print("enConvL1", enConvL1)
    # Add activation function
    enConvL1Af = layers.Activation(tf.nn.relu)(enConvL1)
    del enConvL1
    enConvL1AfMP = layers.MaxPooling1D(2)(enConvL1Af)
    del enConvL1Af
    # Encoder convolution layer 2
    enConvL2= layers.Conv1D(filters = filters[2], kernel_size = knSize,padding='same')(enConvL1AfMP)
    print("enConvL2", enConvL2)
    del enConvL1AfMP
    # Add activation function
    enConvL2Af = layers.Activation(tf.nn.relu)(enConvL2)
    del enConvL2
    enConvL2AfMP = layers.MaxPooling1D(2)(enConvL2Af)
    del enConvL2Af 
    # Encoder convolution layer 3
    enConvL3= layers.Conv1D(filters = filters[3], kernel_size = knSize,padding='same')(enConvL2AfMP)
    print("enConvL3", enConvL3)
    del enConvL2AfMP 
    # Add activation function
    enConvL3Af = layers.Activation(tf.nn.relu)(enConvL3)
    del enConvL3
    enConvL3AfMP = layers.MaxPooling1D(2)(enConvL3Af)
    del enConvL3Af
    # Encoder convolution layer 4
    enConvL4= layers.Conv1D(filters = filters[4], kernel_size = knSize,padding='same')(enConvL3AfMP)
    print("enConvL4",enConvL4)
    del enConvL3AfMP
    # Add activation function
    enConvL4Af = layers.Activation(tf.nn.relu)(enConvL4)
    # del enConvL4
    enConvL4AfMP = layers.MaxPooling1D(2)(enConvL4Af)
    del enConvL4Af
    # Encoder convolution layer 5
    enConvL5= layers.Conv1D(filters = filters[5], kernel_size = knSize,padding='same')(enConvL4AfMP)
    print("enConvL5",enConvL5)
    del enConvL4AfMP
    # Add activation function
    enConvL5Af = layers.Activation(tf.nn.relu)(enConvL5)
    del enConvL5
    enConvL5AfMP = layers.MaxPooling1D(2)(enConvL5Af)
    del enConvL5Af
    # Encoder convolution layer 6
    enConvL6= layers.Conv1D(filters = filters[6], kernel_size = knSize,padding='same')(enConvL5AfMP)
    print("enConvL6",enConvL6)
    del enConvL5AfMP
    # Add activation function
    enConvL6Af = layers.Activation(tf.nn.relu)(enConvL6)
    del enConvL6
    enConvL6AfMP = layers.MaxPooling1D(2)(enConvL6Af)
    del enConvL6Af
    # Encoder convolution layer 7
    enConvL7= layers.Conv1D(filters = filters[7], kernel_size = knSize,padding='same')(enConvL6AfMP)
    print("enConvL7",enConvL7)
    del enConvL6AfMP
    # Add activation function
    enConvL7Af = layers.Activation(tf.nn.relu)(enConvL7)
    del enConvL7
    enConvL7AfMP = layers.MaxPooling1D(2)(enConvL7Af)
    del enConvL7Af
    # Encoder convolution layer 8
    enConvL8= layers.Conv1D(filters = filters[8], kernel_size = knSize,padding='same')(enConvL7AfMP)
    print("enConvL8", enConvL8)
    del enConvL7AfMP
    # Add activation function
    enConvL8Af = layers.Activation(tf.nn.relu)(enConvL8)
    del enConvL8
    enConvL8AfMP = layers.MaxPooling1D(2)(enConvL8Af)
    print("enConvL8AfMP",enConvL8AfMP)
    # Decoder convolution transpose layer 1
    deConvL1= conv1DTranspose(enConvL8AfMP,filters = filters[7], kernel_size = knSize)
    print("deConvL1", deConvL1)
    del enConvL8Af
    # Add activation function
    deConvL1Af = layers.Activation(tf.nn.relu)(deConvL1)
    del deConvL1
    # Decoder convolution layer 2
    deConvL2= conv1DTranspose(deConvL1Af,filters = filters[6], kernel_size = knSize)
    print("deConvL2", deConvL2)
    del deConvL1Af
    # Add activation function
    deConvL2Af = layers.Activation(tf.nn.relu)(deConvL2)
    del deConvL2
    # Decoder convolution layer 3
    deConvL3= conv1DTranspose(deConvL2Af,filters = filters[5], kernel_size = knSize)
    print("deConvL3", deConvL3)
    del deConvL2Af
    # Add activation function
    deConvL3Af = layers.Activation(tf.nn.relu)(deConvL3)
    del deConvL3
    # Decoder convolution layer 4
    deConvL4= conv1DTranspose(deConvL3Af,filters = filters[4], kernel_size = knSize)
    print("deConvL4", deConvL4)
    del deConvL3Af
    # Add activation function
    deConvL4Af = layers.Activation(tf.nn.relu)(deConvL4)
    del deConvL4
    # Decoder convolution layer 5
    deConvL5= conv1DTranspose(deConvL4Af,filters = filters[3], kernel_size = knSize)
    print("deConvL5", deConvL5)
    del deConvL4Af
    skipConnection2 = layers.add([enConvL4,deConvL5])
    # Add activation function
    deConvL5Af = layers.Activation(tf.nn.relu)(skipConnection2)
    del skipConnection2
    del deConvL5
    # Decoder convolution layer 6
    deConvL6= conv1DTranspose(deConvL5Af,filters = filters[2], kernel_size = knSize)
    print("deConvL6", deConvL6)
    del deConvL5Af
    # skip connection2
    # skipConnection2 = layers.add([enConvL2, deConvL6])
    # Add activation function
    deConvL6Af = layers.Activation(tf.nn.relu)(deConvL6)
    del deConvL6
    # Decoder convolution layer 7
    deConvL7= conv1DTranspose(deConvL6Af,filters = filters[1], kernel_size = knSize, padding='same')
    print("deConvL7", deConvL7)
    del deConvL6Af
    # Add activation function
    deConvL7Af = layers.Activation(tf.nn.relu)(deConvL7)
    del deConvL7
    # Decoder convolution layer 8
    deConvL8= conv1DTranspose(deConvL7Af,filters = filters[0], kernel_size = knSize)
    print("deConvL8", deConvL8)
    # del deConvL7Af
    skipConnection1 = layers.add([inp, deConvL8])
    # Add activation function
    deConvL8Af = layers.Activation(tf.nn.relu)(skipConnection1)
    return inp, deConvL8Af

def loadData(segmentLen):
    trainOutput = []
    groundTruthOutput = []
    testingOutput = []
#     fileNameList = glob.glob('./data/training/*.csv')
#     fileNameGTList = glob.glob('./data/ground_truth/*.csv')
    trainDir = "./data/training/"
    gtDir = "./data/ground_truth/"
    fileNameTestingList = glob.glob('./data/testing/*.csv')
    fileNameList = os.listdir(trainDir)
    #-----------load training data------------------------
    for filename in fileNameList:
        #openning the csv file which is in the same location of this python file
        fEcgFile = open(trainDir + filename)
        #reading the File with the help of csv.reader()
        fEcgReader = csv.reader(fEcgFile)
        #storing the values contained in the Reader into Data
        fEcgData = list(fEcgReader)
        #printing the each line of the Data in the console
        for data in fEcgData:
            segNum = int(len(data)/segmentLen)
            for n in range(segNum):
                segDataTmp = [np.abs(float(x)) for x in data[n*segmentLen:(n+1)*segmentLen]]
                #normalize abs(segDataTmp) to range[0,1]
                minSD = min(segDataTmp)
                maxSD = max(segDataTmp)
                rangeSD = maxSD-minSD
                segDataAbsNorm = [(x-minSD)/rangeSD for x in segDataTmp]
                segData = np.array([segDataAbsNorm]).T
                trainOutput.append(segData)
        fEcgFile.close()
    trainOutput = np.array(trainOutput)
    #-----------load ground truth data------------------------
    for filename in fileNameList:
        #openning the csv file which is in the same location of this python file
        filenameSplit = filename.split("_")
        #openning the csv file which is in the same location of this python file
        fEcgFile = open(gtDir+filenameSplit[0]+"_gt"+filenameSplit[1])
        #reading the File with the help of csv.reader()
        fEcgReader = csv.reader(fEcgFile)
        #storing the values contained in the Reader into Data
        fEcgData = list(fEcgReader)
        #printing the each line of the Data in the console
        for data in fEcgData:
            segNum = int(len(data)/segmentLen)
            for n in range(segNum):
                segDataTmp = [np.abs(float(x)) for x in data[n*segmentLen:(n+1)*segmentLen]]
                #normalize abs(segDataTmp) to range[0,1]
                minSD = min(segDataTmp)
                maxSD = max(segDataTmp)
                rangeSD = maxSD-minSD
                segDataAbsNorm = [(x-minSD)/rangeSD for x in segDataTmp]
                segData = np.array([segDataAbsNorm]).T
                groundTruthOutput.append(segData)
        fEcgFile.close()
    groundTruthOutput = np.array(groundTruthOutput)
    #-----------load testing truth data------------------------
    for filename in fileNameTestingList:
        #openning the csv file which is in the same location of this python file
        fEcgFile = open(filename)
        #reading the File with the help of csv.reader()
        fEcgReader = csv.reader(fEcgFile)
        #storing the values contained in the Reader into Data
        fEcgData = list(fEcgReader)
        #printing the each line of the Data in the console
        for data in fEcgData:
            segNum = int(len(data)/segmentLen)
            for n in range(segNum):
                segDataTmp = [np.abs(float(x)) for x in data[n*segmentLen:(n+1)*segmentLen]]
                #normalize abs(segDataTmp) to range[0,1]
                minSD = min(segDataTmp)
                maxSD = max(segDataTmp)
                rangeSD = maxSD-minSD
                segDataAbsNorm = [(x-minSD)/rangeSD for x in segDataTmp]
                segData = np.array([segDataAbsNorm]).T
                testingOutput.append(segData)
        fEcgFile.close()
    testingOutput = np.array(testingOutput)
    return trainOutput, groundTruthOutput, testingOutput

def loadChallenge2013Data(segmentLen):
    trainOutput = []
    groundTruthOutput = []
    testingOutput = []
#     fileNameList = glob.glob('./data/training/*.csv')
#     fileNameGTList = glob.glob('./data/ground_truth/*.csv')
    trainDir = "./challenge_2013/training/"
    gtDir = "./challenge_2013/ground_truth/"
    fileNameList = os.listdir(trainDir)
    print(fileNameList)
    #-----------load training data------------------------
    for filename in fileNameList:
        #openning the csv file which is in the same location of this python file
        fEcgFile = open(trainDir + filename)
        #reading the File with the help of csv.reader()
        fEcgReader = csv.reader(fEcgFile)
        #storing the values contained in the Reader into Data
        fEcgData = list(fEcgReader)
        #printing the each line of the Data in the console
        for data in fEcgData:
            segNum = int(len(data)/segmentLen)
            for n in range(segNum):
                segDataTmp = [np.abs(float(x)) for x in data[n*segmentLen:(n+1)*segmentLen]]
                #normalize abs(segDataTmp) to range[0,1]
                minSD = min(segDataTmp)
                maxSD = max(segDataTmp)
                rangeSD = maxSD-minSD
                segDataAbsNorm = [(x-minSD)/rangeSD for x in segDataTmp]
                segData = np.array([segDataAbsNorm]).T
                trainOutput.append(segData)
        fEcgFile.close()
    trainOutput = np.array(trainOutput)
    #-----------load ground truth data------------------------
    for filename in fileNameList:
        #openning the csv file which is in the same location of this python file
        filenameSplit = filename.split(".")
        #openning the csv file which is in the same location of this python file
        fEcgFile = open(gtDir+filenameSplit[0]+"_gt."+filenameSplit[1])
        #reading the File with the help of csv.reader()
        fEcgReader = csv.reader(fEcgFile)
        #storing the values contained in the Reader into Data
        fEcgData = list(fEcgReader)
        #printing the each line of the Data in the console
        for data in fEcgData:
            segNum = int(len(data)/segmentLen)
            for n in range(segNum):
                segDataTmp = [np.abs(float(x)) for x in data[n*segmentLen:(n+1)*segmentLen]]
                #normalize abs(segDataTmp) to range[0,1]
                minSD = min(segDataTmp)
                maxSD = max(segDataTmp)
                rangeSD = maxSD-minSD
                segDataAbsNorm = [(x-minSD)/rangeSD for x in segDataTmp]
                segData = np.array([segDataAbsNorm]).T
                groundTruthOutput.append(segData)
        fEcgFile.close()
    groundTruthOutput = np.array(groundTruthOutput)
    return trainOutput, groundTruthOutput