# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:33:48 2022

@author: Hochun
"""
import cv2
import numpy as np
import math
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model

class sad(object):
    def __init__(self):
        self.img_L = np.zeros(())
        self.img_R = np.zeros(())
        self.disparity_depth = np.zeros(())
        self.disparity_depth_r = np.zeros(())
        
    # calculate pixel(x,y) corresponding value
    # w=(7*7) region or windows
    def pixel_SAD_left(self, x, y, d):
        cost = 0

        for i in range(x-3,x+4):
            for j in range (y-3,y+4):
               cost = cost + abs(int(self.img_L[i][j]) - int(self.img_R[i][j+d]))
        
               cost = cost / 49.0
            
        return abs(int(self.img_L[i][j]) - int(self.img_R[i][j+d]))
        
    def read_img(self, path_L, path_R, maxd):
        
        self.imgColorL = cv2.imread(path_L)
        self.imgColorR = cv2.imread(path_R)
        self.img_L = np.zeros((self.imgColorL.shape[0], self.imgColorL.shape[1]))
        self.img_R = np.zeros((self.imgColorL.shape[0], self.imgColorL.shape[1]))
        self.disparity_depth = np.zeros((maxd, self.imgColorL.shape[0], self.imgColorL.shape[1]))
        self.disparity_depth_r = np.zeros((maxd, self.imgColorL.shape[0], self.imgColorL.shape[1]))
        
        for i in range(self.imgColorL.shape[0]):
            for j in range(self.imgColorL.shape[1]):
                self.img_L[i][j] = (int(self.imgColorL[i][j][0]) + int(self.imgColorL[i][j][1]) + int(self.imgColorL[i][j][2])) / 3
                self.img_R[i][j] = (int(self.imgColorR[i][j][0]) + int(self.imgColorR[i][j][1]) + int(self.imgColorR[i][j][2])) / 3
        return 
    
    def calculate_cost_aggression_left(self, x,y):
        result = 0
        for i in range(-5,6):
            for j in range(-5,6):
                distance = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
                intensity = abs(self.img_L[x][y] - self.img_L[x+i][y+j])
                result = math.exp(-distance/289) * math.exp(-intensity/0.09)
        return result
    
    def calculate_cost_aggression_left_bileteral(self, x,y,d):
        result = 0
        for i in range(-5,6):
            for j in range(-5,6):
                distance = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
                intensity = abs(self.disparity_depth[d][x][y] - self.disparity_depth[d][x+i][y+j])
                result = math.exp(-distance/289) * math.exp(-intensity/0.09)
        return result
    
    def calculate_cost_aggression_right(self, x,y):
        result = 0
        for i in range(-5,6):
            for j in range(-5,6):
                distance = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
                intensity = math.pow((self.img_R[x][y] - self.img_R[x+i][y+j]), 2)
                result = math.exp(-distance/289) * math.exp(-intensity/0.09)
        return result                
    def pixel_SAD_right(self, x, y, d):
        cost = 0
        for i in range(x-2,x+2):
            for j in range (y-2,y+2):
               cost = cost + abs(int(self.img_R[i][j]) - int(self.img_L[i][j-d]))
               
        return abs(int(self.img_R[i][j]) - int(self.img_L[i][j-d]))
    
    def feature_extract(self, backbone='vgg'):
        """
        Construct CNN feature maps in multi-layer.
        Use pre-trained VGG19 or ResNet50
        """
        if backbone == 'vgg':
            cnn = VGG19(weights='imagenet', include_top=False, input_shape=self.imgColorL.shape)
        else:
            cnn = ResNet50(weights='imagenet', include_top=False, input_shape=self.imgColorL.shape)
        # print(vgg.summary())
        
        # List of extracted features
        self.featuresL = []
        self.featuresR = []
        
        # Preprocess the input images
        imgL = cv2.cvtColor(self.imgColorL, cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(self.imgColorR, cv2.COLOR_BGR2RGB)
        xL = np.expand_dims(imgL, axis=0)
        xR = np.expand_dims(imgR, axis=0)
        if backbone == 'vgg':
            xL = vgg_preprocess(xL)
            xR = vgg_preprocess(xR)
            layerName = ['block1_conv2', 'block2_conv2', 'block3_conv4' , 'block4_conv4', 'block5_conv4']
            self.layerW = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            xL = resnet_preprocess(xL)
            xR = resnet_preprocess(xR)
            layerName = ['conv2_block3_out']#['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
            self.layerW = [1.0, 0, 0, 0]
        
        # Extract features from these layers
        for l in range(len(layerName)):
            model = Model(inputs=cnn.input, outputs=cnn.get_layer(layerName[l]).output)
            featL = model.predict(xL)
            featR = model.predict(xR)
            featL = np.squeeze(featL, axis=0)
            featR = np.squeeze(featR, axis=0)
            if featL.shape[2] <= 512:
                featL = cv2.resize(featL, dsize=(imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_CUBIC)
                featR = cv2.resize(featR, dsize=(imgR.shape[1], imgR.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                fL = np.empty((imgL.shape[0], imgL.shape[1], featL.shape[2]))
                fR = np.empty((imgR.shape[0], imgR.shape[1], featR.shape[2]))
                for j in range(featL.shape[2]):
                    fL[:, :, j] = cv2.resize(featL[:, :, j], dsize=(imgL.shape[1], imgL.shape[0]))
                    fR[:, :, j] = cv2.resize(featR[:, :, j], dsize=(imgR.shape[1], imgR.shape[0]))
                featL = fL
                featR = fR
            #normMin = min(featL.min(), featR.min())
            #normMax = max(featL.max(), featR.max())
            #featL = 255.0 * (featL - normMin) / (normMax - normMin)
            #featR = 255.0 * (featR - normMin) / (normMax - normMin)
            self.featuresL.append(featL)
            self.featuresR.append(featR)
        
        return self.featuresL, self.featuresR
    
    def read_featureMap_img(self,featureL,featureR):
        self.img_L = featureL
        self.img_R = featureR
        
        return
