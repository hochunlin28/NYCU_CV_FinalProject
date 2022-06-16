# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:20:29 2022

@author: Hochun
"""

from SAD import sad
import numpy as np
import cv2
import matplotlib.pyplot as plt

maxd = 60
sad = sad()

sad.read_img("./img/imgL2.png","./img/imgR2.png", maxd)
print(sad.img_L.shape)

featuresL, featuresR = sad.feature_extract()
print("feature extract done")
print(featuresL[0].shape[2])
layer = 2

featuresL_avg = np.zeros((sad.img_L.shape[0],sad.img_L.shape[1]))
featuresR_avg = np.zeros((sad.img_L.shape[0],sad.img_L.shape[1]))
for map_size in range(featuresL[0].shape[2]):
    featuresL_avg = featuresL_avg + featuresL[layer][:,:,map_size]
    featuresR_avg = featuresR_avg + featuresR[layer][:,:,map_size]
    
featuresL_avg = featuresL_avg / featuresL[0].shape[2]
featuresR_avg = featuresR_avg / featuresL[0].shape[2]

# extract feature map for input
#sad.read_featureMap_img(np.uint8(featuresL_avg), np.uint8(featuresR_avg))


disparity = np.zeros((sad.img_L.shape[0],sad.img_L.shape[1]))
disparity_r = np.zeros((sad.img_L.shape[0],sad.img_L.shape[1]))


# calculate matching cost
for d in range(maxd):
    for x in range(3, sad.img_L.shape[0]-3):
        for y in range(3,sad.img_L.shape[1]-4):
            if y+d+5 < sad.img_L.shape[1]:
                #print(x,y,d)
                cost = abs(sad.pixel_SAD_left(x, y, d))
                cost_r = abs(sad.pixel_SAD_right(x, y, d))
                sad.disparity_depth[d][x][y] = cost
                sad.disparity_depth_r[d][x][y] = cost_r


# bilateral filter

for d in range(maxd):
    sad.disparity_depth[d,:,:] = cv2.bilateralFilter(sad.disparity_depth[d,:,:].astype('uint8'),7,41,41)
    #sad.disparity_depth_r[d,:,:] = cv2.bilateralFilter(sad.disparity_depth_r[d,:,:].astype('uint8'),7,41,41)

#WTA
for x in range(3, sad.img_L.shape[0]-3):
    for y in range(3,sad.img_L.shape[1]-4):
        disparity[x][y] = np.argmin(sad.disparity_depth[:,x,y])
        #disparity_r[x][y] = np.argmin(sad.disparity_depth_r[:,x,y])

plt.imshow(disparity)
plt.show()



#Disparity refinement
disparity = cv2.medianBlur(np.uint8(disparity), 7) 
disparity_r = cv2.medianBlur(np.uint8(disparity_r), 7) 
disparity = cv2.bilateralFilter(np.uint8(disparity), 11, 50,50)

# hole filling
'''
for x in range(sad.img_L.shape[0]):
    for y in range(sad.img_L.shape[1]):
        if(disparity[x][y] != disparity_r[x][int(y - disparity[x][y])]):
            
            disparity[x][y] = -1

plt.imshow(disparity)
plt.show()
'''




max_value = np.amax(disparity)
print(max_value)
disparity = disparity * 255.0 / max_value 
plt.imshow(disparity)
plt.show()

disparity = disparity.astype('uint8')   
cv2.imshow("123",disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
