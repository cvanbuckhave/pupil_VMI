# -*- coding: utf-8 -*-
"""
Modify the original stimuli to match the desired luminance.
Created on Thu Mar  9 17:13:47 2023

@author: Claire
"""

# Import the useful packages
import os
import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.util import img_as_ubyte

################### begin part to modify
# Working directory
cwd=os.getcwd() # auto
cwd = 'D:\pupilometry-main' # manual
cwd='C:/Users/cvanb/Documents/pupilometry-main'
print('Current working directory:', cwd)

# Input folders
imagesfiles=cwd +'/__pool__/'
# Output folders
datafolder=cwd +'/__newpool2__/'
################### end part to modify

## ------- Get files variables
images=os.listdir(imagesfiles)
print('Number of stimuli:', len(images))

newtxt, newbckg = 73, 100
## ------- Do some image processing
# For each stimulus-image
for image in images:
    img=skio.imread(imagesfiles+image) # get the image
    new_img=np.copy(img) # get a copy of the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j][0]<86: # ensure all 85 values are coded as the new value
                new_img[i,j]=img[i,j]*(newtxt/85)
            else: # ensure all 119 values are coded as the new value
                new_img[i,j]=img[i,j]*(newbckg/119)
                
     # Save the transformed image into the new stim folder           
    skio.imsave(datafolder+image, img_as_ubyte(new_img))

# Display the last image to see the result
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1);plt.imshow(img);plt.grid(False)
plt.subplot(1, 2, 2);plt.imshow(new_img);plt.grid(False)
plt.show()

