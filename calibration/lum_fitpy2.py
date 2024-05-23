# -*- coding: utf-8 -*-
"""
Find the digital values associated with the desired luminance by doing a model fitting.
Created on Thu Mar  9 15:53:59 2023

@author: Claire
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd

################### begin part to modify
# Working directory
cwd=os.getcwd() # automatic
cwd='D:/pupilometry-main/' # manual
cwd='C:/Users/cvanb/Documents/pupilometry-main'
print('Current working directory:', cwd)

# Input folder
datafolder=cwd+'/calibration/'
################## end part to modify
# Load measured luminance in candela by digital value (0 to 255)
data = pd.read_csv(datafolder+'data_calib_screen2.csv')

x_data = list(data['DV']) # extract the digital values
y_data = list(data['Lum']) # extract the associated lum values (candela)

# Transform to float numbers 
#y_data = [float(y.replace(',', '.')) for y in y_data]

# Integers
x_data = [int(x) for x in x_data]

# Plot the values
plt.figure(figsize=(10,8))
plt.scatter(x_data , y_data);plt.title('The measured values', fontsize=20)
plt.xlabel('Digital values (from 0 = pure black, 255 = pure white)', fontsize=16);plt.ylabel('Luminance in candela (cd/m²)', fontsize=16)
plt.grid()
plt.show()

##################### Find the parameter values
# Step 1: define the model function
def model(x, A, gamma):
  return A*(x**gamma)

# Step 2: get the estimated optimized value for A and gamma (and covariance, but we don't care here)
opt, cov = curve_fit(model, x_data, y_data)

# popt: estimated optimized value for A and gamma
A,gamma=opt[0],opt[1]
print('A =', A, 'Gamma =', gamma)

#################### Test the model
x_model = np.arange(min(x_data), max(x_data))
y_model = model(x_model, A, gamma) 

plt.figure(figsize=(15,8))
plt.subplot(1,2,1);plt.scatter(x_data, y_data, color='b');plt.plot(x_model, y_model, color='b', label=r'$y = Ax^{Gamma}$')
plt.xlabel('Digital values (from 0 = pure black, 255 = pure white)', fontsize=16);plt.ylabel('Luminance in candela (cd/m²)', fontsize=16)
plt.grid()
plt.legend(fontsize=16)

################### Reverse the function 
def compute_digit(candela, A, gamma):
    return (candela/A)**(1/gamma)

################### Compute the exact digital values associated with the desired lum
# Method 1: round to the nearest integer the exact value
opt_txt = compute_digit(8.5, A, gamma)
opt_bckg = compute_digit(17.4, A, gamma)

print(f'The digital value for the background should be {round(opt_bckg)} and {round(opt_txt)} for the text.')

# Method 2: Manually select the best value (allowed difference of 0.2)
for y, x in zip(y_model, x_model):
    if 17.2 < y < 17.6: # we are looking for 17.4
        print(y, x)
        
for y, x in zip(y_model, x_model):
    if 8.3 < y < 8.7: # we are looking for 8.5
        print(y, x)

################### Test the model 
y_points = np.linspace(0, 256, 256)
x_points = compute_digit(y_points, A, gamma) 

################### Plot it
plt.subplot(1,2,2);plt.scatter(y_data, x_data, color='r');plt.plot(y_points, x_points, color='r', label=r'$x = (y/A)^{1/Gamma}$')
plt.grid()
plt.ylabel('Digital values (from 0 = pure black, 255 = pure white)', fontsize=16);plt.xlabel('Luminance in candela (cd/m²)', fontsize=16)
plt.suptitle('Relationship between digital value (rgb255) and the screen luminance (cd/m²)\n', fontsize=20)
plt.legend(fontsize=16)
plt.show()
