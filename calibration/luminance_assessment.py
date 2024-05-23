# -*- coding: utf-8 -*-
"""
Display different colored screen (digit values) to assess the luminance of 
the colors on the screen (DELL monitor, see Methods).
Created on Thu Mar  9 15:30:17 2023

We ran this code and used a SpyderElite 5 calibration device to measure the 
associated luminance in candela, and wrote the results in a .csv file 
('data_calib_screen.csv')

@author: Claire
"""

## Import libraries
from psychopy import visual, core, event
import numpy as np
from win32api import GetSystemMetrics
import os
from random import sample

## Set the background and text colors to display
bck_119=119 # hex code #777777 (Sonic Silver) = hue of 359°, 0% saturation and a brightness value of 47%.
txt_85=85# hex code #555555 (Davy's Grey) = hue of 359°, 0% saturation and a brightness value of 33%.
gray_255=list(np.arange(0, 255, 10))
gray_255.append(255)
print((gray_255))

#################### begin part to modify
stim_folder='E:/pupilometry-main/__newpool__/' # Change the path to assess the luminance of the transformed stim after calibration
#################### end part to modify

all_stim=os.listdir(stim_folder)

## Define the functions to show the windows and colors
resolution=[GetSystemMetrics(0), GetSystemMetrics(1)] # lab = [2560, 1600]

def window(resolution, col):
    """Define the window where screens appear."""
    fullScreen=True
    win=visual.Window(resolution, units="pix", color=col, colorSpace='rgb255', fullscr=fullScreen, monitor="testMonitor", allowGUI=False)
    win.setMouseVisible(False)
    return win

def color_rectangle(win, color, resolution):
    """Define the color screens."""
    col=visual.Rect(win, fillColor=color, colorSpace='rgb255', width=resolution[0], height=resolution[1])
    col.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
            core.quit()
    print(color, 'OK')

def show_stim(win, path, stim):
    word=visual.ImageStim(win, image=path+stim)
    word.draw()
    win.flip()
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
            core.quit()
    print(stim, 'OK')
    
## Measure the luminance in cd/m²
win=window(resolution, bck_119)

#Uncomment to assess the original parameters
print('Start testing background (119) and text colors (85):')
for color in [bck_119, txt_85]:
    color_rectangle(win, color, resolution)
    
print('Start testing more gray levels:')
for color in gray_255:
    color_rectangle(win, color, resolution)

# After calibration:
# bck_76=76
# txt_55=55
# n=5

# win=window(resolution, bck_76)

# print('Start testing background (76) and text colors (55):')
# for color in [bck_76, txt_55]:
#     color_rectangle(win, color, resolution)

# print(f'Start testing {n} random stimuli')
# to_test=sample(all_stim, k=n)
# for stim in to_test:
#     show_stim(win, stim_folder, stim)