# -*- coding: utf-8 -*-
"""Mental imagery and sense of brightness of words experiment using pupillometry.

Created on Mon Nov  7 15:15:13 2022

@author: Claire Vanbuckhave
"""

# Import the useful packages
from extra.my_functions import RunExperiment
import os
from psychopy import gui, core
import pandas as pd
import webbrowser

################### begin part to modify
# Working directory
cwd=os.getcwd()
print('Current working directory:', cwd)
#cwd = 'C:/Users/Alan/Desktop/pupilometry-main'
# Input folders
stimfolder=cwd +'/stim/'
imagesfiles=cwd +'/__newpool__/'
# Output folders
datafolder=cwd +'/data/'
# Limesurvey link
url="https://enquetes-screen.msh-alpes.fr/index.php/427199?lang=fr"
################### end part to modify

## ------- Get files variables
trainfile, words_involuntaryfile, words_voluntaryfile = 'words_involuntary-train.csv', 'words_involuntary.csv', 'words_voluntary.csv'
images=os.listdir(imagesfiles)
print(images, len(images))

training = pd.read_table(stimfolder + trainfile, sep = ',')
involuntary = pd.read_table(stimfolder + words_involuntaryfile, sep = ',')
voluntary = pd.read_table(stimfolder + words_voluntaryfile, sep = ',')

print(len(involuntary) + len(training), len(voluntary))

## ------- Create dict with words
training_stim = dict()
involuntary_stim = dict()
voluntary_stim = dict()

## ---- Add words
## Training (involuntary)
for row in range(0, len(training)):
    # Initialize vars
    correct_resp = training.loc[row][0]
    word = training.loc[row][1] 
    condition = training.loc[row][2]
    
    # Add to dict
    if condition not in training_stim.keys():
        training_stim[word] = {condition: correct_resp}
    else:
        training_stim[word].update({condition: correct_resp})

## Involuntary
for row in range(0, len(involuntary)):
    # Initialize vars
    correct_resp = involuntary.loc[row][0]
    word = involuntary.loc[row][1]
    condition = involuntary.loc[row][2]
    
    # Add to dict 
    if condition not in involuntary_stim.keys():
        involuntary_stim[word] = {condition: correct_resp}
    else:
        involuntary_stim[word].update({condition: correct_resp})

## Voluntary
for row in range(0, len(voluntary)):
    # Initialize vars
    text = voluntary.loc[row][0]
    condition = voluntary.loc[row][1]
    
    # Add to dict
    if text not in voluntary_stim.keys():
        voluntary_stim[text] = condition
    else:
        voluntary_stim[text].update(condition)
print(voluntary_stim.keys())
n_stim_total=len(voluntary_stim)+len(involuntary_stim)+len(training_stim)
print(f'Trials train: {len(training_stim)}\nTrials vol: {len(voluntary_stim)}\nTrials invol: {len(involuntary_stim)}\nTotal number of trials: {n_stim_total}')

## ---- Create trial_id table
all_stim=[]
[all_stim.append(key) for key in training_stim.keys()]
[all_stim.append(key) for key in involuntary_stim.keys()]
[all_stim.append(key) for key in voluntary_stim.keys()]
df_stim=pd.DataFrame(data=all_stim, index=range(1, n_stim_total+1), columns=['stim'])
df_stim.to_csv(cwd+'/edf_stim_codes.csv') # Save it

im=[imag.split('.png')[0] for imag in images]
print(im)
for i in all_stim:
    if i in im:
        pass
    else:
        print(i)

## ----- Run the experiment
MODE='exp' # if testing, choose MODE='short' for only 2+5 trials for the involuntary task and 1+2 trials for the voluntary task, else use 'exp'
RunExperiment(datafolder, training_stim, involuntary_stim, voluntary_stim, MODE, all_stim)

## ----- Redirect to questionnaire
webbrowser.open(url)  # Go to LimeSurvey questionnaire  