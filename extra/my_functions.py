# -*- coding: utf-8 -*-
"""Functions used for the experiment [title].

Created on Mon Nov  7 15:18:05 2022

@author: Claire Vanbuckhave
"""

# Import packages
import csv
import os
from psychopy.iohub.util import hideWindow, showWindow
from psychopy import visual, core, event, gui, hardware
from psychopy import data as dt
import datetime # Used to register the date of the experiment
import pandas as pd # Used to save the data as csv easily
from win32api import GetSystemMetrics
from random import sample
import numpy as np
import psychopy.iohub as io
from psychopy.iohub import launchHubServer
from psychopy.hardware import keyboard

## ------ Setting constants parameters 
## Some colours in RGB space
gray=[128, 128, 128]
black=[0, 0, 0]
white=[255, 255, 255]
red=[120, 0, 0] # hex code #780000
green=[0, 85, 0] # hex code #005500

## Window parameters
resolution=[GetSystemMetrics(0), GetSystemMetrics(1)] # lab = [2560, 1600]
fontsize=28 # in pixels
ft='DroidSansMono.ttf' # font (police d'écriture)
ft_file=[os.getcwd()+"/extra/"+ft] # Where to find the font
print(ft_file)
bck_col=76 # hex code #777777 (Sonic Silver) = hue of 359°, 0% saturation and a brightness value of 47%.
ft_col=55 # hex code #555555 (Davy's Grey) = hue of 359°, 0% saturation and a brightness value of 33%.
print(f'Screen resolution: {resolution}\nFont: {ft}\nFontsize: {fontsize} px\nBackground color: {bck_col}')

## Other useful stuff
cwd=os.getcwd()
imagesfiles=cwd +'/__newpool__/'

def window(resolution):
    """Define the window where screens appear."""
    fullScreen=True
    win=visual.Window(resolution, units="pix", color=bck_col, colorSpace='rgb255', fullscr=fullScreen, monitor="testMonitor", allowGUI=False)
    win.setMouseVisible(False)
    return win

def InstructionsTrain(win, eyetracker):
    """Show the instructions for training phases."""
    msg_text=visual.TextStim(win, text="Faisons un entraînement. \n\n Veuillez appuyer sur la barre d'espace pour débuter l'entraînement.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    msg_text.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(secs=0.1) # give it some time
                print('Eyetracker disconnected')
                core.quit()

def EndTrain(win, eyetracker):
    """Show text at the end of the training."""
    msg_text=visual.TextStim(win, text="L'entraînement est terminé, nous allons maintenant passer à la véritable phase de l'expérience. \n\n Veuillez appuyer sur la barre d'espace pour débuter l'expérience.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert= 'center', anchorHoriz = 'center')
    msg_text.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(secs=0.1) # give it time
                print('Eyetracker disconnected')
                core.quit()
    
def InstructionsVol(win, eyetracker):
    """Show instructions for voluntary imagery task."""
    msg_text=visual.TextStim(win, text="Dans cette partie, une scène familière vous sera décrite brièvement. Puis, une fois que vous aurez la scène en tête, vous aurez alors 7 secondes pour imaginer la scène aussi précisément et clairement que possible. \n\n Appuyez sur la barre d'espace pour continuer.", height=fontsize, font=ft, fontFiles=ft_file, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    msg_text.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(secs=0.1) # give some time for initialization
                print('Eyetracker disconnected')
                core.quit()

def EndScreen(win, eyetracker):
    """Show text at the end of the experiment."""
    msg=visual.TextStim(win, text="La partie expérimentale de cette session est maintenant terminée. \n\n Appuyez sur la barre 'espace' pour répondre à un rapide questionnaire.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    msg.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", 'escape'])
    if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(secs=0.1) # give some time for initialization
                print('Eyetracker disconnected')
                core.quit()

def Involuntary(win, resolution, ioServer, stimuli, participant, phase, train, eyetracker, MODE, all_stim):
    """Do the involuntary imagery task."""
    trialClock = core.Clock() # always keep track of time
    stimOnset = core.Clock()  # to record response time
    data=[]
    
    ## Initialize number of trials and stimuli
    if train == True: # if training phase
        InstructionsTrain(win, eyetracker) # show instructions training
        if MODE=='short':
            ntrials=2 # number of train trials is only 2
        else:
            ntrials=len(stimuli.keys()) # all training trials
    elif train == False: # if not
        EndTrain(win, eyetracker) # show end of training text
        if MODE=='short':
            ntrials=5 # only five trials
        elif type(MODE)==int:
            ntrials=MODE # only n trials
        else:
            ntrials=len(stimuli.keys()) # all trials
        
    word_stimuli=sample(list(stimuli.keys()), k=ntrials) # randomly select order, add k=i for only i trials (testing)
    n=0 # count n stim (for pause)
    
    if train == False: 
        eyetracker.setRecordingState(True) # Start recording of pupil data
        core.wait(secs=0.1) # give some time for initialization
        
    ## ---- For each trial
    for stim in word_stimuli:
        correct=list(stimuli[stim].values())[0] # the correct answer 
        condition=list(stimuli[stim].keys())[0] # dark, light, animal or ctrl
        n+=1 # add one to stim count
        
        trial_id=all_stim.index(stim)+1 # trial id is the row index + 1
        
        ## Do drift correction before each trial
        DriftCorrection(eyetracker, win, resolution, ioServer)
        
        if train == False: 
            eyetracker.setRecordingState(True) # Start recording of pupil data
            core.wait(secs=0.1) # give some time for initialization
        
        ## ------ START TRIAL
        if train == False:
            eyetracker.sendMessage(f'start_trial {trial_id}') # start trial
            eyetracker.sendMessage(f'var participant {participant}') # the participant's ID
            eyetracker.sendMessage(f'var experiment {phase}') # start of involuntary task
            eyetracker.sendMessage(f'var condition {condition}') # dark, light, animal or ctrl 
        
        ### ----- Start baseline with Fixation cross 
        msg='start_phase baseline'
        fixation=visual.TextStim(win, text="o", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
        fixation.draw()
        win.flip()
        trialClock.reset()
        
        if train == False: 
            eyetracker.sendMessage(msg) # start baseline
        
        time=trialClock.getTime()
        data.append([participant, time, None, stim, condition, None, msg, phase])

        core.wait(secs=3) # wait 3 seconds
        
        msg='end_phase baseline'
        if train == False: 
            eyetracker.sendMessage(msg) # end baseline
        
        time=trialClock.getTime()
        data.append([participant, time, None, stim, condition, None, msg, phase])
    
        ### -------- End baseline, start test: Show word and wait for key presses or wait 3 seconds        
        msg='start_phase imagery'
        path_to_image_file=imagesfiles+stim+'.png'
        word=visual.ImageStim(win, image=path_to_image_file)
        word.draw()
        win.flip()
        stimOnset.reset()
        
        if train == False: 
            eyetracker.sendMessage(msg) # start imagine
        
        time=trialClock.getTime()
        data.append([participant, time, None, stim, condition, correct, msg, phase])

        keys=event.waitKeys(keyList=['space', 'escape', 'c'], maxWait=float(3)) # wait 3 seconds max for keypress
        keypress_time=stimOnset.getTime()
        ## Save the time of keypress for animal names 
        if condition=='animal':
            eyetracker.sendMessage(f'var tkeypress {keypress_time}') # save the time of the keypress
        
        msg='end_phase imagery'
        if train == False: 
            eyetracker.sendMessage(msg) # end imagine
        
        time=trialClock.getTime()
                
        if keys: # if a key was pressed
            keys=keys[0] # get the first element
            if keys=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(secs=0.1) # wait for it
                print('Eyetracker disconnected')
                core.quit()
            if keys=='c':
                print('Run setup procedure')
                hideWindow(win) # Minimize the PsychoPy window if needed
                result=eyetracker.runSetupProcedure() # Display calibration gfx window and run calibration.
                print("Calibration returned: ", result)
                showWindow(win) # Maximize the PsychoPy window if needed
                print('End setup procedure')
                win.clearBuffer() # clear
            else: # if it wasn't the escape key
                data.append([participant, time, keys, stim, condition, correct, msg, phase])
        else: # if not
            keys='None' # key is None
            data.append([participant, time, keys, stim, condition, correct, msg, phase])

        ### ------ End imagine, Give visual feedback
        if train == False: eyetracker.sendMessage('start_phase feedback') # start feedback
        if keys == correct:
            feedback=visual.TextStim(win, text="o", font=ft, fontFiles=ft_file, height=fontsize, color=green, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
            if train == False: eyetracker.sendMessage('var feedback correct') # right answer 
        else:
            feedback=visual.TextStim(win, text="o", font=ft, fontFiles=ft_file, height=fontsize, color=red, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
            if train == False: eyetracker.sendMessage('var feedback incorrect') # wrong answer
            
        feedback.draw()
        win.flip()
        
        core.wait(secs=0.5) # show feedback for 0.5 seconds
        
        if train == False:
            eyetracker.sendMessage('end_phase feedback') # end feedback
            eyetracker.sendMessage('end_trial') # end trial
        
        ## ----- END TRIAL

        ## PAUSE
        if n in [25, 50, 75, 100]: # Check for the stim count, to propose a pause
            eyetracker.setRecordingState(False) # Stop recording pupil data
            core.wait(secs=0.1) # wait for instruction to reach ET
            
            pause=visual.TextStim(win, text="Temps de pause. \n\nLorsque vous êtes prêt.e à continuer, appuyez sur la barre d'espace de votre clavier. ", fontFiles=ft_file, font=ft, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
            pause.draw()
            win.flip()
            
            keys=event.waitKeys(keyList = ['space', 'escape'])
            if keys[0]=='escape':
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(secs=0.1) # wait for instruction to reach ET
                print('Eyetracker disconnected')
                core.quit()
                
            if n == 50:
                ## RUN SET UP PROCEDURE
                instr=visual.TextStim(win, text='Appuyez sur la barre "espace" pour refaire une calibration.', font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
                instr.draw()
                win.flip()
                keys=event.waitKeys(keyList = ['space', 'escape'])
                if keys[0]=='escape':
                    eyetracker.setConnectionState(False) # Close connection to eyetracker device
                    core.wait(secs=0.1) # wait for instruction to reach ET
                    print('Eyetracker disconnected')
                    core.quit()
                print('Run setup procedure')
                hideWindow(win) # Minimize the PsychoPy window if needed
                result=eyetracker.runSetupProcedure() # Display calibration gfx window and run calibration.
                print("Calibration returned: ", result)
                showWindow(win) # Maximize the PsychoPy window if needed
                print('End setup procedure')
                win.clearBuffer() # clear
                
            if train == False:
                eyetracker.setRecordingState(True) # Start recording pupil data
                core.wait(secs=0.1) # wait for instruction to reach ET

            
        else:
            pass
    
    if train == False: 
        eyetracker.setRecordingState(False) # Stop recording pupil data
        core.wait(secs=0.1) # wait for instruction to reach ET
    
    return data

def Voluntary(win, resolution, ioServer, stimuli, participant, phase, train, eyetracker, MODE, all_stim):
    """Define the screen.
    In psychopy you define the window where all the screens are going to be displayed like this
    """
    trialClock=core.Clock() # keep track of the time
    data=[]
    
    ## Initialize number of trials and stimuli to show
    if train == True: # if training mode
        Texts=['Vous regardez un arbre fleuri.']
        instr=visual.TextStim(win, text='Très bien. \n\n Nous allons passer à la seconde partie de cette expérience.\n\nAppuyez sur la barre "espace" pour refaire une calibration.', font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
        instr.draw()
        win.flip()
        
        keys=event.waitKeys(keyList=['space', 'escape'])
        if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(0.1) # wait a bit
                print('Eyetracker disconnected')
                core.quit()
        
        ## RUN SET UP PROCEDURE
        print('Run setup procedure')
        hideWindow(win) # Minimize the PsychoPy window if needed
        result=eyetracker.runSetupProcedure() # Display calibration gfx window and run calibration.
        print("Calibration returned: ", result)
        showWindow(win) # Maximize the PsychoPy window if needed
        print('End setup procedure')
        win.clearBuffer() # clear
        
        InstructionsTrain(win, eyetracker) # show instructions for training
        InstructionsVol(win, eyetracker) # show instructions for voluntary task
        
    elif train == False: # if not training mode
        if MODE=='short':
            ntrials=2 # only 2 trials
        elif type(MODE)==int:
            ntrials=MODE # only n trials
        else:
            ntrials = len(stimuli.keys()) # all trials
            
        Texts=sample(list(stimuli.keys()), k=ntrials) # randomly select order, add k=i for only i trials (testing)
        
        EndTrain(win, eyetracker) # show end of training phase text
        #InstructionsVol(win, eyetracker) # show instructions for voluntary task again
    
        eyetracker.setRecordingState(True) # Start recording pupil data
        core.wait(0.1) # give time to initiate the ET
    
    msg_text=visual.TextStim(win, text="Appuyez sur la barre d'espace pour lire le scénario à imaginer. Lorsque vous aurez la scène en tête, appuyez de nouveau sur la barre d'espace pour commencer à imaginer.\n\n Veuillez garder les yeux ouverts et fixer la cible centrale pendant que vous pensez au scénario.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    msg_text.draw()
    win.flip() # Elements are only displayed after the flip command is executed

    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
            eyetracker.setConnectionState(False) # Close connection to eyetracker device
            core.wait(secs=0.1) # give time to initiation of ET
            print('Eyetracker disconnected')
            core.quit()
                
    ## ----- For each trial:
    for Text in Texts:
        ## Do drift correction before each trial
        DriftCorrection(eyetracker, win, resolution, ioServer)
        
        if train == False: 
            eyetracker.setRecordingState(True) # Start recording of pupil data
            core.wait(secs=0.1) # give some time for initialization
        
        txt='-'.join(Text.split()) # to save the exact stim in the csv file, join the words with '-'
        
        if train == True: # Detect if training phase or not
            condition='ctrl' 
            trial_id=len(all_stim)+1 # trial id is n_total_stim + 1
        else: 
            condition=stimuli[Text] # dark or light
            trial_id=all_stim.index(Text)+1 # trial id is row index + 1 
        
        ## ---- START TRIAL
            eyetracker.sendMessage(f'start_trial {trial_id}') # start trial
            eyetracker.sendMessage(f'var participant {participant}') # the participant's ID
            eyetracker.sendMessage(f'var experiment {phase}') # voluntary imagery task
            eyetracker.sendMessage(f'var condition {condition}') # dark or light
        
        ### ----- Start baseline with fixation cross
        msg='start_phase baseline'
        fixation=visual.TextStim(win, text="o", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
        fixation.draw()
        win.flip()
        
        trialClock.reset()
        
        if train == False:
            eyetracker.sendMessage(msg) # start baseline
        
        msg='end_phase baseline'
        time=trialClock.getTime()
        data.append([participant, time, None, txt, condition, None, msg, phase])

        core.wait(secs=1) # wait for 1 second
        
        if train == False: 
            eyetracker.sendMessage(msg) # end baseline
        
        time=trialClock.getTime()
        data.append([participant, time, None, txt, condition, None, msg, phase])
        
        ### ------- End baseline, start reading the text
        msg='start_phase read'
        text=visual.TextStim(win, text=Text, font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
        text.draw()
        win.flip()
        
        if train == False: 
            eyetracker.sendMessage(msg) # start reading
        
        time=trialClock.getTime()
        data.append([participant, time, None, txt, condition, None, msg, phase])

        keys=event.waitKeys(keyList = ["space", "escape"]) # wait until keypress
        if keys[0]=='escape':
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(0.1) # give time to initiation of ET
                print('Eyetracker disconnected')
                core.quit()
        
        msg='end_phase read'
        time=trialClock.getTime()
        data.append([participant, time, None, txt, condition, None, msg, phase])
        
        if train == False: 
            eyetracker.sendMessage(msg) # end reading
        
        ### -------- End reading, Imagine for 7secs
        msg='start_phase imagery'
        fixation.draw()
        win.flip()
        
        if train == False: 
            eyetracker.sendMessage(msg) # start imagine
        
        time=trialClock.getTime()
        data.append([participant, time, None, txt, condition, None, msg, phase])

        core.wait(secs=7) # wait for 7 seconds
        
        msg='end_phase imagery' 
        stop=visual.TextStim(win, text='stop', font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
        stop.draw()
        win.flip()
        
        if train == False: 
            eyetracker.sendMessage(msg) # end imagine
        
        time=trialClock.getTime()
        data.append([participant, time, None, txt, condition, None, msg, phase])
        
        core.wait(0.5) # show stop text for 0.5 seconds
        
        ## Rate the effort of imagining the scenario 
        rating = RateImagery(win)
        
        if train == False: 
            eyetracker.sendMessage(f'var rating {rating}')
            eyetracker.sendMessage('end_trial') # end of the trial
        
        ## ----- END TRIAL

    if train == False: 
        eyetracker.setRecordingState(False) # Stop recording pupil data
        core.wait(0.1) # wait for it
    
    return data    

def Color_rectangles(win, eyetracker, participant, resolution, ioServer):
    """Define the color screens."""
    
    bl=visual.Rect(win, fillColor=bck_col, colorSpace='rgb255', width=resolution[0], height=resolution[1])

    colors=[black, white, green, red]
    ids=[999, 998, 997, 996]
    conds=['dark', 'light', 'ctrl', 'ctrl']
    phase='luminance'
    
    msg_text=visual.TextStim(win, text="Très bien.\n Nous allons terminer l'expérience par une simple mesure de votre réponse pupillaire à la lumière. Veuillez garder les yeux au centre de l'écran et fixer l'écran, puis appuyez sur la barre ESPACE pour commencer.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    msg_text.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(0.1) # wait for connection to close properly
                print('Eyetracker disconnected')
                core.quit()
    
    eyetracker.setRecordingState(True) # Start recording pupil data
    core.wait(0.1) # give time to initiate the ET
    
    for color, trial_id, condition in zip(colors, ids, conds):
        for i in range(0, 3):
            ## Do drift correction before each trial
            DriftCorrection(eyetracker, win, resolution, ioServer)
            
            ## ---- START TRIAL
            eyetracker.sendMessage(f'start_trial {trial_id}') # start trial
            eyetracker.sendMessage(f'var participant {participant}') # the participant's ID
            eyetracker.sendMessage(f'var experiment {phase}') # luminance assessment
            eyetracker.sendMessage(f'var condition {condition}') # dark or light
            
            bl.draw()
            win.flip() # Elements are only displayed after the flip command is executed
            
            eyetracker.sendMessage('start_phase gray') # send message to eyetracker
            
            core.wait(3)
            
            eyetracker.sendMessage('end_phase gray') # send message to eyetracker
            
            col=visual.Rect(win, fillColor=color, colorSpace='rgb255', width=resolution[0], height=resolution[1])
            col.draw()
            win.flip() # Elements are only displayed after the flip command is executed
            
            eyetracker.sendMessage('start_phase lum') # send message to eyetracker
            
            core.wait(3)
            
            eyetracker.sendMessage('end_phase lum') # send message to eyetracker
            
            eyetracker.sendMessage('end_trial')
    
    eyetracker.setRecordingState(False) # Stop recording pupil data
    core.wait(0.1) # wait for it

    
def StartScreen(win, eyetracker):
    """Show start screen with welcome text and instructions."""
    msg_text=visual.TextStim(win, text="Bienvenue. \n\n Lors de cette expérience, vous verrez des mots s'afficher à l'écran. Le but est d'appuyer sur la barre d'espace lorsque vous verrez des noms d'animaux s'afficher et de n'appuyer sur aucune touche dans le cas contraire.\n\n Appuyez sur la barre 'espace' pour continuer.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    msg_text.draw()
    win.flip() # Elements are only displayed after the flip command is executed
    
    keys=event.waitKeys(keyList = ["space", "escape"])
    if keys[0]=='escape': # check for escape key
                eyetracker.setConnectionState(False) # Close connection to eyetracker device
                core.wait(0.1) # wait for connection to close properly
                print('Eyetracker disconnected')
                core.quit()

def startScreensAndRecordData(win, resolution, ioServer, expInfo, training_stim, involuntary_stim, voluntary_stim, eyetracker, MODE, all_stim):
    """Start screens and record data."""
    win.clearBuffer() # clear
    
    StartScreen(win, eyetracker) # Show start screen
    
    print('Start training involuntary')
    data_train1 = Involuntary(win, resolution, ioServer, training_stim, expInfo['participant'], 'training1', True, eyetracker, MODE, all_stim)
    print('Stop training involuntary')
    
    print('Start Involuntary experiment')
    data_invol = Involuntary(win, resolution, ioServer, involuntary_stim, expInfo['participant'], 'involuntary', False, eyetracker, MODE, all_stim)
    print('Stop Involuntary experiment')
    
    print('Start training voluntary')
    data_train2 = Voluntary(win, resolution, ioServer, voluntary_stim, expInfo['participant'], 'training2', True, eyetracker, MODE, all_stim)
    print('Stop training voluntary')
    
    print('Start Voluntary experiment')
    data_vol = Voluntary(win, resolution, ioServer, voluntary_stim, expInfo['participant'], 'voluntary', False, eyetracker, MODE, all_stim)
    print('Stop Voluntary experiment')
    
    print('Start Luminance assessment')
    Color_rectangles(win=win, resolution=resolution, ioServer=ioServer, participant=expInfo['participant'], eyetracker=eyetracker)
    print('Stop.')
    
    EndScreen(win, eyetracker) # Show end screen
    
    return data_train1, data_invol, data_train2, data_vol

def RateImagery(win):
    ratingScale = visual.RatingScale(win, low=-2, high=2, markerStart=0, scale=None, 
                    showAccept=True, acceptKeys=['return', 'space'], skipKeys=None, size=1.5,
                    textColor='DarkGray', textSize=0.5, tickMarks=range(-2, 3), noMouse=True,
                    labels=['Très faible', 'Faible', 'Ni faible ni élevé', 'Élevé', 'Très élevé'],
                    lineColor='DarkGray', acceptPreText='##', acceptText="Appuyez sur la barre d'espace pour valider.")
    item=visual.TextStim(win, text="Quel était le degré d'effort nécessaire pour imaginer ce scénario ?\n\nSélectionnez votre réponse avec les flèches du clavier, puis appuyez sur la barre d'espace pour valider.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    while ratingScale.noResponse:
        item.draw()
        ratingScale.draw()
        win.flip()
    rating = ratingScale.getRating()
    print('Rating =', rating)
    return rating

def InitializeEyeTracker(win, simulation_mode, TRACKER, expInfo, datafolder):
    """Set up eyetracking device from iohub."""
    filename=expInfo['fileName'] # Get useful variables from expInfo
    
    ## ---- Setup eyetracking
    if TRACKER == 'mouse': # Mouse simulation mode
        ioDevice='eyetracker.hw.mouse.EyeTracker'
        ioConfig={ioDevice: {'name': 'tracker',
                               'calibration': dict(auto_pace=True,
                                                target_duration=1.5,
                                                target_delay=1.0,
                                                screen_background_color=bck_col,
                                                type='FIVE_POINTS',
                                                unit_type=None,
                                                color_type=None
                                                                       )
                                                }}
        ioSession='1'
        if 'session' in expInfo:
            ioSession=str(expInfo['session'])
        ioServer=io.launchHubServer(window=win, experiment_code='experiment', session_code=ioSession, datastore_name="pupil_data/"+filename, **ioConfig)
        eyetracker=ioServer.getDevice('tracker')
    else: # If not, we're using SR Research Eyelink 1000Plus
        ioDevice='eyetracker.hw.sr_research.eyelink.EyeTracker'
        ioConfig={
            ioDevice: {
                'name': 'tracker',
                'model_name': 'EYELINK 1000 DESKTOP',
                'simulation_mode': simulation_mode,
                'network_settings': '100.1.1.1',
                'default_native_data_file_name': 'EXPFILE',
                'calibration': dict(auto_pace=True,
                    target_duration=1.5,
                    target_delay=1.0,
                    screen_background_color=bck_col,
                    type='FIVE_POINTS',
                    unit_type=None,
                    color_type=None,
                    ),
                'runtime_settings': {
                    'sampling_rate': 1000.0,
                    'track_eyes': 'RIGHT',
                    'sample_filtering': {
                        'sample_filtering': 'FILTER_LEVEL_OFF',
                        'elLiveFiltering': 'FILTER_LEVEL_2',
                    },
                    'vog_settings': {
                        'pupil_measure_types': 'PUPIL_AREA',
                        'tracking_mode': 'PUPIL_CR_TRACKING',
                        'pupil_center_algorithm': 'ELLIPSE_FIT',
                    }
                }
            }
        }
        ioSession='1'
        if 'session' in expInfo:
            ioSession=str(expInfo['session'])
        ioServer=io.launchHubServer(window=win, experiment_code='experiment', session_code=ioSession, datastore_name="pupil_data/"+filename, **ioConfig)
        eyetracker=ioServer.getDevice('tracker')
    
    defaultKeyboard=keyboard.Keyboard() # create a default keyboard (e.g. to check for escape)
    kb=ioServer.getDevice('keyboard')
    win.mouseVisible = False

    ## RUN SET UP PROCEDURE
    print('Run setup procedure')
    hideWindow(win) # Minimize the PsychoPy window if needed
    result=eyetracker.runSetupProcedure() # Display calibration gfx window and run calibration.
    print("Calibration returned: ", result)
    showWindow(win) # Maximize the PsychoPy window if needed
    print('End setup procedure')
    defaultKeyboard.clearEvents() # clear any keypresses from during calibration so they don't interfere with the experiment
    
    return eyetracker, defaultKeyboard, kb, ioServer

def DriftCorrection(eyetracker, win, resolution, ioServer):
    defaultKeyboard=keyboard.Keyboard() # create a default keyboard (e.g. to check for escape)
    kb=ioServer.getDevice('keyboard')
    x_pos, y_pos = resolution[0], resolution[1]
    win.clearBuffer() # clear
    trgt=visual.TextStim(win, text="+", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255', anchorVert='center', anchorHoriz='center')
    trgt.draw()
    win.flip()
    print('Do drift corr')
    params=[int(x_pos/2.0), int(y_pos/2.0), 0, 1]
    print('Parameters:', params)
    dc=eyetracker.sendCommand('doDriftCorrect', params)
    print('Drift corr done:', dc)
    #keys=event.waitKeys(keyList=["space", 'escape'])
    #if keys[0]=='escape': # check for escape key
    #    eyetracker.setConnectionState(False) # Close connection to eyetracker device
    #    core.wait(0.1) # wait for connection to close properly
    #    print('Eyetracker disconnected')
    #    core.quit()
    #print('Drift corr done:', dc)
    defaultKeyboard.clearEvents() # clear any keypresses from during calibration so they don't interfere with the experiment

    
def Init():
    """ Initialize the experiment."""
    ## Store info about the experiment session
    expName='BrightPupilExpMI'  # initialize experiment name
    expInfo={'participant': '', 'session': '001',}

    ## --- Show participant info dialog
    dlg=gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK==False:
        core.quit()  # user pressed cancel
    
    ## --- Initialize windows settings and exp info       
    win=window(resolution)
    ## Store framerate of monitor if we can measure it
    expInfo['frameRate']=win.getActualFrameRate()
    if expInfo['frameRate']!=None:
        expInfo['frameDur'] = 1.0 / round(expInfo['frameRate'])
    else:
        expInfo['frameDur'] = 1.0 / 60.0  # could not measure, so guess
    
    ## Create file name
    filename=expName + '_' + expInfo['participant'] + '_' + '_'.join(str(datetime.datetime.today()).split()).replace(':', '-') + '.csv'
    expInfo['fileName']=filename
    expInfo['date']=dt.getDateStr()  # add a simple timestamp
    expInfo['expName']=expName
    
    return win, expInfo

def new_participant(identifier, today_date, start_time, end_time, filename):
    duration = round((end_time - start_time)/60, 3)
    new = [identifier, filename, today_date, start_time, end_time, duration]
    
    # Create a file object for this file
    with open('participants.csv', 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = csv.writer(f_object, delimiter=';', quotechar='"')
     
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(new)
     
        # Close the file object
        f_object.close()
    return True

def RunExperiment(datafolder, training_stim, involuntary_stim, voluntary_stim, MODE, all_stim):
    """Put all together and save it. Finally we put everything together and save the file as a CSV using pandas.""" 
    ## --- Set up basic stuff
    resolution=[GetSystemMetrics(0), GetSystemMetrics(1)]
    win, expInfo=Init()
    filename=expInfo['fileName']
    ## --- First screen
    calib=visual.TextStim(win, text="Appuyez sur la barre d'espace pour commencer la calibration.", font=ft, fontFiles=ft_file, height=fontsize, color=ft_col, colorSpace='rgb255')
    calib.draw()
    win.flip() # show screen
    
    keys=event.waitKeys(keyList=["space", 'escape'])
    if keys[0]=='escape': # check for escape key
                core.quit()
    
    ## --- Set up eyetracking device
    SIMULATION_MODE=False
    TRACKER="eyelink" # 'mouse' or 'eyelink'
    #win.clearBuffer() # clear
    eyetracker, defaultKeyboard, kb, ioServer = InitializeEyeTracker(win, SIMULATION_MODE, TRACKER, expInfo, datafolder)
    
    ## --- Print useful information
    t1, t2 = core.getTime(), eyetracker.trackerTime()
    print(f'START TIME\nReal time: {t1} seconds \nEyetracker device time: {t2} seconds\nDifference: {t1-t2} seconds')
    print('Eyetracker connected:', eyetracker.isConnected()) # Check if eyetracker device is connected
    
    ## --- Do drift correction
    #DriftCorrection(eyetracker, win, resolution, ioServer)
    
    ## ------ Start Experiment
    data_train1, data_invol, data_train2, data_vol = startScreensAndRecordData(win, resolution, ioServer, expInfo, training_stim, involuntary_stim, voluntary_stim, eyetracker, MODE, all_stim)
    
    ## ------- End Experiment
    t3, t4 = core.getTime(), eyetracker.trackerTime()
    print(f'END TIME\nReal time: {t3} seconds \nEyetracker device time: {t4} seconds \nDifference: {t3-t4} seconds')
    print(f'TOTAL DURATION\nReal time: {round((t3-t1)/60,2)} minutes \nEyetracker device time: {round((t4-t2)/60, 2)} minutes')

    eyetracker.setConnectionState(False) # Close connection to eyetracker device
    print('Eyetracker disconnected')
    
    ## ------ Save data
    all_data=pd.DataFrame(data = None, columns = ['participant', 'time', 'key', 'word', 'condition', 'correct_resp', 'message', 'phase'])
    
    ## ------ Shape them
    minimum,maximum = [],[]
    for data in [data_train1, data_invol, data_train2, data_vol]:
        maximum.append(len(data))
        for i, a in zip(range(int(np.sum(minimum)), int(np.sum(maximum))), range(0, len(data))):
            all_data.loc[i] = data[a]
        minimum.append(len(data))
    
    all_data.to_csv(datafolder + filename)
    print("Experiment saved as:", filename)
    
    print('Eyetracker connected:', eyetracker.isConnected()) # Check if eyetracker device is deconnected
    
    ## -------- Save minimal information: keep track of what just happened
    new_participant(expInfo['participant'], expInfo['date'], t2, t4, filename)
    
    ## ------- End experiment
    win.close()