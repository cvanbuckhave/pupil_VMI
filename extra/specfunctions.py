# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:10:21 2023

@author: cvanbuck
https://github.com/cvanbuckhave/pupil_VMI
"""
# =============================================================================
# Import libraries
# =============================================================================
from statsmodels.formula.api import mixedlm
from eyelinkparser import parse, defaulttraceprocessor
from datamatrix import series as srs, NAN, FloatColumn
from datamatrix import operations as ops
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from datamatrix import plot
from datamatrix.colors.tango import orange, blue, green, gray, red
from datamatrix import DataMatrix, operations
import functools
from scipy.stats import spearmanr
import warnings
import datamatrix
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_white

# =============================================================================
# Create useful functions
# =============================================================================
@functools.cache
def get_data(cwd, folder, split=True):

    dm = parse(
        maxtracelen=700,
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,
            downsample=10,
            mode='advanced',
            ),
        gaze_pos=False,  # Don't store gaze-position information to save memory
        time_trace=False, # Don't store absolute timestamps to save memory
        folder=cwd+folder,
        multiprocess=16)
    
    # To save memory, we keep only a subset of relevant columns.
    columns_to_remove = [i for i in dm.column_names if (i.startswith('blinket') or i.startswith('t_o'))]
    print(columns_to_remove)
    columns_to_keep = [i for i in dm.column_names if i not in set(columns_to_remove)]
    print(columns_to_keep)
    dm = dm[columns_to_keep]
    
    print(dm)
    
    if split==True:
        print('Split by task...')
        dm.condition[dm.trialid == 77] = 'ctrl' # Correct typo in condition for trialid 77 (insalubre classified as 'dark')
        dm1, dm2 = ops.split(dm.experiment, 'involuntary', 'voluntary') # split by experiment type

        return dm1, dm2
    else:
        dm.condition[dm.trialid == 77] = 'ctrl' # Correct typo in condition for trialid 77 (insalubre classified as 'dark')
        dm.participant[dm.participant == 'S005'] = 'N005' # Correct typo in participants' ID
        dm.participant[dm.participant == 'S024'] = 'N025' # Correct typo in participants' ID

        return dm


def filter_dm(dm_cleaned, EXP):
                
    if EXP=='invol':
        # Change all 1s to -1
        dm_cleaned.feedback[dm_cleaned.feedback == 'correct'] = 1
        dm_cleaned.feedback[dm_cleaned.feedback == 'incorrect'] = 0
        
        # id_with_errors = []
        dm_cleaned.meanfb = ''
        for identifier, sdm in ops.split(dm_cleaned.participant):
            dm_cleaned.meanfb[sdm] = (sdm.feedback.sum / len(sdm.feedback))
            #if dm_cleaned.meanfb[sdm].unique[0]*100 != 100:
                #id_with_errors.append(identifier)

        # dm_cleaned.meanfb_cond = ''
        # for cond, sdm in ops.split(dm_cleaned.condition):
        #     dm_cleaned.meanfb_cond[sdm] = (sdm.feedback.sum / len(sdm.feedback[sdm.condition==cond]))
        #     print(f"For condition {cond}: {dm_cleaned.meanfb_cond[sdm].unique[0]*100} of correct responses")

        print('Before removing incorrect trials: N(trial) = {}'.format(len(dm_cleaned)))
        dm_cleaned = dm_cleaned.feedback == 1
        print('After removing incorrect trials: N(trial) = {}'.format(len(dm_cleaned)))
        
        # Print final N 
        print('N Total =', len(dm_cleaned.participant.unique))
        
        return dm_cleaned
    else:
        rate_bright = dm_cleaned.rating[dm_cleaned.condition == 'light']
        rate_dark = dm_cleaned.rating[dm_cleaned.condition == 'dark'] 
    
        print(f'Mean effort for dark scenarios: {np.mean(rate_dark)}, std={np.std(rate_dark)} and bright: {np.mean(rate_bright)} ({np.std(rate_bright)})')
        
        # Print final N 
        print('N Total =', len(dm_cleaned.participant.unique))
        
        return dm_cleaned

def preprocess(dm):
    """
    Exclude trials with unrealistic pupil sizes (+/- 2 SD) and only keep 
    participants with more than 5 trials remaining ()
    """
    dm_ = dm.condition != 'animal' # don't consider the animal trials

    # By-participant (within subjects)
    dm_.z_baseline, dm_.bl = '', ''
    for participant, block, sdm in ops.split(dm_.participant, dm_.blocks):
        dm_.bl[sdm] = srs.reduce(sdm.baseline)
        dm_.z_baseline[sdm] = ops.z(sdm.bl)
    
    before = len(dm_[dm_.condition!='ctrl'])
    print(f'Before removing outliers: N(trial) = {before}')
    dm_ = dm_.z_baseline > -2.0
    dm_ = dm_.z_baseline < 2.0
    
    # Between participants
    dm_.bl = ''
    for block, sdm in ops.split(dm_.blocks):
        sdm.bl[sdm] = srs.reduce(sdm.baseline)
        dm_.z_baseline[sdm] = ops.z(sdm.bl)

    dm_ = dm_.z_baseline > -2.0
    dm_ = dm_.z_baseline < 2.0
    after = len(dm_[dm_.condition != "ctrl"])
    print(f'After removing outliers: N(trial) = {after}')
    
    print('Number of trials removed:', before-after, ((before-after)/before) * 100, '%')
    
    # How many trials left per participant?
    dm_.nbtrials = ''
    for participant, cond, sdm in ops.split(dm_.participant, dm_.condition):
        dm_.nbtrials[sdm] = len(sdm[sdm.condition==cond])
    
    if len(dm_.blocks.unique) > 1: # if involuntary task
        too_few_trials = dm_.participant[dm_.nbtrials < 5]
    else: # if voluntary task
        too_few_trials = dm_.participant[dm_.nbtrials < 2]
    
    dm_ = dm_.participant != set(too_few_trials)
    print(f'{len(dm.participant.unique) - len(dm_.participant.unique)} participants removed.')
    
    return dm_

def count_nonnan(a):
    return np.sum(~np.isnan(a))

def check_blinks(dm, baseline=False, time=float):
    dm_ = dm.condition != 'animal'
    dm_ = dm_.condition != 'ctrl'
    if baseline==True:
        dm_.n_blinks = srs.reduce(dm_.blinkstlist_baseline, count_nonnan)  / time
        title='Fixation'
    else:
        title='Imagery'
        dm_.n_blinks = srs.reduce(dm_.blinkstlist_imagery, count_nonnan)  / time
    
    # aggregate data by subject and condition
    pm = ops.group(dm_, by=[dm_.participant, dm_.condition])
        
    # calculate mean blink rate per condition
    pm.mean_blink_rate = srs.reduce(pm.n_blinks)
    
    for identifier in dm_.participant.unique:
        print(f"{identifier}: {round(dm_.n_blinks[dm_.participant == identifier].mean,3)}")
    # Plot the mean blink rate as a function of experimental condition and participant
    x = sns.pointplot(
        x="condition",
        y="n_blinks",
        hue="participant",
        data=dm_,
        ci=None,
        palette=sns.color_palette(['indianred']),
        markers='.')
    plt.setp(x.lines, alpha=.4)
    plt.setp(x.collections, alpha=.4)
    sns.pointplot(
        x="condition",
        y="mean_blink_rate",
        data=pm,
        linestyles='solid',
        color='crimson',
        markers='o',
        scale=2)
    plt.ylim([-0.1,1.5])
    plt.xlabel('Condition', color='black', fontsize=20)
    plt.ylabel('Mean number of blinks per second', color='black', fontsize=20)
    plt.legend([], [], frameon=False)
    plt.title(title)

def dm_involuntary(dm, exp='invol'):
    """Processing of the pupil traces for the involuntary task."""
    dm_ = dm.condition != '' # to not overwrite the dm

    # Set depth
    dm_.ptrace_baseline.depth = 300 # 3s
    dm_.ptrace_imagery.depth = 300 # 3s
    
    # Check participants with an abusive number of blinks during imagery (+ 2 SD)
    dm_.blinks = ''
    for p, sdm in ops.split(dm_.participant):
        dm_.blinks[sdm] = np.nansum(srs.reduce(sdm.blinkstlist_imagery, count_nonnan)) # sum of blinks across trials for each participant
    
    dm_.z_blinks = ops.z(dm_.blinks) # z-transform
    
    blinky = set(dm_.participant[dm_.z_blinks > 3.0])
    
    print(f'Participant with a lot of blinks = {blinky} ({dm_.blinks[dm_.z_blinks > 3.0].unique}; M = {dm_.blinks.mean}, STD = {dm_.blinks.std})')    
    
    # Smooth
    dm_.ptrace_baseline = srs.smooth(dm_.ptrace_baseline, 21)
    dm_.ptrace_imagery = srs.smooth(dm_.ptrace_imagery, 21)

    # The whole trial
    dm_.pupil = dm_.ptrace_imagery[:,:]
    
    # Remove incorrect trials
    dm1 = filter_dm(dm_, EXP=exp)
    
    # Print reaction time
    mean_rt=dm1.tkeypress[dm1.condition=='animal'].mean
    std_rt=dm1.tkeypress[dm1.condition=='animal'].std
    se_rt= std_rt / len(dm1.tkeypress[dm1.condition=='animal'])**.5
    print(f'Mean response time (animal names): {mean_rt} s (std={std_rt} s, SE={se_rt}s)')
    
    # Print accuracy
    mean_accuracy = dm1.meanfb.mean
    print('Mean accuracy:', mean_accuracy*100, '% of correct responses')
    
    # Remove bad quality data
    # We take this baseline period of 50 ms after word onset because it's the max 
    # baseline duration that we use. Additionally, we also filter trials with inconsistent
    # pupil sizes during the 200 ms before word onset, because this can also cause 
    # inconsistent pupil sizes between fixation and the 50 ms after word onset. 
    dm1.baseline = srs.concatenate(dm1.ptrace_baseline[:, 280:300], dm1.pupil[:, 0:5])
    dm1 = preprocess(dm1)
    
    print('N=', len(dm1.participant.unique))
    
    # Apply baseline correction
    dm1.pupil_imagery_ = srs.baseline(dm1.pupil, dm1.pupil, 0, 1, method='divisive')
    dm1.pupil_imagery = srs.baseline(dm1.pupil, dm1.pupil, 0, 5, method='subtractive')
    
    # Compute the mean pupil size during word presentation 
        # 1st baseline
    dm1.mean_pupil_ = srs.reduce(dm1.pupil_imagery_[:, 100:140]) # to compute pupil size changes
    dm1 = dm1.mean_pupil_ != NAN # don't keep missing points

    dm1.pupil_change_=''
    for p, sdm in ops.split(dm1.participant):
        diff = sdm.mean_pupil_[sdm.condition =='dark'].mean - sdm.mean_pupil_[sdm.condition =='light'].mean
        dm1.pupil_change_[sdm] = diff
        
        # 2nd baseline
    dm1.mean_pupil = srs.reduce(dm1.pupil_imagery[:, 100:150]) # to compute pupil size changes
    dm1 = dm1.mean_pupil != NAN # don't keep missing points

    dm1.pupil_change=''
    for p, sdm in ops.split(dm1.participant):
        diff = sdm.mean_pupil[sdm.condition =='dark'].mean - sdm.mean_pupil[sdm.condition == 'light'].mean
        dm1.pupil_change[sdm] = diff
        
    # Plots
    #############################   
    # Per block and per condition
    plt.figure(figsize=(10,10))
    ax1=plt.subplot(1,1,1)
    dm_bis = dm1.condition != 'animal'
    dm_b1, dm_b2, dm_b3, dm_b4, dm_b5 = ops.split(dm_bis.blocks, 1, 2, 3, 4, 5)
    plot.trace(dm_b1.pupil_imagery, color=blue[0], label='Block 1 (N=%d)' % len(dm_b1))
    plot.trace(dm_b2.pupil_imagery, color=green[1], label='Block 2 (N=%d)' % len(dm_b2))
    plot.trace(dm_b3.pupil_imagery, color=gray[3], label='Block 3 (N=%d)' % len(dm_b3))
    plot.trace(dm_b4.pupil_imagery, color=orange[1], label='Block 4 (N=%d)' % len(dm_b4))
    plot.trace(dm_b5.pupil_imagery, color=red[1], label='Block 5 (N=%d)' % len(dm_b5))
    plt.title('Pupil size during each experimental block', fontsize=25)
    plt.legend(ncol=1, loc='upper right', fontsize=25)
    plt.xticks(range(0, 400, 100), range(0, 4, 1))
    plt.xlim([0,300])
    plt.xlabel('Time Since Word Onset (s)');plt.ylabel('Baseline-corrected Pupil Size (a.u.)')
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    # Per block all conditions confounded
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(25,20))
    ax1=plt.subplot(2,3,1)
    dm_dark, dm_light, dm_ctrl = ops.split(dm1.condition, 'dark', 'light', 'ctrl')
    plot.trace(dm_dark.pupil_imagery, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.pupil_imagery, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    plt.title('All\n')
    plt.legend(ncol=1, loc='upper right', fontsize=25)
    plt.xticks(range(0, 400, 100), range(0, 4, 1))
    plt.xlim([0,300])
    plt.xlabel('Time Since Word Onset (s)');plt.ylabel('Baseline-corrected Pupil Size (a.u.)')
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    i=2
    for phase, sdm in ops.split(dm1.blocks):
        dm_dark, dm_light, dm_ctrl = ops.split(sdm.condition, 'dark', 'light', 'ctrl')
        ax = plt.subplot(2,3,i)
        plt.title(f'Block #{phase}\n')
        plot.trace(dm_dark.pupil_imagery, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
        plot.trace(dm_light.pupil_imagery, color=orange[1], label='Bright (N=%d)' % len(dm_light))
        plot.trace(dm_ctrl.pupil_imagery, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
        plt.legend(ncol=1, loc='upper right', fontsize=25)
        plt.xticks(range(0, 400, 100), range(0, 4, 1))
        plt.xlabel('Time Since Word Onset (s)')
        plt.xlim([0,300])
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i==4: plt.ylabel('Baseline-corrected Pupil Size (a.u.)')
        i+=1
        del ax
    plt.tight_layout()
    plt.show()
    
    # Raw
    plt.figure(figsize=(25,10))
    plt.suptitle('Before trial exclusion')
    dm_dark, dm_light, dm_ctrl = ops.split(dm_.condition, 'dark', 'light', 'ctrl')
    ax=plt.subplot(1,2,1)
    plt.title('Fixation\n')
    plot.trace(dm_dark.ptrace_baseline, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.ptrace_baseline, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.ptrace_baseline, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(ncol=1, loc='lower right')
    plt.xticks(range(0, 400, 100), range(-3, 1, 1))
    plt.xlabel('Time Since Word Onset (s)');plt.ylabel('Pupil Size (a.u.)')
    
    ax2=plt.subplot(1,2,2)
    plt.title('Reading\n')
    plot.trace(dm_dark.pupil, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.pupil, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    plt.legend(ncol=1, loc='upper right')
    plt.xticks(range(0, 400, 100), range(0, 4, 1))
    plt.xlabel('Time Since Word Onset (s)');plt.ylabel('Pupil Size (a.u.)')
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_visible(True)
    ax2.spines['left'].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(25,10))
    plt.suptitle('After trial exclusion')
    dm_dark, dm_light, dm_ctrl = ops.split(dm1.condition, 'dark', 'light', 'ctrl')
    ax=plt.subplot(1,2,1)
    plt.title('Fixation\n')
    plot.trace(dm_dark.ptrace_baseline, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.ptrace_baseline, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.ptrace_baseline, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(ncol=1, loc='lower right')
    plt.xticks(range(0, 400, 100), range(-3, 1, 1))
    plt.xlabel('Time Since Word Onset (s)');plt.ylabel('Pupil Size (a.u.)')
    
    ax2=plt.subplot(1,2,2)
    plt.title('Reading\n')
    plot.trace(dm_dark.pupil, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.pupil, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    plt.legend(ncol=1, loc='upper right')
    plt.xticks(range(0, 400, 100), range(0, 4, 1))
    plt.xlabel('Time Since Word Onset (s)');plt.ylabel('Pupil Size (a.u.)')
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_visible(True)
    ax2.spines['left'].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    # baseline corrected
    fig, axes = plt.subplots(1, 2, figsize=(25, 12), sharex=True)
    ax0=plt.subplot(1,2,1)
    plt.title('A\n', loc='left')
    plot.trace(dm_dark.pupil_imagery_, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery_, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.pupil_imagery_, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    ax0.spines['bottom'].set_visible(True)
    ax0.spines['bottom'].set_color('black')
    ax0.spines['left'].set_visible(True)
    ax0.spines['left'].set_color('black')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    plt.xticks(range(0, 400, 100), range(0, 4, 1), color='black');plt.yticks(color='black')
    plt.axvline(mean_rt*100, color='black', linestyle='dashed', label='Mean RT (s)')
    plt.xlabel('Time Since Word Onset (s)', color='black');plt.ylabel('Baseline-corrected Pupil Size (a.u.)', color='black')
    plt.legend('')
    plt.xlim([0,300])
    plt.axvline(100, color=gray[1])
    plt.axvline(140, color=gray[1])
    
    ax1=plt.subplot(1,2,2)
    plt.title('B\n', loc='left')
    dm_dark, dm_light, dm_ctrl = ops.split(dm1.condition, 'dark', 'light', 'ctrl')
    plot.trace(dm_dark.pupil_imagery, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plot.trace(dm_ctrl.pupil_imagery, color=gray[3], label='Neutral (N=%d)' % len(dm_ctrl))
    plt.xticks(range(0, 400, 100), range(0, 4, 1), color='black');plt.yticks(color='black')
    plt.axvline(mean_rt*100, color='black', linestyle='dashed', label='Mean RT (s)')
    plt.xlabel('Time Since Word Onset (s)', color='black')#;plt.ylabel('Baseline-corrected Pupil Size (a.u.)', color='black')
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, ncol=4, bbox_to_anchor=(0.99, -0.03), frameon=False, fontsize=40)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(4)
    plt.tight_layout()
    plt.axvline(90, color=gray[1])
    plt.axvline(150, color=gray[1])
    plt.xlim([0,300])
    plt.show()
    
    
    return dm1

def dm_voluntary(dm, exp='vol'):
    """Processing of the pupil traces for the voluntary task."""
    dm_ = dm.condition != "" # to not overwrite the dm
    
    # Set depth
    dm_.ptrace_baseline.depth = 100 # 1s
    dm_.ptrace_read.depth = 300 # 3s
    dm_.ptrace_imagery.depth = 700 # 7s

    # Check participants with an abusive number of blinks during imagery (+ 2 SD)
    dm_.blinks = ''
    for p, sdm in ops.split(dm_.participant):
        dm_.blinks[sdm] = np.nansum(srs.reduce(sdm.blinkstlist_imagery, count_nonnan)) # sum of blinks across trials for each participant
    
    dm_.z_blinks = ops.z(dm_.blinks) # z-transform
    
    blinky = set(dm_.participant[dm_.z_blinks > 3.0])
    
    print(f'Participant with a lot of blinks = {blinky} ({dm_.blinks[dm_.z_blinks > 3.0].unique}; M = {dm_.blinks.mean}, STD = {dm_.blinks.std})')    

    # The whole trial
    dm_.pupil = srs.concatenate(dm_.ptrace_baseline, dm_.ptrace_read, dm_.ptrace_imagery)
    print(dm_.pupil.depth)

    # Compute mean effort per condition
    dm1 = filter_dm(dm_, EXP=exp)
    
    # Smooth 
    dm1.pupil = srs.smooth(dm1.pupil, 21)
    
    # Remove bad quality data
    dm1.baseline = dm1.pupil[:, 0:100]
    dm1 = preprocess(dm1)
    
    print('N=', len(dm1.participant.unique))
      
    # Apply baseline correction
    dm1.pupil_imagery_ = srs.baseline(dm1.pupil, dm1.pupil, 0, 100, method='subtractive')
    dm1.pupil_imagery = srs.baseline(dm1.pupil, dm1.pupil, 95, 100, method='subtractive')

    # Compute the mean pupil size during word presentation 
        # 1st baseline
    dm1.mean_pupil_ = srs.reduce(dm1.pupil_imagery_[:, 400:]) # to compute pupil size changes
    dm1 = dm1.mean_pupil_ != NAN # don't keep missing points

    dm1.pupil_change_=''
    for p, sdm in ops.split(dm1.participant):
        diff = sdm.mean_pupil_[sdm.condition =='dark'].mean - sdm.mean_pupil_[sdm.condition =='light'].mean
        dm1.pupil_change_[sdm] = diff
        
        # 2nd baseline
    dm1.mean_pupil = srs.reduce(dm1.pupil_imagery[:, 400:]) # to compute pupil size changes
    dm1 = dm1.mean_pupil != NAN # don't keep missing points

    dm1.pupil_change=''
    for p, sdm in ops.split(dm1.participant):
        diff = sdm.mean_pupil[sdm.condition =='dark'].mean - sdm.mean_pupil[sdm.condition == 'light'].mean
        dm1.pupil_change[sdm] = diff
    
    # Plots
    #############################   
    # Plot raw
    dm_dark, dm_light = ops.split(dm_.condition, 'dark', 'light')
    
    plt.figure(figsize=(25,10))
    plt.suptitle('Before trial exclusion')
    plt.subplot(1,3,1)
    plt.title('Fixation', fontsize=40)
    plot.trace(dm_dark.pupil[:,0:100], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil[:,0:100], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.legend(ncol=1, loc='lower left', fontsize=30)
    plt.xticks(range(0, 200, 100), range(-4, -2, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Pupil Size (a.u.)')

    plt.subplot(1,3,2)
    plt.title('Reading', fontsize=40)
    plot.trace(dm_dark.pupil[:, 100:400], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil[:, 100:400], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.legend(ncol=1, loc='lower left', fontsize=30)
    plt.xticks(range(0, 400, 100), range(-3, 1, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Pupil Size (a.u.)')
    
    plt.subplot(1,3,3)
    plt.title('Imagine', fontsize=40)
    plot.trace(dm_dark.pupil[:,400:1100], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil[:,400:1100], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.legend(ncol=1, loc='lower left', fontsize=30)
    plt.xticks(range(0, 800, 100), range(0, 8, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Pupil Size (a.u.)')
    
    plt.tight_layout()
    plt.show()
    
    dm_dark, dm_light = ops.split(dm1.condition, 'dark', 'light')

    plt.figure(figsize=(25,10))
    plt.suptitle('After trial exclusion')
    plt.subplot(1,3,1)
    plt.title('Fixation', fontsize=40)
    plot.trace(dm_dark.pupil[:,0:100], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil[:,0:100], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.legend(ncol=1, loc='lower left', fontsize=30)
    plt.xticks(range(0, 200, 100), range(-4, -2, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Pupil Size (a.u.)')

    plt.subplot(1,3,2)
    plt.title('Reading', fontsize=40)
    plot.trace(dm_dark.pupil[:, 100:400], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil[:, 100:400], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.legend(ncol=1, loc='lower left', fontsize=30)
    plt.xticks(range(0, 400, 100), range(-3, 1, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Pupil Size (a.u.)')
    
    plt.subplot(1,3,3)
    plt.title('Imagine', fontsize=40)
    plot.trace(dm_dark.pupil[:,400:1100], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil[:,400:1100], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.legend(ncol=1, loc='lower left', fontsize=30)
    plt.xticks(range(0, 800, 100), range(0, 8, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Pupil Size (a.u.)')
    
    plt.tight_layout()
    plt.show()
    
    # After baseline correction [1]
    fig = plt.figure(figsize=(25,20))
    dm_dark, dm_light = ops.split(dm1.condition, 'dark', 'light')
    ax1=plt.subplot(2,2,1)
    plt.title('Reading')
    plot.trace(dm_dark.pupil_imagery[:, 100:400], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery[:, 100:400], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #plt.legend(ncol=1, loc='lower left')
    plt.xticks(range(0, 400, 100), range(-3, 1, 1), color='black');plt.yticks(color='black')
    plt.xlabel('Time Since Start Imagine (s)', color='black')#plt.ylabel('Baseline-corrected Pupil Size (a.u.)', color='black')
    plt.xlim([0,300])

    ax2=plt.subplot(2,2,2)
    plt.title('Imagining')
    plot.trace(dm_dark.pupil_imagery[:, 400:], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery[:, 400:], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_visible(True)
    ax2.spines['left'].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #plt.legend(ncol=1, loc='lower left')
    plt.xticks(range(0, 800, 100), range(0, 8, 1), color='black');plt.yticks(color='black')
    plt.xlabel('Time Since Start Imagine (s)', color='black')#;plt.ylabel('Baseline-corrected Pupil Size (a.u.)', color='black')
    plt.xlim([0,700])

    ax3=plt.subplot(2,1,2)
    plt.title('Full trial')
    plot.trace(dm_dark.pupil_imagery, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.axvline(100, linestyle='solid', color='black')
    plt.axvline(400, linestyle='solid', color='black')
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_visible(True)
    ax3.spines['left'].set_color('black')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    #plt.legend(ncol=1, loc='upper right')
    plt.xticks(range(0, 1200, 100), range(-4, 8, 1), color='black');plt.yticks(color='black')
    plt.xlabel('Time Since Start Imagine (s)', color='black')
    plt.xlim([0,1100])
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, ncol=4, bbox_to_anchor=(0.825, -0.03), frameon=False, fontsize=50)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(4)
    axes = [ax1, ax2, ax3]
    fig.supylabel('Baseline-corrected Pupil Size (a.u.)', color='black', ha='center', va='center', fontsize=55)
    plt.tight_layout()
    plt.show()
    
    # After baseline correction [2]
    plt.figure(figsize=(25,20))
    plt.suptitle('With the whole 1-s fixation period as baseline')
    dm_dark, dm_light = ops.split(dm1.condition, 'dark', 'light')
    ax1=plt.subplot(2,2,1)
    plt.title('Reading')
    plot.trace(dm_dark.pupil_imagery_[:, 100:400], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery_[:, 100:400], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.legend(ncol=1, loc='lower left')
    plt.xticks(range(0, 400, 100), range(-3, 1, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Baseline-corrected Pupil Size (a.u.)')
    plt.xlim([0,300])

    ax2=plt.subplot(2,2,2)
    plt.title('Imagining')
    plot.trace(dm_dark.pupil_imagery_[:, 400:], color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery_[:, 400:], color=orange[1], label='Bright (N=%d)' % len(dm_light))
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_visible(True)
    ax2.spines['left'].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.legend(ncol=1, loc='lower left')
    plt.xticks(range(0, 800, 100), range(0, 8, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Baseline-corrected Pupil Size (a.u.)')
    plt.xlim([0,700])

    ax3=plt.subplot(2,1,2)
    plt.title('Full trial')
    plot.trace(dm_dark.pupil_imagery_, color=blue[1], label='Dark (N=%d)' % len(dm_dark))
    plot.trace(dm_light.pupil_imagery_, color=orange[1], label='Bright (N=%d)' % len(dm_light))
    plt.axvline(100, linestyle='solid', color='black')
    plt.axvline(400, linestyle='solid', color='black')
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_visible(True)
    ax3.spines['left'].set_color('black')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.legend(ncol=1, loc='upper right')
    plt.xticks(range(0, 1200, 100), range(-4, 8, 1))
    plt.xlabel('Time Since Start Imagine (s)');plt.ylabel('Baseline-corrected Pupil Size (a.u.)')
    plt.xlim([0,1100])
    axes = [ax1, ax2, ax3]
    plt.tight_layout()
    plt.show()
    
    return dm1

# Bar plots
def plot_bars(dm, x='condition', y='mean_pupil', hue=None, order=None, hue_order=None, xlab='Condition', ylab='Mean Pupil Size (a.u.)',pal='colorblind'):
    """Plot the mean pupil size as bar plots."""
    #plt.figure(figsize=(8,8))
    sns.barplot(x=x, y=y, hue=hue, data=dm, order=order, hue_order=hue_order, palette=pal)
    plt.axhline(0, linestyle='solid', color='black')
    plt.xlabel(xlab, color='black');plt.ylabel(ylab, color='black')
    plt.yticks(color='black');plt.xticks(color='black')
    plt.tight_layout()
    plt.legend().remove()
    #plt.show()

def group_by_cond(dm):
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    dm_valid_data = ops.group(dm, by=[dm.participant, dm.condition]) # add dm_sub.response_lang if necessary
    type_ = type(dm.condition)
    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_valid_data.column_names:
        if type(dm_valid_data[col]) != type_:
            dm_valid_data[col] = srs.reduce(dm_valid_data[col]) # Compute the mean 
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    return dm_valid_data

def group_dm(dm, by='condition', cond=True):
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    dm = dm.condition != {'animal', 'ctrl'}

    if by!='condition':
        if cond==True:
            dm_valid_data = ops.group(dm, by=[dm['participant'], dm.blocks, dm.condition]) #
            type_ = type(dm.blocks)
        else:
            dm_valid_data = ops.group(dm, by=[dm['participant'], dm[by]]) # 
            type_ = type(dm[by])
    else:
        dm_valid_data = ops.group(dm, by=[dm['participant'], dm.condition]) # 
        type_ = type(dm.condition)

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_valid_data.column_names:
        if type(dm_valid_data[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            if type(dm_valid_data[col]) != type_:
                dm_valid_data[col] = srs.reduce(dm_valid_data[col]) # Compute the mean 
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    return dm_valid_data

def test_stats(dm, formula=False, re_formula=None, reml=True, method='Powell'):
    """Mixed linear model with Powell's optimization method."""
    dm_tst = dm.condition != {'animal', 'ctrl'}
    
    dm_valid_data = dm_tst.mean_pupil != NAN
    dm_valid_data = dm_valid_data.pupil_change != NAN

    md = mixedlm(formula, dm_valid_data, 
                     groups='participant',
                     re_formula=re_formula)
        
    mdf = md.fit(reml=reml, method=method)

    print(mdf.summary())
    
    return mdf


def size_se(dm, start=0, end=None):
    """
	desc:
		Gets the pupil-size standard error during a time window.

	arguments:
		dm:
			type: DataMatrix

	keywords:
		start:	The start time.
		end:	The end time, or None for trace end.

	returns:
		type:
			ndarrray
	"""
    s = srs.reduce(srs.window(dm.pupil_imagery, start=start, end=end))
    print(len(s))
    return s.mean, s.std / len(s)**.5
    
def word_summary(dm, EXP, df, low, high):
    """desc:
		Plots the mean pupil size for dark and bright words as a bar plot. The
		time window is indicated by the PEAKWIN constant. This data is also
		written to a .csv file.

	arguments:
		dm:
			type: DataMatrix
    """
    dm = (dm.condition == 'light') | (dm.condition == 'dark')
    df = df[df.code > 10] # remove the trials from training phase
    df = df[df.code != 77] # remove the trials from ctrl condition
    df = df[df.code < 80] # remove the trials from ctrl condition
    
    dicts = dict({135: 'ciel ensoleillé', 
                136: 'pièce sombre',
                137: 'ciel nuageux',
                138: 'ciel nocturne',
                139: 'visage soleil',
                140: 'visage pénombre',
                141: 'pièce éclairée', 
                142: 'main soleil',
                143: 'main obscurité',
                144: 'écran blanc',
                145: 'écran noir'})
    
    # set parameters
    if EXP=='invol':
        x = np.arange(dm.pupil_imagery[:, low:high].depth)
    else:
        x = np.arange(dm.pupil_imagery[:, low:high].depth)
    sm = DataMatrix(length=len(dm.trialid.unique))
    sm.trialid = 0
    sm.condition = 0
    sm.stim = 0
    sm.participant = 0
    sm.pupil_win = FloatColumn
    sm.pupil_win_se = FloatColumn
    sm.pupil_full = FloatColumn
    sm.pupil_full_se = FloatColumn
    
    for i, w in enumerate(dm.trialid.unique):
        _dm = dm.trialid == w
        sm.trialid[i] = w
        sm.condition[i] = (dm.trialid == w).condition[0]
        sm.pupil_win[i], sm.pupil_win_se[i] = size_se(_dm, start=low, end=high)
        sm.pupil_full[i], sm.pupil_full_se[i] = size_se(_dm, start=low, end=high)
        if EXP=='invol':
            sm.stim[i] = list(df.stim[df.code == int(w)])[0]
    
    sm = operations.sort(sm, sm.pupil_win)
    if EXP=='invol':
        plot.new(size=(20,7))
        s=20
    else:
        plot.new(size=(15,7))
        s=30
    
    i=0
    for color, cond_ in ((orange[1], 'light'), (blue[1],'dark')):
        sm_ = sm.condition == cond_
        x = np.arange(len(sm_))
        if i==0:
            plt.plot(sm_.pupil_win, 'o-', color=color, markersize=s)

            if cond_ == 'dark':
                yerr = (np.zeros(len(sm_)), sm_.pupil_win_se)
            else:
                yerr = (sm_.pupil_win_se, np.zeros(len(sm_)))
                
            plt.errorbar(x, sm_.pupil_win, yerr=yerr, linestyle='-', color=color, capsize=0)
            plt.yticks(color='black')
            plt.ylabel('Mean Baseline corrected Pupil Size (a.u.)', color='black')

            if EXP=='vol':
                labs=[dicts[i] for i in sm_.trialid]
                plt.xticks(x, labels=labs, rotation=20, color='black')
                plt.xlabel('\nBright Scenes', color='black')
            else: # labels can be trialid or stim (names)
                plt.xticks(x, labels=sm_.stim, rotation=90, color='black')
                plt.xlabel('\nBright Words', color='black')
        else:
            plt.twiny()
            plt.plot(sm_.pupil_win, 'o-', color=color, markersize=s)
            plt.yticks(color='black')
            plt.ylabel('Mean Baseline corrected Pupil Size (a.u.)')
            if cond_ == 'dark':
                yerr = (np.zeros(len(sm_)), sm_.pupil_win_se)
            else:
                yerr = (sm_.pupil_win_se, np.zeros(len(sm_)))
            plt.errorbar(x, sm_.pupil_win, yerr=yerr, linestyle='', color=color, capsize=0)
            if EXP=='vol':
                labs=[dicts[i] for i in sm_.trialid]
                plt.xticks(x, labels=labs, rotation=20, color='black')
                plt.xlabel('Dark Scenes\n', color='black')
            else:
                plt.xlabel('Dark Words\n', color='black')
                plt.xticks(x, labels=sm_.stim, rotation=90, color='black')
        i+=1

    plt.ylabel('Baseline-Corrected Pupil size (a.u.)')
    plt.show()
    
def test_correlation(dm_c, x, y, alt='two-sided', pcorr=1):
    """Test the correlations between pupil measures and questionnaire measures using Spearman's correlation."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Group per participant 
    dm_cor = ops.group(dm_c, by=dm_c.participant)

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_cor.column_names:
        if type(dm_cor[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_cor[col] = srs.reduce(dm_cor[col], operation=np.nanmean)
            
    # The variables to test the correlation
    x, y = dm_cor[x], dm_cor[y]
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    # Compute spearman's rank correlation
    cor=spearmanr(x, y, alternative=alt)

    N = len(x)
    p = cor.pvalue * pcorr # Apply Bonferroni correction (multiply p-values by the number of tests)

    if p > 1.0:
        p = 1.0
    if p < 0.001:
        p = '{:.3e}'.format(p)
    else:
        p = np.round(p, 3)
    print(fr'{x.name} and {y.name} ({chr(961)} = {round(cor.correlation, 3)}, p = {p}, n = {N})')
    
    # Plot the correlations (linear regression model fit)
    plt.figure(figsize=(10,8))
    plt.xlabel(x.name);plt.ylabel(y.name)
    sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=False, color='red', label=f'{chr(961)} = {round(cor.correlation, 3)}, p = {p}, n = {N}', x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.4}, robust=True)
    plt.legend(markerscale=0, frameon=False)
    # use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression)
    sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=True, color='red', label=f'{chr(961)} = {round(cor.correlation, 3)}, p = {p}, n = {N}', x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.0}, line_kws={'linestyle': 'dashed', 'alpha':0.5})
    plt.tight_layout()
    plt.show()

    return str(f'{x.name} and {y.name} ({chr(961)} = {round(cor.correlation, 3)}, p = {p}, n = {N})')


def check_assumptions(model):
    """Check assumptions for normality of residuals and homoescedasticity.
    Code from: https://www.pythonfordatascience.org/mixed-effects-regression-python/#assumption_check"""
    plt.rcParams['font.size'] = 40
    print('Assumptions check:')
    fig = plt.figure(figsize = (25, 16))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    fig.suptitle(f'{model.model.formula} (n = {model.model.n_groups})')
    # Normality of residuals
    ax1 = plt.subplot(2,2,1)
    sns.distplot(model.resid, hist = True, kde_kws = {"fill" : True, "lw": 4}, fit = stats.norm)
    ax1.set_title("KDE Plot of Model Residuals (Red)\nand Normal Distribution (Black)", fontsize=30)
    ax1.set_xlabel("Residuals")
    
    # Q-Q PLot
    ax2 = plt.subplot(2,2,2)
    sm.qqplot(model.resid, dist = stats.norm, line = 's', ax = ax2, alpha=0.5, markerfacecolor='black', markeredgecolor='black')
    ax2.set_title("Q-Q Plot", fontsize=30)
    
    # Shapiro
    labels1 = ["Statistic", "p-value"]
    norm_res = stats.shapiro(model.resid)
    print('Shapir-Wilk test of normality')
    for key, val in dict(zip(labels1, norm_res)).items():
        print(key, val)
    lab1 = f'Shapiro (normality): Statistic = {np.round(norm_res[0],3)}, p = {np.round(norm_res[1],3)}'

    # Homogeneity of variances
    ax3 = plt.subplot(2,2,3)
    sns.scatterplot(y = model.resid, x = model.fittedvalues, alpha=0.8)
    ax3.set_title("RVF Plot", fontsize=30)
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("Residuals")
    
    ax4 = plt.subplot(2,2,4)
    sns.boxplot(x = model.model.groups, y = model.resid)
    plt.xticks(range(0, len(model.model.group_labels)), range(1, len(model.model.group_labels)+1), fontsize=15)
    ax4.set_title("Distribution of Residuals for Weight by Litter", fontsize=30)
    ax4.set_ylabel("Residuals")
    ax4.set_xlabel("Litter")
    
    # White’s Lagrange Multiplier Test for Heteroscedasticity
    print('White’s Lagrange Multiplier Test for Heteroscedasticity')
    het_white_res = het_white(model.resid, model.model.exog)
    labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    for key, val in dict(zip(labels, het_white_res)).items():
        print(key, val)
    lab2 = f'LM Test (homoscedasticity): LM Statistic = {np.round(het_white_res[0],3)}, p = {np.round(het_white_res[1],3)}'
    
    fig.supxlabel(f'{lab1}\n{lab2}')
    plt.tight_layout()
    plt.show()
    
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=UserWarning)

