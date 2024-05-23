# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:20:51 2023

@author: cvanbuck
"""
# =============================================================================
# Set up 
# =============================================================================
# Import packages
###################################
from datamatrix.colors.tango import orange, blue, gray
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import os 
from datamatrix import series as srs, operations as ops, NAN
from extra.specfunctions import ( get_data,
                               dm_involuntary,
                               dm_voluntary,
                               plot_bars,
                               check_blinks,
                               test_stats,
                               check_assumptions,
                               word_summary,
                               test_correlation,
                               group_dm )

import pingouin as pg # to compute Cronbach's alpha
import seaborn as sns
from datamatrix import convert
from scipy.stats import wilcoxon

######################################
# Edit the font, font size, grid color and axes width
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 4
np.random.seed(111)  # Fix random seed for predictable outcomes

######################## begin part to modify
cwd = os.getcwd() # automatic
cwd='C:/Users/cvanb/Documents/pupilometry-main/' # manual
folder = 'pupil_data/data' # folder with all data
folder0, folder1 ='pupil_data/data_invol', 'pupil_data/data_vol' # separated folders by task
questfile = '/quest_data/results-survey.csv'
outputfolder= cwd+'output/'
######################## end part to modify

# =============================================================================
# Preprocessing
# =============================================================================
# Parsing data 
# Get dm (experiments) 
    # if all data in one file/folder
dm1_A, dm2_A = get_data(cwd, folder, split=True) # involuntary & voluntary
    # if data in separate folders/files:
dm1_B = get_data(cwd, folder0, split=False) # involuntary
dm2_B = get_data(cwd, folder1, split=False) # voluntary

# Index the trials and blocks to see possible impact of time 
trials=[[1]*25, [2]*25, [3]*25, [4]*25, [5]*24]
trials = [item for sublist in trials for item in sublist]

for dm in [dm1_A, dm1_B]:
    dm.blocks = int
    dm.trials = int
    for participant in dm.participant.unique:
        dm.blocks[dm.participant == participant] = trials
        dm.trials[dm.participant == participant] = range(0, len(dm.participant[dm.participant == participant]))

for dm in [dm2_A, dm2_B]:
    dm.blocks = 1
    dm.trials = int
    for participant in dm.participant.unique:
        dm.trials[dm.participant == participant] = range(0, len(dm.participant[dm.participant == participant]))

# Combine the datamatrix per task 
dm1_AB = dm1_A << dm1_B # involuntary
dm2_AB = dm2_A << dm2_B # voluntary

# Check blinks
# for dm, t in zip([dm1_AB, dm2_AB], [[3, 3], [1, 7]]):
#     plt.figure(figsize=(16,8))
#     plt.suptitle(dm.experiment.unique[0])
#     plt.subplot(1,2,1)
#     check_blinks(dm, True, time=t[0])
#     plt.subplot(1,2,2)
#     check_blinks(dm, False, time=t[1])
#     plt.tight_layout()
#     plt.show()

# Process the pupil data and plot
dm1_AB_ = dm_involuntary(dm1_AB)
dm2_AB_ = dm_voluntary(dm2_AB)

# Check that the same participants have been excluded for both tasks
print(dm2_AB_.participant.unique == dm1_AB_.participant.unique)

# Get the average duration of the reading phase for the voluntary imagery 
print(f'Mean duration of imagery phase: {round(dm2_AB_.trace_length_read.mean)} ms (SD = {round(dm2_AB_.trace_length_read.std)} ms)')
print(f'Max = {round(dm2_AB_.trace_length_read.max)}; Min = {round(dm2_AB_.trace_length_read.min)} ms)')

# =============================================================================
# Statistical analyses
# =============================================================================
from scipy.stats.distributions import chi2
def compare_models(model1, model2, ddf):
    """Null hypothesis: The simpler model is true. 
    Log-likelihood of the model 1 for H0 must be <= LLF of model 2."""
    print(f'Log-likelihood of model 1 <= model 2: {model1.llf <= model2.llf}')
    
    ratio = (model1.llf - model2.llf)*-2
    p = chi2.sf(ratio, ddf) # How many more DoF does M2 has as compared to M1?
    if p >= .05:
        print(f'The simpler model is the better one (LLF M1: {round(model1.llf,3)}, LLF M2: {round(model2.llf,3)}, ratio = {round(ratio,3)}, df = {ddf}, p = {round(p,4)})')
    else:
        print(f'The simpler model is not the better one (LLF M1: {round(model1.llf,3)}, LLF M2: {round(model2.llf,3)}, ratio = {round(ratio,3)}, df = {ddf}, p = {round(p,4)})')

# Involuntary
dm1_test = group_dm(dm1_AB_, by='condition') # does not really change the results to average by cond
# but it helps with meeting the assumption checks

    # With the original 10 ms divisive method
m0 = test_stats(dm1_test, formula='mean_pupil_ ~ condition', re_formula='1 + condition')
check_assumptions(m0) # OK
    # With the 50 ms subtractive method
m1 = test_stats(dm1_test, formula='mean_pupil ~ condition', re_formula='1 + condition')
check_assumptions(m1) # OK
    # Checking the interaction with the experimental block (time)
dm1_test_ = group_dm(dm1_AB_, by='blocks') # same
m2a = test_stats(dm1_test_, formula='mean_pupil ~ condition + blocks', re_formula='1 + condition', reml=False)
check_assumptions(m2a) # /!\ violates normality statistically but QQ plot ok

m2b = test_stats(dm1_test_, formula='mean_pupil ~ condition * blocks', re_formula='1 + condition', reml=False) 
check_assumptions(m2b) # /!\ violates normality statistically but QQ plot ok

compare_models(m2a, m2b, 1) # compare models

    # Another way to test it
# sdm1_1 = dm1_AB_.blocks <= 3
# sdm1_test1 = group_dm(sdm1_1, by='condition')
# m3a = test_stats(sdm1_test1, formula='mean_pupil ~ condition', re_formula='1 + condition')
# check_assumptions(m3a) # OK

# sdm1_2 = dm1_AB_.blocks > 3
# sdm1_test2 = group_dm(sdm1_2, by='condition')
# m3b = test_stats(sdm1_test2, formula='mean_pupil ~ condition', re_formula='1 + condition')
# check_assumptions(m3b) # OK

# Voluntary experiment
dm2_test = group_dm(dm2_AB_)

    # With the original 1 s subtractive method
md0 = test_stats(dm2_test, formula='mean_pupil_ ~ condition', re_formula="1 + condition")
check_assumptions(md0) # violates normality but only slightly 

    # With the 50 ms subtractive method
md1 = test_stats(dm2_test, formula='mean_pupil ~ condition', re_formula="1 + condition", reml=False)
check_assumptions(md1) #OK

    # Check the interaction with effort ratings
md2 = test_stats(dm2_test, formula='mean_pupil ~ condition * rating', re_formula="1 + condition", reml=False)
check_assumptions(md2) #OK

compare_models(md1, md2, 2)

# =============================================================================
# Visualise the data as barplots
# =============================================================================
dm1_AB_ = ops.sort(dm1_AB_, by=dm1_AB_.mean_pupil)
dm_df1 = convert.to_pandas(dm1_AB_)
plt.figure(figsize=(20,10))
ax1=plt.subplot(1,2,1)
plot_bars(dm_df1, order=['light', 'dark', 'ctrl'], pal=[orange[1], blue[1], gray[2]])
plt.xticks(ticks=[0, 1, 2], labels=['Light', 'Dark', 'Neutral'], color='black')
ax2=plt.subplot(1,2,2)
plot_bars(dm_df1, x='blocks', y='mean_pupil', hue='condition', hue_order=['light', 'dark', 'ctrl'], xlab='Experimental Block rank', pal=[orange[1], blue[1], gray[2]])
for ax in [ax1, ax2]:
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.show()

plt.figure(figsize=(30,10))
ax1=plt.subplot(1,3,1)
dm2_AB_ = ops.sort(dm2_AB_, by=dm2_AB_.pupil_change)
dm_df2 = convert.to_pandas(dm2_AB_)
plot_bars(dm_df2, order=['light', 'dark'], pal=[orange[1], blue[1]])
plt.xticks(ticks=[0, 1], labels=['Light', 'Dark'])
ax2=plt.subplot(1,3,2)
dm2_AB_ = ops.sort(dm2_AB_, by=dm2_AB_.pupil_change)
dm_df2 = convert.to_pandas(dm2_AB_)
plot_bars(dm_df2, x='rating', y='mean_pupil', hue='condition', hue_order=['light', 'dark'], xlab='Effort ratings', pal=[orange[1], blue[1]], ylab='')
plt.xticks(color='black', fontsize=30)
ax3=plt.subplot(1,3,3)
plot_bars(dm_df2, x='trials', y='mean_pupil', hue='condition', hue_order=['light', 'dark'], xlab='Trial rank', pal=[orange[1], blue[1]], ylab='')
plt.xticks(ticks=range(0,11), labels=range(1,12), color='black', fontsize=30)
for ax in [ax1, ax2, ax3]:
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.show()

plt.figure(figsize=(10,10))
plot_bars(dm_df2, y='trace_length_read', pal=[orange[1], blue[1]], order=['light', 'dark'], ylab='Mean reading time (ms)')
plt.show()

# Check negative scores rate (or positive)
N = len(dm2_AB_.participant.unique)
neg_vol = set(dm2_AB_.participant[dm2_AB_.pupil_change < 0])
neg_invol = set(dm1_AB_.participant[dm1_AB_.pupil_change_ < 0])
pos_vol = set(dm2_AB_.participant[dm2_AB_.pupil_change > 0])
pos_invol = set(dm1_AB_.participant[dm1_AB_.pupil_change > 0])
both_pos = [i for i in pos_vol if i in pos_invol]

print(f'Voluntary: {len(pos_vol)/N * 100}% ({len(pos_vol)}/{N})')
print(f'Involuntary: {len(pos_invol)/N * 100}% ({len(pos_invol)}/{N})')
print('Both:', len(both_pos)/N * 100)

# =============================================================================
# Process the questionnaire data
# =============================================================================
# Get questionnaires data
df = pd.read_csv(cwd+questfile,sep=',')

# Match participants
new, df['excluded'] = dm1_AB_.participant.unique, 0
for i in list(df.index):
    if df.Q00ID[i] in new: 
        df.loc[i, 'excluded'] = 0
    else: 
        df.loc[i, 'excluded'] = 1
df = df[df.excluded == 0].reset_index()
print(list(np.sort(df.Q00ID)) == list(np.sort(new)))

# Rename columns
df.rename(columns={'INVOL01': 'vivid_stories', 'INVOL02': 'vivid_dreams','Q01EXP': 'freq_invol', 'Q02EXP': 'freq_vol'}, inplace=True)

# Appropriately change scale (8 = 'only for specific words/scenarios' -> recoded as 'half ot the time')
df.freq_invol[df.freq_invol == 8] = 4
df.freq_vol[df.freq_vol == 8] = 4

# Retrieve the QMI columns 
QMI = [i for i in df.columns if i.startswith('QMI')]
QMI_visual = [i for i in df.columns if i.startswith('QMIQ02')]
QMI_audio = [i for i in df.columns if i.startswith('QMIQ03')]
QMI_touch = [i for i in df.columns if i.startswith('QMIQ04')]
QMI_motor = [i for i in df.columns if i.startswith('QMIQ05')]
QMI_taste = [i for i in df.columns if i.startswith('QMIQ06')]
QMI_smell = [i for i in df.columns if i.startswith('QMIQ07')]
QMI_feeling = [i for i in df.columns if i.startswith('QMIQ08')]

# Compute the mean QMI scores
df['QMI_total'] = df[QMI].mean(axis=1)
df['QMI_visual'] = df[QMI_visual].mean(axis=1)
df['QMI_audio'] = df[QMI_audio].mean(axis=1)
df['QMI_touch'] = df[QMI_touch].mean(axis=1)
df['QMI_motor'] = df[QMI_motor].mean(axis=1)
df['QMI_taste'] = df[QMI_taste].mean(axis=1)
df['QMI_smell'] = df[QMI_smell].mean(axis=1)
df['QMI_feeling'] = df[QMI_feeling].mean(axis=1)

# Compute the mean scores for items about vividness of involuntary imagery
df['vivid_invol'] = (df.vivid_dreams + df.vivid_stories)/2

# Cronbach's alpha for QMI questionnaire
alpha = pg.cronbach_alpha(data=df[QMI])
print(f"Cronbach's alpha: {alpha}")

# Descriptive stats for demographic items
demo = [i for i in df.columns if i.startswith('Q00')]
desc_stats_demo = np.round(df[demo].describe(include='all'), 2).transpose()
df.groupby('Q00SEX').Q00AGE.describe(include='all') # Age by sex
desc_stats_demo.to_csv(outputfolder+'descriptives_demo.csv', encoding='utf-8', index=True)

# Descriptive stats for QMI and other items
questlist = [i for i in df.columns if (i.startswith('QMI_') or i.startswith('freq_') or i.startswith('vivid_'))]
desc_stats_quest = np.round(df[questlist].describe(include='all'), 2).transpose() # get descriptives and round to the 2nd decimal
desc_stats_quest.to_csv(outputfolder+'descriptives_quest.csv', encoding='utf-8', index=True)

# See scores of those who have a mean score under 3 for the visual subscale of the QMI
for a in list(df.Q00ID[df.QMI_visual<=3]):
    print(a, df.QMI_visual[df.Q00ID==a])

# Check if any aphant
df.Q00Aphant2.describe(include='all')
df.Q00Aphant.describe(include='all')

# Print their scores
for a in list(df.Q00ID[df.Q00Aphant2=='Y']):
    print(a, df.QMI_visual[df.Q00ID==a])
for a in list(df.Q00ID[df.Q00Aphant2=='M']):
    print(a, df.QMI_visual[df.Q00ID==a])

# Add columns of interest to the datamatrix
dm1_AB_.QMI_visual, dm1_AB_.freq_vol = '', ''
dm1_AB_.vivid_invol, dm1_AB_.freq_invol = '', ''
dm1_AB_.aphant = ''
for p, sdm in ops.split(dm1_AB_.participant):
    sub_df = df[df['Q00ID'] == p] # Subset the df too
    dm1_AB_.QMI_visual[sdm] = float(sub_df.QMI_visual.iloc[0])
    dm1_AB_.freq_vol[sdm] = float(sub_df.freq_vol.iloc[0])
    
    dm1_AB_.vivid_invol[sdm] = float(sub_df.vivid_invol.iloc[0])
    dm1_AB_.freq_invol[sdm] = float(sub_df.freq_invol.iloc[0])
    
    dm1_AB_.aphant[sdm] = str(sub_df.Q00Aphant2.iloc[0])
    
dm2_AB_.QMI_visual, dm2_AB_.freq_vol = '', ''
dm2_AB_.vivid_invol, dm2_AB_.freq_invol = '', ''
dm2_AB_.aphant = ''
for p, sdm in ops.split(dm2_AB_.participant):
    sub_df = df[df['Q00ID'] == p] # Subset the df too
    dm2_AB_.QMI_visual[sdm] = float(sub_df.QMI_visual.iloc[0])
    dm2_AB_.freq_vol[sdm] = float(sub_df.freq_vol.iloc[0])
    
    dm2_AB_.vivid_invol[sdm] = float(sub_df.vivid_invol.iloc[0])
    dm2_AB_.freq_invol[sdm] = float(sub_df.freq_invol.iloc[0])

    dm2_AB_.aphant[sdm] = str(sub_df.Q00Aphant2.iloc[0])
    
# =============================================================================
# Visualise the individual effects per participant and per stimulus
# =============================================================================
plt.rcParams['font.size'] = 40
dm1_AB_ = ops.sort(dm1_AB_, by=dm1_AB_.pupil_change)
dm_df1 = convert.to_pandas(dm1_AB_)

dm2_AB_ = ops.sort(dm2_AB_, by=dm2_AB_.pupil_change)
dm_df2 = convert.to_pandas(dm2_AB_)

fig = plt.figure(figsize=(20,13)) # 25 13
ax1=plt.subplot(1,1,1)
#plt.title('\n\n\nA) Involuntary\n', loc='left')
sns.pointplot(data=dm_df1, x='participant', y='pupil_change', hue='aphant', hue_order=['N', 'Y', 'M'], scale=3.0, palette=['black', 'red', 'orange'])
plt.xlabel('Participants', color='black');plt.ylabel('Pupil Size Mean Differences\n(Dark - Bright) (a.u)', color='black', fontsize=40)
handles, labels = ax1.get_legend_handles_labels()
plt.legend('')
fig.legend(handles, ['No', 'Yes', 'Maybe'], loc='upper center', title='Do you think this definition of aphantasia fits you?', ncol=3, frameon=False, fontsize=40)
ax1.spines['bottom'].set_visible(True)
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_color('black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.xticks([]);plt.yticks(color='black')
plt.xlim([-1, 51])
plt.axhline(0, color='black')
plt.show()

fig = plt.figure(figsize=(20,13))
ax2=plt.subplot(1,1,1)
#plt.title('\n\n\nB) Voluntary\n', loc='left')
sns.pointplot(data=dm_df2, x='participant', y='pupil_change', hue='aphant', hue_order=['N', 'Y', 'M'], scale=3.0, palette=['black', 'red', 'orange'])
plt.xlabel('Participants', color='black');plt.ylabel('Pupil Size Mean Differences\n(Dark - Bright) (a.u)', color='black', fontsize=40)
ax2.spines['bottom'].set_visible(True)
ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_visible(True)
ax2.spines['left'].set_color('black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.xticks([]);plt.yticks(color='black')
plt.legend('')
fig.legend(handles, ['No', 'Yes', 'Maybe'], loc='upper center', title='Do you think this definition of aphantasia fits you?', ncol=3, frameon=False, fontsize=40)
plt.tight_layout()
plt.xlim([-1, 51])
plt.axhline(0, color='black')
plt.show()

# See pupil changes per word
df_codes = pd.read_csv(cwd+'edf_stim_codes.csv', sep=',')

word_summary(dm1_AB_, 'invol', df_codes, 100, 150) # during the 1000 to 1500 ms period
word_summary(dm2_AB_, 'vol', df_codes, 400, 1100) # during the whole 7-s imagery period

# Plot relationship between objective and subjective measures
plt.figure(figsize=(10,10))
plt.rcParams['font.size'] = 35
ax1=plt.subplot(1,1,1)
sns.regplot(x=dm_df1.vivid_invol, y=dm_df1.pupil_change, label='Involuntary', lowess=True, color='red')
sns.regplot(x=dm_df1.vivid_invol, y=dm_df1.pupil_change, lowess=False, color='red', ci=95)

sns.regplot(x=dm_df2.QMI_visual, y=dm_df2.pupil_change, label='Voluntary', lowess=True, color='blue')
sns.regplot(x=dm_df2.QMI_visual, y=dm_df2.pupil_change, lowess=False, color='blue')

ax1.spines['bottom'].set_visible(True)
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_color('black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.xlabel('Subjective Vividness ratings');plt.ylabel('Pupil Size Mean Differences (a.u.)', fontsize=35)
plt.legend()
plt.show()

# =============================================================================
# Test the correlations between subjective and objective measures
# =============================================================================
dm1_AB_.pupil_change_vol = NAN # add column to datamatrix 
for p, sdm in ops.split(dm1_AB_.participant):
    dm1_AB_.pupil_change_vol[sdm] = dm2_AB_.pupil_change[dm2_AB_.participant == p].mean
    
# Spearman's rank correlations
test_correlation(dm1_AB_, x='pupil_change', y='vivid_invol', alt='greater')
test_correlation(dm1_AB_, x='pupil_change', y='blocks', alt='less')
test_correlation(dm1_AB_, x='pupil_change', y='mean_pupil', alt='less')

test_correlation(dm2_AB_, x='pupil_change', y='QMI_visual', alt='greater')
test_correlation(dm2_AB_, x='pupil_change', y='rating', alt='less')
test_correlation(dm2_AB_, x='pupil_change', y='mean_pupil', alt='less')

test_correlation(dm1_AB_, x='QMI_visual', y='vivid_invol', alt='greater')
test_correlation(dm2_AB_, x='QMI_visual', y='rating', alt='less')

test_correlation(dm1_AB_, x='pupil_change', y='pupil_change_vol', alt='greater')

# =============================================================================
# Assess the emotional intensity of the stimuli
# =============================================================================
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

# Get scenarios
df_scenarios = pd.read_csv(cwd+'/stim/words_voluntary.csv',sep=',')

# VALENCE from https://realpython.com/python-nltk-sentiment-analysis/
sia = SentimentIntensityAnalyzer()
df_scenarios['EC']=0
df_scenarios['Effort']=0
for cond in set(df_scenarios.condition_vol):
    df_=df_scenarios[df_scenarios.condition_vol==cond]
    val=[]
    for text in set(df_.text):
        EC=np.round(sia.polarity_scores(text)['compound'],2)
        df_scenarios.EC[df_scenarios.text==text] = EC
        code=df_codes.code[df_codes.stim==text]
        dm_r = dm2_AB_.trialid == list(code)[0]
        df_scenarios.Effort[df_scenarios.text==text] = np.round(dm_r.rating.mean,2)
        val.append(EC)
    print(cond, val, np.mean(val))
    
# Write into csv file
df_scenarios.to_csv(outputfolder+'EC_voluntary.csv', encoding='utf-8', index=True)

# Get words
df_words = pd.read_csv(cwd+'/stim/words_involuntary.csv',sep=',')

# VALENCE from https://realpython.com/python-nltk-sentiment-analysis/
sia = SentimentIntensityAnalyzer()
df_words['EC']=0
for cond in set(df_words.condition_in):
    df_=df_words[df_words.condition_in==cond]
    val=[]
    for word in set(df_.word):
        EC=np.round(sia.polarity_scores(word)['compound'],2)
        df_words.EC[df_words.word==word] = EC
        val.append(EC)
        
    print(cond, val, np.mean(val))
    
# Write into csv file
df_words.to_csv(outputfolder+'EC_involuntary.csv', encoding='utf-8', index=True)
