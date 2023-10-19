# -*- coding: utf-8 -*-
"""
@author: robfren
"""

import numpy as np

# Defining functions to calculate similarity metrics

def get_sim_metric(baseline_ts, comp_ts):
    
    """
    Calculate similarity metric from a baseline timeseries and comparison
    timeseries. This function returns both the pearsons correlation and the 
    fisher r-z transformed correlation.
    """
    
    baseline_ts = np.array(baseline_ts)        # ensure ts is numpy array
    comp_ts = np.array(comp_ts)                # ensure ts is numpy array
    
    nanidx = np.argwhere(np.isnan(comp_ts))    # get NaN indices in comp_ts
    comp_ts = np.delete(comp_ts, nanidx)       # remove NaNs in comp_ts
    temp_mean = np.delete(baseline_ts, nanidx) # remove corresponding indices in baseline_ts
    
    r = np.corrcoef(temp_mean, comp_ts)[0][1]  # get pearsons r
    rz = np.arctanh(r)                         # fisher r-z transform r
    
    return r, rz

def baseline_robustness(baseline):
    
    """
    Get a similarity metric for all baseline samples. `baseline_ts` in this 
    example is a rotating Leave-One-Out (LOO) mean where the one left out is 
    the `comp_ts`. Resulting correlations are averaged to assess general 
    alignment across baseline consensus.
    """
    
    all_rs = [] 
    
    for idx, comp_ts in enumerate(baseline):   # iterate over all baseline timeseries 
        
        # drop this iteration's timeseries from the basline (LOO)
        LOO_baseline = np.append(baseline[:idx], baseline[idx+1:], axis=0)
        
        # get average of this new baseline
        baseline_ts = np.nanmean(LOO_baseline, axis=0)
        
        # pull r (unstandardized)
        r, _ = get_sim_metric(baseline_ts, comp_ts) 
        
        all_rs.append(r)
    
    mean_corr = np.mean(all_rs)
    
    return mean_corr

def even_subdivide(timeseries_length, divisions):
    """
    Create even subdivisions of an array of length `timeseries_length`
    subdivisions are forced to be integers (floor division) so the final division
    may be slightly larger than the others. Resulting array that is returned is
    a list of tuples, where each tuple is the (start, stop) of all divisions
    """
    
    stops = []
    for i in range(divisions):
        stops.append(timeseries_length//divisions * (i+1))
    starts = [0] + stops[:-1]
    stops[-1] = timeseries_length
    start_stop = []
    for start, stop in zip(starts, stops):
        start_stop.append((start, stop))
    
    return start_stop

#-----------------------------------------------------------------------------#
#%%

# creating similarity metric and baseline robustness score

# load baseline and full sample timeseries
baseline = np.loadtxt(
    './baseline_timeseries/baseline_timeseries-standardized.csv',
    delimiter=',')

# load baseline subject ids - these are ordered corresponding to the timeseries
baseline_ids = np.loadtxt(
    './baseline_timeseries/baseline_subid.csv',
    delimiter=',', dtype=str)


allsubjects = np.loadtxt(
    './all_subjects_timeseries/all_subjects_timeseries-standardized.csv',
    delimiter=',')

# load all subject ids - these are ordered corresponding to the timeseries
 
allsubject_ids = np.loadtxt(
    './all_subjects_timeseries/all_subjects_subid.csv',
    delimiter=',', dtype=str)

 
#-----------------------------------------------------------------------------#
#%% 

# create mean "ground truth" consensus from baseline
baseline_ts = np.nanmean(baseline, axis=0)

# creating similarity metrics (rz) by comparing baseline to individual timeseries
similarity_metrics = [get_sim_metric(baseline_ts, i)[1] for i in allsubjects]

# As these scores are already present in the file "OA_YA_static-response_awk-sim_public.csv"
# compare the similarity score calculated here to the one in the csv file

import pandas as pd

# load file
similarity_validation = pd.read_csv('OA_YA_static-response_awk-sim_public.csv')

# ALL subjects are in the `similarity_metrics` variable- including baseline.
# let's drop the baseline subjects as they are not-independent from the mean

bool_baseline = np.isin(allsubject_ids, baseline_ids) # find overlap

non_baseline_ts = allsubjects[~bool_baseline] # get the non-baseline time series
non_baseline = np.array(similarity_metrics)[~bool_baseline] # get the non-baseline subject scores
non_baseline_ids = allsubject_ids[~bool_baseline] # same as above but with ids

# pull `similarity_validation` scores and compare them to the ones we just calculated

# sort both based on ID to ensure same order

# check that sorting the ids results in the same order
check = np.equal(np.sort(non_baseline_ids), np.sort(similarity_validation.ID))
print(check) # should be all True

sim_score_orig = similarity_validation.awk_sim.values[np.argsort(similarity_validation.ID)]
sim_score_new = non_baseline[np.argsort(non_baseline_ids)]

# not exactly equal to 0 due to rounding/floating point error when exporting/loading, but very close
sim_score_comparison = sim_score_orig - sim_score_new
print(sim_score_comparison)

# when rounded up to 9th decimal they are equal
sim_score_comp_rounded = sim_score_orig.round(9) - sim_score_new.round(9)
print(sim_score_comp_rounded)

#-----------------------------------------------------------------------------#
#%%

## Reliability and robustness tests ##

# assessing robustness of baseline by checking average correlation
# acts as one measure of variable reliability for the baseline sample

br = baseline_robustness(baseline)
print(br)

# assessing reliability of test sample via split half correlation

# divide each participant's full rating into 2 evenly split sub ratings
start_stop_subrating = even_subdivide(len(baseline_ts), 2)

# init lists to hold dataframe information
sub_scores = []
sub_ID = []
section = []

# iterate through timeseries (and ID)
for ID, ts in zip(non_baseline_ids, non_baseline_ts):
    sect = 1 # reset this count each subject loop
    
    # iterate through the 3 divided subsections per subject and get awk. sim for each
    for start, stop in start_stop_subrating:
        sub_scores.append(get_sim_metric(baseline_ts[start:stop], ts[start:stop])[1])
        sub_ID.append(ID)
        section.append(sect)
        sect += 1

# put this into a dataframe to analyze more easily
split_awk_sim = pd.DataFrame()
split_awk_sim['ID'] = sub_ID
split_awk_sim['subsection'] = section
split_awk_sim['awk_sim'] = sub_scores

# split half correlation
pivoted = pd.pivot_table(split_awk_sim, index='ID', columns='subsection')
pivoted.columns = pivoted.columns.droplevel()

r_half = np.corrcoef(pivoted[1], pivoted[2])[0][1]
# spearman-brown 
r_full = r_half * 2 / (1 + r_half)

#-----------------------------------------------------------------------------#
#%%

## Hypothesis 1 tests ##

# Hypothesis 1A: Shapirio-Wilk tests
from scipy import stats

# static-response test
static_resp_W, static_resp_p = stats.shapiro(stats.zscore(similarity_validation.mean_correct))
# skew and kurtosis
static_resp_skew = stats.skew(stats.zscore(similarity_validation.mean_correct), bias=True)
static_resp_kurt = stats.kurtosis(stats.zscore(similarity_validation.mean_correct), bias=True)

# awkwardness similarity score test
awk_sim_W, awk_sim_p = stats.shapiro(similarity_validation.awk_sim)
# skew and kurtosis
awk_sim_skew = stats.skew(similarity_validation.awk_sim, bias=True)
awk_sim_kurt = stats.kurtosis(similarity_validation.awk_sim, bias=True)

# format results in table for viewing
shapiro_wilk_results = pd.DataFrame()
shapiro_wilk_results['Static-Response'] = [static_resp_W, static_resp_p, static_resp_skew, static_resp_kurt]
shapiro_wilk_results['Awkwardness Similarity Metric'] = [awk_sim_W, awk_sim_p, awk_sim_skew, awk_sim_kurt]
shapiro_wilk_results.index = ['W', 'p', 'skew', 'kurtosis']
print(shapiro_wilk_results)


# Hypothesis 1B: regression
import statsmodels.formula.api as smf

# filter dataframe to Young and Older Adults separately
OA_only = similarity_validation[similarity_validation['Age']=='OA']
YA_only = similarity_validation[similarity_validation['Age']=='YA']

# run regression (Ordinary Least Squares)
young_adult_validation = smf.ols('mean_correct ~ awk_sim', data=YA_only).fit()
print(young_adult_validation.summary())

#-----------------------------------------------------------------------------#
#%%

# Hypothesis 2 tests

# t-tests on age differences in both measures
# static-response
static_resp_ttest = stats.ttest_ind(YA_only.mean_correct, OA_only.mean_correct)
print(static_resp_ttest)

# awkwardness similarity metric
awk_sim_ttest = stats.ttest_ind(YA_only.awk_sim, OA_only.awk_sim)
print(awk_sim_ttest)

# heirarchical regression
# step 1 multiple regression with Age (dummy coded)
step_1 = smf.ols('mean_correct ~ awk_sim + Age', data=similarity_validation).fit()

# step 2 same regression model as above, with the new addition of an interaction term
step_2 = smf.ols('mean_correct ~ awk_sim + Age + Age:awk_sim', data=similarity_validation).fit()

# model comparison between step 1 and step 2
from statsmodels.stats.anova import anova_lm

model_comparison = anova_lm(step_1,step_2)

# print model outputs
# step 1
print(f"Step 1 of model\n{step_1.summary()}\n\n")

# step 2
print(f"Step 2 of model\n{step_1.summary()}\n\n")

# model comparison between step 1 and two
print(f"Comparison of Step 1 and Step 2\n{model_comparison}\n")

#-----------------------------------------------------------------------------#
#%%

# Hypothesis 3 tests

# read in fMRI false-belief contrast ROI csv
fmri = pd.read_csv('YA_false-belief_awk-sim_public.csv')

# run regression models for each ROI

# right temporoparietal junction
rTPJ = smf.ols('rtpj ~ awk_sim', data=fmri).fit()
print('right temporoparietal junction\n', rTPJ.summary(), '\n')

# posterior cingulate cortex
PCC = smf.ols('pcc ~ awk_sim', data=fmri).fit()
print('posterior cingulate cortex\n', PCC.summary(), '\n')

# dorsomedial prefrontal cortex
dmPFC = smf.ols('dmpfc ~ awk_sim', data=fmri).fit()
print('dorsomedial prefrontal cortex\n', dmPFC.summary(), '\n')

# ventromedial prefrontal cortex
vmPFC = smf.ols('vmpfc ~ awk_sim', data=fmri).fit()
print('ventromedial prefrontal cortex\n', vmPFC.summary(), '\n')

#-----------------------------------------------------------------------------#
#%%
# additional measures- intercorrelation matrix

# merge fmri and YA_only into larger dataframe for correlation across all measures

fullYA = similarity_validation.merge(fmri[['ID', 'rtpj', 'pcc', 'dmpfc', 'vmpfc']], on='ID')
fullYA = fullYA[fullYA.columns[-6:]] # drop subscales

# create intercorrelation matrix of all tested variables
intercorrelation = fullYA.corr()
print(intercorrelation.to_string())


#%%

# full pairwise regressions (permutations of variables) to get significance of correlation table
from itertools import combinations

sig_cor = {}

for t in combinations(['mean_correct', 'awk_sim', 'rtpj', 'pcc', 'dmpfc', 'vmpfc'], 2):
    res = smf.ols(f'{t[0]} ~ {t[1]}', data=fullYA).fit()
    if res.f_pvalue < 0.05:
        sig_cor[f'{t[0]} ~ {t[1]}'] = True
    else:
        sig_cor[f'{t[0]} ~ {t[1]}'] = False
