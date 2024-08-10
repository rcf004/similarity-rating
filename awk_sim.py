# -*- coding: utf-8 -*-
"""
@author: robfren
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection


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

def LOO_cross_validation(baseline):
    
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

## NEW ##
# load in OA baseline and timeseries
oa_baseline = np.loadtxt(
    './baseline_timeseries/oa_baseline_timeseries-standardized.csv',
    delimiter=',')

# load baseline subject ids - these are ordered corresponding to the timeseries
oa_baseline_ids = np.loadtxt(
    './baseline_timeseries/oa_baseline_subid.csv',
    delimiter=',', dtype=str)
## --- ##


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

# create mean "ground truth" consensus using Older Adult baseline
oa_baseline_ts = np.nanmean(oa_baseline, axis=0)

# creating similarity metrics (rz) by comparing baseline to individual timeseries
similarity_metrics = [get_sim_metric(baseline_ts, i)[1] for i in allsubjects]
similarity_metrics_nontr = [get_sim_metric(baseline_ts, i)[0] for i in allsubjects]

# create similarity metric using older adult basline for all subjects
similarity_metrics_oa = [get_sim_metric(oa_baseline_ts, i)[1] for i in allsubjects]
# create similarity metric using respective age baseline (YA use YA baseline, OA use OA baseline) 

## --- ##

# As these scores are already present in the file "OA_YA_static-response_awk-sim_public.csv"
# compare the similarity score calculated here to the one in the csv file

# load file
similarity_validation = pd.read_csv('OA_YA_static-response_awk-sim_public.csv')

# ALL subjects are in the `similarity_metrics` variable- including baseline.
# let's drop the baseline subjects as they are not-independent from the mean

bool_baseline = np.isin(allsubject_ids, baseline_ids) # find overlap|

non_baseline_ts = allsubjects[~bool_baseline] # get the non-baseline time series
non_baseline = np.array(similarity_metrics)[~bool_baseline] # get the non-baseline subject scores
non_baseline_ids = allsubject_ids[~bool_baseline] # same as above but with ids

# pull `similarity_validation` scores and compare them to the ones we just calculated

# sort both based on ID to ensure same order

# check that sorting the ids results in the same order
check = np.equal(np.sort(non_baseline_ids), np.sort(similarity_validation.ID))
print(np.all(check)) # should be True
print('\n----------------------------\n')

sim_score_orig = similarity_validation.awk_sim_ya_baseline.values[np.argsort(similarity_validation.ID)]
sim_score_new = non_baseline[np.argsort(non_baseline_ids)]

# not exactly equal to 0 due to rounding/floating point error when exporting/loading, but very close
sim_score_comparison = sim_score_orig - sim_score_new
print(sim_score_comparison)
print('\n----------------------------\n')

# when rounded up to 9th decimal they are equal
sim_score_comp_rounded = sim_score_orig.round(9) - sim_score_new.round(9)
print(sim_score_comp_rounded)
print('\n----------------------------\n')



#-----------------------------------------------------------------------------#
#%% 

# add in older adult baseline version of similarity score

# sort pandas dataframe by subject ID, to match input of new similarity score
similarity_validation.sort_values(by=['ID'], inplace=True)

# sort older adult baseline similarity score by this same ID
non_baseline_oa = np.array(similarity_metrics_oa)[~bool_baseline] # get the non-baseline subject scores
non_baseline_ids = allsubject_ids[~bool_baseline] # pull the non-baseline id numbers for sorting
sim_score_oa = non_baseline_oa[np.argsort(non_baseline_ids)]

# add in additional non transformed version of similarity score
non_baseline_nontf = np.array(similarity_metrics_nontr)[~bool_baseline]
sim_score_ya_nontf = non_baseline_nontf[np.argsort(non_baseline_ids)]


# ensure sorting worked correctly, similarity_validation ID column should be equal to the sorted non_baseline_ids
check = np.equal(np.sort(non_baseline_ids), similarity_validation.ID)
print(np.all(check))

# given same subject order- add in new column to similarity_validation

similarity_validation['awk_sim_oa_baseline'] = sim_score_oa

# create a new column that uses an OA baseline for OA and a YA baseline for YA
similarity_validation['awk_sim'] = np.where(similarity_validation['Age']=='OA',
                                            similarity_validation['awk_sim_oa_baseline'],
                                            similarity_validation['awk_sim_ya_baseline'])

# linear regression to assess relation between different baseline similarity scores
bl_comp = smf.ols('awk_sim_ya_baseline ~ awk_sim_oa_baseline', data=similarity_validation).fit()
bl_comp.summary()



#-----------------------------------------------------------------------------#
#%%

## Reliability and robustness tests ##

# assessing between subject reliability of baseline sample by performing a LOO
# cross validation with all baseline subjects.

br_ya = LOO_cross_validation(baseline)

# additional version with OA baseline
br_oa = LOO_cross_validation(oa_baseline)
## --- ##

print(f'leave-one-out cross validation, young adult baseline: {br_ya}')
print(f'leave-one-out cross validation, older adult baseline: {br_oa}')
print('\n----------------------------\n')

# assessing reliability of test sample via split half correlation

# divide each participant's full rating into 2 evenly split sub ratings
start_stop_subrating = even_subdivide(len(baseline_ts), 2)

# init lists to hold dataframe information
sub_scores_ya = []
sub_scores_oa = []
sub_scores_cong = []
sub_ID = []
section = []

# iterate through timeseries (and ID)
for ID, ts in zip(non_baseline_ids, non_baseline_ts):
    sect = 1 # reset this count each subject loop
    
    # iterate through the subsections per subject and get awk. sim for each
    for start, stop in start_stop_subrating:
        sub_scores_ya.append(get_sim_metric(baseline_ts[start:stop], ts[start:stop])[1])
        
        ## NEW ##
        # add in new baselines, oa and congruent
        sub_scores_oa.append(get_sim_metric(oa_baseline_ts[start:stop], ts[start:stop])[1])
        if 'YA' in ID:
            sub_scores_cong.append(get_sim_metric(baseline_ts[start:stop], ts[start:stop])[1])
        else:
            sub_scores_cong.append(get_sim_metric(oa_baseline_ts[start:stop], ts[start:stop])[1])
        ## --- ##
        
        sub_ID.append(ID)
        section.append(sect)
        sect += 1

# put this into a dataframe to analyze more easily
split_awk_sim = pd.DataFrame()
split_awk_sim['ID'] = sub_ID
split_awk_sim['subsection'] = section
split_awk_sim['awk_sim_ya_bl'] = sub_scores_ya
split_awk_sim['awk_sim_oa_bl'] = sub_scores_oa
split_awk_sim['awk_sim'] = sub_scores_cong

# split half correlation
for col, title in zip(['awk_sim_ya_bl', 'awk_sim_oa_bl', 'awk_sim'], ['ya baseline', 'oa basline', 'congruent basline']):
    truncated_df = split_awk_sim[['ID', 'subsection', col]]
    pivoted = pd.pivot_table(truncated_df, index='ID', columns='subsection')
    pivoted.columns = pivoted.columns.droplevel()

    r_half = np.corrcoef(pivoted[1], pivoted[2])[0][1]
    # spearman-brown 
    r_full = r_half * 2 / (1 + r_half)
    print(f'spearman-brown corrected split half correlation, {title}: {r_full}')

print('\n----------------------------\n')

#-----------------------------------------------------------------------------#
#%%

## Hypothesis 1 tests ##

# Hypothesis 1A: Shapirio-Wilk tests


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
print('\n----------------------------')



# Hypothesis 1B: regression



# filter dataframe to Young and Older Adults separately
OA_only = similarity_validation[similarity_validation['Age']=='OA']
YA_only = similarity_validation[similarity_validation['Age']=='YA']

# run regression (Ordinary Least Squares)
young_adult_validation = smf.ols('mean_correct ~ awk_sim', data=YA_only).fit()
print('young adult mean correct by awk. similarity regression')
print(young_adult_validation.summary())
print('\n----------------------------')

# check awk sim against all subscales, order by variance explained.
var_explained_subscales = {}
for subscale in ['perc_emotion', 'perc_belief', 'perc_motivation', 'perc_fauxpas', 
                 'perc_deception']:
    cor_res = stats.pearsonr(YA_only['awk_sim'], YA_only[subscale])
    print(f'{subscale} - R: {cor_res[0]}, p: {cor_res[1]}')
    W, p = stats.shapiro(YA_only[subscale])
    print(f'{subscale} - W: {W}, p: {p}')
    
    res = smf.ols(f'{subscale} ~ awk_sim', data=YA_only).fit()
    
    # additionally pull skew and kurtosis as measures of ceiling effects
    sk = stats.skew(YA_only[f'{subscale}'], bias=True)
    ku = stats.kurtosis(YA_only[f'{subscale}'], bias=True)
    
    var_explained_subscales[subscale] = [res.rsquared, res.pvalues[1], sk, ku]

r2 = np.array([i[0] for i in var_explained_subscales.values()])
pval = np.array([i[1] for i in var_explained_subscales.values()])
subsc = np.array([i for i in var_explained_subscales.keys()])
skew = np.array([i[2] for i in var_explained_subscales.values()])
kurt = np.array([i[3] for i in var_explained_subscales.values()])

# sort r2, pvalues and subscale names by order of decreasing variance explained
r2_sort = r2[np.argsort(r2)][::-1]
pval_sort = pval[np.argsort(r2)][::-1]
subsc_sort = subsc[np.argsort(r2)][::-1]
skew_sort = skew[np.argsort(r2)][::-1]
kurt_sort = kurt[np.argsort(r2)][::-1]

clean = [f'{s}\t\t|{r:.4f}\t|{p:.4f}\t|{sk:.3f}\t|{ku:.3f}' if len(s) < 12 else f'{s}\t|{r:.4f}\t|{p:.4f}\t|{sk:.3f}\t|{ku:.3f}' for s, r, p, sk, ku in zip(subsc_sort, r2_sort, pval_sort, skew_sort, kurt_sort)]
clean = ['subscale\t\t|r2\t\t|pvalue\t|skew\t|kurtosis', '-'*45] + clean
print('variance explained for all subscales of static response task ordered')
print('\n'.join(clean))
print('\n----------------------------')

# fdr correction for multiple comparisons
fdr_pval = fdrcorrection(pval)

## --- ##


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
# dummy code age: YA=1, OA=0
similarity_validation['Age_dummy'] = np.where(similarity_validation['Age'] == 'YA', 1, 0)

# initial step without age
step_1 = smf.ols('mean_correct ~ awk_sim_ya_baseline', data=similarity_validation).fit()

# step 1 multiple regression with Age (dummy coded)
step_2 = smf.ols('mean_correct ~ awk_sim_ya_baseline + C(Age_dummy)', data=similarity_validation).fit()

# step 2 same regression model as above, with the new addition of an interaction term
step_3 = smf.ols('mean_correct ~ awk_sim_ya_baseline + C(Age_dummy) + Age_dummy:awk_sim_ya_baseline', data=similarity_validation).fit()

# model comparison between step 1, 2, 3
from statsmodels.stats.anova import anova_lm

model_comparison = anova_lm(step_1,step_2,step_3)

# print model outputs
# step 1
print(f"Step 1 of model\n{step_1.summary()}\n\n")

# step 2
print(f"Step 2 of model\n{step_2.summary()}\n\n")

# step 3
print(f"Step 3 of model\n{step_3.summary()}\n\n")

# model comparison between model steps
print(f"Comparison of Steps 1-3\n{model_comparison}\n")



#-----------------------------------------------------------------------------#
#%%

# Hypothesis 3 tests

# read in fMRI false-belief contrast ROI + accuraccy + RT csv
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

print('\n----------------------------')
# False Discovery Rate correction for hypothesis 3 multiple comparisions
fdr_fmri_pval = fdrcorrection([rTPJ.pvalues['awk_sim'], 
                               PCC.pvalues['awk_sim'], 
                               dmPFC.pvalues['awk_sim'], 
                               vmPFC.pvalues['awk_sim']])

print('\n----------------------------')
print(f'fdr correction for results: {fdr_fmri_pval}')


## --- ##

#-----------------------------------------------------------------------------#
#%%
# additional measures- intercorrelation matrix

# merge fmri and YA_only into larger dataframe for correlation across all measures

fullYA = similarity_validation.merge(fmri[['ID', 'rtpj', 'pcc', 'dmpfc', 'vmpfc']], on='ID')
fullYA = fullYA[fullYA.columns[7:]] # drop subscales
fullYA = fullYA.drop(columns=['awk_sim','awk_sim_oa_baseline', 'Age_dummy']) # drop dummy coded age variable and alternate baseline
 

# create intercorrelation matrix of all tested variables
intercorrelation = fullYA.corr()
print(intercorrelation.to_string())

#-----------------------------------------------------------------------------#
#%%

# full pairwise regressions (permutations of variables) to get significance of correlation table
from itertools import combinations

sig_cor = {}

for t in combinations(['mean_correct', 'awk_sim_ya_baseline', 'rtpj', 'pcc', 'dmpfc', 'vmpfc'], 2):
    res = smf.ols(f'{t[0]} ~ {t[1]}', data=fullYA).fit()
    if res.f_pvalue < 0.05:
        sig_cor[f'{t[0]} ~ {t[1]}'] = True
    else:
        sig_cor[f'{t[0]} ~ {t[1]}'] = False

#-----------------------------------------------------------------------------#
#%%

# 
