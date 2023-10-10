# -*- coding: utf-8 -*-
"""
@author: robfren
"""

import numpy as np

# Defining functions for later

def get_sim_metric(baseline_ts, comp_ts):
    
    """calculate similarity metrics"""
    
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
    Get a similarity metric for all baseline samples, baseline_ts in this 
    example is a rotating Leave-One-Out (LOO) mean where the one left out is 
    the comp_ts. Resulting correlations are averaged to assess general 
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

#-----------------------------------------------------------------------------#

# creating similarity metric and baseline robustness score

# load baseline and full sample timeseries
baseline = np.loadtxt(
    './baseline_timeseries/baseline_timeseries-standardized.csv',
    delimiter=',')

allsubjects = np.loadtxt(
    './all_subjects_timeseries/all_subjects_timeseries-standardized.csv',
    delimiter=',')

# create mean "ground truth" consensus from baseline
baseline_ts = np.nanmean(baseline, axis=0)

# creating similarity metrics (rz) by comparing baseline to individual timeseries
similarity_metrics = [get_sim_metric(baseline_ts, i)[1] for i in allsubjects]

# assessing robustness of baseline by checking average correlation
br = baseline_robustness(baseline)

print(br)
