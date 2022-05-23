# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:17:46 2020

@author: orteg
"""
import scipy.stats as stats
import pandas as pd
import numpy as np

### Functions ####
def imandavenport_test(num_models, num_datasets, model_ranks):

  chisqF = (12*num_datasets/(num_models*(num_models+1)))*(sum(model_ranks**2)-(num_models*(num_models+1)**2)/4)
  Ff = (num_datasets-1)*chisqF/(num_datasets*(num_models-1)-chisqF)

  df1 = num_models - 1
  df2 = (num_models-1)*(num_datasets-1)
  pvalue = 1 - stats.f.cdf(Ff, df1, df2)
  print('p-value ROC ranks: ', pvalue)
  return pvalue
    
def bonfholm_test(num_models, num_datasets, model_names, model_ranks, alpha, control = None):
  denominator_ = (num_models*(num_models+1)/(6*num_datasets))**0.5
  list_pv_ = np.array(model_ranks)
  if control == None:
    ix_min_ = np.where(list_pv_==np.min(model_ranks))[0]
  else:
    ix_min_ = np.where(model_names == control)[0]

  list_pv = np.delete(list_pv_, ix_min_)
  model_names_ = np.delete(model_names, ix_min_)

  z_scores = np.asarray([ (list_pv[i] - model_ranks[ix_min_] )/denominator_ for i, _ in enumerate(list_pv) ])
  p_values = stats.norm.sf(abs(z_scores)) #one-sided
  p_values = p_values.reshape(1,-1).flatten()
  ix_sort = np.argsort(p_values)
  decision = np.ones(num_models - 1, dtype=bool)

  for m, i in enumerate(p_values[ix_sort]):  
    if i <= alpha/(num_models-1-m):
      decision[ix_sort[m]] = False
      
  
  return model_names[ix_min_], model_names_[decision], p_values

## Load Data

df = pd.read_csv('experimental_results.csv')


cols = ['data_partition','flip_ratio','label_noise']

cond = 'flip_ratio == "0.25"'

df_ = df.copy()
df_ = df_.query(cond)
df_filtered = df_.drop(columns = cols)

# Average Metrics
print(df_filtered.copy().groupby('dataset').mean().mean(axis = 0).to_string() )
print(df_filtered.copy().groupby('dataset').std().mean(axis = 0).to_string())

## EXPORT EXCEL FILES
# df_filtered.copy().groupby('dataset').mean().mean(axis = 0).to_csv('mean75_aggregate_results.csv')
# df_filtered.copy().groupby('dataset').std().mean(axis = 0).to_csv('sd75_aggregate_results.csv')

# df_filtered.copy().groupby('dataset').mean().transpose().to_csv('mean50_detailed_dataset_results.csv')
# df_filtered.copy().groupby('dataset').std().transpose().to_csv('std75_detailed_dataset_results.csv')

#### TECHNIQUES ####
roc_models = ['roc_srf_puhd', 'roc_pubag','roc_elkno','roc_welog',
             'roc_spyem','roc_rnkpr','roc_rf',
             'roc_rf_puhd','roc_dt_puhd','roc_dt_hd',
             'roc_upu','roc_nnpu','roc_imbnnpu']

ap_models = ['ap_srf_puhd', 'ap_pubag','ap_elkno','ap_welog',
             'ap_spyem','ap_rnkpr','ap_rf',
             'ap_rf_puhd','ap_dt_puhd','ap_dt_hd',
             'ap_upu','ap_nnpu','ap_imbnnpu']

f1max_models = ['f1_max_srf_puhd', 'f1_max_pubag', 'f1_max_elkno',
                'f1_max_spyem','f1_max_rf', 'f1_max_rf_puhd',
                'f1_max_dt_puhd','f1_max_dt_hd','f1_max_upu',
                'f1_max_nnpu','f1_max_imbnnpu']

## Evaluation ##

# Optimal F1-score
df_f1scoremax_ = df_[f1max_models].copy().rank(axis = 1, ascending = False)
df_f1scoremax_['dataset'] = df_['dataset']

df_f1_max_transpose = df_f1scoremax_.groupby('dataset').mean().T
df_f1scoremax_rank_mean_ = df_f1_max_transpose.mean(axis = 1)

# PR-AUC
df_ap_ = df_[ap_models].copy().rank(axis = 1, ascending = False)
df_ap_['dataset'] = df_['dataset']

df_ap_transpose = df_ap_.groupby('dataset').mean().T
df_ap_rank_mean_ = df_ap_transpose.mean(axis = 1)

# ROC-AUC 
df_roc_ = df_[roc_models].copy().rank(axis = 1, ascending = False)
df_roc_['dataset'] = df_['dataset']

df_roc_transpose = df_roc_.groupby('dataset').mean().T
df_roc_rank_mean_ = df_roc_transpose.mean(axis = 1)

#### HYPOTHESIS TESTING ####

## Optimal F1-Score
N = len(df_f1scoremax_.index)
k = len(df_f1scoremax_rank_mean_)

model_names_ = np.asarray(list(df_f1scoremax_rank_mean_.index))

imandavenport_test(k, N, df_f1scoremax_rank_mean_)

# Holm's Test F1 Score Max
lowest_rank_f1scoremax, no_rejected_f1scoremax, adjpvalues_f1scoremax = bonfholm_test(k, N, model_names_, df_f1scoremax_rank_mean_, 0.05, control=None)

## PR-AUC
N = len(df_ap_.index)
k = len(df_ap_rank_mean_)

model_names_ = np.asarray(list(df_ap_rank_mean_.index))

imandavenport_test(k, N, df_ap_rank_mean_)

# Holm's Test PR
lowest_rank_pr, no_rejected_pr, adjpvalues_pr = bonfholm_test(k, N, model_names_, df_ap_rank_mean_, 0.05, control=None)

## ROC-AUC
N = len(df_roc_.index)
k = len(df_roc_rank_mean_)

model_names_ = np.asarray(list(df_roc_rank_mean_.index))

imandavenport_test(k, N, df_roc_rank_mean_)

# Holm's Test PR
lowest_rankroc_, no_rejectedroc_, adjpvaluesroc_ = bonfholm_test(k, N, model_names_, df_roc_rank_mean_, 0.05, control=None)