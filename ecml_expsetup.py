# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:01:47 2021

@author: orteg
"""
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from pos_noisyneg.spyEM import SpyEM
from pos_noisyneg.elkannoto import WeightedElkanotoPuClassifier
from pos_noisyneg.rankpruning import RankPruning
from pos_noisyneg.utils import make_noisy_negatives
from pos_noisyneg.PU_bagging import BaggingPuClassifier
from pu_tree_simplified._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified._pu_classes import DecisionTreeClassifier

from imblearn.over_sampling import ADASYN
from ecsmodels.methodologies.cs_boost import CSBoost


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import expit

# List of metrics
list_roc_upu = []
list_ap_upu = []
list_f1_upu = []
list_f1_ci_upu = []
list_f1_max_upu = []

list_roc_nnpu = []
list_ap_nnpu = []
list_f1_nnpu = []
list_f1_ci_nnpu = []
list_f1_max_nnpu = []

list_roc_imbnnpu = []
list_ap_imbnnpu = []
list_f1_imbnnpu = []
list_f1_ci_imbnnpu = []
list_f1_max_imbnnpu = []

list_ap_pubag = []
list_roc_pubag = []
list_f1_pubag = []
list_f1_ci_pubag = []
list_f1_max_pubag = []

list_ap_elkno = []
list_roc_elkno = []
list_f1_elkno = []
list_f1_ci_elkno = []
list_f1_max_elkno = []

list_ap_welog = []
list_roc_welog = []
list_f1_welog = []
list_f1_ci_welog = []
list_f1_max_welog = []

list_ap_spyem = []
list_roc_spyem = []
list_f1_spyem = []
list_f1_ci_spyem = []
list_f1_max_spyem = []

list_ap_rnkpr = []
list_roc_rnkpr = []
list_f1_rnkpr = []
list_f1_ci_rnkpr = []
list_f1_max_rnkpr = []

list_ap_rf = []
list_roc_rf = []
list_f1_rf = []
list_f1_ci_rf = []
list_f1_max_rf = []

list_ap_rf_puhd = []
list_roc_rf_puhd = []
list_f1_rf_puhd = []
list_f1_ci_rf_puhd = []
list_f1_max_rf_puhd = []

list_ap_srf_puhd = []
list_roc_srf_puhd = []
list_f1_srf_puhd = []
list_f1_ci_srf_puhd = []
list_f1_max_srf_puhd = []

list_ap_dt_puhd = []
list_roc_dt_puhd = []
list_f1_dt_puhd = []
list_f1_ci_dt_puhd = []
list_f1_max_dt_puhd = []

list_ap_dt_hd = []
list_roc_dt_hd = []
list_f1_dt_hd = []
list_f1_ci_dt_hd = []
list_f1_max_dt_hd = []

set_thresholds = np.linspace(0.01, 0.99, num=100)


def binarize_prob(prob, threshold=0.50):
  
  labels = np.zeros(prob.shape, dtype=int)
  labels[prob>threshold] = 1
  return labels

#### Generate Data ####

list_dataset = [
                # pd.read_csv('fraud_car_insurance.csv', header = None),
                # pd.read_csv('scaled_creditcard.csv', header = None),
                # pd.read_csv('scaled_forestcover_full.csv', header = None),
                # pd.read_csv('scaled_mammography_full.csv', header = None),
                # pd.read_csv('scaled_piechart2.csv', header = None),
                # pd.read_csv('scaled_pizzacutter1.csv', header = None),
                # pd.read_csv('scaled_satellite.csv', header = None),
                # pd.read_csv('scaled_winequality-red-4.csv', header = None),
                # pd.read_csv('korean5_scaled_imbalanced.csv', header = None),
                # pd.read_csv('chile_scaled_imbalanced.csv', header = None),
                # pd.read_csv('car-good_scaled.csv', header = None),
                # pd.read_csv('abalone19_scaled.csv', header = None),
                # pd.read_csv('shuttle-c0-vs-c4_scaled.csv', header = None),
                # pd.read_csv('yeast6_scaled.csv', header = None),
                # pd.read_csv('poker-8-9_vs_5_scaled.csv', header = None),
                pd.read_csv('musk_scaled.csv', header = None),
                pd.read_csv('pendigits_scaled.csv', header = None)
                ]

list_random_state = list(np.arange(20))

list_label_noise = list(ParameterGrid({'flip_ratio': [0.25, 0.50, 0.75],
                                       'label_noise': ['uniform']}))

for r in tqdm(list_random_state, desc='Data Partition'):
  
  for ix_d, d in tqdm(enumerate(list_dataset), desc='Data Set'):
    
    d = np.asarray(d)

    
    if len(d) > 10000:
      
      rs = StratifiedShuffleSplit(n_splits = 1, train_size = 10000, 
                                  random_state = r)
      aux_tr, aux_ts = [(a, b) for a, b in rs.split(d[:,:-1], d[:,-1])][0]
      d = d[aux_tr]
      
      
    X = d[:,:-1]
    y = d[:,-1]
    
    
    for l in tqdm(list_label_noise, desc='Label Noise'):
      noisy_y = make_noisy_negatives(y, 
                                     X = X, 
                                     flip_ratio = l['flip_ratio'], 
                                     label_noise = l['label_noise'],
                                     random_state = r)

      ## Class Y Generation
      class_y = y.copy()
      class_y[np.logical_and(np.array(noisy_y == 0), np.array(y == 1))] = 2
  
      conditions = [class_y == 0, class_y == 2, class_y == 1]
      values = ['negs', 'noisy_negs', 'pos']
      class_label = np.select(conditions, values)
      contamination_ratio = np.mean(class_label == 'noisy_negs')

      rs = StratifiedShuffleSplit(n_splits = 1, test_size = 0.30, 
                                  random_state = r)
      ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, class_label)][0]

      rs_trval = StratifiedShuffleSplit(n_splits = 1, test_size = 0.50, 
                                        random_state = r)
      ix_tr_tr, ix_tr_val = [(a, b) for a, b in rs_trval.split(X[ix_tr], class_label[ix_tr])][0]
      
      prior_y = np.mean(y)
      
      ### MODELING ####
      
      ### Decision Tree ###
      
      ##  Hellinger Distance Decision Tree ##
      
      # F1 Score Maximization
      
      m_dt_hd = DecisionTreeClassifier(random_state=r, max_depth=5)
      m_dt_hd.fit(X[ix_tr_tr], noisy_y[ix_tr_tr], p_y = np.mean(noisy_y))
      
      prob = m_dt_hd.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      ##  Hellinger Distance Decision Tree ##
      m_dt_hd = DecisionTreeClassifier(random_state=r, max_depth=5)
      m_dt_hd.fit(X[ix_tr], noisy_y[ix_tr], p_y = np.mean(noisy_y))
      
      prob_dt_hd = m_dt_hd.predict_proba(X[ix_ts])[:,1]
      label_dt_hd = binarize_prob(prob_dt_hd)
      label_ci_dt_hd = binarize_prob(prob_dt_hd, threshold=prior_y)
      label_f1max_dt_hd = binarize_prob(prob_dt_hd, threshold=opt_threshold)

      list_roc_dt_hd.append(roc_auc_score(y[ix_ts], prob_dt_hd))
      list_ap_dt_hd.append(average_precision_score(y[ix_ts], prob_dt_hd))
      list_f1_dt_hd.append(f1_score(y[ix_ts], label_dt_hd))
      list_f1_ci_dt_hd.append(f1_score(y[ix_ts], label_ci_dt_hd))
      list_f1_max_dt_hd.append(f1_score(y[ix_ts], label_f1max_dt_hd))

      print('AP DT HD score: ', average_precision_score(y[ix_ts],  prob_dt_hd))
    
    
      ## Elkan-Noto PU Hellinger Distance Decision Tree ##
      
      # F1 Score Maximization
      
      m_dt_puhd = DecisionTreeClassifier(random_state=r, max_depth=5)
      m_dt_puhd.fit(X[ix_tr_tr], noisy_y[ix_tr_tr], p_y = np.mean(noisy_y))
      
      prob = m_dt_puhd.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 

      m_dt_puhd = DecisionTreeClassifier(random_state=r, max_depth=5)
      m_dt_puhd.fit(X[ix_tr], noisy_y[ix_tr], p_y = np.mean(y))
      
      prob_dt_puhd = m_dt_puhd.predict_proba(X[ix_ts])[:,1]
      label_dt_puhd = binarize_prob(prob_dt_puhd)
      label_ci_dt_puhd = binarize_prob(prob_dt_puhd, threshold=prior_y)
      label_f1max_dt_puhd = binarize_prob(prob_dt_puhd, threshold=opt_threshold)

      list_roc_dt_puhd.append(roc_auc_score(y[ix_ts], prob_dt_puhd))
      list_ap_dt_puhd.append(average_precision_score(y[ix_ts], prob_dt_puhd))
      list_f1_dt_puhd.append(f1_score(y[ix_ts], label_dt_puhd))
      list_f1_ci_dt_puhd.append(f1_score(y[ix_ts], label_ci_dt_puhd))
      list_f1_max_dt_puhd.append(f1_score(y[ix_ts], label_f1max_dt_puhd))

      print('AP DT PU HD score: ', average_precision_score(y[ix_ts],  prob_dt_puhd))
      
      ### Random Forest ###
      
      ## Random Forest ##
      
      # F1 Score Maximization
      
      try:
        resampler = ADASYN(random_state=r)
        X_noisy_rs, noisy_y_rs = resampler.fit_resample(X[ix_tr_tr], noisy_y[ix_tr_tr])
      except:
        X_noisy_rs, noisy_y_rs = X[ix_tr_tr], noisy_y[ix_tr_tr]
      
      m_rf = RandomForestClassifier(random_state=r, n_jobs = -1)
      m_rf.fit(X_noisy_rs, noisy_y_rs)
      
      prob = m_rf.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      try:
        resampler = ADASYN(random_state=r)
        X_noisy_rs, noisy_y_rs = resampler.fit_resample(X[ix_tr], noisy_y[ix_tr])
      except:
        X_noisy_rs, noisy_y_rs = X[ix_tr], noisy_y[ix_tr]
        
      m_rf = RandomForestClassifier(random_state=r, n_jobs = -1)
      m_rf.fit(X_noisy_rs, noisy_y_rs)
      
      prob_rf = m_rf.predict_proba(X[ix_ts])[:,1]
      label_rf = binarize_prob(prob_rf)
      label_ci_rf = binarize_prob(prob_rf, threshold=prior_y)
      label_f1max_rf = binarize_prob(prob_rf, threshold=opt_threshold)

      list_roc_rf.append(roc_auc_score(y[ix_ts], prob_rf))
      list_ap_rf.append(average_precision_score(y[ix_ts], prob_rf))
      list_f1_rf.append(f1_score(y[ix_ts], label_rf))
      list_f1_ci_rf.append(f1_score(y[ix_ts], label_ci_rf))
      list_f1_max_rf.append(f1_score(y[ix_ts], label_f1max_rf))

      print('AP RF Label Noise: ', average_precision_score(y[ix_ts], prob_rf))
      
      ## Elkan-Noto Stratified Bootstrap Random Forest Hellinger Distance ##
      
      # F1 Score Maximization
      
      m_srf_puhd = PURF_SIMP(random_state=r, max_samples=sum(noisy_y[ix_tr_tr] == 0), pu_biased_bootstrap=True)
      m_srf_puhd.fit(X[ix_tr_tr], noisy_y[ix_tr_tr], p_y = np.mean(y))
      
      prob = m_srf_puhd.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_srf_puhd = PURF_SIMP(random_state=r, max_samples=sum(noisy_y[ix_tr] == 0), pu_biased_bootstrap=True)
      m_srf_puhd.fit(X[ix_tr], noisy_y[ix_tr], p_y = np.mean(y))
      
      prob_srf_puhd = m_srf_puhd.predict_proba(X[ix_ts])[:,1]
      label_srf_puhd = binarize_prob(prob_srf_puhd)
      label_ci_srf_puhd = binarize_prob(prob_srf_puhd, threshold=prior_y)
      label_f1max_srf_puhd = binarize_prob(prob_srf_puhd, threshold=opt_threshold)

      list_roc_srf_puhd.append(roc_auc_score(y[ix_ts], prob_srf_puhd))
      list_ap_srf_puhd.append(average_precision_score(y[ix_ts], prob_srf_puhd))
      list_f1_srf_puhd.append(f1_score(y[ix_ts], label_srf_puhd))
      list_f1_ci_srf_puhd.append(f1_score(y[ix_ts], label_ci_srf_puhd))
      list_f1_max_srf_puhd.append(f1_score(y[ix_ts], label_f1max_srf_puhd))

      print('AP Biased RF HD score: ', average_precision_score(y[ix_ts],  prob_srf_puhd))
      
      ## Elkan-Noto PU Random Forest Hellinger Distance ##

      # F1 Score Maximization
      
      m_rf_puhd = PURF_SIMP(random_state=r, pu_biased_bootstrap=False)
      m_rf_puhd.fit(X[ix_tr_tr], noisy_y[ix_tr_tr], p_y = np.mean(y))
      
      prob = m_rf_puhd.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_rf_puhd = PURF_SIMP(random_state=r, pu_biased_bootstrap=False)
      m_rf_puhd.fit(X[ix_tr], noisy_y[ix_tr], p_y = np.mean(y))

      prob_rf_puhd = m_rf_puhd.predict_proba(X[ix_ts])[:,1]
      label_rf_puhd = binarize_prob(prob_rf_puhd)
      label_ci_rf_puhd = binarize_prob(prob_rf_puhd, threshold=prior_y)
      label_f1max_rf_puhd = binarize_prob(prob_rf_puhd, threshold=opt_threshold)

      list_roc_rf_puhd.append(roc_auc_score(y[ix_ts], prob_rf_puhd))
      list_ap_rf_puhd.append(average_precision_score(y[ix_ts], prob_rf_puhd))
      list_f1_rf_puhd.append(f1_score(y[ix_ts], label_rf_puhd))
      list_f1_ci_rf_puhd.append(f1_score(y[ix_ts], label_ci_rf_puhd))
      list_f1_max_rf_puhd.append(f1_score(y[ix_ts], label_f1max_rf_puhd))

      print('AP RF HD score: ', average_precision_score(y[ix_ts],  prob_rf_puhd))
      
      ## PU Bagging ##
      
      # F1 Score Maximization
      
      m_pubag = BaggingPuClassifier(SVC(kernel = 'linear'),
                                    n_estimators = 100, 
                                    n_jobs = -1, 
                                    max_samples = sum(noisy_y[ix_tr_tr] == 1),  # Each training sample will be balanced
                                    random_state = r)
      
      m_pubag.fit(X[ix_tr_tr], noisy_y[ix_tr_tr])
      
      prob = m_pubag.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_pubag = BaggingPuClassifier(SVC(kernel = 'linear'),
                                    n_estimators = 100, 
                                    n_jobs = -1, 
                                    max_samples = sum(noisy_y[ix_tr] == 1),  # Each training sample will be balanced
                                    random_state = r)
      
      m_pubag.fit(X[ix_tr], noisy_y[ix_tr])
      
      prob_pubag = m_pubag.predict_proba(X[ix_ts])[:,1]
      label_pubag = binarize_prob(prob_pubag)
      label_ci_pubag = binarize_prob(prob_pubag, threshold=prior_y)
      label_f1max_pubag = binarize_prob(prob_pubag, threshold=opt_threshold)

      list_roc_pubag.append(roc_auc_score(y[ix_ts], prob_pubag))
      list_ap_pubag.append(average_precision_score(y[ix_ts], prob_pubag))
      list_f1_pubag.append(f1_score(y[ix_ts], label_pubag))
      list_f1_ci_pubag.append(f1_score(y[ix_ts], label_ci_pubag))
      list_f1_max_pubag.append(f1_score(y[ix_ts], label_f1max_pubag))

      print('PR AUC PU Bagging: ', average_precision_score(y[ix_ts], prob_pubag))
           
      ## SpyEM ##
      
      # F1 Score Maximization
      
      m_spyem = SpyEM(random_state = r, resampler = False)
      m_spyem.fit(X[ix_tr_tr], noisy_y[ix_tr_tr])
      
      prob = m_spyem.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_spyem = SpyEM(random_state = r, resampler = False)
      m_spyem.fit(X[ix_tr], noisy_y[ix_tr])
      
      prob_spyem = m_spyem.predict_proba(X[ix_ts])[:,1]
      label_spyem = binarize_prob(prob_spyem)
      label_ci_spyem = binarize_prob(prob_spyem, threshold=prior_y)
      label_f1max_spyem = binarize_prob(prob_spyem, threshold=opt_threshold)

      list_roc_spyem.append(roc_auc_score(y[ix_ts], prob_spyem))
      list_ap_spyem.append(average_precision_score(y[ix_ts], prob_spyem))
      list_f1_spyem.append(f1_score(y[ix_ts], label_spyem))
      list_f1_ci_spyem.append(f1_score(y[ix_ts], label_ci_spyem))
      list_f1_max_spyem.append(f1_score(y[ix_ts], label_f1max_spyem))

      print('PR AUC Spy-EM: ', average_precision_score(y[ix_ts], prob_spyem)) 
      
      ## Weighted PU Logistic Regression
      
      # F1 Score Maximization
      
      m_welog = LogisticRegression(class_weight = {0: np.mean(noisy_y[ix_tr_tr]), 
                                                    1: 1 - np.mean(noisy_y[ix_tr_tr]) },
                                    n_jobs = -1,
                                    penalty = 'none')
      m_welog.fit(X[ix_tr_tr], noisy_y[ix_tr_tr])
      
      prob = m_welog.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_welog = LogisticRegression(class_weight = {0: np.mean(noisy_y[ix_tr]), 
                                                    1: 1 - np.mean(noisy_y[ix_tr]) },
                                    n_jobs = -1,
                                    penalty = 'none')
      m_welog.fit(X[ix_tr], noisy_y[ix_tr])
      
      prob_welog = m_welog.predict_proba(X[ix_ts])[:,1]
      label_welog = binarize_prob(prob_welog)
      label_ci_welog = binarize_prob(prob_welog, threshold=prior_y)
      label_f1max_welog = binarize_prob(prob_welog, threshold=opt_threshold)

      list_roc_welog.append(roc_auc_score(y[ix_ts], prob_welog))
      list_ap_welog.append(average_precision_score(y[ix_ts], prob_welog))
      list_f1_welog.append(f1_score(y[ix_ts], label_welog))
      list_f1_ci_welog.append(f1_score(y[ix_ts], label_ci_welog))
      list_f1_max_welog.append(f1_score(y[ix_ts], label_f1max_welog))

      print('PR AUC Weighted Log. Reg.: ', average_precision_score(y[ix_ts], prob_welog)) 
      
      ## Rank Pruning
      
      # F1 Score Maximization
      
      m_rnkpr = RankPruning(clf = LogisticRegression(penalty = 'none'), 
                            frac_neg2pos = 0)
      
      m_rnkpr.fit(X[ix_tr_tr], noisy_y[ix_tr_tr])
      
      prob = m_rnkpr.predict_proba(X[ix_tr_val])[:,1]
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_rnkpr = RankPruning(clf = LogisticRegression(penalty = 'none'), 
                            frac_neg2pos = 0)
      
      m_rnkpr.fit(X[ix_tr], noisy_y[ix_tr])
      
      prob_rnkpr = m_rnkpr.predict_proba(X[ix_ts])[:,1]
      label_rnkpr = binarize_prob(prob_rnkpr)
      label_ci_rnkpr = binarize_prob(prob_rnkpr, threshold=prior_y)
      label_f1max_rnkpr = binarize_prob(prob_rnkpr, threshold=opt_threshold)

      list_roc_rnkpr.append(roc_auc_score(y[ix_ts], prob_rnkpr))
      list_ap_rnkpr.append(average_precision_score(y[ix_ts], prob_rnkpr))
      list_f1_rnkpr.append(f1_score(y[ix_ts], label_rnkpr))
      list_f1_ci_rnkpr.append(f1_score(y[ix_ts], label_ci_rnkpr))
      list_f1_max_rnkpr.append(f1_score(y[ix_ts], label_f1max_rnkpr))

      print('PR AUC Rank Pruning: ', average_precision_score(y[ix_ts], prob_rnkpr))
     
      ## Elkan-Noto
      
      # F1 Score Maximization
      
      m_elkno = WeightedElkanotoPuClassifier(estimator=SVC(kernel = 'linear', probability = True), 
                                              labeled= np.sum(noisy_y[ix_tr_tr] == 1), 
                                              unlabeled = np.sum(noisy_y[ix_tr_tr] == 0), 
                                              hold_out_ratio = 0.20)
      
      m_elkno.fit(X[ix_tr_tr], noisy_y[ix_tr_tr])
      
      prob = m_elkno.predict_proba(X[ix_tr_val])
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m_elkno = WeightedElkanotoPuClassifier(estimator=SVC(kernel = 'linear', probability = True), 
                                              labeled= np.sum(noisy_y[ix_tr] == 1), 
                                              unlabeled = np.sum(noisy_y[ix_tr] == 0), 
                                              hold_out_ratio = 0.20)
      
      m_elkno.fit(X[ix_tr], noisy_y[ix_tr])
      
      prob_elkno = m_elkno.predict_proba(X[ix_ts])
      label_elkno = binarize_prob(prob_elkno)
      label_ci_elkno = binarize_prob(prob_elkno, threshold=prior_y)
      label_f1max_elkno = binarize_prob(prob_elkno, threshold=opt_threshold)

      list_roc_elkno.append(roc_auc_score(y[ix_ts], prob_elkno))
      list_ap_elkno.append(average_precision_score(y[ix_ts], prob_elkno))
      list_f1_elkno.append(f1_score(y[ix_ts], label_elkno))
      list_f1_ci_elkno.append(f1_score(y[ix_ts], label_ci_elkno))
      list_f1_max_elkno.append(f1_score(y[ix_ts], label_f1max_elkno))

      print('PR AUC Elkan-Noto: ', average_precision_score(y[ix_ts], prob_elkno))

      ### UNBIASED PU LEARNING ###

      prior_s = np.mean(noisy_y)
      c = prior_s/prior_y
      
      cost_matrix_tr_tr = np.zeros((len(X[ix_tr_tr]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
      cost_matrix_tr_tr[:, 0, 0] = 0.0
      cost_matrix_tr_tr[:, 0, 1] = 1.0
      cost_matrix_tr_tr[:, 1, 0] = 1.0
      cost_matrix_tr_tr[:, 1, 1] = 0.0
      
      # F1 Score Maximization
      
      m1_xgb_upu = CSBoost(obj='puwce', 
                          prior_y=prior_y, 
                          prob_lab=prior_s,
                          validation=False,
                          random_state=r).fit(X[ix_tr_tr], 
                                                noisy_y[ix_tr_tr],
                                                cost_matrix_train=cost_matrix_tr_tr)

      prob = expit(m1_xgb_upu.inplace_predict(X[ix_tr_val]))
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      # Cost Matrix for Training
      cost_matrix_tr = np.zeros((len(X[ix_tr]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
      cost_matrix_tr[:, 0, 0] = 0.0
      cost_matrix_tr[:, 0, 1] = 1.0
      cost_matrix_tr[:, 1, 0] = 1.0
      cost_matrix_tr[:, 1, 1] = 0.0
      
      ## unbiased PU learning ##
      
      m1_xgb_upu = CSBoost(obj='puwce', 
                          prior_y=prior_y, 
                          prob_lab=prior_s,
                          validation=False,
                          random_state=r).fit(X[ix_tr], 
                                                noisy_y[ix_tr],
                                                cost_matrix_train=cost_matrix_tr)

      m1_upu_prob = expit(m1_xgb_upu.inplace_predict(X[ix_ts]))
      m1_upu_label = np.array([m1_upu_prob[i]>=0.50 for i in range(m1_upu_prob.shape[0])])
      label_ci_upu = np.array([m1_upu_prob[i]>=prior_y for i in range(m1_upu_prob.shape[0])])
      label_f1max_upu = np.array([m1_upu_prob[i]>=opt_threshold for i in range(m1_upu_prob.shape[0])])

      list_roc_upu.append(roc_auc_score(y[ix_ts], m1_upu_prob))      
      list_ap_upu.append(average_precision_score(y[ix_ts], m1_upu_prob))
      list_f1_upu.append(f1_score(y[ix_ts], m1_upu_label))
      list_f1_ci_upu.append(f1_score(y[ix_ts], label_ci_upu))
      list_f1_max_upu.append(f1_score(y[ix_ts], label_f1max_upu))

      print('PR AUC uPU: ', average_precision_score(y[ix_ts], m1_upu_prob) )       
      
      ## non-negative unbiased PU learning ##

      # F1 Score Maximization
      
      m1_xgb_nnpu = CSBoost(obj='nnpu',
                            prior_y=prior_y, 
                            prob_lab=prior_s,
                            validation=False,
                            random_state=r).fit(X[ix_tr_tr], 
                                                noisy_y[ix_tr_tr],
                                                cost_matrix_train=cost_matrix_tr_tr)

      prob = expit(m1_xgb_nnpu.inplace_predict(X[ix_tr_val]))
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m1_xgb_nnpu = CSBoost(obj='nnpu',
                            prior_y=prior_y, 
                            prob_lab=prior_s,
                            validation=False,
                            random_state=r).fit(X[ix_tr], 
                                                noisy_y[ix_tr],
                                                cost_matrix_train=cost_matrix_tr)

      m1_nnpu_prob = expit(m1_xgb_nnpu.inplace_predict(X[ix_ts]))
      m1_nnpu_label = np.array([m1_nnpu_prob[i]>=0.50 for i in range(m1_nnpu_prob.shape[0])])
      label_ci_nnpu = np.array([m1_nnpu_prob[i]>=prior_y for i in range(m1_nnpu_prob.shape[0])])
      label_f1max_nnpu = np.array([m1_nnpu_prob[i]>=opt_threshold for i in range(m1_nnpu_prob.shape[0])])

      list_roc_nnpu.append(roc_auc_score(y[ix_ts], m1_nnpu_prob))      
      list_ap_nnpu.append(average_precision_score(y[ix_ts], m1_nnpu_prob))
      list_f1_nnpu.append(f1_score(y[ix_ts], m1_nnpu_label))
      list_f1_ci_nnpu.append(f1_score(y[ix_ts], label_ci_nnpu))
      list_f1_max_nnpu.append(f1_score(y[ix_ts], label_f1max_nnpu))

      print('PR AUC nnPU: ', average_precision_score(y[ix_ts], m1_nnpu_prob) ) 
      
      ## XGBoost imbalanced nnPU ## 
      
      # F1 Score Maximization
      
      m1_xgb_imbnnpu = CSBoost(obj='imbnnpu',
                              oversampling = 0.50,
                              prior_y=np.mean(y),
                              prob_lab=np.mean(noisy_y),  
                              validation=False,
                              random_state=r).fit(X[ix_tr_tr], 
                                                  noisy_y[ix_tr_tr],
                                                  cost_matrix_train=cost_matrix_tr_tr)

      prob = expit(m1_xgb_imbnnpu.inplace_predict(X[ix_tr_val]))
      
      set_pred_labels = [binarize_prob(prob, threshold=i) for i in set_thresholds]
      set_f1_scores = [f1_score(y[ix_tr_val], i) for i in set_pred_labels]
      max_f1score = np.max(set_f1_scores)
      ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
      opt_threshold = set_thresholds[ix_fscore]
      
      # Training optimal 
      
      m1_xgb_imbnnpu = CSBoost(obj='imbnnpu',
                              oversampling = 0.50,
                              prior_y=np.mean(y),
                              prob_lab=np.mean(noisy_y),  
                              validation=False,
                              random_state=r).fit(X[ix_tr], 
                                                    noisy_y[ix_tr],
                                                    cost_matrix_train=cost_matrix_tr)
      
      m1_imbnnpu_prob = expit(m1_xgb_imbnnpu.inplace_predict(X[ix_ts]))
      m1_imbnnpu_label = np.array([m1_imbnnpu_prob[i]>=0.50 for i in range(m1_imbnnpu_prob.shape[0])])
      label_ci_imbnnpu = np.array([m1_imbnnpu_prob[i]>=prior_y for i in range(m1_imbnnpu_prob.shape[0])])
      label_f1max_imbnnpu = np.array([m1_imbnnpu_prob[i]>=opt_threshold for i in range(m1_imbnnpu_prob.shape[0])])

      list_roc_imbnnpu.append(roc_auc_score(y[ix_ts], m1_imbnnpu_prob))
      list_ap_imbnnpu.append(average_precision_score(y[ix_ts], m1_imbnnpu_prob))
      list_f1_imbnnpu.append(f1_score(y[ix_ts], m1_imbnnpu_label))
      list_f1_ci_imbnnpu.append(f1_score(y[ix_ts], label_ci_imbnnpu))
      list_f1_max_nnpu.append(f1_score(y[ix_ts], label_f1max_imbnnpu))

      print('PR AUC imbnnPU: ', average_precision_score(y[ix_ts], m1_imbnnpu_prob) ) 
  
#### EXPORT RESULTS IN CSV or EXCEL ###########################################

len_rs = len(list_random_state)

names_datasets = [
                  # 'fraud_car_insur', 
                  # 'fraud_creditcard', 
                  # 'cover', 
                  # 'mammography',
                  # 'piechart2', 
                  # 'pizzacutter1', 
                  # 'satellite', 
                  # 'winequality',
                  # 'korean5',
                  # 'chile',
                  # 'cargood',
                  # 'abalone19',
                  # 'shuttle_c0_vs_c4',
                  # 'yeast6',
                  # 'poker_89_vs_5',
                  'musk',
                  'pendigits'
                  ]
                                

data_part_col = list(np.repeat(np.arange(len_rs), len(list_dataset)*len(list_label_noise) ))

dataset_col = list(np.repeat(names_datasets, len(list_label_noise)))*len(np.arange(len_rs))

label_noise_col = list_label_noise*len(list_random_state)*len(list_dataset)
  
df_ref = pd.DataFrame()

df_ref['data_partition'] = data_part_col
df_ref['dataset'] = dataset_col
df_ref['flip_ratio'] = [label_noise_col[i]['flip_ratio'] for i, _ in enumerate(label_noise_col)]
df_ref['label_noise'] = [label_noise_col[i]['label_noise'] for i, _ in enumerate(label_noise_col)]

df_ref['roc_upu'] = list_roc_upu
df_ref['ap_upu'] = list_ap_upu
df_ref['f1_upu'] = list_f1_upu
df_ref['f1_ci_upu'] = list_f1_ci_upu
df_ref['f1_max_upu'] = list_f1_max_upu

df_ref['roc_nnpu'] = list_roc_nnpu
df_ref['ap_nnpu'] = list_ap_nnpu
df_ref['f1_nnpu'] = list_f1_nnpu
df_ref['f1_ci_nnpu'] = list_f1_ci_nnpu
df_ref['f1_max_nnpu'] = list_f1_max_nnpu

df_ref['roc_imbnnpu'] = list_roc_imbnnpu
df_ref['ap_imbnnpu'] = list_ap_imbnnpu
df_ref['f1_imbnnpu'] = list_f1_imbnnpu
df_ref['f1_ci_imbnnpu'] = list_f1_ci_imbnnpu
df_ref['f1_max_imbnnpu'] = list_f1_max_imbnnpu

df_ref['roc_dt_hd'] = list_roc_dt_hd
df_ref['ap_dt_hd'] = list_ap_dt_hd
df_ref['f1_dt_hd'] = list_f1_dt_hd
df_ref['f1_ci_dt_hd'] = list_f1_ci_dt_hd
df_ref['f1_max_dt_hd'] = list_f1_max_dt_hd

df_ref['roc_dt_puhd'] = list_roc_dt_puhd
df_ref['ap_dt_puhd'] = list_ap_dt_puhd
df_ref['f1_dt_puhd'] = list_f1_dt_puhd
df_ref['f1_ci_dt_puhd'] = list_f1_ci_dt_puhd
df_ref['f1_max_dt_puhd'] = list_f1_max_dt_puhd

df_ref['roc_rf'] = list_roc_rf
df_ref['ap_rf'] = list_ap_rf
df_ref['f1_rf'] = list_f1_rf
df_ref['f1_ci_rf'] = list_f1_ci_rf
df_ref['f1_max_rf'] = list_f1_max_rf

df_ref['roc_rf_puhd'] = list_roc_rf_puhd
df_ref['ap_rf_puhd'] = list_ap_rf_puhd
df_ref['f1_rf_puhd'] = list_f1_rf_puhd
df_ref['f1_ci_rf_puhd'] = list_f1_ci_rf_puhd
df_ref['f1_max_rf_puhd'] = list_f1_max_rf_puhd

df_ref['roc_srf_puhd'] = list_roc_srf_puhd
df_ref['ap_srf_puhd'] = list_ap_srf_puhd
df_ref['f1_srf_puhd'] = list_f1_srf_puhd
df_ref['f1_ci_srf_puhd'] = list_f1_ci_srf_puhd
df_ref['f1_max_srf_puhd'] = list_f1_max_srf_puhd

df_ref['roc_pubag'] = list_roc_pubag
df_ref['ap_pubag'] = list_ap_pubag
df_ref['f1_pubag'] = list_f1_pubag
df_ref['f1_ci_pubag'] = list_f1_ci_pubag
df_ref['f1_max_pubag'] = list_f1_max_pubag

df_ref['roc_elkno'] = list_roc_elkno
df_ref['ap_elkno'] = list_ap_elkno
df_ref['f1_elkno'] = list_f1_elkno
df_ref['f1_ci_elkno'] = list_f1_ci_elkno
df_ref['f1_max_elkno'] = list_f1_max_elkno

df_ref['roc_welog'] = list_roc_welog
df_ref['ap_welog'] = list_ap_welog
df_ref['f1_welog'] = list_f1_welog
df_ref['f1_ci_welog'] = list_f1_ci_welog
df_ref['f1_max_welog'] = list_f1_max_welog

df_ref['roc_spyem'] = list_roc_spyem
df_ref['ap_spyem'] = list_ap_spyem
df_ref['f1_spyem'] = list_f1_spyem
df_ref['f1_ci_spyem'] = list_f1_ci_spyem
df_ref['f1_max_spyem'] = list_f1_max_spyem

df_ref['roc_rnkpr'] = list_roc_rnkpr
df_ref['ap_rnkpr'] = list_ap_rnkpr
df_ref['f1_rnkpr'] = list_f1_rnkpr
df_ref['f1_ci_rnkpr'] = list_f1_ci_rnkpr
df_ref['f1_max_rnkpr'] = list_f1_max_rnkpr

df_ref.to_csv('ecml_expsetup_f1score_extraDatasets.csv', index=False)