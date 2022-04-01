# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:04:49 2020

@author: orteg
"""

# Own packages, put that package in whatever PATH you use in python
from preprocesamiento.utils import CatNarrow
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.compose import ColumnTransformer

# Standard
import numpy as np
import pandas as pd

# sklearn
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# scipy
import scipy.io


#### MAT FORMAT ####
df = pd.read_csv('solarflare.csv', header=None)

X, y = df.iloc[:,:-1], np.array(df.iloc[:,-1]).reshape(-1,1)

# num_vars = [0]
# num_vars = [0,4,5,9,12,15,16,22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
num_vars = [15,16,17,18,19,20]
# cat_vars = 0, 1,2,3,6,7,8,10,11,13,14,17,18,19,20,21
cat_vars = [0,1,2,3,4, 5, 6, 7,8,9,10]


numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                                ('cat', categorical_transformer, cat_vars),
                                                # ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

target = np.zeros_like(y)
target[y==' positive'] = 1

X_trans = pipe.fit_transform(X, target)

df_trans = pd.DataFrame(np.hstack((X_trans, target.reshape(-1,1) )))
df_trans.to_csv('solarflare_scaled.csv', index=False, header=False)
######################## ODDS Datasets #######################################

#### MAT FORMAT ####
mat = scipy.io.loadmat('musk.mat')

X, y = mat['X'], mat['y']
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df = pd.DataFrame(np.hstack((X_trns,y)))

df.isna().any()
df.to_csv('musk_scaled.csv', index = False, header = False)

### Loading and Filtering Data

th = pd.read_csv('thyroid.csv', header = None)
wi = pd.read_csv('wine.csv', header = None)
pi = pd.read_csv('pima.csv', header = None)
co = pd.read_csv('cover.csv', header = None)
ca = pd.read_csv('cardio.csv', header = None)

#  http://odds.cs.stonybrook.edu/

tuple_ = [(th, 'thyroid_scaled.csv'), 
          (wi, 'wine_scaled.csv'), 
          (pi, 'pima_scaled.csv'), 
          (co, "cover_scaled.csv"), 
          (ca, 'cardio_scaled.csv')
          ]

for df, filename in tuple_:
  X = df.iloc[:,:-1]
  y = np.array(df.iloc[:,-1]).reshape(-1,1)

  scaler = StandardScaler()
  X_trns = scaler.fit_transform(df.iloc[:, :-1])
  df_trns = pd.DataFrame(np.hstack((X_trns, y)))
  df_trns.to_csv(filename, index = False,  header = False)
  
  
######################## CHURN #########################################

## KOREAN 5


df = pd.read_csv('korean5.csv')

num_vars = ['AGE','AVSRATIO','USAGE']

cat_vars = ['SEX','JOB','REGION','METHOD','CUSTGR']

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                               ('cat', categorical_transformer, cat_vars),
                                               ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

target = np.zeros_like(df['TARGET'])
target[df['TARGET']=='Churn'] = 1

df_trans = pipe.fit_transform(df.drop(columns = 'TARGET'), target)

df_trans = pd.DataFrame(np.hstack((df_trans, target.reshape(-1,1) )))

desired_py = 0.05

n_pos = np.sum(df_trans.iloc[:,-1]==0)*desired_py/(1-desired_py)

ix = np.random.RandomState(123).choice(np.where(target==1)[0], size = int(n_pos), replace = False)

ix_ = np.hstack((np.where(target==0)[0], ix))

df_trans.iloc[ix_].to_csv('korean5_scaled_imbalanced.csv', index=False, header=False)

## KOREAN 3

df = pd.read_csv('korean3.csv')

num_vars = ['ALDT','AIDT','LOCALT','DISCNT','AGE','CHILD','AMT']

cat_vars = ['SEX','METHOD','TLCP','TLDCP','MARRY','CAR']

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                               ('cat', categorical_transformer, cat_vars),
                                               ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

target = np.zeros_like(df['TARGET'])
target[df['TARGET']=='Y'] = 1

df_trans = pipe.fit_transform(df.drop(columns = 'TARGET'), target)

df_trans = pd.DataFrame(np.hstack((df_trans, target.reshape(-1,1) )))

desired_py = 0.05

n_pos = np.sum(df_trans.iloc[:,-1]==0)*desired_py/(1-desired_py)

ix = np.random.RandomState(123).choice(np.where(target==1)[0], size = int(n_pos), replace = False)

ix_ = np.hstack((np.where(target==0)[0], ix))

df_trans.iloc[ix_].to_csv('korean3_scaled_imbalanced.csv', index=False, header=False)
  
## KOREAN 2

df = pd.read_csv('korean2.csv')

num_vars = ['AGE','VAF','USAGE']

cat_vars = ['SEX','METHOD','JOB','GRP','REGION','PAYT']

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                               ('cat', categorical_transformer, cat_vars),
                                               ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

target = np.zeros_like(df['TARGET'])
target[df['TARGET']=='Y'] = 1

df_trans = pipe.fit_transform(df.drop(columns = 'TARGET'), target)

df_trans = pd.DataFrame(np.hstack((df_trans, target.reshape(-1,1) )))

desired_py = 0.05

n_pos = np.sum(df_trans.iloc[:,-1]==0)*desired_py/(1-desired_py)

ix = np.random.RandomState(123).choice(np.where(target==1)[0], size = int(n_pos), replace = False)

ix_ = np.hstack((np.where(target==0)[0], ix))

df_trans.iloc[ix_].to_csv('korean2_scaled_imbalanced.csv', index=False, header=False)

## Chile

df = pd.read_csv('chile.csv')

num_vars = ['ACTIVE_DAYS', 'ACTIVE_WEEKS',
           'ACTIVE_MONTHS', 'PREPAID_BEFORE', 'COLLECTIONS',
           'PAYMENT_DELAY', 'COUNT_PAY_DELAYS', 'ANNUAL_PAY_DELAY',
           'RECEIPT_DELAYS', 'COMPLAINT_2WEEKS', 'COMPLAINT_3MONTHS',
           'COMPLAINT_6MONTHS', 'COMPLAINT_1WEEK', 'COMPLAINT_1MONTH', 'ARPU',
           'COUNT_OFFNET_CALLS_1WEEK', 'COUNT_ONNET_CALLS_1WEEK',
           'AVG_INC_OFFNET_1MONTH', 'AVG_INC_ONNET_1MONTH', 'AVG_DATA_3MONTH',
           'COUNT_CONNECTIONS_3MONTH', 'AVG_DATA_1MONTH',
           'COUNT_SMS_INC_ONNET_6MONTH', 'COUNT_SMS_OUT_OFFNET_6MONTH',
           'COUNT_SMS_INC_OFFNET_1MONTH', 'COUNT_SMS_INC_OFFNET_WKD_1MONTH',
           'COUNT_SMS_INC_ONNET_1MONTH', 'COUNT_SMS_INC_ONNET_WKD_1MONTH',
           'COUNT_SMS_OUT_OFFNET_1MONTH', 'COUNT_SMS_OUT_OFFNET_WKD_1MONTH',
           'COUNT_SMS_OUT_ONNET_1MONTH', 'COUNT_SMS_OUT_ONNET_WKD_1MONTH',
           'AVG_MINUTES_INC_OFFNET_1MONTH', 'AVG_MINUTES_INC_ONNET_1MONTH',
           'MINUTES_INC_OFNET_WKD_1MONTH', 'MINUTES_INC_ONNET__WKD_1MONTH',
           'AVG_MINUTES_OUT_OFFNET_1MONTH', 'AVG_MINUTES_OUT_ONNET_1MONTH',
           'MINUTES_OUT_OFFNET_WKD_1MONTH', 'MINUTES_OUT_ONNET_WKD_1MONTH',
           'MINUTES_INC_ONNET_3MONTH', 'MINUTES_INC_OFFNET_3MONTH']

cat_vars = []

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                               ('cat', categorical_transformer, cat_vars),
                                               ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

target = np.zeros_like(df['CHURN'])
target[df['CHURN']==1] = 1

df_trans = pipe.fit_transform(df.drop(columns = 'CHURN'), target)

df_trans = pd.DataFrame(np.hstack((df_trans, target.reshape(-1,1) )))

desired_py = 0.05

n_pos = np.sum(df_trans.iloc[:,-1]==0)*desired_py/(1-desired_py)

ix = np.random.RandomState(123).choice(np.where(target==1)[0], size = int(n_pos), replace = False)

ix_ = np.hstack((np.where(target==0)[0], ix))

df_trans.iloc[ix_].to_csv('chile_scaled_imbalanced.csv', index=False, header=False)
  
## KOREAN 2

df = pd.read_csv('korean2.csv')

num_vars = ['AGE','VAF','USAGE']

cat_vars = ['SEX','METHOD','JOB','GRP','REGION','PAYT']

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                               ('cat', categorical_transformer, cat_vars),
                                               ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

target = np.zeros_like(df['TARGET'])
target[df['TARGET']=='Y'] = 1

df_trans = pipe.fit_transform(df.drop(columns = 'TARGET'), target)

df_trans = pd.DataFrame(np.hstack((df_trans, target.reshape(-1,1) )))

desired_py = 0.05

n_pos = np.sum(df_trans.iloc[:,-1]==0)*desired_py/(1-desired_py)

ix = np.random.RandomState(123).choice(np.where(target==1)[0], size = int(n_pos), replace = False)

ix_ = np.hstack((np.where(target==0)[0], ix))

df_trans.iloc[ix_].to_csv('korean2_scaled_imbalanced.csv', index=False, header=False)


######################## ULB CREDIT FRAUD #####################################


df = pd.read_csv('creditcard.csv')
df = df[df['Amount']!=0]
y = np.asarray(df['Class'])
amount = np.asarray(df['Amount'])

#### Data Transformation X ####
X = df.iloc[:,1:-2]
X['L_Amount'] = np.log(amount)
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df = pd.DataFrame(np.hstack((X_trns, y.reshape(-1,1))))
df.to_csv('scaled_creditcard.csv', index = False,  header = False)

######################## ORACLE INSURANCE FRAUD  #####################################
 

df = pd.read_csv('oracle_insurance_fraud.csv')

num_vars = ['DriverAge']

cat_vars = ['AddressChangeClaim', 'AgeOfPolicyHolder', 'AgeOfVehicle',
            'AgentType', 'BasePolicy', 'DaysPolicyAccident', 'DaysPolicyClaim',
            'DeclarationDayOfWeek', 'DeclarationMonth',
            'DeclarationWeekOfMonth', 'Deductible', 'DriverGender',
            'DriverMaritalStatus', 'DriverRating', 'Fault',
            'IncidentAddressType', 'IncidentDayOfWeek', 'IncidentMonth',
            'IncidentWeekOfMonth', 'IncidentYear', 'Make', 'NumberOfCars',
            'NumberOfSuppliments', 'PastNumberOfClaims', 'PoliceReportFiled',
            'RepairerDetailID', 'VehicleCategory', 'VehiclePrice',
            'WitnessPresent']

#### PIPELINES ####

### FEATURE ENGINEERING

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.00)),
                                            ('woe', ce.WOEEncoder())
                                            ]
                                   )
                                   

preprocessor = ColumnTransformer(transformers=[
                                               ('cat', categorical_transformer, cat_vars),
                                               ('num', numeric_transformer, num_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

df_trans = pipe.fit_transform(df.drop(columns = 'FraudFound'), df['FraudFound'])

df_trans = pd.DataFrame(np.hstack((df_trans, np.asarray(df['FraudFound']).reshape(-1,1) )))



df_trans.to_csv('fraud_car_insurance.csv', index=False, header=False)
