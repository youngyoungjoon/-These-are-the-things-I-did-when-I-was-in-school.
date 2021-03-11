import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor
from neptunecontrib.versioning.data import log_data_version
from neptunecontrib.monitoring.reporting import send_binary_classification_report


validation_params = {'seed':1234,
                     'n_folds':5,
                     'validation_schema': 'kfold'}

folds = KFold(n_splits=validation_params['n_folds'], random_state=validation_params['seed'])
model_params = {'num_leaves': 256,
                  'min_child_samples': 79,
                  'objective': 'binary',
                  'max_depth': 15,
                  'learning_rate': 0.02,
                  "boosting_type": "gbdt",
                  "subsample_freq": 3,
                  "subsample": 0.9,
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 0.3,
                  'reg_lambda': 0.3,
                  'colsample_bytree': 0.9
                 }

training_params = {'num_boosting_rounds':5000,
                   'early_stopping_rounds':200
               }