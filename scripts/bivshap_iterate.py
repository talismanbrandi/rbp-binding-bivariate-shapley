# %% =====================================================
# Libraries
import os
import sys
import json
import pickle
import logging
import time
import argparse
import uuid
import lzma

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/work/jdy/davin/rbp-binding-bivariate-shapley')
sys.path.append('./')
from BivariateShapley.BivariateShapley.shapley_kernel import Bivariate_KernelExplainer
from BivariateShapley.BivariateShapley.utils_shapley import g2h
from scripts.xgbdt import *
from shared_utils import *


np.random.seed(0)

from argparse import ArgumentParser 
parser = ArgumentParser(description='get phi plus matrices')

parser.add_argument('--dataset_min_index', type = int,default=0,
                    help='iterate over dataset starting from min_index')
parser.add_argument('--dataset_samples', type = int,default=5,
                    help='number of samples, starting from min_index')
parser.add_argument('--seed', type = int,default=0,
                    help='seed')

args = parser.parse_args()

utils.print_args(args)

# %% =====================================================
# Functions
class val_function():
    '''
    wrapper for xgboost model for KernelSHAP
    '''
    def __init__(self, model):
        self.model = model
    def __call__(self, x):
        pred = self.model.predict(x)
        return pred


# %% =====================================================
# Load Model

model_path = '/work/jdy/davin/rbp-binding-bivariate-shapley/models/bdt-xgb-models-HepG2-100-8414c135/HepG2-100-50-50-ensemble-models.pkl.xz'

with lzma.open(model_path, mode='rb') as f:
    model_dict = pickle.load(f)
model = model_dict['0']['model'] # select first model

# %% =====================================================
# Load Data

# train
data_path = '/work/jdy/davin/rbp-binding-bivariate-shapley/models/bdt-xgb-models-HepG2-100-8414c135/HepG2-100-50-50-train-data.dat.xz'
with lzma.open(data_path, mode='rb') as f:
    df_train = pd.read_csv(f)

# validation
data_path = '/work/jdy/davin/rbp-binding-bivariate-shapley/models/bdt-xgb-models-HepG2-100-8414c135/HepG2-100-50-50-validation-data.dat.xz'

with lzma.open(data_path, mode='rb') as f:
    df_test = pd.read_csv(f)


x_train = df_train.drop(['target'], axis = 1).values
x_test = df_test.drop(['target', 'psi_hat'], axis = 1).values

# %% =====================================================
# Calculate Explanations

baseline = x_train.mean(axis = 0).reshape(1,-1) # baseline when using fixed baseline
output_dict = {}

for i in range(args.dataset_min_index,(args.dataset_min_index+args.dataset_samples)):
    starttime = time.time()
    # select samples
    x = x_test[i:i+1,:]

    # initialize explainer
    val = val_function(model)
    explainer = Bivariate_KernelExplainer(val, baseline) # initialize BivShap-K

    # Explain Sample
    shap = explainer.shap_values(x, l1_reg = False) # univariate shapley values (KernelShap)
    bshap_G = explainer.phi_b # bivariate shapley values (G Graph)

    output_dict[i] = {}
    output_dict[i]['kernelshap'] = shap
    output_dict[i]['bivshap_G'] = bshap_G
    output_dict[i]['time'] = time.time() - starttime

save_path = '/work/jdy/davin/rbp-binding-bivariate-shapley/scripts/results'
filename = 'bivshap_min' + str(args.dataset_min_index) + '_samples' + str(args.dataset_samples)  + '_seed' + str(args.seed) + '.pkl'
utils.save_dict(output_dict, os.path.join(save_path, filename))