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

os.chdir('/home/davinhill/Code/rbp-binding-bivariate-shapley/')
from BivariateShapley.BivariateShapley.shapley_kernel import Bivariate_KernelExplainer
from BivariateShapley.BivariateShapley.utils_shapley import g2h
from xgbdt import *

np.random.seed(0)



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

model_path = './models/bdt-xgb-models-HepG2-100-8414c135/HepG2-100-50-50-ensemble-models.pkl.xz'

with lzma.open(model_path, mode='rb') as f:
    model_dict = pickle.load(f)
model = model_dict['0']['model'] # select first model

# %% =====================================================
# Load Data
model_path = '/home/davinhill/Code/rbp-binding-bivariate-shapley/models/bdt-xgb-models-K562-100-cf156139/K562-100-50-50-train-data.dat.xz'

with lzma.open(model_path, mode='rb') as f:
    df = pd.read_csv(f)



# %% =====================================================
# Calculate Explanations

# select samples
x = df.iloc[0:1,:-1].values
x_train = df.sample(n = 1000).values[:,:-1] # baseline samples for kernelshap

# initialize explainer
val = val_function(model)
explainer = Bivariate_KernelExplainer(val, x_train) # initialize BivShap-K

# Explain Sample
shap = explainer.shap_values(x, l1_reg = False) # univariate shapley values (KernelShap)
bshap_G = explainer.phi_b # bivariate shapley values (G Graph)

gamma = 1e-3 # threshold for generating H Graph. All edge weights < gamma in G graph are set to 1 in H Graph
bshap_H = g2h(bshap_G, gamma) # bivariate shapley values (H Graph)


# %% =====================================================
# Plot Bivariate Shapley (G Graph)

fig, axes = plt.subplots(1, 1, figsize=(40, 40), sharey=False, sharex = False)
node_labels = df.columns[:-1]
annot_flag = False

min_value = min(shap.min(), bshap_G.min())
max_value = max(shap.max(), bshap_G.max())

tmp = pd.DataFrame(bshap_G.round(2))
tmp.columns = node_labels
tmp.index = node_labels

sns.heatmap(tmp, square=True, annot = annot_flag, center = 0,cbar_kws = dict(use_gridspec=False,location="bottom"), vmin = min_value, vmax = max_value)

axes.set_title('Bivariate Shapley')
plt.show()
# %%
