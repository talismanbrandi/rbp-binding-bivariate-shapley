#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.metrics import r2_score
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sys
import logging
import json
import uuid
import os
import shutil
import argparse
import pyreadr as pyr
import shap
import lzma
import random
import pickle



def load_data(config):
    ''' a function to load the data
        argument:
            config: the configuration file for the run containing the pat to the data source
        returns:
            df: uint8 encoded input data
    '''
    logging.info(' loading data from file {}'.format(config['data_dir']))
    
    # input files
    config['data_dir'] = config['data_dir']+'/' if config['data_dir'][-1] != '/' else config['data_dir']
    config['data_fname'] = config['data_dir']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50/regression_'+config['cell_line'] +'_binding_psi_psip_KD.rds'
    config['rbp_list'] = config['data_dir']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50/rbp_'+config['cell_line'] +'.rds'
    
    df = pd.DataFrame(pyr.read_r(config['data_fname'])[None])
    config['n_rbp'] = pd.DataFrame(pyr.read_r(config['rbp_list'])[None]).shape[0] * 6
    config['n_samples'] = {}
    config['n_samples']['total'] = df.shape[0]
    
    if config['annotate_cell_lines']:
        if 'cell_line' not in df.columns:
                logging.error(' cell line annotation not possible as cell line is not made explicit in the data set')
                sys.exit()
        else:
            unique_cl = df.cell_line.unique()
            for cl in unique_cl:
                config['n_samples'][cl] = {}
                config['n_samples'][cl]['n_samples'] = df[df.cell_line == cl].shape[0]
                config['n_samples'][cl]['fraction'] = df[df.cell_line == cl].shape[0]/config['n_samples']['total']
    
    if config['sample_size'] == 0 or config['sample_size'] > df.shape[0]:
        df = df.sample(frac=1)
    else:
        df = df.sample(n=config['sample_size'])
        
    df_int = df.iloc[:,:config['n_rbp']].astype('uint8').copy()
    df = pd.concat([df_int, df.iloc[:,config['n_rbp']:]], axis=1, copy=False).copy()
    del df_int
    
    df_control = df[df.RBP_KD=='NONE'].copy()
    df_control['target'] = df_control.psi
    df_knockdown = df[df.RBP_KD!='NONE'].copy()
    df_knockdown['target'] = df_knockdown.psip
    df = pd.concat([df_control, df_knockdown], copy=False).sample(frac=1).copy()
    
    if config['annotate_cell_lines']:
        map_cell = {unique_cl[i]: i for i in range(len(unique_cl))}
        df['id_cell_line'] = df.cell_line.apply(lambda x: map_cell[x]).astype('uint8')
        df = pd.concat([df.iloc[:,:config['n_rbp']], df[['id_cell_line', 'target']]], axis=1, copy=False)
    else:
        df = pd.concat([df.iloc[:,:config['n_rbp']], df['target']], axis=1, copy=False)
    
    del df_control
    del df_knockdown
    
    return df


def trainXGB(df, config):
    ''' a function for training individual BDTs
        arguments:
            df: a dataframe with the data sample
            config: dictionary with configuration parameters
        returns:
            xgbregressor: the trained regressor (xgboost regressor object)
            r2: the r-squared of the model
            df_test: the test data for the BDT
    '''
    random_state = random.randint(1,1000000000)
    if config['test_size'] != 0:
        df_test = df.sample(frac=config['test_size'], random_state=random_state).copy()
        df_train = df.drop(df_test.index).copy()
    else:
        df_train = df.copy()
        df_test = pd.DataFrame()
        
    # dump the train set
    df_train.to_csv(config['train_df_file'], index=False)
        
    if config['annotate_cell_lines']:
        x_train = pd.concat([df_train.iloc[:,:config['n_rbp']], df_train.id_cell_line], axis=1).values
    else:
        x_train = df_train.iloc[:,:config['n_rbp']].values
        
    config['n_features'] = np.shape(x_train)[1]
    config['n_samples']['n_model_test_sample'] = df_test.shape[0]
    
    y_train = df_train['target'].values
    
    # build training pipeline with early-stopping
    if config["early_stopping_rounds"] > 0:
        if config["early_stopping_size"] == 0: config["early_stopping_size"] = 0.25
        x_train, x_test, y_train, y_test = ms.train_test_split(x_train, 
                                                               y_train, 
                                                               test_size=config["early_stopping_size"], 
                                                               random_state=random_state)
        eval_set = [(x_train, y_train), (x_test, y_test)]
        config['n_samples']['n_training_samples'] = np.shape(x_train)[0]
        config['n_samples']['n_early_stop_valid_samples'] = np.shape(x_test)[0]

        xgbregressor = xgb.XGBRegressor(max_depth=config['max_depth'], 
                                        n_estimators=config['n_estimators'],
                                        objective='reg:logistic',
                                        reg_alpha=config['reg_alpha'],
                                        subsample=config['subsample'], 
                                        n_jobs=config['n_jobs'], 
                                        random_state=random_state, 
                                        early_stopping_rounds=config["early_stopping_rounds"],
                                        eval_metric=config["eval_metric"])

        xgbregressor = xgbregressor.fit(x_train, 
                                        y_train, 
                                        eval_set=eval_set,
                                        verbose=False)
        
    # training pipeline without early stopping
    else:
        xgbregressor = xgb.XGBRegressor(max_depth=config['max_depth'], 
                                        n_estimators=config['n_estimators'],
                                        objective='reg:logistic',
                                        reg_alpha=config['reg_alpha'],
                                        subsample=config['subsample'], 
                                        n_jobs=config['n_jobs'], 
                                        random_state=random_state)

        xgbregressor = xgbregressor.fit(x_train, 
                                        y_train)
    
    # compute r2 for the individual model if a test set is provided
    if config['test_size'] != 0:
        if config['annotate_cell_lines']:
            x_test = pd.concat([df_test.iloc[:,:config['n_rbp']], df_test.id_cell_line], axis=1).values
        else:
            x_test = df_test.iloc[:,:config['n_rbp']].values
            
        y_pred = xgbregressor.predict(x_test)
        r2 = r2_score(df_test['target'].values, y_pred)
        df_test['psi_hat'] = y_pred
        
        r2_cl = {}
        if 'cell_line' in df.columns:
            for cl in df_test.cell_line.unique():
                df_cl = df_test[df_test.cell_line == cl]
                r2_cl[cl] = r2_score(df_cl['target'].values, df_cl['psi_hat'].values)
        
    else:
        r2 = 0
        r2_cl = {}
        
    
    if not df_test.empty:
        # dump the train set
        df_test.to_csv(config['test_df_file'], index=False)
    
    return xgbregressor, r2, r2_cl, df_test


def trainXGB_ensemble(df, config):
    ''' function for training the ensemble of trees
        arguments:
            df: a dataframe with the data sample
            config: dictionary with configuration parameters
    '''

    df_val = df.sample(frac=config['valid_size'], random_state=config['seed']).copy()
    df_train = df.drop(df_val.index).copy()
    config['n_samples']['n_ensemble_training_samples'] = df_train.shape[0]
    config['n_samples']['n_ensemble_valid_samples'] = df_val.shape[0]
    
    ensemble_dict = {}
    config['r2'] = {}
    if config['annotate_cell_lines']:
        for cl in df_train.cell_line.unique():
            config['r2'][cl] = {}
    config['num_trees'] = {}
    
    # loop over ensemble
    for i in range(config['n_ensemble']):
        logging.info(' training xgbdt model no. {}'.format(i+1))
        ensemble_dict[str(i)] = {}
        r1, r2, r2_cl, _ = trainXGB(df_train, config)
        
        # the model
        ensemble_dict[str(i)]['model'] = r1
        dump_list = r1.get_booster().get_dump()
        config['num_trees']['model-'+str(i)] = len(dump_list)
        
        # r2 for the models
        ensemble_dict[str(i)]['r2'] = r2
        if r2 != 0: 
            logging.info(' R2 score of model '+str(i)+': %.3f' % (r2))
            config['r2']['model-'+str(i)] = r2
        if r2_cl:
            for key, val in r2_cl.items():
                logging.info(' R2 score of model '+str(i)+' for %s: %.3f' % (key, val))
                config['r2'][key]['model-'+str(i)] = val
    
    # make predictions
    logging.info(' making predictions using the xgbdt ensemble')
    df_predict = pd.DataFrame(columns=ensemble_dict.keys())
    for key, val in ensemble_dict.items():

        if config['annotate_cell_lines']:
            x_val = pd.concat([df_val.iloc[:,:config['n_rbp']], df_val.id_cell_line], axis=1).values
        else:
            x_val = df_val.iloc[:,:config['n_rbp']].values

        df_predict[key] = val['model'].predict(x_val)
        
    df_predict.index = df_val.index # NOT NEEDED
    df_val['psi_hat'] = df_predict.mean(axis=1)
    
    # r2 for the model
    config['r2']['ensemble'] = r2_score(df_val['target'].values, df_val['psi_hat'].values)
    logging.info(' R2 score of the xgbdt ensemble: %.3f' % (config['r2']['ensemble']))
    
    if 'cell_line' in df.columns:
        for cl in df_val.cell_line.unique():
            df_cl = df_val[df_val.cell_line == cl]
            config['r2'][cl]['ensemble'] = r2_score(df_cl['target'].values, df_cl['psi_hat'].values)
            logging.info(' R2 score of the xgbdt ensemble for %s: %.3f' % (cl, config['r2'][cl]['ensemble']))
    
    # pickle the ensemble
    with lzma.open(config['pickle_file'], "wb") as f:
        pickle.dump(ensemble_dict, f)
        
    # dump the test set
    df_val.to_csv(config['validation_df_file'], index=False)
    
    return ensemble_dict, df_val
    
    
def computeShap(model, df, config):
    ''' function to compute the shapley value and its correlation with the predictors
        arguments:
            model: the model that is being explained
            df: the input dataframe to be sampled from
            config: the configuration file
    '''
    logging.info(" computing the Shapley values from the ensemble")
    n_loop = min(config['n_ensemble'], len(model.keys()))
    smaple_size = min(config['shap_sample_size'], df.shape[0])
    df_sample = df.sample(n=smaple_size, random_state=config['seed'])
    
    if config['annotate_cell_lines']:
        x_sample = pd.concat([df_sample.iloc[:,:config['n_rbp']], df_sample.id_cell_line], axis=1)
    else:
        x_sample = df_sample.iloc[:,:config['n_rbp']]
    
    shap_values = None
    for i in range(n_loop):
        logging.info(" computing Shapley values from model {}".format(i+1))
        explainer = shap.TreeExplainer(model[str(i)]['model'])
        if shap_values is None:
            sh = explainer.shap_values(x_sample)
            shap_values = sh
            shap_values_sq = sh * sh
        else:
            sh = explainer.shap_values(x_sample)
            shap_values += sh
            shap_values_sq += sh * sh
    shap_values /= n_loop
    shap_values_sigma = shap_values_sq/n_loop - shap_values * shap_values
    
    df_shap = pd.DataFrame()
    df_shap['RBP_site'] = x_sample.columns
    df_shap['abs_shap'] = np.einsum('ij->j',np.abs(shap_values))/np.shape(shap_values)[0]
    df_shap['abs_shap_sigma'] = np.sqrt(np.einsum('ij->j', shap_values_sigma))/np.shape(shap_values_sq)[0]
    
    # merge the sample and the Shapley values
    df_all_shap = pd.DataFrame(shap_values)
    df_all_shap.columns = x_sample.columns + '_shap'
    df_all_shap.index = df_sample.index
    df_all_shap = pd.concat([df_sample, df_all_shap], axis=1)
    
    # Make a copy of the input data
    shap_v = pd.DataFrame(shap_values)
    feature_list = x_sample.columns
    shap_v.columns = feature_list
    # df_v = df_sample.copy().reset_index(drop=True).iloc[:,:config['n_rbp']]

    # Determine the correlation between the abs_shap and sample
    logging.info(" computing the correlations of the Shapley values with the predictors")
    corr_list = list()
    for i in feature_list:
        corr_list.append(np.corrcoef(shap_v[i],df_sample[i])[1][0])
    df_shap['correlations'] = pd.Series(corr_list).fillna(0)
    
    # dump the sample set with Shapley values
    df_all_shap.to_csv(config['shap_sample_df_file'], index=False)
    
    # dump the abs shap with the correlations
    df_shap.to_csv(config['abs_shap_df_file'], index=False)
    
    
def timediff(x):
    ''' utility funtion to convert seconds to hh:mm:ss
        argument:
            x: time in seconds
        returns:
            string formated as hh:mm:ss
    '''
    return "{}:{}:{}".format(int(x/3600), str(int(x/60%60)).zfill(2), str(round(x - int(x/3600)*3600 - int(x/60%60)*60)).zfill(2))


def main():
    start = time.time()
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    
    # parse input arguments
    parser = argparse.ArgumentParser(description="A python script implementing XGB BDTs models",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="configuration file for the run")
    args = vars(parser.parse_args())
    
    # set up the config
    with open(args['config'], 'r') as f:
        config = json.load(f)
        
    # start time
    config["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    
    logging.info(' ensemble training started at {}'.format(config["start_time"]))
    
    #  create directory structure
    if config['model-uuid'] == "UUID":
        m_uuid = str(uuid.uuid4())[:8]
        config['model-uuid'] = m_uuid
    else:
        m_uuid = config['model-uuid']
    
    if config['base_directory'] != '':
        config['base_directory'] = config['base_directory']+'/' if config['base_directory'][-1] != '/' else config['base_directory']
        dir_name = config['base_directory']+'bdt-xgb-models-'+config['cell_line']+'-'+config['exon_buffer']+'-'+m_uuid+'/'
    else:
        dir_name = 'bdt-xgb-models-'+config['cell_line']+'-'+config['exon_buffer']+'-'+m_uuid+'/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    config['directory'] = dir_name
    
    # input files
    config['data_dir'] = config['data_dir']+'/' if config['data_dir'][-1] != '/' else config['data_dir']
    config['data_fname'] = config['data_dir']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50/regression_'+config['cell_line'] +'_binding_psi_psip_KD.rds'
    config['rbp_list'] = config['data_dir']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50/rbp_'+config['cell_line'] +'.rds'
    
    # output files
    config['pickle_file'] = config['directory']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50-ensemble-models'+'.pkl.xz'
    config['train_df_file'] = config['directory']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50-train-data'+'.dat.xz'
    config['test_df_file'] = config['directory']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50-test-data'+'.dat.xz'
    config['validation_df_file'] = config['directory']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50-validation-data'+'.dat.xz'
    config['shap_sample_df_file'] = config['directory']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50-shap-sample-data'+'.dat.xz'
    config['abs_shap_df_file'] = config['directory']+config['cell_line']+ '-'+config['exon_buffer']+'-50-50-abs-shap-correlation'+'.dat.xz'
    
    # save the config
    with open(config['directory']+'/config-'+config['model-uuid']+'-prelim.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # load data
    df = load_data(config)
    data_load_end_time = time.time()
    config["data_load_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    config["data_load_time"] = timediff(data_load_end_time - start)
    logging.info(' data loading ended at {}'.format(config["data_load_end_time"]))
    logging.info(' total time taken for loading data: {}'.format(config["data_load_time"]))
    
    # run ensemble
    ensemble, df_val = trainXGB_ensemble(df, config)
    
    # time taken for fitting models
    fit_end = time.time()
    config["fit_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    config["fit_time"] = timediff(fit_end - data_load_end_time)
    logging.info(' ensemble training ended at {}'.format(config["fit_end_time"]))
    logging.info(' total time taken for fitting XGBDTs: {}'.format(config["fit_time"]))
    
    # get the Shapley values
    if config['compute_shap']:
        # save the config
        with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
            json.dump(config, f, indent=4)
        computeShap(ensemble, df_val, config)
        # time for computing Shapley values
        config["shap_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
        config["shap_time"] = timediff(time.time() - fit_end)
        logging.info(' Shapley value computation ended at {}'.format(config["shap_end_time"]))
        logging.info(' total time taken by SHAP: {}'.format(config["shap_time"]))

    # end time
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    
    # time taken
    config["total_time"] = timediff(time.time() - start)
    
    logging.info(' ML pipeline ended at {}'.format(config["end_time"]))
    logging.info(' total time taken: {}'.format(config["total_time"]))
    
    # save the config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # remove preliminary config file
    os.remove(config['directory']+'/config-'+config['model-uuid']+'-prelim.json')
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()