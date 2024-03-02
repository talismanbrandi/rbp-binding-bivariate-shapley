### Code for running the neural networks

```
python xgbdt.py configuration_xgbdt.json
```

Configuration file:

```{json}
{
    "cell_line": "HepG2",            # the cell line
    "exon_buffer": "100",            # the exon buffer. Its a characteristic of the data.
    "data_dir": "../data/",          # the directory where the data is
    "seed": 42,                      # seed is always 42
    "test_size": 0,                  # the test size. Set to 0 if ensemble = 1
    "valid_size": 0.2,               # the validation size
    "max_depth": 18,                 # the max depth of the trees
    "n_jobs": 40,                    # the number of threads to use
    "n_estimators": 10000,           # the maximum number of extimators
    "early_stopping_rounds": 10,     # the patience for early stopping
    "early_stopping_size": 0.2,      # the fraction of the training set used for early stopping
    "eval_metric": [                 # the evaluation metrics for early stopping
        "rmse",
        "logloss"
    ],
    "reg_alpha": 0.1,                # the regularization parameter
    "subsample": 1,                  # sbsample as defined by xgboost
    "colsample_bytree": 1,           # as defined by xgboost
    "n_ensemble": 1,                 # the number of XGBDT models to be build in the same
    "shap_sample_size": 10000,       # the sample taken from the validations set to compute Shapley values
    "sample_size": 500000,           # the total sample size = train + test + validation
    "annotate_cell_lines": false,    # set to false for single cell lines and true when using both cell lines
    "compute_shap": false,           # to compute Shapley values or not
    "model-uuid": "UUID",            # the UUID of for the model. putting "UUID" autogenerates UUID
    "base_directory": "../models/"   # the directory where the models will be stored
}
```
