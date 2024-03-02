## Getting the data

run

```{sh}
sh getModels.sh
```

The files in the model folders are:
```
.
├── bdt-xgb-models-HepG2-100-8414c135
│   ├── config-8414c135.json
│   ├── HepG2-100-50-50-ensemble-models.pkl.xz
│   ├── HepG2-100-50-50-train-data.dat.xz
│   └── HepG2-100-50-50-validation-data.dat.xz
└──bdt-xgb-models-K562-100-cf156139
    ├── config-cf156139.json
    ├── K562-100-50-50-ensemble-models.pkl.xz
    ├── K562-100-50-50-train-data.dat.xz
    └── K562-100-50-50-validation-data.dat.xz


```

- The two folders contain the two models for the two cell lines
- The training and test data are separated
- the model is stored as a pickled dictionary. The key '0' should acces the first model