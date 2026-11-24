"""Smith-MMA LightGBM saved hyperparameters, evaluated on the new split.
Original ordinal encoding is retained to provide a faithful baseline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data'
x=pd.read_csv(DATA/'training_set_features.csv').drop(columns='respondent_id')
y=pd.read_csv(DATA/'training_set_labels.csv')[['h1n1_vaccine','seasonal_vaccine']]
z=np.load(ROOT/'artifacts/split.npz');tr,va=z['dev'],z['holdout']
params=[dict(n_estimators=475,learning_rate=.036,max_depth=9,reg_alpha=7.2497,reg_lambda=5.2895,subsample=.6174,colsample_bytree=.4625,class_weight=None),dict(n_estimators=450,learning_rate=.0942,max_depth=10,reg_alpha=4.8069,reg_lambda=96.7952,subsample=.2598,colsample_bytree=.3932,class_weight='balanced')]
cat=['race','sex','marital_status','rent_or_own','hhs_geo_region','census_msa','employment_industry','employment_occupation']
ordinal=['age_group','education','income_poverty','employment_status']
pred=np.zeros((len(va),2));rows=[]
for ti,t in enumerate(y):
 prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('ord',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),ordinal)],remainder='passthrough',sparse_threshold=0)
 m=Pipeline([('preprocessing',prep),('model',LGBMClassifier(**params[ti],n_jobs=2,verbosity=-1,random_state=20261115))])
 m.fit(x.iloc[tr],y.iloc[tr,ti]);pred[:,ti]=m.predict_proba(x.iloc[va])[:,1]
 rows.append({'target':t,'auc':roc_auc_score(y.iloc[va,ti],pred[:,ti])})
 print(rows[-1],flush=True)
np.savez_compressed(ROOT/'artifacts/baseline_holdout.npz',pred=pred,holdout_ids=va)
(ROOT/'baseline_metrics.json').write_text(json.dumps({'model':'Smith-MMA LightGBM saved notebook hyperparameters','targets':rows,'mean_auc':np.mean([r['auc'] for r in rows]),'note':'Historical parameters originated in original notebook; same reserved rows as new experiment.'},indent=2))
