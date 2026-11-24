from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent
# 优先读取仓库内的 data/，同时兼容把数据放在上一级的旧用法
DATA=ROOT/'data' if (ROOT/'data'/'training_set_features.csv').exists() else ROOT.parent/'data'
T=['h1n1_vaccine','seasonal_vaccine']
template=pd.read_csv(DATA/'submission_format.csv')
results={}
for f in sorted(ROOT.glob('submission_*.csv')):
 d=pd.read_csv(f)
 assert list(d)==['respondent_id']+T and d.shape==template.shape
 assert d.respondent_id.equals(template.respondent_id) and d.respondent_id.is_unique
 a=d[T].to_numpy();assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
 results[f.name]={'rows':len(d),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'range':[float(a.min()),float(a.max())]}
assert len(results)>=3
z=np.load(ROOT/'artifacts/predictions.npz'); y=pd.read_csv(DATA/'training_set_labels.csv')[T].to_numpy()
recipe=json.loads((ROOT/'artifacts/frozen_recipe.json').read_text())
from experiment import combine
pred=np.column_stack([combine(z['holdout'][:,:,i],recipe['weights'][i],recipe['modes'][i]) for i in range(2)])
base=np.load(ROOT/'artifacts/baseline_holdout.npz');assert np.array_equal(z['holdout_ids'],base['holdout_ids'])
yh=y[z['holdout_ids']]
auc=lambda yy,pp:np.mean([roc_auc_score(yy[:,i],pp[:,i]) for i in range(2)])
rng=np.random.default_rng(20261115); diffs=[]
for _ in range(1000):
 ix=rng.integers(len(yh),size=len(yh));diffs.append(auc(yh[ix],pred[ix])-auc(yh[ix],base['pred'][ix]))
results['paired_holdout_comparison']={'new_auc':auc(yh,pred),'smith_lgb_auc':auc(yh,base['pred']),'difference':auc(yh,pred)-auc(yh,base['pred']),'bootstrap_95pct_difference':np.quantile(diffs,[.025,.975]).tolist(),'bootstrap_replicates':1000,'note':'Paired row bootstrap, conditional on the trained models; does not cover training variability or official test distribution.'}
if (ROOT/'artifacts/joint_final.npz').exists():
 j=np.load(ROOT/'artifacts/joint_final.npz')
 w=np.asarray(json.loads((ROOT/'artifacts/joint_frozen_recipe.json').read_text())['joint_weights'])
 jp=pred*(1-w)+j['holdout']*w
 rng=np.random.default_rng(20261115);delta=[]
 for _ in range(1000):
  ix=rng.integers(len(yh),size=len(yh));delta.append(auc(yh[ix],jp[ix])-auc(yh[ix],base['pred'][ix]))
 results['joint_paired_holdout_comparison']={'new_auc':auc(yh,jp),'smith_lgb_auc':auc(yh,base['pred']),'difference':auc(yh,jp)-auc(yh,base['pred']),'bootstrap_95pct_difference':np.quantile(delta,[.025,.975]).tolist(),'bootstrap_replicates':1000,'note':'Conditional row bootstrap; not official test uncertainty.'}
(ROOT/'verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
