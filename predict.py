"""Reproduce full-seed submissions from saved trusted models, without fitting."""
import argparse,json
import numpy as np
import pandas as pd
import joblib
import experiment as E

def main(limit):
 E.TRAIN=pd.read_csv(E.DATA/'training_set_features.csv')
 test=pd.read_csv(E.DATA/'test_set_features.csv')
 if limit:test=test.iloc[:limit]
 recipe=json.loads((E.OUT/'frozen_recipe.json').read_text())
 pred=np.zeros((3,len(test),2))
 for c in recipe['models']:
  k,ti=c['kind'],c['target'];ki=['lgb','xgb','cat'].index(k)
  x,_=E.features(test,k,c['engineering'])
  for seed in [E.SEED,42,2026]:
   m=joblib.load(E.OUT/f'model_{k}_{ti}_{seed}.joblib')
   pred[ki,:,ti]+=m.predict_proba(x)[:,1]/3
 base=np.column_stack([E.combine(pred[:,:,ti],recipe['weights'][ti],recipe['modes'][ti]) for ti in range(2)])
 original=pd.read_csv(E.ROOT/'submission_fullbag.csv')[E.TARGETS].to_numpy()[:len(test)]
 assert np.allclose(base,original,atol=1e-9,rtol=0)
 result={'rows':len(test),'base_max_abs_error':float(np.max(abs(base-original)))}
 jp=E.OUT/'joint_frozen_recipe.json'
 if jp.exists():
  j=json.loads(jp.read_text());x,_=E.features(test,'lgb',j['joint']['engineering'])
  from joint import marg
  p=sum(marg(joblib.load(E.OUT/f'model_joint_{seed}.joblib').predict_proba(x))/3 for seed in [E.SEED,42,2026])
  w=np.asarray(j['joint_weights']);p=base*(1-w)+p*w
  original=pd.read_csv(E.ROOT/'submission_joint_fullbag.csv')[E.TARGETS].to_numpy()[:len(test)]
  assert np.allclose(p,original,atol=1e-9,rtol=0)
  result['joint_max_abs_error']=float(np.max(abs(p-original)))
 E.dump(E.ROOT/'inference_verification.json',result);print(result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);main(p.parse_args().limit)
