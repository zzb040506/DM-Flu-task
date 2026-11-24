"""Joint four-state target model; no observed target is used as input.
Blend weights selected using development OOF, before joint holdout is scored.
Run experiment.py first or run --screen while it trains, then run without flags.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier,early_stopping
import experiment as E
ROOT=E.ROOT;OUT=E.OUT

def marg(p):return np.column_stack([p[:,2]+p[:,3],p[:,1]+p[:,3]])
def metric(y,p):
 yy=np.column_stack([np.asarray(y,dtype=int)//2,np.asarray(y,dtype=int)%2])
 return 'mean_auc',E.score(yy,marg(p)),True

def make(p,n,seed):return LGBMClassifier(**p,n_estimators=n,objective='multiclass',num_class=4,metric='None',verbosity=-1,n_jobs=2,random_state=seed,subsample_freq=1)

def main(screen):
 E.TRAIN=pd.read_csv(E.DATA/'training_set_features.csv');test=pd.read_csv(E.DATA/'test_set_features.csv')
 y=pd.read_csv(E.DATA/'training_set_labels.csv')[E.TARGETS].to_numpy();yc=y[:,0]*2+y[:,1]
 split=np.load(OUT/'split.npz');dev,hold=split['dev'],split['holdout']
 folds=list(StratifiedKFold(3,shuffle=True,random_state=E.SEED+7).split(dev,yc[dev]))
 candidates=[]
 for eng in [False,True]:
  for leaves in [7,15,25]:
   candidates.append({'engineering':eng,'params':dict(num_leaves=leaves,learning_rate=.035,min_child_samples=70,colsample_bytree=.8,subsample=.9,reg_lambda=8,cat_smooth=30,max_cat_to_onehot=32)})
 rows=[]
 for ci,c in enumerate(candidates):
  cache=OUT/f'joint_search_{ci}.npz'
  if cache.exists():z=np.load(cache);pred=z['oof'];its=z['iterations'].tolist()
  else:
   x,_=E.features(E.TRAIN,'lgb',c['engineering']);pred=np.zeros((len(dev),2));its=[]
   for tr,va in folds:
    m=make(c['params'],1600,E.SEED)
    m.fit(x.iloc[dev[tr]],yc[dev[tr]],eval_set=[(x.iloc[dev[va]],yc[dev[va]])],eval_metric=metric,callbacks=[early_stopping(80,first_metric_only=True,verbose=False)])
    pred[va]=marg(m.predict_proba(x.iloc[dev[va]]));its.append(m.best_iteration_)
   np.savez_compressed(cache,oof=pred,iterations=its)
  rows.append(dict(index=ci,auc=E.score(y[dev],pred),iterations=its,**c))
  E.log(f'JOINT SCREEN {rows[-1]}')
 best=max(rows,key=lambda r:r['auc']);E.dump(OUT/'joint_selected.json',best)
 if screen:return
 recipe=json.loads((OUT/'frozen_recipe.json').read_text())
 base=np.zeros((len(dev),2))
 for ti in range(2):
  cs=[c for c in recipe['models'] if c['target']==ti]
  ps=np.array([np.load(OUT/f'tune_{c["kind"]}_{E.TARGETS[ti]}_{c["trial"]}.npz')['oof'] for c in cs])
  base[:,ti]=E.combine(ps,recipe['weights'][ti],recipe['modes'][ti])
 jp=np.load(OUT/f'joint_search_{best["index"]}.npz')['oof']
 # Predefined small grid, independently selected for each target on development rows.
 weights=[]
 for ti in range(2):
  w=max([0,.15,.3,.5],key=lambda a:roc_auc_score(y[dev,ti],(1-a)*base[:,ti]+a*jp[:,ti]))
  weights.append(w)
 E.dump(OUT/'joint_frozen_recipe.json',{'joint':best,'joint_weights':weights,'selection_data':'development OOF only','holdout_used_for_selection':False})
 E.log(f'JOINT WEIGHTS {weights}')
 x,_=E.features(E.TRAIN,'lgb',best['engineering']);xt,_=E.features(test,'lgb',best['engineering'])
 n=max(80,int(np.median(best['iterations'])*1.12))
 cache=OUT/'joint_final.npz'
 if cache.exists():z=np.load(cache);hp=z['holdout'];op=z['oof'];tp=z['test'];fp=z['full']
 else:
  m=make(best['params'],n,E.SEED);m.fit(x.iloc[dev],yc[dev]);hp=marg(m.predict_proba(x.iloc[hold]))
  op=np.zeros((len(y),2));tp=np.zeros((len(test),2));fp=tp.copy()
  for fi,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=E.SEED+91).split(x,yc)):
   m=make(best['params'],n,E.SEED+fi*17);m.fit(x.iloc[tr],yc[tr]);op[va]=marg(m.predict_proba(x.iloc[va]));tp+=marg(m.predict_proba(xt))/5
   E.log(f'JOINT FINAL fold={fi} auc={E.score(y[va],op[va]):.6f}')
  for seed in [E.SEED,42,2026]:
   m=make(best['params'],int(n*1.08),seed);m.fit(x,yc);fp+=marg(m.predict_proba(xt))/3
   joblib.dump(m,OUT/f'model_joint_{seed}.joblib',compress=3)
  np.savez_compressed(cache,holdout=hp,oof=op,test=tp,full=fp)
 z=np.load(OUT/'predictions.npz');w=np.asarray(weights)
 bh=np.column_stack([E.combine(z['holdout'][:,:,ti],recipe['weights'][ti],recipe['modes'][ti]) for ti in range(2)])
 bo=np.column_stack([E.combine(z['oof'][:,:,ti],recipe['weights'][ti],recipe['modes'][ti]) for ti in range(2)])
 for name,jarr in [('robust',.5*(tp+fp)),('foldbag',tp),('fullbag',fp)]:
  sub=pd.read_csv(ROOT/f'submission_{name}.csv');sub[E.TARGETS]=(1-w)*sub[E.TARGETS].to_numpy()+w*jarr
  sub.to_csv(ROOT/f'submission_joint_{name}.csv',index=False,float_format='%.10f')
 result={'joint_weights':weights,'joint_development_auc':best['auc'],'base_development_auc':E.score(y[dev],base),'blended_development_auc':E.score(y[dev],base*(1-w)+jp*w),'joint_holdout_auc':E.score(y[hold],hp),'blended_holdout_auc':E.score(y[hold],bh*(1-w)+hp*w),'blended_descriptive_cv_auc':E.score(y,bo*(1-w)+op*w),'official_score':None}
 E.dump(ROOT/'joint_metrics.json',result);E.log(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--screen',action='store_true');main(p.parse_args().screen)
