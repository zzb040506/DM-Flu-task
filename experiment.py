"""Reproducible competition experiment. No external data or test labels.
Tune on development folds; freeze recipe before evaluating reserved holdout.
Final CV is descriptive (parameters selected using overlapping development rows).
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse, json, hashlib, time, sys, importlib, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.special import expit, logit
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data'
OUT=ROOT/'artifacts'; OUT.mkdir(exist_ok=True)
TARGETS=['h1n1_vaccine','seasonal_vaccine']
SEED=20261115

def log(s): print(s,flush=True)
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def score(y,p): return float(np.mean([roc_auc_score(y[:,i],p[:,i]) for i in range(2)]))

def features(df,kind,engineering):
    x=df.drop(columns='respondent_id').copy()
    cats=list(x.select_dtypes(include=['object','str']).columns)
    if engineering:
        x['missing_count']=x.isna().sum(axis=1)
        for prefix in ['behavioral_','opinion_','doctor_recc_','employment_']:
            cols=[c for c in df if c.startswith(prefix)]
            x[prefix+'missing']=df[cols].isna().sum(axis=1)
            if prefix!='employment_': x[prefix+'sum']=df[cols].sum(axis=1,min_count=1)
        for s in ['h1n1','seas']:
            a,b,c=[f'opinion_{s}_{v}' for v in ['vacc_effective','risk','sick_from_vacc']]
            x[s+'_opinion_net']=x[a]+x[b]-x[c]
            x[s+'_eff_risk']=x[a]*x[b]
        x['risk_difference']=x.opinion_h1n1_risk-x.opinion_seas_risk
        x['effective_difference']=x.opinion_h1n1_vacc_effective-x.opinion_seas_vacc_effective
        x['household_total']=x.household_adults+x.household_children
        x['knowledge_concern']=x.h1n1_knowledge*x.h1n1_concern
        for c,levels in {'age_group':['18 - 34 Years','35 - 44 Years','45 - 54 Years','55 - 64 Years','65+ Years'],'education':['< 12 Years','12 Years','Some College','College Graduate'],'income_poverty':['Below Poverty','<= $75,000, Above Poverty','> $75,000']}.items():
            x[c+'_ordered']=x[c].map(dict(zip(levels,range(len(levels))))).astype(float)
    for c in cats:
        if kind=='cat': x[c]=x[c].fillna('__missing__').astype(str)
        else:
            levels=sorted(TRAIN[c].dropna().unique())
            if kind=='lgb': x[c]=pd.Categorical(x[c],categories=levels)
            else: x[c]=x[c].map(dict(zip(levels,range(len(levels))))).astype(float)
    if kind=='cat':
        num=x.columns.difference(cats); x[num]=x[num].fillna(-999)
    return x,cats

def model(kind,p,iterations,seed):
    if kind=='lgb': return LGBMClassifier(**p,n_estimators=iterations,objective='binary',verbosity=-1,n_jobs=4,random_state=seed,subsample_freq=1)
    if kind=='xgb': return XGBClassifier(**p,n_estimators=iterations,objective='binary:logistic',tree_method='hist',eval_metric='auc',n_jobs=4,random_state=seed)
    return CatBoostClassifier(**p,iterations=iterations,loss_function='Logloss',eval_metric='AUC',thread_count=4,random_seed=seed,verbose=False,allow_writing_files=False)

def fit(m,kind,x,y,cats,valid=None):
    kw={}
    if kind=='cat': kw['cat_features']=cats
    if valid is not None:
        if kind=='lgb': kw.update(eval_set=[valid],eval_metric='auc',callbacks=[early_stopping(90,verbose=False)])
        elif kind=='xgb':
            m.set_params(early_stopping_rounds=90); kw.update(eval_set=[valid],verbose=False)
        else: kw.update(eval_set=valid,early_stopping_rounds=90)
    m.fit(x,y,**kw)
    return m

def iterations(m,k):
    return int(m.best_iteration_ if k=='lgb' else m.best_iteration+1 if k=='xgb' else m.get_best_iteration()+1)

def get_params(t,k):
    eng=t.suggest_categorical('engineering',[False,True])
    if k=='lgb':
        p=dict(learning_rate=t.suggest_float('learning_rate',.018,.065,log=True),num_leaves=t.suggest_int('num_leaves',7,42),min_child_samples=t.suggest_int('min_child_samples',25,150),colsample_bytree=t.suggest_float('colsample_bytree',.55,1),subsample=t.suggest_float('subsample',.75,1),reg_lambda=t.suggest_float('reg_lambda',.1,30,log=True),reg_alpha=t.suggest_float('reg_alpha',.001,3,log=True),cat_smooth=t.suggest_float('cat_smooth',5,80),max_cat_to_onehot=t.suggest_categorical('max_cat_to_onehot',[4,32]))
    elif k=='xgb':
        p=dict(learning_rate=t.suggest_float('learning_rate',.025,.065,log=True),max_depth=t.suggest_int('max_depth',3,6),min_child_weight=t.suggest_float('min_child_weight',3,20),colsample_bytree=t.suggest_float('colsample_bytree',.6,1),subsample=t.suggest_float('subsample',.75,1),reg_lambda=t.suggest_float('reg_lambda',1,30,log=True),reg_alpha=t.suggest_float('reg_alpha',.001,2,log=True))
    else:
        p=dict(learning_rate=t.suggest_float('learning_rate',.035,.075,log=True),depth=t.suggest_int('depth',4,7),l2_leaf_reg=t.suggest_float('l2_leaf_reg',2,20,log=True),random_strength=t.suggest_float('random_strength',.05,2),one_hot_max_size=t.suggest_categorical('one_hot_max_size',[2,12,32]),rsm=t.suggest_float('rsm',.65,1))
    return eng,p

def tune(kind,ti,ntrials):
    name=f'{kind}_{TARGETS[ti]}'
    folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED+7).split(DEV,Y[DEV,0]*2+Y[DEV,1]))
    study=optuna.create_study(study_name=name,storage=f'sqlite:///{OUT / "search.sqlite"}',load_if_exists=True,direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED+ti,n_startup_trials=5))
    def objective(t):
        eng,p=get_params(t,kind)
        x,cats=features(TRAIN,kind,eng)
        pred=np.zeros(len(DEV)); its=[]
        for tr,va in folds:
            m=fit(model(kind,p,2200,SEED),kind,x.iloc[DEV[tr]],Y[DEV[tr],ti],cats,(x.iloc[DEV[va]],Y[DEV[va],ti]))
            pred[va]=m.predict_proba(x.iloc[DEV[va]])[:,1]; its.append(iterations(m,kind))
        val=roc_auc_score(Y[DEV,ti],pred)
        t.set_user_attr('iterations',its)
        np.savez_compressed(OUT/f'tune_{name}_{t.number}.npz',oof=pred)
        log(f'TUNE {name} trial={t.number} auc={val:.6f} iterations={its}')
        return val
    remaining=max(0,ntrials-len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]))
    study.optimize(objective,n_trials=remaining)
    best=study.best_trial
    p=best.params.copy();eng=p.pop('engineering')
    return dict(kind=kind,target=ti,engineering=eng,params=p,iterations=int(np.median(best.user_attrs['iterations'])),trial=best.number,dev_auc=best.value)

def combine(pred,weights,mode):
    p=np.clip(pred,1e-6,1-1e-6)
    return np.tensordot(weights,p,axes=1) if mode=='probability' else expit(np.tensordot(weights,logit(p),axes=1))

def main(args):
    global TRAIN,Y,DEV
    TRAIN=pd.read_csv(DATA/'training_set_features.csv'); test=pd.read_csv(DATA/'test_set_features.csv')
    labels=pd.read_csv(DATA/'training_set_labels.csv'); template=pd.read_csv(DATA/'submission_format.csv')
    assert TRAIN.respondent_id.equals(labels.respondent_id)
    assert template.respondent_id.equals(test.respondent_id)
    assert TRAIN.respondent_id.is_unique and test.respondent_id.is_unique
    assert set(TRAIN.respondent_id).isdisjoint(test.respondent_id)
    Y=labels[TARGETS].to_numpy()
    DEV,hold=train_test_split(np.arange(len(Y)),test_size=.22,stratify=Y[:,0]*2+Y[:,1],random_state=SEED)
    np.savez(OUT/'split.npz',dev=DEV,holdout=hold)
    dump(OUT/'provenance.json',{'seed':SEED,'python':sys.version,'versions':{m:importlib.import_module(m).__version__ for m in ['numpy','pandas','sklearn','lightgbm','xgboost','catboost','optuna']},'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.glob('*.csv')}})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg=[]
    for k,n in [('lgb',args.lgb_trials),('xgb',args.xgb_trials),('cat',args.cat_trials)]:
        for ti in range(2):
            cfg.append(tune(k,ti,n));dump(OUT/'selected_models.json',cfg)
    # Freeze blend using DEVELOPMENT OOF only, never the holdout.
    weights=[];modes=[]
    for ti in range(2):
        cs=[c for c in cfg if c['target']==ti]
        pred=np.array([np.load(OUT/f'tune_{c["kind"]}_{TARGETS[ti]}_{c["trial"]}.npz')['oof'] for c in cs])
        choices=[]
        # Small, predefined convex grid; prevents extreme single-model selection.
        for w in [[1/3,1/3,1/3],[.5,.25,.25],[.25,.5,.25],[.25,.25,.5],[.4,.2,.4],[.2,.4,.4]]:
            for mode in ['probability','logit']:
                choices.append((roc_auc_score(Y[DEV,ti],combine(pred,w,mode)),w,mode))
        best=max(choices,key=lambda z:z[0]);weights.append(best[1]);modes.append(best[2])
        log(f'FROZEN BLEND {TARGETS[ti]} dev_auc={best[0]:.6f} weights={best[1]} mode={best[2]}')
    recipe={'models':cfg,'weights':weights,'modes':modes,'holdout_used_for_selection':False}
    dump(OUT/'frozen_recipe.json',recipe)
    hp=np.zeros((3,len(hold),2)); final_oof=np.zeros((3,len(Y),2)); final_test=np.zeros((3,len(test),2)); full_test=final_test.copy()
    fold_ids=np.zeros(len(Y),int)
    folds=list(StratifiedKFold(args.folds,shuffle=True,random_state=SEED+91).split(TRAIN,Y[:,0]*2+Y[:,1]))
    for fi,(_,va) in enumerate(folds):fold_ids[va]=fi
    for c in cfg:
        k,ti=c['kind'],c['target'];ki=['lgb','xgb','cat'].index(k)
        x,cats=features(TRAIN,k,c['engineering']);xt,_=features(test,k,c['engineering'])
        cache=OUT/f'final_{k}_{ti}_{args.folds}.npz'
        if cache.exists():
            z=np.load(cache);hp[ki,:,ti]=z['holdout'];final_oof[ki,:,ti]=z['oof'];final_test[ki,:,ti]=z['test'];full_test[ki,:,ti]=z['full'];continue
        # Retrain on all development rows with iterations chosen without holdout labels.
        rounds=max(80,int(c['iterations']*1.12))
        m=fit(model(k,c['params'],rounds,SEED),k,x.iloc[DEV],Y[DEV,ti],cats)
        hp[ki,:,ti]=m.predict_proba(x.iloc[hold])[:,1]
        log(f'HOLDOUT {k} {TARGETS[ti]} auc={roc_auc_score(Y[hold,ti],hp[ki,:,ti]):.6f}')
        for fi,(tr,va) in enumerate(folds):
            m=fit(model(k,c['params'],rounds,SEED+fi*17),k,x.iloc[tr],Y[tr,ti],cats)
            final_oof[ki,va,ti]=m.predict_proba(x.iloc[va])[:,1]
            final_test[ki,:,ti]+=m.predict_proba(xt)[:,1]/len(folds)
            log(f'FINAL {k} {TARGETS[ti]} fold={fi} auc={roc_auc_score(Y[va,ti],final_oof[ki,va,ti]):.6f}')
        for seed in [SEED,42,2026]:
            m=fit(model(k,c['params'],int(rounds*1.08),seed),k,x,Y[:,ti],cats)
            full_test[ki,:,ti]+=m.predict_proba(xt)[:,1]/3
            import joblib
            joblib.dump(m,OUT/f'model_{k}_{ti}_{seed}.joblib',compress=3)
        np.savez_compressed(cache,holdout=hp[ki,:,ti],oof=final_oof[ki,:,ti],test=final_test[ki,:,ti],full=full_test[ki,:,ti])
    blend_hold=np.column_stack([combine(hp[:,:,ti],weights[ti],modes[ti]) for ti in range(2)])
    blend_oof=np.column_stack([combine(final_oof[:,:,ti],weights[ti],modes[ti]) for ti in range(2)])
    report={'holdout_n':len(hold),'holdout_mean_auc':score(Y[hold],blend_hold),'holdout_target_auc':[roc_auc_score(Y[hold,ti],blend_hold[:,ti]) for ti in range(2)],'final_cv_mean_auc':score(Y,blend_oof),'final_cv_note':'Descriptive, not independent of hyperparameter selection. Holdout evaluated after recipe freeze.','official_score':None,'official_rank':None}
    for name,arr in [('foldbag',final_test),('fullbag',full_test),('robust',.5*final_test+.5*full_test)]:
        pred=np.column_stack([combine(arr[:,:,ti],weights[ti],modes[ti]) for ti in range(2)])
        sub=template.copy();sub[TARGETS]=pred
        assert sub.shape==(26708,3) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
        sub.to_csv(ROOT/f'submission_{name}.csv',index=False,float_format='%.10f')
    np.savez_compressed(OUT/'predictions.npz',holdout_ids=hold,holdout=hp,oof=final_oof,fold_ids=fold_ids,test=final_test,full=full_test)
    dump(ROOT/'metrics.json',report);log(json.dumps(report,indent=2));log('DONE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lgb-trials',type=int,default=14);p.add_argument('--xgb-trials',type=int,default=8);p.add_argument('--cat-trials',type=int,default=6);p.add_argument('--folds',type=int,default=5)
    main(p.parse_args())
