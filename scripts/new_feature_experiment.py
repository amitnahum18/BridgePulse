import itertools
import json
import shutil
from datetime import datetime
from pathlib import Path
import h5py
import nbformat
import numpy as np
from scipy.fft import dct
from scipy.signal import get_window
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

def signal_features(x, fs):
    """16 LFCC summaries plus 8 relative Haar energies; one recording at a time."""
    frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
    power = np.abs(np.fft.rfft(frames * get_window('hann',512), axis=1))**2
    frequencies = np.fft.rfftfreq(512,1/fs)
    edges = np.linspace(0,fs/2,18)
    filters = np.maximum(0,np.minimum((frequencies[None,:]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None],
                         (edges[2:,None]-frequencies[None,:])/(edges[2:]-edges[1:-1])[:,None]))
    log_power = 10*np.log10(np.maximum(power @ filters.T,1e-10))
    coefficients = dct(log_power,type=2,norm='ortho',axis=1)[:,:8]
    result = list(coefficients.mean(axis=0)) + list(coefficients.std(axis=0))
    # Orthonormal Haar decomposition; trim a recording locally to a multiple of 128.
    a = x[:len(x)//128*128].copy()
    energies=[]
    for _ in range(7):
        detail=(a[::2]-a[1::2])/np.sqrt(2)
        a=(a[::2]+a[1::2])/np.sqrt(2)
        energies.append(float(np.sum(detail**2)))
    energies.append(float(np.sum(a**2)))
    total=sum(energies)
    result += [e/total if total else 0 for e in energies]
    return np.asarray(result)

def run_experiment():
    with np.load(ROOT/'data/processed/all_acceleration_features.npz') as source:
        ids=source['record_ids'].astype(str); y=source['y']; names=source['feature_names'].astype(str)
        old_X=source['X']; mfcc=old_X[:,['_mfcc_' in n for n in names]]
    sensors=sorted({n.split('_')[0] for n in names})
    new_X=np.empty((25,len(sensors)*2*24))
    with h5py.File(ROOT/'data/raw/Rt345Bridge.h5','r') as raw:
        for i,identity in enumerate(ids):
            state,test=identity.split('/'); values=[]
            for sensor in sensors:
                group=raw[f'{state}/acceleration/{test}/{sensor}']
                fs=float(np.asarray(group.attrs['sample_rate']).item())
                for axis in ('y','z'):
                    values.extend(signal_features(group[axis][...].reshape(-1).astype(float),fs))
            new_X[i]=values
    assert np.isfinite(new_X).all()
    new_names=[f'{sensor}_{axis}_{feature}' for sensor in sensors for axis in ('y','z')
               for feature in ([f'lfcc_{i}_{stat}' for stat in ('mean','std') for i in range(8)]+
                               [f'haar_fraction_{i}' for i in range(1,9)])]
    lfcc=new_X[:,['_lfcc_' in n for n in new_names]]
    wave=new_X[:,['_haar_' in n for n in new_names]]
    families={'MFCC':mfcc,'LFCC':lfcc,'Haar relative energies':wave,
              'MFCC + Haar':np.column_stack([mfcc,wave])}
    models={'Logistic L2':LogisticRegression(C=.01,class_weight='balanced',max_iter=5000),
            'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=2,
                class_weight='balanced',random_state=42,n_jobs=1)}
    states=np.array([r.split('/')[0] for r in ids]); folds=[]; allocations=[]
    for allocation,damages in enumerate(itertools.combinations([f'damage_{i}' for i in range(1,7)],3),1):
        side={'baseline_1',*damages}
        mask=np.isin(states,list(side))
        for family,X in families.items():
            for name,model in models.items():
                oof=np.empty_like(y)
                for fold,hold in enumerate((mask,~mask),1):
                    a=np.flatnonzero(~hold);b=np.flatnonzero(hold)
                    assert set(states[a]).isdisjoint(states[b])
                    pipe=make_pipeline(VarianceThreshold(),StandardScaler(),clone(model))
                    pipe.fit(X[a],y[a]); prediction=pipe.predict(X[b]);oof[b]=prediction
                    score=pipe.decision_function(X[b]) if hasattr(pipe,'decision_function') else pipe.predict_proba(X[b])[:,1]
                    train_ba=float(balanced_accuracy_score(y[a],pipe.predict(X[a])))
                    val_ba=float(balanced_accuracy_score(y[b],prediction))
                    folds.append({'allocation':allocation,'fold':fold,'family':family,'model':name,
                        'train_BA':train_ba,'validation_BA':val_ba,'gap':train_ba-val_ba,
                        'damage_AP':float(average_precision_score(y[b],score)),
                        'baseline_AP':float(average_precision_score(1-y[b],-score))})
                allocations.append({'allocation':allocation,'family':family,'model':name,
                                    'OOF_BA':float(balanced_accuracy_score(y,oof))})
        if allocation%5==0:print(f'Completed allocation {allocation}/20',flush=True)
    summary=[]
    for family,X in families.items():
        for name in models:
            selected=[r for r in folds if r['family']==family and r['model']==name]
            ba=np.array([r['OOF_BA'] for r in allocations if r['family']==family and r['model']==name])
            summary.append({'family':family,'model':name,'features':X.shape[1],
                'BA_mean':float(ba.mean()),'BA_split_SD':float(ba.std()),
                **{k:float(np.mean([r[k] for r in selected])) for k in ('train_BA','gap','damage_AP','baseline_AP')}})
    summary.sort(key=lambda r:r['BA_mean'],reverse=True)
    np.savez_compressed(ROOT/'data/processed/lfcc_haar_features.npz',X=new_X,y=y,record_ids=ids,feature_names=new_names)
    report={'summary':summary,'folds':folds,'allocations':allocations,
        'limitations':'Exploratory comparison on previously examined data, fixed parameters, state grouping only, one bridge. Split SD is not a confidence interval.',
        'feature_definitions':'LFCC: 16 linear triangular filterbanks, log power, DCT, 8 coefficients summarized by mean/std. Haar: normalized detail energies at levels 1-7 and approximation energy level 7. No acquisition-time labels or window samples.'}
    (ROOT/'outputs/new_features_grouped_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return report

if __name__=='__main__':
    report=run_experiment()
    path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
    shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_new_features_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    description=nbformat.v4.new_markdown_cell('## 19. New feature families: LFCC and Haar wavelet energies\n\nCompare fixed Logistic L2 and shallow Random Forest using the same 20 state-grouped allocations as stage 18. Results remain exploratory. Haar relative energies add temporal-scale information without creating separately labeled windows. LFCC uses linear frequency filters. Raw H5 is unchanged. The full reusable implementation is in `scripts/new_feature_experiment.py`.')
    code=nbformat.v4.new_code_cell('import sys\nif str(ROOT / "scripts") not in sys.path: sys.path.insert(0,str(ROOT / "scripts"))\nfrom new_feature_experiment import run_experiment\nnew_feature_report = run_experiment()\nshow_analysis_table(new_feature_report["summary"])')
    description['id']='new-families-00';code['id']='new-families-01'
    code.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps(report['summary'],indent=2))]
    nb.cells[-1:-1]=[description,code];nbformat.validate(nb);nbformat.write(nb,path)
