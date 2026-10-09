import base64
import itertools
import json
import shutil
from datetime import datetime
from pathlib import Path
import h5py
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]

def run_robust_audit():
    with np.load(ROOT/'data/processed/mfcc_frame_features.npz') as data:
        X=data['4s_robust_change'];y=data['y'];ids=data['record_ids'].astype(str)
    with np.load(ROOT/'data/processed/all_acceleration_features.npz') as source:
        assert np.array_equal(ids,source['record_ids'].astype(str))
        assert np.array_equal(y,source['y'])
        sensors=sorted({n.split('_')[0] for n in source['feature_names'].astype(str)})
    assert X.shape==(25,1392) and np.isfinite(X).all() and len(set(ids))==25
    assert 'A14' not in sensors
    assert np.array_equal(y,np.array([int(not identity.startswith('baseline_')) for identity in ids]))
    duplicates=[(ids[i],ids[j]) for i in range(25) for j in range(i) if np.array_equal(X[i],X[j])]
    assert not duplicates
    # Independent reconstruction in reverse recording order: local signal only, no y.
    max_error=0.
    with h5py.File(ROOT/'data/raw/Rt345Bridge.h5','r') as raw:
        for i in reversed(range(25)):
            state,test=ids[i].split('/'); rebuilt=[]
            for sensor in sensors:
                group=raw[f'{state}/acceleration/{test}/{sensor}']
                fs=float(np.asarray(group.attrs['sample_rate']).item())
                for axis in ('y','z'):
                    signal=group[axis][...].reshape(-1).astype(float)
                    c=librosa.feature.mfcc(y=signal,sr=fs,n_mfcc=8,n_mels=16,n_fft=int(fs*4),
                        hop_length=int(fs*2),fmin=0,fmax=fs/2,center=False)
                    for coefficient in c:
                        rebuilt.extend([float(np.median(coefficient)),
                            float(np.percentile(coefficient,75)-np.percentile(coefficient,25)),
                            float(np.mean(np.abs(coefficient[1:]-coefficient[:-1])))])
            max_error=max(max_error,float(np.max(np.abs(X[i]-rebuilt))))
            assert np.allclose(X[i],rebuilt,rtol=1e-12,atol=1e-12)
    audit={'raw_reconstruction_all_25_passed':True,'max_absolute_reconstruction_error':max_error,
        'labels_aligned':True,'one_row_per_recording':True,'exact_duplicates':duplicates,
        'label_metadata_not_in_features':True,'A14_excluded':True,
        'direct_preprocessing_leakage_found':False,
        'remaining_bias':'4s robust summaries were selected after examining six configurations on these same data. Feature-family/model selection optimism remains. State grouping cannot prove independent excitation conditions or generalization to another bridge.'}
    (ROOT/'outputs/robust_mfcc_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Raw reconstruction and alignment audit passed.',flush=True)
    models={
        'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=1),
        'Logistic L2':LogisticRegression(C=.01,class_weight='balanced',max_iter=5000),
        'SVM RBF':SVC(C=.1,gamma='scale',class_weight='balanced'),
        'Extra Trees':ExtraTreesClassifier(n_estimators=200,max_depth=2,min_samples_leaf=4,class_weight='balanced',random_state=42,n_jobs=1),
        'Histogram Boosting':HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=3,min_samples_leaf=5,l2_regularization=1.,class_weight='balanced',early_stopping=False,random_state=42),
    }
    states=np.array([r.split('/')[0] for r in ids]);folds=[];allocations=[];predictions=[]
    for allocation,damages in enumerate(itertools.combinations([f'damage_{i}' for i in range(1,7)],3),1):
        mask=np.isin(states,['baseline_1',*damages])
        for name,estimator in models.items():
            oof=np.empty_like(y);seen=np.zeros(len(y),dtype=int)
            for fold,hold in enumerate((mask,~mask),1):
                a=np.flatnonzero(~hold);b=np.flatnonzero(hold)
                assert set(states[a]).isdisjoint(states[b]) and set(ids[a]).isdisjoint(ids[b])
                assert set(y[a])==set(y[b])=={0,1}
                model=make_pipeline(StandardScaler(),clone(estimator)) if name in ('Logistic L2','SVM RBF') else clone(estimator)
                model.fit(X[a],y[a])
                if name in ('Logistic L2','SVM RBF'):
                    scaler=model[0]
                    assert np.allclose(scaler.mean_,X[a].mean(axis=0))
                    assert np.allclose(scaler.var_,X[a].var(axis=0))
                    assert int(scaler.n_samples_seen_)==len(a)
                    before=scaler.mean_.copy();scaler.transform(X[b]+1e6)
                    assert np.array_equal(scaler.mean_,before)
                predicted=model.predict(X[b]);oof[b]=predicted;seen[b]+=1
                scores=model.decision_function(X[b]) if hasattr(model,'decision_function') else model.predict_proba(X[b])[:,1]
                train_ba=float(balanced_accuracy_score(y[a],model.predict(X[a])))
                val_ba=float(balanced_accuracy_score(y[b],predicted))
                folds.append({'allocation':allocation,'fold':fold,'model':name,'train_BA':train_ba,
                    'validation_BA':val_ba,'gap':train_ba-val_ba,
                    'damage_AP':float(average_precision_score(y[b],scores)),
                    'baseline_AP':float(average_precision_score(1-y[b],-scores))})
                for i,p,s in zip(b,predicted,scores):predictions.append({'allocation':allocation,'fold':fold,
                    'model':name,'recording':str(ids[i]),'true_label':int(y[i]),'prediction':int(p),'score':float(s)})
            assert np.all(seen==1)
            allocations.append({'allocation':allocation,'model':name,'OOF_BA':float(balanced_accuracy_score(y,oof))})
        if allocation%5==0:print(f'Completed audited comparison {allocation}/20',flush=True)
    summary=[]
    for name in models:
        selected=[r for r in folds if r['model']==name];ba=np.array([r['OOF_BA'] for r in allocations if r['model']==name])
        summary.append({'model':name,'BA_mean':float(ba.mean()),'BA_split_SD':float(ba.std()),
            **{key:float(np.mean([r[key] for r in selected])) for key in ('train_BA','gap','damage_AP','baseline_AP')}})
    rf=next(r for r in summary if r['model']=='Random Forest')
    assert np.isclose(rf['BA_mean'],.781578947368421)
    previous=json.loads((ROOT/'outputs/mfcc_frame_report.json').read_text())
    reference={r['allocation']:r['OOF_BA'] for r in previous['allocations'] if r['configuration']=='4s robust_change'}
    assert all(np.isclose(r['OOF_BA'],reference[r['allocation']]) for r in allocations if r['model']=='Random Forest')
    audit.update({'state_overlap_checks_passed':True,'scaler_train_only_checks_passed':True,'RF_all_20_scores_reproduced':True})
    (ROOT/'outputs/robust_mfcc_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    summary.sort(key=lambda r:r['BA_mean'],reverse=True)
    report={'audit':audit,'summary':summary,'folds':folds,'allocations':allocations,'predictions':predictions,
        'features':1392,'parameters':{name:model.get_params() for name,model in models.items()},
        'limitations':'Fixed conservative parameters; no grouped inner tuning feasible with only two baseline states. All results exploratory. Split SD is not a confidence interval.'}
    (ROOT/'outputs/robust_mfcc_models_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');p=np.arange(len(summary))
    axes[0].bar(p,[r['BA_mean'] for r in summary],yerr=[r['BA_split_SD'] for r in summary],capsize=4)
    axes[0].axhline(.5,color='gray',ls='--');axes[0].set(ylabel='Grouped OOF balanced accuracy',ylim=(0,1.05))
    axes[1].bar(p-.18,[r['damage_AP'] for r in summary],.36,label='Damage AP')
    axes[1].bar(p+.18,[r['baseline_AP'] for r in summary],.36,label='Baseline AP');axes[1].legend();axes[1].set_ylim(0,1.05)
    for ax in axes:ax.set_xticks(p,[r['model'] for r in summary],rotation=25,ha='right')
    plot=ROOT/'outputs/robust_mfcc_models.png';fig.savefig(plot,dpi=150);plt.close(fig)
    path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
    shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_robust_audit_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    md=nbformat.v4.new_markdown_cell('## 24. Improved MFCC audit and model comparison\n\nIndependently reconstruct every 4s median/IQR/inter-frame-change row from raw H5 in reverse order; verify labels, absence of metadata inputs, uniqueness, state separation and exactly one held-out prediction per recording per allocation. Confirm scaler means/variances use only training rows. Reproduce every previous Random Forest allocation score. Compare five fixed models on identical features/splits. Passing these checks does not remove post-hoc feature-family selection optimism or establish physical independence or external validity.')
    cell=nbformat.v4.new_code_cell('import sys\nif str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))\nfrom audit_compare_robust_mfcc import run_robust_audit\nrobust_comparison=run_robust_audit()\nprint(robust_comparison["audit"])\nshow_analysis_table(robust_comparison["summary"])\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/"outputs/robust_mfcc_models.png")))')
    md['id']='robust-audit-00';cell['id']='robust-audit-01'
    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps({'audit':audit,'summary':summary},indent=2)),nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(plot.read_bytes()).decode()},metadata={})]
    nb.cells=[c for c in nb.cells if not c.get('id','').startswith('robust-audit-')];nb.cells[-1:-1]=[md,cell]
    nbformat.validate(nb);nbformat.write(nb,path)
    print(json.dumps(summary,indent=2),flush=True)
    return report

if __name__=='__main__':run_robust_audit()
