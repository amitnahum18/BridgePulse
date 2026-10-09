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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1]

def run_frame_experiment():
    with np.load(ROOT/'data/processed/all_acceleration_features.npz') as data:
        ids=data['record_ids'].astype(str);y=data['y'];names=data['feature_names'].astype(str)
        old=data['X'][:,['_mfcc_' in n for n in names]]
    sensors=sorted({n.split('_')[0] for n in names});states=np.array([r.split('/')[0] for r in ids])
    matrices={f'{seconds}s {summary}':[] for seconds in (2,4,8) for summary in ('mean_std','robust_change')}
    frame_counts={}
    with h5py.File(ROOT/'data/raw/Rt345Bridge.h5','r') as raw:
        for identity in ids:
            state,test=identity.split('/');rows={key:[] for key in matrices}
            for sensor in sensors:
                group=raw[f'{state}/acceleration/{test}/{sensor}'];fs=float(np.asarray(group.attrs['sample_rate']).item())
                for axis in ('y','z'):
                    x=group[axis][...].reshape(-1).astype(float)
                    for seconds in (2,4,8):
                        n_fft=int(seconds*fs)
                        coefficients=librosa.feature.mfcc(y=x,sr=fs,n_mfcc=8,n_mels=16,
                            n_fft=n_fft,hop_length=n_fft//2,fmin=0,fmax=fs/2,center=False)
                        frame_counts.setdefault(str(seconds),set()).add(coefficients.shape[1])
                        # Match prior feature order: each coefficient's mean, then std.
                        rows[f'{seconds}s mean_std'].extend(np.column_stack([coefficients.mean(axis=1),coefficients.std(axis=1)]).ravel())
                        med=np.median(coefficients,axis=1)
                        iqr=np.quantile(coefficients,.75,axis=1)-np.quantile(coefficients,.25,axis=1)
                        change=np.mean(np.abs(np.diff(coefficients,axis=1)),axis=1)
                        rows[f'{seconds}s robust_change'].extend(np.column_stack([med,iqr,change]).ravel())
            for key in matrices:matrices[key].append(rows[key])
    matrices={key:np.asarray(value) for key,value in matrices.items()}
    assert np.allclose(matrices['4s mean_std'],old)
    assert all(np.isfinite(X).all() for X in matrices.values())
    folds=[];allocations=[]
    for allocation,damages in enumerate(itertools.combinations([f'damage_{i}' for i in range(1,7)],3),1):
        mask=np.isin(states,['baseline_1',*damages])
        for configuration,X in matrices.items():
            oof=np.empty_like(y)
            for fold,hold in enumerate((mask,~mask),1):
                a=np.flatnonzero(~hold);b=np.flatnonzero(hold)
                assert set(states[a]).isdisjoint(states[b])
                model=RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=2,
                    class_weight='balanced',random_state=42,n_jobs=1)
                # No normalization needed for trees. Feature extraction uses each recording alone.
                model.fit(X[a],y[a]);prediction=model.predict(X[b]);oof[b]=prediction
                score=model.predict_proba(X[b])[:,1]
                train=float(balanced_accuracy_score(y[a],model.predict(X[a])))
                validation=float(balanced_accuracy_score(y[b],prediction))
                folds.append({'allocation':allocation,'fold':fold,'configuration':configuration,
                    'train_BA':train,'validation_BA':validation,'gap':train-validation,
                    'damage_AP':float(average_precision_score(y[b],score)),
                    'baseline_AP':float(average_precision_score(1-y[b],-score))})
            allocations.append({'allocation':allocation,'configuration':configuration,
                'OOF_BA':float(balanced_accuracy_score(y,oof))})
        if allocation%5==0:print(f'Completed frame allocation {allocation}/20',flush=True)
    summary=[]
    for configuration,X in matrices.items():
        rows=[r for r in folds if r['configuration']==configuration]
        ba=np.array([r['OOF_BA'] for r in allocations if r['configuration']==configuration])
        summary.append({'configuration':configuration,'features':X.shape[1],'BA_mean':float(ba.mean()),
            'BA_split_SD':float(ba.std()),**{key:float(np.mean([r[key] for r in rows])) for key in ('train_BA','gap','damage_AP','baseline_AP')}})
    reference=next(r for r in summary if r['configuration']=='4s mean_std')
    assert np.isclose(reference['BA_mean'],.6535087719298246)
    summary.sort(key=lambda r:r['BA_mean'],reverse=True)
    report={'summary':summary,'folds':folds,'allocations':allocations,
        'frames_per_recording':{key:sorted(value) for key,value in frame_counts.items()},
        'model':'Fixed shallow Random Forest; 200 trees, depth 2, min leaf 2, balanced weights',
        'definitions':'robust_change = per-coefficient median, interquartile range, mean absolute difference between consecutive overlapping frames. No window-level labels.',
        'limitations':'Post-hoc comparison on previously explored one-bridge data. Winning configuration is exploratory, not independently validated. Repeated splits are dependent. Robust summaries also have more columns than mean/std, so their comparison changes both representation and feature count.'}
    (ROOT/'outputs/mfcc_frame_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'data/processed/mfcc_frame_features.npz',record_ids=ids,y=y,
        **{key.replace(' ','_'):X for key,X in matrices.items()})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');positions=np.arange(len(summary))
    axes[0].bar(positions,[r['BA_mean'] for r in summary],yerr=[r['BA_split_SD'] for r in summary],capsize=4)
    axes[0].axhline(.5,color='gray',ls='--');axes[0].set(ylabel='Grouped OOF balanced accuracy',ylim=(0,1))
    axes[1].bar(positions-.18,[r['damage_AP'] for r in summary],.36,label='Damage AP')
    axes[1].bar(positions+.18,[r['baseline_AP'] for r in summary],.36,label='Baseline AP')
    axes[1].set(ylim=(0,1));axes[1].legend()
    for ax in axes:ax.set_xticks(positions,[r['configuration'] for r in summary],rotation=30,ha='right')
    plot=ROOT/'outputs/mfcc_frame_comparison.png';fig.savefig(plot,dpi=150);plt.close(fig)
    path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
    shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_frames_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    md=nbformat.v4.new_markdown_cell('## 23. MFCC frame duration and recording summaries\n\nCompare 2/4/8-second frames with 50% overlap. Summarize by mean/std or median/IQR/mean absolute inter-frame change. Each recording is still one sample. Fixed shallow Random Forest, same 20 state-separated allocations, no label-derived feature selection or normalization. Reproduces existing 4s mean/std features and score. Results are exploratory; the best configuration needs new data for confirmation. Different summary families have different feature counts.')
    cell=nbformat.v4.new_code_cell('import sys\nif str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))\nfrom mfcc_frame_experiment import run_frame_experiment\nframe_report=run_frame_experiment()\nshow_analysis_table(frame_report["summary"])\nprint("Frames per recording:",frame_report["frames_per_recording"])\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/"outputs/mfcc_frame_comparison.png")))')
    md['id']='mfcc-frames-00';cell['id']='mfcc-frames-01'
    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps(summary,indent=2)),
        nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(plot.read_bytes()).decode()},metadata={})]
    nb.cells=[c for c in nb.cells if not c.get('id','').startswith('mfcc-frames-')];nb.cells[-1:-1]=[md,cell]
    nbformat.validate(nb);nbformat.write(nb,path)
    print(json.dumps({'summary':summary,'frame_counts':report['frames_per_recording']},indent=2),flush=True)
    return report

if __name__=='__main__':run_frame_experiment()
