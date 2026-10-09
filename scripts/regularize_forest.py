import base64
import itertools
import json
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]

def run_regularization():
    with np.load(ROOT/'data/processed/mfcc_frame_features.npz') as data:
        X=data['4s_robust_change'];y=data['y'];ids=data['record_ids'].astype(str)
    with np.load(ROOT/'data/processed/all_acceleration_features.npz') as source:
        sensors=sorted({n.split('_')[0] for n in source['feature_names'].astype(str)})
    names=[f'{sensor}_{axis}_mfcc_{i}_{stat}' for sensor in sensors for axis in ('y','z') for i in range(8) for stat in ('median','iqr','change')]
    lookup={name:i for i,name in enumerate(names)}
    previous=json.loads((ROOT/'outputs/robust_shap_selection_report.json').read_text())
    ranking={(r['allocation'],r['fold']):r['top25'] for r in previous['rankings']}
    states=np.array([r.split('/')[0] for r in ids]);folds=[];allocations=[]
    configurations=[('reference',2,2,'sqrt'),('depth1_leaf2',1,2,'sqrt'),
                    ('depth2_leaf3',2,3,'sqrt'),('depth2_leaf4',2,4,'sqrt'),
                    ('depth1_leaf3',1,3,'sqrt'),('depth2_leaf3_fraction',2,3,.2)]
    for allocation,damages in enumerate(itertools.combinations([f'damage_{i}' for i in range(1,7)],3),1):
        mask=np.isin(states,['baseline_1',*damages])
        for configuration,depth,leaf,max_features in configurations:
            for count in (25,1392):
                oof=np.empty_like(y)
                for fold,hold in enumerate((mask,~mask),1):
                    a=np.flatnonzero(~hold);b=np.flatnonzero(hold)
                    assert set(states[a]).isdisjoint(states[b])
                    # Reuse rankings from precisely the same training-only split, never held-out rows.
                    columns=[lookup[n] for n in ranking[(allocation,fold)]] if count==25 else np.arange(1392)
                    model=RandomForestClassifier(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,
                        max_features=max_features,class_weight='balanced',random_state=42,n_jobs=1)
                    model.fit(X[a][:,columns],y[a]);pred=model.predict(X[b][:,columns]);oof[b]=pred
                    scores=model.predict_proba(X[b][:,columns])[:,1]
                    train=float(balanced_accuracy_score(y[a],model.predict(X[a][:,columns])))
                    val=float(balanced_accuracy_score(y[b],pred))
                    folds.append({'allocation':allocation,'fold':fold,'configuration':configuration,'features':count,
                        'train_BA':train,'validation_BA':val,'gap':train-val,
                        'damage_AP':float(average_precision_score(y[b],scores)),
                        'baseline_AP':float(average_precision_score(1-y[b],-scores))})
                allocations.append({'allocation':allocation,'configuration':configuration,'features':count,
                    'OOF_BA':float(balanced_accuracy_score(y,oof))})
        if allocation%5==0:print(f'Completed regularization allocation {allocation}/20',flush=True)
    summary=[]
    for configuration,depth,leaf,max_features in configurations:
        for count in (25,1392):
            selected=[r for r in folds if r['configuration']==configuration and r['features']==count]
            ba=np.array([r['OOF_BA'] for r in allocations if r['configuration']==configuration and r['features']==count])
            summary.append({'configuration':configuration,'features':count,'max_depth':depth,'min_samples_leaf':leaf,
                'max_features':max_features,'OOF_BA':float(ba.mean()),'split_SD':float(ba.std()),
                **{k:float(np.mean([r[k] for r in selected])) for k in ('train_BA','gap','damage_AP','baseline_AP')}})
    for count in (25,1392):
        reference=next(r for r in summary if r['configuration']=='reference' and r['features']==count)
        expected=next(r['BA_mean'] for r in previous['summary'] if r['features']==count)
        assert np.isclose(reference['OOF_BA'],expected)
    summary.sort(key=lambda r:r['OOF_BA'],reverse=True)
    result={'summary':summary,'folds':folds,'allocations':allocations,
        'limitations':'Exploratory comparison after multiple rounds on same eight experimental states. Selecting winning parameters uses validation results and is not unbiased final performance. Split SD is not confidence interval. Rankings reused only for identical training-only splits. Smaller gap alone can reflect underfitting.'}
    (ROOT/'outputs/forest_regularization_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained');pos=np.arange(len(summary))
    axes[0].bar(pos-.18,[r['train_BA'] for r in summary],.36,label='Training')
    axes[0].bar(pos+.18,[r['OOF_BA'] for r in summary],.36,yerr=[r['split_SD'] for r in summary],capsize=3,label='Grouped validation')
    axes[0].set_ylim(0,1.05);axes[0].axhline(.5,color='gray',ls='--');axes[0].legend()
    axes[1].bar(pos,[r['gap'] for r in summary]);axes[1].set_ylabel('Mean fold training minus validation')
    for ax in axes:ax.set_xticks(pos,[f"{r['configuration']}\n{r['features']} features" for r in summary],rotation=65,ha='right')
    plot=ROOT/'outputs/forest_regularization.png';fig.savefig(plot,dpi=150);plt.close(fig)
    path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
    shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_regularization_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    md=nbformat.v4.new_markdown_cell('## 26. Stronger forest regularization: training versus grouped validation\n\nCompare six explicit tree settings with all 1392 features or training-only SHAP top 25. Reuse exactly the same state splits and rankings; reproduce both reference scores. Restrict depth, increase minimum leaf samples and vary features considered per split. Report training, grouped validation, AP and gap together: a small gap is not useful if validation collapses. This is exploratory parameter comparison, not tuning against a final test. More trials increase selection optimism.')
    cell=nbformat.v4.new_code_cell('import sys\nif str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))\nfrom regularize_forest import run_regularization\nregularization_report=run_regularization()\nshow_analysis_table(regularization_report["summary"])\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/"outputs/forest_regularization.png")))')
    md['id']='regularization-00';cell['id']='regularization-01'
    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps(summary,indent=2)),nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(plot.read_bytes()).decode()},metadata={})]
    nb.cells[-1:-1]=[md,cell];nbformat.validate(nb);nbformat.write(nb,path)
    print(json.dumps(summary,indent=2),flush=True)
    return result

if __name__=='__main__':run_regularization()
