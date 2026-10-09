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
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]

def run_shap_experiment():
    with np.load(ROOT/'data/processed/all_acceleration_features.npz') as data:
        names=data['feature_names'].astype(str); columns=np.flatnonzero(['_mfcc_' in n for n in names])
        X=data['X'][:,columns];names=names[columns];y=data['y'];ids=data['record_ids'].astype(str)
    states=np.array([r.split('/')[0] for r in ids])
    settings=[10,25,50,100,928];folds=[];allocations=[];rankings=[]
    frequencies={k:np.zeros(928,dtype=int) for k in settings[:-1]}
    def forest():
        return RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=1)
    for allocation,damages in enumerate(itertools.combinations([f'damage_{i}' for i in range(1,7)],3),1):
        mask=np.isin(states,['baseline_1',*damages]);oof={k:np.empty_like(y) for k in settings}
        for fold,hold in enumerate((mask,~mask),1):
            a=np.flatnonzero(~hold);b=np.flatnonzero(hold)
            assert set(states[a]).isdisjoint(states[b])
            # Trees need no scaling; all fitting, SHAP background and ranking use training rows only.
            full=forest().fit(X[a],y[a])
            explainer=shap.TreeExplainer(full,feature_perturbation='tree_path_dependent')
            values=explainer.shap_values(X[a],check_additivity=True)
            values=values[:,:,1] if isinstance(values,np.ndarray) and values.ndim==3 else values[1] if isinstance(values,list) else values
            assert values.shape==(len(a),928)
            importance=np.mean(np.abs(values),axis=0)
            order=np.argsort(-importance,kind='stable')
            nonzero=int(np.count_nonzero(importance>0))
            rankings.append({'allocation':allocation,'fold':fold,'nonzero_SHAP_features':nonzero,
                'top25':[str(n) for n in names[order[:25]]]})
            for k in settings:
                chosen=order[:k] if k<928 else np.arange(928)
                if k<928:frequencies[k][chosen]+=1
                fitted=full if k==928 else forest().fit(X[a][:,chosen],y[a])
                predicted=fitted.predict(X[b][:,chosen]);oof[k][b]=predicted
                scores=fitted.predict_proba(X[b][:,chosen])[:,1]
                train_ba=float(balanced_accuracy_score(y[a],fitted.predict(X[a][:,chosen])))
                val_ba=float(balanced_accuracy_score(y[b],predicted))
                folds.append({'allocation':allocation,'fold':fold,'features':k,'train_BA':train_ba,
                    'validation_BA':val_ba,'gap':train_ba-val_ba,'nonzero_ranking_features':nonzero,
                    'damage_AP':float(average_precision_score(y[b],scores)),
                    'baseline_AP':float(average_precision_score(1-y[b],-scores))})
        for k in settings:
            allocations.append({'allocation':allocation,'features':k,'OOF_BA':float(balanced_accuracy_score(y,oof[k]))})
        if allocation%5==0:print(f'Completed SHAP allocation {allocation}/20',flush=True)
    summary=[]
    baseline=np.array([r['OOF_BA'] for r in allocations if r['features']==928])
    for k in settings:
        selected=[r for r in folds if r['features']==k]
        ba=np.array([r['OOF_BA'] for r in allocations if r['features']==k])
        summary.append({'features':k,'BA_mean':float(ba.mean()),'BA_split_SD':float(ba.std()),
            'paired_BA_change_vs_all':float(np.mean(ba-baseline)),
            **{key:float(np.mean([r[key] for r in selected])) for key in ('train_BA','gap','damage_AP','baseline_AP')}})
    assert np.isclose(baseline.mean(),.6535087719298246)
    stability=[{'feature':str(names[i]),'selected_top25_folds':int(frequencies[25][i]),'fraction':float(frequencies[25][i]/40)}
               for i in np.argsort(-frequencies[25],kind='stable')[:30]]
    report={'summary':summary,'folds':folds,'allocations':allocations,'rankings':rankings,'top25_selection_frequency':stability,
        'shap_version':shap.__version__,'training_only_selection':True,
        'limitations':'Fixed candidate feature counts, no nested selection of winning count; best result remains exploratory. 40 folds are dependent. SHAP explains fitted model, not causal sensor importance. Correlated features can share importance.'}
    (ROOT/'outputs/shap_selection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    p=np.arange(len(summary))
    axes[0].bar(p,[r['BA_mean'] for r in summary],yerr=[r['BA_split_SD'] for r in summary],capsize=4)
    axes[0].axhline(.5,color='gray',ls='--');axes[0].set_xticks(p,[str(r['features']) for r in summary]);axes[0].set(xlabel='Number of features',ylabel='State-held-out balanced accuracy',ylim=(0,1))
    axes[1].barh([r['feature'] for r in stability[:15]][::-1],[r['fraction'] for r in stability[:15]][::-1])
    axes[1].set(xlabel='Fraction of training fits selecting feature in top 25',xlim=(0,1))
    fig.savefig(ROOT/'outputs/shap_selection.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    print('Top stable features:',json.dumps(stability[:8],indent=2),flush=True)
    return report

if __name__=='__main__':
    report=run_shap_experiment()
    path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
    shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_shap_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    markdown=nbformat.v4.new_markdown_cell('## 20. Training-only SHAP feature selection\n\nShallow Random Forest on MFCC. For each state-separated training set, rank by mean absolute damage-class TreeSHAP on training rows, refit with 10/25/50/100 features and evaluate on held-out states. Compare with all 928 features. Trees require no normalization. Fixed counts are experimental comparisons, not an independently validated choice of best count. SHAP is model attribution, not causality; correlated features may share credit. Forty folds reuse eight states, so split variability is not a confidence interval. See `scripts/shap_feature_selection.py` for implementation.')
    code=nbformat.v4.new_code_cell('import sys\nif str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))\nfrom shap_feature_selection import run_shap_experiment\nshap_selection_report=run_shap_experiment()\nshow_analysis_table(shap_selection_report["summary"])\nshow_analysis_table(shap_selection_report["top25_selection_frequency"][:15])\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/"outputs/shap_selection.png")))')
    markdown['id']='shap-selection-00';code['id']='shap-selection-01'
    code.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps(report['summary'],indent=2))]
    nb.cells[-1:-1]=[markdown,code];nbformat.validate(nb);nbformat.write(nb,path)
