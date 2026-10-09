import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import nbformat
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,accuracy_score,average_precision_score,confusion_matrix,recall_score
from bridge_model import FeatureContract

ROOT=Path(__file__).resolve().parents[1]
with np.load(ROOT/'data/processed/fixed89_mfcc_features.npz') as data:
    X=pd.DataFrame(data['X'],columns=data['feature_names']);y=data['y'];ids=data['record_ids'].astype(str)
states=np.array([r.split('/')[0] for r in ids])
# Fixed, label-independent random selection among damage states. No retries or tuning.
rng=np.random.default_rng(2026)
hold_states={'baseline_2',*rng.choice([f'damage_{i}' for i in range(1,7)],3,replace=False).tolist()}
hold=np.flatnonzero(np.isin(states,list(hold_states)));train=np.flatnonzero(~np.isin(states,list(hold_states)))
assert set(states[train]).isdisjoint(states[hold])
model=Pipeline([('contract',FeatureContract()),('classifier',RandomForestClassifier(n_estimators=200,max_depth=2,
    min_samples_leaf=4,class_weight='balanced',random_state=42,n_jobs=1))])
model.fit(X.iloc[train],y[train])
metrics={}
predictions=[]
for split,indices in [('train',train),('holdout',hold)]:
    pred=model.predict(X.iloc[indices]);score=model.predict_proba(X.iloc[indices])[:,1]
    metrics[split]={'records':len(indices),'baseline_count':int(sum(y[indices]==0)),
        'damage_count':int(sum(y[indices]==1)),'balanced_accuracy':float(balanced_accuracy_score(y[indices],pred)),
        'accuracy':float(accuracy_score(y[indices],pred)),
        'damage_AP':float(average_precision_score(y[indices],score)),
        'baseline_AP':float(average_precision_score(1-y[indices],-score)),
        'baseline_recall':float(recall_score(y[indices],pred,pos_label=0)),
        'damage_recall':float(recall_score(y[indices],pred,pos_label=1)),
        'confusion_matrix_true_rows_pred_columns_0_1':confusion_matrix(y[indices],pred,labels=[0,1]).tolist()}
    for i,p,s in zip(indices,pred,score):predictions.append({'split':split,'recording':str(ids[i]),
        'true_label':int(y[i]),'prediction':int(p),'damage_probability':float(s)})
report={'train_states':sorted(set(states[train])),'holdout_states':sorted(hold_states),'split_seed':2026,
    'parameters_frozen':True,'holdout_used_for_fitting':False,'metrics':metrics,'predictions':predictions,
    'limitation':'All records were previously explored and used for feature/parameter choices; this is retrospective grouped holdout, not an independent final test. No retries or threshold tuning performed.'}
(ROOT/'outputs/group_holdout_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
joblib.dump(model,ROOT/'models/bridge_damage_rf_group_holdout.joblib')
import mlflow,mlflow.sklearn
from mlflow.models import infer_signature
registration=json.loads((ROOT/'outputs/mlflow_registration.json').read_text())
mlflow.set_tracking_uri(registration['tracking_uri'])
with mlflow.start_run(experiment_id=registration['experiment_id'],run_name='retrospective-group-holdout-seed2026') as run:
    mlflow.set_tags({'evaluation':'retrospective_group_holdout','independent_test':'false','post_hoc_selection_bias':'present'})
    mlflow.log_params({'split_seed':2026,'train_states':','.join(report['train_states']),
        'holdout_states':','.join(report['holdout_states']),'n_estimators':200,'max_depth':2,'min_samples_leaf':4,'threshold':.5,'duration_seconds':89})
    for split,values in metrics.items():mlflow.log_metrics({split+'_'+k:v for k,v in values.items() if isinstance(v,(int,float))})
    mlflow.log_artifact(str(ROOT/'outputs/group_holdout_report.json'))
    info=mlflow.sklearn.log_model(model,name='train_only_holdout_model',serialization_format='cloudpickle',
        code_paths=[str(ROOT/'scripts/bridge_model.py')],signature=infer_signature(X.iloc[train[:2]],model.predict(X.iloc[train[:2]])),input_example=X.iloc[train[:1]])
    loaded=mlflow.sklearn.load_model(info.model_uri)
    assert np.allclose(model.predict_proba(X.iloc[hold]),loaded.predict_proba(X.iloc[hold]))
    report['mlflow_run_id']=run.info.run_id
(ROOT/'outputs/group_holdout_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_holdout_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
md=nbformat.v4.new_markdown_cell('## 28. Retrospective grouped holdout, frozen model settings\n\nSeed 2026 assigns three damage states and baseline_2 to holdout. Train a fresh model on remaining states, never use the existing all-record candidate for this evaluation. Fixed 89s features, RF depth 2/min leaf 4/200 trees, threshold .5. No tuning or retries. All records were previously explored; this is not a new independent final test. Run and predictions logged separately in MLflow.')
c=nbformat.v4.new_code_cell('import json\nholdout_report=json.loads((ROOT/"outputs/group_holdout_report.json").read_text())\nprint("Train:",holdout_report["train_states"],"Holdout:",holdout_report["holdout_states"])\nshow_analysis_table([{"split":name,**values} for name,values in holdout_report["metrics"].items()])\nshow_analysis_table([r for r in holdout_report["predictions"] if r["split"]=="holdout"])')
md['id']='holdout-00';c['id']='holdout-01';c.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps(report,indent=2))]
nb.cells[-1:-1]=[md,c];nbformat.validate(nb);nbformat.write(nb,path)
print(json.dumps(report,indent=2))
