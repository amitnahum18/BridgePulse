import ast
import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/experiment.ipynb'
original = nbformat.read(path, as_version=4)
shutil.copy2(path, root / 'archive/recovery' / f'experiment_before_grouped_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
nb = copy.deepcopy(original)
nb.cells = [c for c in nb.cells if not c.get('id', '').startswith('grouped-models-')]
description = '''## 18. Corrected evaluation: hold out entire experimental states

This stage supersedes stages 16–17 for assessing unseen experimental states. All recordings of a state stay together. There are only two baseline states: two outer folds are feasible with both labels in training and validation; nested grouped tuning with both labels in every inner partition is not feasible after holding out one baseline state. Use explicit conservative fixed parameters, not GridSearchCV. These parameters and the MFCC family were informed by earlier exploratory work, so these results are still exploratory, not an untouched final test.

Evaluate all 20 complementary allocations of the six damage states (three per side), anchoring baseline_1 on one side and baseline_2 on the other. Each allocation has two folds; reuse the same allocations for all models. Repetition does not create independent samples or confidence intervals. Per-allocation out-of-fold metrics cover all 25 recordings; their spread measures split sensitivity. AP is additionally computed within each fold to avoid pooling scores with different scales. Include minority-baseline AP. Each preprocessing pipeline is fitted only on its training states. There is one bridge, so this cannot validate prediction on a new bridge. Original states may also share experimental history; state grouping does not establish full physical independence.
'''
code = '''import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, average_precision_score, recall_score

with np.load(ROOT / "data/processed/all_acceleration_features.npz") as artifact:
    grouped_names = artifact["feature_names"].astype(str)
    grouped_X = artifact["X"][:, [i for i,n in enumerate(grouped_names) if "_mfcc_" in n]]
    grouped_y = artifact["y"]
    grouped_ids = artifact["record_ids"].astype(str)
grouped_states = np.asarray([r.split('/')[0] for r in grouped_ids])
assert grouped_X.shape == (25,928) and len(set(grouped_ids)) == 25
assert np.isfinite(grouped_X).all()
grouped_specs = {
    "Logistic L2": LogisticRegression(C=.01, class_weight="balanced", max_iter=5000),
    "SVM RBF": SVC(C=.1, gamma="scale", class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200,max_depth=2,min_samples_leaf=4,class_weight="balanced",random_state=42,n_jobs=1),
    "Histogram Boosting": HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=3,min_samples_leaf=5,l2_regularization=1.,class_weight="balanced",early_stopping=False,random_state=42),
}
damage_states = [f"damage_{i}" for i in range(1,7)]
grouped_folds, grouped_allocations, grouped_predictions = [], [], []
for allocation, damages in enumerate(itertools.combinations(damage_states,3),1):
    side_a = {"baseline_1", *damages}
    side_b = set(grouped_states) - side_a
    for name, estimator in grouped_specs.items():
        oof = np.empty_like(grouped_y)
        for fold, validation_states in enumerate((side_a,side_b),1):
            validation = np.flatnonzero(np.isin(grouped_states,list(validation_states)))
            train = np.flatnonzero(~np.isin(grouped_states,list(validation_states)))
            assert set(grouped_states[train]).isdisjoint(grouped_states[validation])
            assert set(grouped_ids[train]).isdisjoint(grouped_ids[validation])
            assert len(np.unique(grouped_y[train])) == len(np.unique(grouped_y[validation])) == 2
            pipeline = Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
                ("nonconstant",VarianceThreshold()),("scaler",StandardScaler()),("classifier",clone(estimator))])
            pipeline.fit(grouped_X[train],grouped_y[train])
            training_selected = pipeline["nonconstant"].transform(pipeline["imputer"].transform(grouped_X[train]))
            assert np.allclose(pipeline["scaler"].mean_,training_selected.mean(axis=0))
            assert int(pipeline["scaler"].n_samples_seen_) == len(train)
            predicted = pipeline.predict(grouped_X[validation]); oof[validation] = predicted
            scores = pipeline.decision_function(grouped_X[validation]) if hasattr(pipeline,"decision_function") else pipeline.predict_proba(grouped_X[validation])[:,1]
            train_ba = float(balanced_accuracy_score(grouped_y[train],pipeline.predict(grouped_X[train])))
            val_ba = float(balanced_accuracy_score(grouped_y[validation],predicted))
            grouped_folds.append({"allocation":allocation,"fold":fold,"model":name,
                "train_states":sorted(set(grouped_states[train])),"validation_states":sorted(validation_states),
                "train_BA":train_ba,"validation_BA":val_ba,"gap":train_ba-val_ba,
                "damage_AP":float(average_precision_score(grouped_y[validation],scores)),
                "baseline_AP":float(average_precision_score(1-grouped_y[validation],-scores)),
                "damage_prevalence":float(grouped_y[validation].mean())})
            for i,p,s in zip(validation,predicted,scores):
                grouped_predictions.append({"allocation":allocation,"fold":fold,"model":name,
                    "recording":str(grouped_ids[i]),"true_label":int(grouped_y[i]),"prediction":int(p),"score":float(s)})
        grouped_allocations.append({"allocation":allocation,"model":name,
            "OOF_BA":float(balanced_accuracy_score(grouped_y,oof)),
            "baseline_recall":float(recall_score(grouped_y,oof,pos_label=0)),
            "damage_recall":float(recall_score(grouped_y,oof,pos_label=1))})
    if allocation % 5 == 0: print("Completed grouped allocations",allocation,"/20")
grouped_summary=[]
for name in grouped_specs:
    folds=[r for r in grouped_folds if r["model"]==name]
    allocations=[r for r in grouped_allocations if r["model"]==name]
    ba=np.asarray([r["OOF_BA"] for r in allocations])
    grouped_summary.append({"model":name,"OOF_BA_mean":float(ba.mean()),"OOF_BA_split_SD":float(ba.std()),
        "OOF_BA_min":float(ba.min()),"OOF_BA_max":float(ba.max()),
        "train_BA_mean":float(np.mean([r["train_BA"] for r in folds])),
        "train_validation_gap_mean":float(np.mean([r["gap"] for r in folds])),
        "damage_AP_mean":float(np.mean([r["damage_AP"] for r in folds])),
        "baseline_AP_mean":float(np.mean([r["baseline_AP"] for r in folds])),
        "baseline_recall_mean":float(np.mean([r["baseline_recall"] for r in allocations])),
        "damage_recall_mean":float(np.mean([r["damage_recall"] for r in allocations]))})
grouped_summary.sort(key=lambda r:r["OOF_BA_mean"],reverse=True)
show_analysis_table(grouped_summary)
fig, axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
positions=np.arange(len(grouped_summary)); labels=[r["model"] for r in grouped_summary]
axes[0].bar(positions,[r["OOF_BA_mean"] for r in grouped_summary],yerr=[r["OOF_BA_split_SD"] for r in grouped_summary],capsize=4)
axes[0].axhline(.5,color="gray",ls="--");axes[0].set_title("State-held-out balanced accuracy | split sensitivity")
axes[1].bar(positions-.18,[r["damage_AP_mean"] for r in grouped_summary],.36,label="Damage AP")
axes[1].bar(positions+.18,[r["baseline_AP_mean"] for r in grouped_summary],.36,label="Baseline AP")
axes[1].axhline(.76,color="C0",ls="--",alpha=.5);axes[1].axhline(.24,color="C1",ls="--",alpha=.5)
axes[1].set_title("Mean fold AP | dashed lines: overall prevalence");axes[1].legend()
for ax in axes: ax.set_xticks(positions,labels,rotation=25,ha="right");ax.set_ylim(0,1.05)
plt.show()
(ROOT/"outputs/grouped_mfcc_models_report.json").write_text(json.dumps({"summary":grouped_summary,
    "folds":grouped_folds,"allocations":grouped_allocations,"predictions":grouped_predictions,
    "fixed_parameters":{name:est.get_params() for name,est in grouped_specs.items()},
    "normalization_train_only_checks_passed":True,"group_overlap_checks_passed":True,
    "limitations":"Exploratory, fixed parameters; 8 states, 2 baseline states, 1 bridge. Repeated splits are dependent. No independent final test."},indent=2),encoding="utf-8")
'''
new = [nbformat.v4.new_markdown_cell(description), nbformat.v4.new_code_cell(code)]
for i,c in enumerate(new): c['id'] = f'grouped-models-{i:02d}'
by_id = {c['id']: c for c in nb.cells}
source = by_id['evaluation-01'].source
helper = next(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='show_analysis_table')
execution = nbformat.v4.new_notebook(cells=[copy.deepcopy(by_id['restored-01']),nbformat.v4.new_code_cell('import html\nfrom IPython.display import HTML,display\n'+helper)]+copy.deepcopy(new),metadata=copy.deepcopy(nb.metadata))
NotebookClient(execution,timeout=600,kernel_name='bridgepulse',resources={'metadata':{'path':str(path.parent)}}).execute()
nb.cells[-1:-1] = execution.cells[-2:]
nbformat.validate(nb)
assert all(next(c for c in nb.cells if c['id']==old['id'])==old for old in original.cells if not old['id'].startswith('grouped-models-'))
nbformat.write(nb,path)
print(json.dumps(json.loads((root/'outputs/grouped_mfcc_models_report.json').read_text())['summary'],indent=2))
