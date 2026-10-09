import ast,copy,datetime,hashlib,json,shutil
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1];path=ROOT/'notebooks/experiment.ipynb'
nb=nbformat.read(path,as_version=4)
backup=ROOT/'archive/recovery'/f'experiment_before_mfcc_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
nb.cells=[c for c in nb.cells if not c.get('id','').startswith('mfcc-models-')]
description='''## 16. אותם 928 פיצ׳רי MFCC — השוואת מודלים ובדיקת אוברפיט
משווים Logistic L2, SVM RBF, Random Forest, Extra Trees ו־Histogram Gradient Boosting באותם שלושה פיצולים חיצוניים מהניסוי המקונן הקודם (seed=42). כל שורה היא הקלטה שלמה. לכל מודל אותם 928 פיצ׳רים מכל 58 הערוצים, ללא A14. אף חיישן או פיצ׳ר לא נבחרים לפי הוולידציה החיצונית.

בחירת הפרמטרים נעשית ב־2-fold CV בתוך כל אימון חיצוני בלבד. Pipeline לומד מילוי, הסרת פיצ׳רים קבועים ותקנון בכל אימון פנימי. האימון החיצוני והוולידציה שלו נפרדים. ההגדרות מוגבלות ומורכבות העצים מרוסנת. רגרסיה לוגיסטית משתמשת באותו טווח C מהניסוי הקודם. הוולידציה החיצונית אינה משמשת לכוונון הפרמטרים או הסף (0.5, וב־SVM סף decision_function=0).

אבחון אוברפיט: מודדים balanced accuracy באימון החיצוני ובוולידציה החיצונית, ובאימון/ולידציה פנימיים. פער גדול ועקבי או פיזור חיצוני גדול הם סימני סיכון, לא מבחן המוכיח נוכחות או היעדר אוברפיט. שישה מצבי בסיס בלבד, פיצולים קטנים ודאטה שכבר נחקר אינם מאפשרים להוכיח שאין אוברפיט או שהמודל מכליל לגשר חדש. בחירת המודל המנצח מבין כמה מודלים מוסיפה אופטימיות; עדיין נדרשות הקלטות חדשות להערכה סופית.
'''
code='''import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score,recall_score,roc_auc_score,accuracy_score,confusion_matrix

with np.load(ROOT/"data/processed/all_acceleration_features.npz") as artifact:
    model_source_X=artifact["X"];model_y=artifact["y"]
    model_names=artifact["feature_names"].tolist();model_record_ids=artifact["record_ids"]
model_mfcc_columns=np.asarray([i for i,name in enumerate(model_names) if "_mfcc_" in name])
model_X=model_source_X[:,model_mfcc_columns]
assert model_X.shape==(25,928) and len(set(model_record_ids))==25
assert np.isfinite(model_X).all()
outer_model_splits=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=42).split(model_X,model_y))
model_specs=[
    ("Logistic L2",LogisticRegression(class_weight="balanced",max_iter=5000),
     {"classifier__C":[.01,.1,1.0]}),
    ("SVM RBF",SVC(kernel="rbf",class_weight="balanced"),
     {"classifier__C":[.1,1.0,10.0],"classifier__gamma":["scale",.0001]}),
    ("Random Forest",RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=1),
     {"classifier__max_depth":[2,4],"classifier__min_samples_leaf":[2,4]}),
    ("Extra Trees",ExtraTreesClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=1),
     {"classifier__max_depth":[2,4],"classifier__min_samples_leaf":[2,4]}),
    ("Histogram Boosting",HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,early_stopping=False,class_weight="balanced",random_state=42),
     {"classifier__max_leaf_nodes":[3,7],"classifier__min_samples_leaf":[2,5],"classifier__l2_regularization":[.1,1.0]}),
]
mfcc_model_folds=[];mfcc_model_predictions=[]
for fold,(train,validation) in enumerate(outer_model_splits,start=1):
    assert set(model_record_ids[train]).isdisjoint(model_record_ids[validation])
    inner_splits=list(StratifiedKFold(n_splits=2,shuffle=True,random_state=50+fold).split(model_X[train],model_y[train]))
    for name,estimator,grid in model_specs:
        pipeline=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
            ("nonconstant",VarianceThreshold()),("scaler",StandardScaler()),("classifier",estimator)])
        search=GridSearchCV(pipeline,grid,scoring="balanced_accuracy",cv=inner_splits,
                           return_train_score=True,n_jobs=1,error_score="raise")
        search.fit(model_X[train],model_y[train]);model=search.best_estimator_
        training_pred=model.predict(model_X[train]);validation_pred=model.predict(model_X[validation])
        validation_score=model.decision_function(model_X[validation]) if hasattr(model,"decision_function") else model.predict_proba(model_X[validation])[:,1]
        train_ba=float(balanced_accuracy_score(model_y[train],training_pred))
        outer_ba=float(balanced_accuracy_score(model_y[validation],validation_pred))
        best_index=search.best_index_
        inner_train=float(search.cv_results_["mean_train_score"][best_index])
        inner_validation=float(search.cv_results_["mean_test_score"][best_index])
        mfcc_model_folds.append({"model":name,"fold":fold,"parameters":search.best_params_,
            "train_balanced_accuracy":train_ba,"outer_balanced_accuracy":outer_ba,
            "train_outer_gap":train_ba-outer_ba,"inner_train_balanced_accuracy":inner_train,
            "inner_validation_balanced_accuracy":inner_validation,"inner_gap":inner_train-inner_validation,
            "baseline_recall":float(recall_score(model_y[validation],validation_pred,pos_label=0,zero_division=0)),
            "damage_recall":float(recall_score(model_y[validation],validation_pred,pos_label=1,zero_division=0)),
            "outer_roc_auc":float(roc_auc_score(model_y[validation],validation_score))})
        for i,p,score in zip(validation,validation_pred,validation_score):
            mfcc_model_predictions.append({"model":name,"fold":fold,"recording":str(model_record_ids[i]),
                "original_state":str(model_record_ids[i]).split('/')[0],"true_label":int(model_y[i]),
                "prediction":int(p),"decision_score":float(score)})
    print("Completed model comparison, outer fold",fold)

mfcc_model_summary=[]
for name,_,_ in model_specs:
    rows=[r for r in mfcc_model_folds if r["model"]==name]
    values=np.asarray([r["outer_balanced_accuracy"] for r in rows])
    predictions=[r for r in mfcc_model_predictions if r["model"]==name]
    assert len(predictions)==25 and len({r["recording"] for r in predictions})==25
    mfcc_model_summary.append({"model":name,
        "train_balanced_accuracy_mean":float(np.mean([r["train_balanced_accuracy"] for r in rows])),
        "outer_balanced_accuracy_mean":float(values.mean()),"outer_balanced_accuracy_std":float(values.std()),
        "train_outer_gap_mean":float(np.mean([r["train_outer_gap"] for r in rows])),
        "inner_train_mean":float(np.mean([r["inner_train_balanced_accuracy"] for r in rows])),
        "inner_validation_mean":float(np.mean([r["inner_validation_balanced_accuracy"] for r in rows])),
        "baseline_recall_mean":float(np.mean([r["baseline_recall"] for r in rows])),
        "damage_recall_mean":float(np.mean([r["damage_recall"] for r in rows]))})
mfcc_model_summary.sort(key=lambda r:r["outer_balanced_accuracy_mean"],reverse=True)
show_analysis_table(mfcc_model_summary)
show_analysis_table(mfcc_model_folds)

fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
positions=np.arange(len(mfcc_model_summary));width=.35
axes[0].bar(positions-width/2,[r["train_balanced_accuracy_mean"] for r in mfcc_model_summary],width,label="Outer training")
axes[0].bar(positions+width/2,[r["outer_balanced_accuracy_mean"] for r in mfcc_model_summary],width,
            yerr=[r["outer_balanced_accuracy_std"] for r in mfcc_model_summary],capsize=4,label="Outer validation")
axes[0].set_ylabel("Balanced accuracy | mean and SD");axes[0].set_ylim(0,1.05)
axes[0].axhline(.5,color="gray",ls="--");axes[0].legend()
axes[1].bar(positions,[r["train_outer_gap_mean"] for r in mfcc_model_summary],color="tab:orange")
axes[1].axhline(0,color="gray");axes[1].set_ylabel("Training minus outer validation | diagnostic gap")
for ax in axes:ax.set_xticks(positions,[r["model"] for r in mfcc_model_summary],rotation=25,ha="right")
plt.show()

previous_nested=json.loads((ROOT/"outputs/diagnostics_nested_report.json").read_text())
expected=next(r for r in previous_nested["nested_summary"] if r["strategy"]=="All MFCC (928)")["outer_balanced_accuracy_mean"]
actual=next(r for r in mfcc_model_summary if r["model"]=="Logistic L2")["outer_balanced_accuracy_mean"]
assert np.isclose(actual,expected),"Reference model no longer matches previous splits"
with (ROOT/"outputs/mfcc_models_overfit_report.json").open("w",encoding="utf-8") as stream:
    json.dump({"features":928,"recordings":25,"summary":mfcc_model_summary,"folds":mfcc_model_folds,
        "predictions":mfcc_model_predictions,"warning":"Cannot establish absence of overfitting with 25 previously inspected recordings; no external test"},stream,indent=2)
'''
new=[nbformat.v4.new_markdown_cell(description),nbformat.v4.new_code_cell(code)]
for i,c in enumerate(new):c['id']=f'mfcc-models-{i:02d}'
nb.cells[-1:-1]=new;by_id={c['id']:c for c in nb.cells}
source=by_id['evaluation-01'].source
helper=next(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='show_analysis_table')
prerequisites=[copy.deepcopy(by_id['restored-01']),nbformat.v4.new_code_cell('import html\nfrom IPython.display import HTML,display\n'+helper)]
execution=nbformat.v4.new_notebook(cells=prerequisites+[copy.deepcopy(c) for c in new],metadata=copy.deepcopy(nb.metadata))
NotebookClient(execution,timeout=300,kernel_name='bridgepulse',resources={'metadata':{'path':str(path.parent)}}).execute()
executed={c['id']:c for c in execution.cells if c.get('id','').startswith('mfcc-models-')}
for i,c in enumerate(nb.cells):
 if c.get('id') in executed:nb.cells[i]=executed[c['id']]
nbformat.validate(nb);nbformat.write(nb,path)
with (ROOT/'data/raw/Rt345Bridge.h5').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==(ROOT/'data/raw/Rt345Bridge.h5.sha256').read_text().split()[0]
print('Saved nested MFCC model comparison and overfitting diagnostics; raw H5 unchanged.')
