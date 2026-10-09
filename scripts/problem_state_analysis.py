import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import nbformat
from sklearn.metrics import roc_auc_score

root=Path(__file__).resolve().parents[1]
with np.load(root/'data/processed/all_acceleration_features.npz') as source:
    X=source['X'];names=source['feature_names'].astype(str);ids=source['record_ids'].astype(str);y=source['y']
states=np.array([r.split('/')[0] for r in ids])
target=states=='damage_1';base=y==0;other=(y==1)&~target
summary=[]
# Fixed summaries across channels, not 58 independent observations.
for feature in ['rms','kurtosis','dominant_frequency','spectral_centroid','fraction_4_8','fraction_8_14','fraction_25_35','fraction_55_64']:
    columns=[i for i,n in enumerate(names) if n.split('_',2)[2]==feature]
    if not columns:continue
    values=np.median(X[:,columns],axis=1)
    summary.append({'feature':feature,'damage_1_recording_values':values[target].tolist(),
        'baseline_median':float(np.median(values[base])),
        'other_damage_median':float(np.median(values[other]))})
ranked=[]
for i,name in enumerate(names):
    values=X[:,i];sd=float(np.std(values[~target]))
    if sd<1e-12:continue
    effect=float((np.median(values[target])-np.median(values[~target]))/sd)
    a=roc_auc_score(target.astype(int),values)
    ranked.append({'feature':str(name),'absolute_descriptive_AUC':float(max(a,1-a)),
        'standardized_median_difference':effect,
        'damage_1_values':values[target].tolist(),'baseline_median':float(np.median(values[base])),
        'other_damage_median':float(np.median(values[other]))})
ranked.sort(key=lambda r:(r['absolute_descriptive_AUC'],abs(r['standardized_median_difference'])),reverse=True)
pred=json.loads((root/'outputs/grouped_mfcc_models_report.json').read_text())['predictions']
errors=[]
for model in ['Random Forest','Logistic L2']:
    for state in sorted(set(states)):
        rows=[r for r in pred if r['model']==model and r['recording'].split('/')[0]==state]
        errors.append({'model':model,'state':str(state),'correct_fraction_across_splits':float(np.mean([r['true_label']==r['prediction'] for r in rows]))})
report={'state_accuracy':errors,'damage_1_aggregate_channel_summaries':summary,'exploratory_top_features':ranked[:20],
    'limitations':'Three damage_1 recordings; rankings selected among 2030 features after observing all data. AUC is descriptive, not validated predictive performance or significance. Repeated split predictions are not independent observations.'}
(root/'outputs/problem_state_features_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'summaries':summary,'top_features':ranked[:6]},indent=2))
path=root/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
shutil.copy2(path,root/'archive/recovery'/f'experiment_before_problem_state_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
md=nbformat.v4.new_markdown_cell('## 21. Problematic experimental states: descriptive feature analysis\n\nDamage_1 and baseline_2 have the weakest held-out identification. Report per-state accuracy across repeated state-separated splits, fixed physical summaries (median across channels per recording), and exploratory damage_1-versus-rest feature rankings. Only three damage_1 recordings exist. Rankings across 2030 features are post-hoc and can overfit; descriptive AUC is not a test score or statistical significance. No model is changed using these findings.')
code=nbformat.v4.new_code_cell('import json\nproblem_state_report=json.loads((ROOT/"outputs/problem_state_features_report.json").read_text())\nshow_analysis_table(problem_state_report["state_accuracy"])\nshow_analysis_table(problem_state_report["damage_1_aggregate_channel_summaries"])\nshow_analysis_table(problem_state_report["exploratory_top_features"][:10])')
md['id']='problem-state-00';code['id']='problem-state-01'
code.outputs=[nbformat.v4.new_output('stream',name='stdout',text=json.dumps(report,indent=2))]
nb.cells[-1:-1]=[md,code];nbformat.validate(nb);nbformat.write(nb,path)
