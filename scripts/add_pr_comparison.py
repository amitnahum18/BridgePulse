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
shutil.copy2(path, root / 'archive/recovery' / f'experiment_before_pr_auc_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
nb = nbformat.read(path, as_version=4)
nb.cells = [c for c in nb.cells if not c.get('id', '').startswith('pr-comparison-')]
description = '''## 17. Precision–Recall comparison on held-out recordings

Average Precision (AP) summarizes the precision–recall curve without trapezoidal interpolation. Report the mean and standard deviation across the same three outer folds, using their held-out scores. Hyperparameters were previously selected using balanced accuracy; this is an additional evaluation, not AP-specific tuning.

Damage (1) is the majority: 19/25 = 0.76. Baseline (0) is the minority: 6/25 = 0.24. Evaluate both directions by reversing the score for class 0. The primary comparison is mean fold AP; pooling scores from separately fitted models can distort rankings because score scales differ. This does not establish generalization to unseen damage states or bridges, or absence of overfitting.
'''
code = '''import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

pr_source = json.loads((ROOT / "outputs/mfcc_models_overfit_report.json").read_text())
pr_predictions = pr_source["predictions"]
pr_folds = []
pr_summary = []
pr_models = sorted({r["model"] for r in pr_predictions})
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for model_index, name in enumerate(pr_models):
    color = f"C{model_index}"
    for fold in (1, 2, 3):
        rows = [r for r in pr_predictions if r["model"] == name and r["fold"] == fold]
        y = np.asarray([r["true_label"] for r in rows])
        scores = np.asarray([r["decision_score"] for r in rows])
        assert len(np.unique(y)) == 2 and np.isfinite(scores).all()
        result = {"model": name, "fold": fold, "damage_prevalence": float(y.mean())}
        for axis, label, positive, score in zip(axes, ("damage", "baseline"), (y, 1-y), (scores, -scores)):
            precision, recall, _ = precision_recall_curve(positive, score)
            result[label + "_AP"] = float(average_precision_score(positive, score))
            result[label + "_trapezoidal_PR_AUC"] = float(auc(recall, precision))
            axis.step(recall, precision, where="post", color=color, alpha=.6,
                      label=name if fold == 1 else None)
        pr_folds.append(result)
    selected = [r for r in pr_folds if r["model"] == name]
    pr_summary.append({"model": name,
        "damage_AP_mean": float(np.mean([r["damage_AP"] for r in selected])),
        "damage_AP_SD": float(np.std([r["damage_AP"] for r in selected])),
        "baseline_AP_mean": float(np.mean([r["baseline_AP"] for r in selected])),
        "baseline_AP_SD": float(np.std([r["baseline_AP"] for r in selected])),
        "damage_trapezoidal_PR_AUC_mean": float(np.mean([r["damage_trapezoidal_PR_AUC"] for r in selected]))})
pr_summary.sort(key=lambda r: r["damage_AP_mean"], reverse=True)
show_analysis_table(pr_summary)
show_analysis_table(pr_folds)
for axis, title, prevalence in zip(axes, ("Damage positive (1)", "Baseline positive (0)"), (.76, .24)):
    axis.axhline(prevalence, color="gray", linestyle="--", label=f"Overall prevalence: {prevalence:.2f}")
    axis.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03))
    axis.legend(fontsize=8)
plt.show()
(ROOT / "outputs/mfcc_pr_auc_report.json").write_text(json.dumps({"summary": pr_summary,
    "folds": pr_folds, "metric": "Mean outer-fold Average Precision; trapezoidal PR-AUC also reported",
    "tuning_metric": "balanced_accuracy", "damage_prevalence": .76, "baseline_prevalence": .24}, indent=2), encoding="utf-8")
'''
new = [nbformat.v4.new_markdown_cell(description), nbformat.v4.new_code_cell(code)]
for i, cell in enumerate(new):
    cell['id'] = f'pr-comparison-{i:02d}'
by_id = {c['id']: c for c in nb.cells}
source = by_id['evaluation-01'].source
helper = next(ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'show_analysis_table')
execution = nbformat.v4.new_notebook(cells=[copy.deepcopy(by_id['restored-01']), nbformat.v4.new_code_cell('import html\nfrom IPython.display import HTML, display\n' + helper)] + copy.deepcopy(new), metadata=copy.deepcopy(nb.metadata))
NotebookClient(execution, timeout=180, kernel_name='bridgepulse', resources={'metadata': {'path': str(path.parent)}}).execute()
nb.cells[-1:-1] = execution.cells[-2:]
nbformat.validate(nb)
nbformat.write(nb, path)
print(json.dumps(json.loads((root / 'outputs/mfcc_pr_auc_report.json').read_text())['summary'], indent=2))
