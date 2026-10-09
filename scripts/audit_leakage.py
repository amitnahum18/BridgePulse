import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parents[1]
with np.load(root / 'data/processed/all_acceleration_features.npz') as data:
    names = data['feature_names'].astype(str)
    columns = [i for i, name in enumerate(names) if '_mfcc_' in name]
    X, y, ids = data['X'][:, columns], data['y'], data['record_ids'].astype(str)
assert X.shape == (25, 928) and np.isfinite(X).all()
assert len(set(ids)) == len(ids)
assert not any(name.startswith('A14_') for name in names[columns])
assert all(name.split('_')[2] == 'mfcc' for name in names[columns])
assert np.array_equal(y, np.array([0 if r.startswith('baseline_') else 1 for r in ids]))
duplicates = [(ids[i], ids[j]) for i in range(25) for j in range(i) if np.array_equal(X[i], X[j])]
assert not duplicates
states = np.array([r.split('/')[0] for r in ids])
checks = []
for fold, (train, validation) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y), 1):
    assert set(ids[train]).isdisjoint(ids[validation])
    splits = [('outer', train, validation)]
    for inner, (a, b) in enumerate(StratifiedKFold(2, shuffle=True, random_state=50+fold).split(X[train], y[train]), 1):
        assert set(train[a]).isdisjoint(validation) and set(train[b]).isdisjoint(validation)
        splits.append((f'inner_{inner}', train[a], train[b]))
    for level, a, b in splits:
        pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('nonconstant', VarianceThreshold()), ('scaler', StandardScaler())])
        pipeline.fit(X[a])
        imputed = pipeline['imputer'].transform(X[a])
        kept = pipeline['nonconstant'].transform(imputed)
        assert np.allclose(pipeline['scaler'].mean_, kept.mean(axis=0))
        assert np.allclose(pipeline['scaler'].var_, kept.var(axis=0))
        assert int(pipeline['scaler'].n_samples_seen_) == len(a)
        before = pipeline['scaler'].mean_.copy()
        poisoned_validation = X[b].copy() + 1e6
        pipeline.transform(poisoned_validation)
        assert np.array_equal(before, pipeline['scaler'].mean_)
        assert set(ids[a]).isdisjoint(ids[b])
        checks.append({'fold': fold, 'level': level, 'train_count': len(a), 'validation_count': len(b),
                       'scaler_train_only': True,
                       'shared_original_states': sorted(set(states[a]) & set(states[b]))})
report = {'records': 25, 'features': 928, 'normalization_checks': checks,
          'exact_duplicate_feature_rows': duplicates,
          'label_and_record_metadata_excluded': True,
          'code_audit': 'GridSearchCV receives only outer training rows; preprocessing is inside its pipeline. MFCC is extracted separately for each recording.',
          'limitations': ['Repeated recordings of the same original experimental state occur in training and validation.',
                          'All recordings come from one bridge; bridge-level generalization cannot be evaluated.',
                          'Feature-family and model selection followed exploratory analysis of this dataset; reported scores are not an untouched final test.',
                          'No exact duplicate feature rows found; overlapping acquisition intervals cannot be ruled out without acquisition metadata.']}
(root / 'outputs/leakage_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
