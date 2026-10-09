# BridgePulse damage classifier — exploratory candidate

Random Forest: 200 trees, maximum depth 2, minimum leaf size 4, balanced class weights, seed 42. Trained on all 25 recordings only after state-grouped evaluation. No untouched final test exists.

Input: 29 acceleration sensors A01–A30 excluding A14, axes y/z, 128 Hz. Feature extraction always uses the first 89 seconds (11,392 samples), producing exactly 43 overlapping 4-second frames. Each of eight MFCC coefficients is summarized by median, interquartile range and mean absolute inter-frame change: 1,392 named features. No recording identifiers, state, label, sample count or timestamps enter the model. No normalization is needed for this forest.

The reusable extractor and schema checks are in `scripts/bridge_model.py`. Supply a dictionary keyed by `(sensor, axis)` to `extract_features`; the returned named DataFrame is accepted by the registered sklearn pipeline. The registered model accepts these extracted features, not an H5 filename. Extra trailing samples cannot affect features; missing channels, short signals, non-finite values and extra/missing feature columns are rejected.

Target: 0 = experimental baseline; 1 = imposed experimental damage. This is not forecasting future crack onset. All recordings are from one bridge. Different experimental states may share excitation/protocol history. Label timing within a recording is not independently established; frames are internal summaries, not independently labeled examples.

Evaluation keeps all recordings of each original experimental state together. Twenty allocations, two folds each, are repeated assessments of the same eight states, not independent experiments. Representation and parameters were selected after exploratory comparisons, so model-selection optimism remains. The fixed-duration audit and metrics are stored in `outputs/final_pipeline_audit.json`; registration details in `outputs/mlflow_registration.json`.

The earlier 84.7% balanced accuracy used slightly different recording lengths and frame counts across classes. The final candidate eliminates that protocol difference; use its audited metrics instead. It is a research candidate, not a production-validated bridge safety system.

For local MLflow, run from the project root:

```powershell
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Model registration uses SQLite locally unless `MLFLOW_TRACKING_URI` is already configured. Registration is not deployment.
