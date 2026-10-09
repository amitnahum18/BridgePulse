# Prediction with BridgePulse

The prediction pipeline reads one recording from an H5 file, extracts MFCC features, and loads the packaged model. It does not train a model or use the recording's label.

Install the prediction dependencies and supply the input file as described in the [project README](../README.md). Run the following command from the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m prediction --h5 data/raw/Rt345Bridge.h5 --recording-path damage_1/acceleration/test_1 --output outputs/predictions/example.json
```

`--recording-path` identifies the sensor group inside the H5 file. Its name is used only to locate the measurements and does not enter the model's features. The command prints a JSON result and saves it to the path supplied with `--output`; omit that option to print only.

## Use from Python or a notebook

When running from the project root:

```python
from prediction.pipeline import BridgePredictor

predictor = BridgePredictor()
result = predictor.predict_h5(
    "data/raw/Rt345Bridge.h5",
    "damage_1/acceleration/test_1",
)
result
```

If the notebook runs from `notebooks/`, first add the project root to `sys.path`. The existing research notebook defines `ROOT` and `file_path`, so use:

```python
import sys
sys.path.insert(0, str(ROOT))

from prediction.pipeline import BridgePredictor

predictor = BridgePredictor()
result = predictor.predict_h5(file_path, "damage_1/acceleration/test_1")
result
```

For a new recording, call `predictor.predict_signals(signals, fs=128)`, where `signals` is a dictionary containing all 58 channels, keyed by `(sensor, axis)`, such as `("A01", "y")`. This accepts signals without assigning the recording to a baseline or damage group.

## Input and output

The input requires sensors `A01` through `A30`, excluding `A14`, with axes `y` and `z`, a sampling rate of 128 Hz, and at least 89 seconds per channel. Each signal must contain finite values. Only the first 89 seconds are used. Missing sensors are not filled in, and no normalization is fitted to the recording. The H5 file is opened read-only.

For H5 input, the recording group must contain sensor groups. Each sensor group must have `y` and `z` datasets and a `sample_rate` attribute equal to `128`.

The output contains `predicted_label` (`0` for baseline or `1` for experimental damage), `predicted_class`, and scores for both classes. The scores are not calibrated probabilities. The model is a research candidate for classifying an existing condition; it does not forecast future crack onset.

## Files

- `pipeline.py`: H5 loading and prediction from signals.
- `bridge_model.py`: input validation and MFCC feature extraction.
- `artifacts/model.joblib`: model trained on all 25 recordings.
- `artifacts/manifest.json`: parameters, feature names, dependency versions, and hashes.
- `artifacts/model_card.md`: model evaluation and limitations.
- `requirements.txt`: dependencies for prediction only.

The packaged model was trained on all available recordings, so predictions for the original recordings are not a holdout evaluation. The separate holdout model is stored at `models/bridge_damage_rf_group_holdout.joblib`, relative to the project root. Research scripts are in `scripts/`, the notebook is in `notebooks/`, and experiment results are in `outputs/`.
