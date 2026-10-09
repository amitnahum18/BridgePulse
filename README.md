# BridgePulse

BridgePulse is a research project for detecting experimental bridge damage from acceleration sensor recordings. Its prediction pipeline reads one recording, extracts signal features using mel-frequency cepstral coefficients (MFCCs), and applies a pretrained Random Forest model to classify the recording as `baseline` or `damage`.

The model classifies the condition represented by a recording. It does not forecast future crack formation. It is an exploratory research candidate and has not been validated for bridge safety decisions.

## Quick start: prediction

The commands below use Windows PowerShell. Run them from the project root after cloning the repository. The existing environment was tested with Python 3.12.3.

### 1. Clone the project and install dependencies

```powershell
git clone https://github.com/amitnahum18/BridgePulse.git
cd BridgePulse
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r prediction/requirements.txt
```

This installs the dependencies needed to use the bundled prediction model. Retraining and MLflow are not required. For the notebook and experiments, see the research environment section below.

### 2. Add the dataset

The source file `Rt345Bridge.h5` is approximately 907 MB and is not included in the repository. Obtain it separately and place it at the following path before running the example or the notebook:

```text
data/raw/Rt345Bridge.h5
```

The original file's SHA-256 checksum is recorded in [data/raw/Rt345Bridge.h5.sha256](data/raw/Rt345Bridge.h5.sha256). Compute the checksum of your local file and compare it with the recorded value:

```powershell
Get-FileHash -Algorithm SHA256 data/raw/Rt345Bridge.h5
```

You can also use another H5 file that meets the input requirements by passing its path through `--h5`.

### 3. Run a prediction

```powershell
.\.venv\Scripts\python.exe -m prediction --h5 data/raw/Rt345Bridge.h5 --recording-path damage_1/acceleration/test_1 --output outputs/predictions/example.json
```

`--recording-path` locates the sensor group inside the H5 file. The group name is used only to find the measurements; it is not used as a model feature or a label. The H5 file is opened read-only.

The result is printed as JSON and saved to the path specified by `--output`. Omit `--output` to print the result without saving it.

| Output field | Meaning |
|---|---|
| `predicted_label` | `0` = experimental baseline; `1` = experimental damage |
| `predicted_class` | `baseline` or `damage` |
| `damage_score`, `baseline_score` | Model scores for the two classes; these are not calibrated probabilities |
| `status` | `exploratory_candidate` |

See [prediction/README.md](prediction/README.md) for Python and notebook usage.

## Input requirements

| Component | Requirement |
|---|---|
| Sensors | `A01` through `A30`, excluding `A14`: 29 sensors |
| Axes | `y` and `z` for each sensor: 58 channels |
| Sampling rate | 128 Hz |
| Duration | At least 89 seconds per channel; prediction uses the first 89 seconds |
| Signal values | One-dimensional arrays containing only finite values |
| H5 structure | A recording group containing sensor groups, each with `y` and `z` datasets and a `sample_rate=128` attribute |

Missing sensors, short signals, or invalid values cause an error. Missing channels are not filled automatically. You can also pass signals directly to `BridgePredictor.predict_signals`; see the prediction documentation for an example.

## Model and evaluation

The bundled Random Forest has 200 trees, a maximum depth of 2, and a minimum of 4 samples per leaf. Feature extraction uses 4-second windows with a 2-second hop. Eight MFCCs per channel are summarized using the median, interquartile range, and mean absolute change between windows, producing 1,392 features per recording.

Evaluation keeps all recordings from each experimental state together, so a state is never shared between the training and validation sets within a split. Mean balanced accuracy under this evaluation is **81.97%**, with a standard deviation of **8.14 percentage points** across allocations. Detailed results are in [outputs/final_pipeline_audit.json](outputs/final_pipeline_audit.json).

The dataset contains 25 recordings from eight experimental states on one bridge. Features and parameters were selected after exploratory experiments, and there is no untouched final test set. The packaged model was ultimately trained on all 25 recordings; predictions on those recordings do not measure performance on new data.

See the [model card](prediction/artifacts/model_card.md) for the full limitations. The separate holdout evaluation model is stored in `models/bridge_damage_rf_group_holdout.joblib`; the prediction package loads its bundled model by default. Earlier experiment results are retained for reference and should be distinguished from the final model's audited results.

## Research environment and notebook

From the project root, use the `.venv` created in the quick start. If it does not exist yet, create it with `python -m venv .venv`. Add the source dataset, then install the research dependencies and register the notebook kernel:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name bridgepulse --display-name "Python (BridgePulse)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/experiment.ipynb
```

Select the **Python (BridgePulse)** kernel in the notebook. The notebook contains signal analysis, model comparisons, and experiment results. Running all cells performs computations and retraining, and may update files under `data/processed/`, `models/`, and `outputs/`.

`requirements-lock.txt` records the research environment's package versions. `requirements.txt` lists direct research dependencies. Dependencies for prediction alone are in `prediction/requirements.txt`.

## Project structure

| Path | Contents |
|---|---|
| `prediction/` | Standalone prediction package, feature extraction, and the pretrained model at `artifacts/model.joblib` |
| `data/raw/` | Source dataset checksum; add the H5 file locally |
| `data/processed/` | Cached features used in experiments |
| `notebooks/experiment.ipynb` | Research notebook and results |
| `scripts/` | Analysis, checks, model comparisons, and MLflow registration |
| `models/` | Research models and model card |
| `outputs/` | JSON reports, plots, and an example prediction under `predictions/` |

The `.venv/` environment, `archive/` recovery copies, logs, and local MLflow stores are excluded from Git.

## Optional local MLflow registration

The original experiment registered the model as `BridgePulseDamageRF`, version 1. Registration details and metrics are in [outputs/mlflow_registration.json](outputs/mlflow_registration.json). Prediction package consistency checks are in [outputs/prediction_package_checks.json](outputs/prediction_package_checks.json).

The original MLflow database and artifacts are not included in the repository. To create a registration in your own environment, install the research dependencies, add the source dataset, and run:

```powershell
.\.venv\Scripts\python.exe scripts/register_bridge_model.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The registration script reruns the model checks, trains the model, and updates artifacts and reports. It creates a local MLflow store by default, or uses `MLFLOW_TRACKING_URI` if that environment variable is set. The UI command above opens the local store at http://127.0.0.1:5000. Registering the model does not deploy it as a service.
