"""Bundle the audited model without retraining and verify raw-to-prediction parity."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prediction.bridge_model import FeatureContract, FEATURE_NAMES
from prediction.pipeline import BridgePredictor

def main():
    original=joblib.load(ROOT/'models/bridge_damage_rf_candidate.joblib')
    with np.load(ROOT/'data/processed/fixed89_mfcc_features.npz') as data:
        X=pd.DataFrame(data['X'],columns=data['feature_names']);ids=data['record_ids'].astype(str)
    packaged=Pipeline([('contract',FeatureContract().fit(X)),('classifier',original.named_steps['classifier'])])
    assert np.allclose(packaged.predict_proba(X),original.predict_proba(X))
    destination=ROOT/'prediction/artifacts/model.joblib'
    joblib.dump(packaged,destination)
    predictor=BridgePredictor(destination)
    assert np.allclose(predictor.model.predict_proba(X),original.predict_proba(X))
    raw_path=ROOT/'data/raw/Rt345Bridge.h5'
    for identity in ['baseline_1/test_1','damage_1/test_2','damage_6/test_3']:
        state,test=identity.split('/')
        result=predictor.predict_h5(raw_path,f'{state}/acceleration/{test}')
        index=np.flatnonzero(ids==identity)[0]
        assert np.isclose(result['damage_score'],original.predict_proba(X.iloc[[index]])[0,1])
        assert result['predicted_label']==original.predict(X.iloc[[index]])[0]
    packages=['numpy','pandas','scipy','scikit-learn','librosa','h5py','joblib']
    requirements={package:version(package) for package in packages}
    (ROOT/'prediction/requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in requirements.items())+'\n',encoding='utf-8')
    registration=json.loads((ROOT/'outputs/mlflow_registration.json').read_text())
    manifest={'name':'BridgePulseDamageRF','status':'exploratory_candidate','source_mlflow':registration,
        'schema':{'sensors':29,'channels':58,'fs':128,'duration_seconds':89,'n_fft':512,'hop_length':256,
                  'n_mfcc':8,'n_mels':16,'summaries':['median','iqr','mean_absolute_change'],'feature_names':FEATURE_NAMES},
        'model_parameters':{'n_estimators':200,'max_depth':2,'min_samples_leaf':4,'class_weight':'balanced','random_state':42},
        'model_sha256':hashlib.file_digest(destination.open('rb'),'sha256').hexdigest(),
        'dependencies':requirements,'verification':{'all_25_feature_predictions_match':True,'three_raw_recordings_match':True,'no_retraining':True},
        'fit_scope':'all 25 original recordings; never use these predictions as test metrics'}
    (ROOT/'prediction/artifacts/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'outputs/prediction_package_checks.json').write_text(json.dumps(manifest['verification'],indent=2),encoding='utf-8')
    print('Prediction package saved; all feature predictions and three raw-recording predictions match.')

if __name__=='__main__':main()
