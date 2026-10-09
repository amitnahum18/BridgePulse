from pathlib import Path
import json
import h5py
import joblib
import numpy as np
from .bridge_model import SENSORS, extract_features

DEFAULT_MODEL=Path(__file__).parent/'artifacts/model.joblib'

class BridgePredictor:
    def __init__(self,model_path=DEFAULT_MODEL):
        self.model=joblib.load(Path(model_path))
        self.classes=self.model.named_steps['classifier'].classes_.tolist()
        if self.classes!=[0,1]:raise ValueError('Unexpected model classes')

    def predict_signals(self,signals,fs=128):
        features=extract_features(signals,fs)
        probabilities=self.model.predict_proba(features)[0]
        label=int(self.model.predict(features)[0])
        return {'predicted_label':label,'predicted_class':'damage' if label else 'baseline',
                'damage_score':float(probabilities[1]),'baseline_score':float(probabilities[0]),
                'threshold':.5,'status':'exploratory_candidate',
                'score_note':'Model scores are not calibrated probabilities.'}

    def predict_h5(self,file_path,recording_path):
        """recording_path locates sensor data only; its text never enters features."""
        signals={}
        with h5py.File(file_path,'r') as raw:
            if recording_path not in raw:raise ValueError(f'Recording group not found: {recording_path}')
            group=raw[recording_path]
            for sensor in SENSORS:
                if sensor not in group:raise ValueError(f'Missing sensor: {sensor}')
                channel=group[sensor]
                if 'sample_rate' not in channel.attrs:raise ValueError(f'Missing sample_rate: {sensor}')
                if float(np.asarray(channel.attrs['sample_rate']).item())!=128:
                    raise ValueError(f'Expected 128 Hz: {sensor}')
                for axis in ('y','z'):
                    if axis not in channel:raise ValueError(f'Missing axis: {sensor}/{axis}')
                    signals[(sensor,axis)]=channel[axis][...].reshape(-1)
        result=self.predict_signals(signals)
        result['recording_path']=recording_path
        return result

def write_result(result,output):
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

