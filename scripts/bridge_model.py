"""Fixed signal contract; no state, label, file name or sample count enters features."""
import numpy as np
import pandas as pd
import librosa
from sklearn.base import BaseEstimator, TransformerMixin

SENSORS=tuple(f'A{i:02d}' for i in range(1,31) if i!=14)
FEATURE_NAMES=tuple(f'{s}_{a}_mfcc_{i}_{stat}' for s in SENSORS for a in ('y','z') for i in range(8) for stat in ('median','iqr','change'))
FS=128
SAMPLES=89*FS

def extract_features(signals,fs=128):
    if fs!=FS:raise ValueError('Expected sampling rate 128 Hz')
    expected={(s,a) for s in SENSORS for a in ('y','z')}
    if set(signals)!=expected:raise ValueError('Expected exactly 58 specified acceleration channels; A14 excluded')
    values=[]
    for s in SENSORS:
        for a in ('y','z'):
            x=np.asarray(signals[(s,a)],dtype=float)
            if x.ndim!=1 or len(x)<SAMPLES or not np.isfinite(x).all():
                raise ValueError('Expected finite 1D signal of at least 89 seconds')
            c=librosa.feature.mfcc(y=x[:SAMPLES],sr=FS,n_mfcc=8,n_mels=16,n_fft=512,
                hop_length=256,fmin=0,fmax=64,center=False)
            assert c.shape==(8,43)
            values.extend(np.column_stack([np.median(c,axis=1),np.quantile(c,.75,axis=1)-np.quantile(c,.25,axis=1),np.mean(np.abs(np.diff(c,axis=1)),axis=1)]).ravel())
    return pd.DataFrame([values],columns=FEATURE_NAMES)

class FeatureContract(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.transform(X)
        self.n_features_in_=len(FEATURE_NAMES)
        return self
    def transform(self,X):
        if not isinstance(X,pd.DataFrame):raise ValueError('Named feature DataFrame required')
        if not X.columns.is_unique or set(X.columns)!=set(FEATURE_NAMES):
            raise ValueError('Missing/extra feature columns; metadata columns forbidden')
        values=X.loc[:,list(FEATURE_NAMES)].to_numpy(dtype=float)
        if not np.isfinite(values).all():raise ValueError('Non-finite features')
        return values
