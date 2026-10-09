import hashlib,itertools,json,os
from pathlib import Path
import h5py,joblib,numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,average_precision_score
from bridge_model import extract_features,FeatureContract,FEATURE_NAMES,SENSORS

ROOT=Path(__file__).resolve().parents[1]

def main():
    with np.load(ROOT/'data/processed/mfcc_frame_features.npz') as data:
        ids=data['record_ids'].astype(str);y=data['y']
    assert len(set(ids))==25
    assert np.array_equal(y,[int(not r.startswith('baseline_')) for r in ids])
    rows=[]
    with h5py.File(ROOT/'data/raw/Rt345Bridge.h5','r') as raw:
        for identity in ids:
            state,test=identity.split('/');signals={}
            for sensor in SENSORS:
                group=raw[f'{state}/acceleration/{test}/{sensor}']
                assert float(np.asarray(group.attrs['sample_rate']).item())==128
                for axis in ('y','z'):signals[(sensor,axis)]=group[axis][...].reshape(-1)
            row=extract_features(signals)
            # Extra trailing samples and their values must not influence the features.
            if not rows:
                changed={k:np.concatenate([v[:89*128],np.full(300,1e6)]) for k,v in signals.items()}
                assert np.array_equal(row.values,extract_features(changed).values)
            rows.append(row)
    X=pd.concat(rows,ignore_index=True);assert X.shape==(25,1392)
    assert not X.duplicated().any()
    states=np.array([r.split('/')[0] for r in ids]);folds=[];allocations=[]
    def model():return Pipeline([('contract',FeatureContract()),('classifier',RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=4,class_weight='balanced',random_state=42,n_jobs=1))])
    for allocation,damages in enumerate(itertools.combinations([f'damage_{i}' for i in range(1,7)],3),1):
        mask=np.isin(states,['baseline_1',*damages]);oof=np.empty_like(y);seen=np.zeros(25,int)
        for fold,hold in enumerate((mask,~mask),1):
            a=np.flatnonzero(~hold);b=np.flatnonzero(hold)
            assert set(states[a]).isdisjoint(states[b])
            fitted=model().fit(X.iloc[a],y[a]);pred=fitted.predict(X.iloc[b]);oof[b]=pred;seen[b]+=1
            score=fitted.predict_proba(X.iloc[b])[:,1]
            folds.append({'allocation':allocation,'fold':fold,'train_BA':float(balanced_accuracy_score(y[a],fitted.predict(X.iloc[a]))),
                'validation_BA':float(balanced_accuracy_score(y[b],pred)),
                'damage_AP':float(average_precision_score(y[b],score)),
                'baseline_AP':float(average_precision_score(1-y[b],-score)),
                'train_states':sorted(set(states[a])),'validation_states':sorted(set(states[b]))})
        assert np.all(seen==1)
        allocations.append(float(balanced_accuracy_score(y,oof)))
    final=model().fit(X,y)
    # Ordering is explicit; metadata cannot be accidentally consumed at inference.
    assert np.allclose(final.predict_proba(X),final.predict_proba(X[X.columns[::-1]]))
    rejected=0
    for bad in [X.assign(label=y),X.drop(columns=X.columns[0]),X.assign(record_id=ids),X.iloc[:,:10],X*float('nan')]:
        try:final.predict(bad)
        except ValueError:rejected+=1
    assert rejected==5
    checksum=hashlib.file_digest((ROOT/'data/raw/Rt345Bridge.h5').open('rb'),'sha256').hexdigest()
    assert checksum==(ROOT/'data/raw/Rt345Bridge.h5.sha256').read_text().split()[0]
    metrics={'grouped_validation_BA_mean':float(np.mean(allocations)),'grouped_split_SD':float(np.std(allocations)),
        'cv_train_BA_mean':float(np.mean([r['train_BA'] for r in folds])),
        'cv_train_validation_gap':float(np.mean([r['train_BA']-r['validation_BA'] for r in folds])),
        'grouped_damage_AP':float(np.mean([r['damage_AP'] for r in folds])),
        'grouped_baseline_AP':float(np.mean([r['baseline_AP'] for r in folds])),
        'final_fit_training_BA':float(balanced_accuracy_score(y,final.predict(X)))}
    report={'metrics':metrics,'folds':folds,'allocations':allocations,'raw_sha256':checksum,
        'checks':{'uniform_89_seconds_43_frames':True,'tail_invariance':True,'metadata_rejected':True,'labels_aligned':True,'state_disjoint':True,'schema_order_checked':True,'no_normalization_needed_for_forest':True},
        'limitations':'Exploratory candidate trained on all 25 recordings after prior feature/parameter selection. No untouched final test; one bridge, eight experimental states; group-level physical dependence may remain. Not production validated.'}
    report_path=ROOT/'outputs/final_pipeline_audit.json';report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    joblib.dump(final,ROOT/'models/bridge_damage_rf_candidate.joblib')
    np.savez_compressed(ROOT/'data/processed/fixed89_mfcc_features.npz',X=X.values,y=y,record_ids=ids,feature_names=FEATURE_NAMES)
    print('Pipeline audit passed:',json.dumps(metrics),flush=True)
    import mlflow,mlflow.sklearn
    from mlflow.models import infer_signature
    tracking=os.environ.get('MLFLOW_TRACKING_URI') or 'sqlite:///'+(ROOT/'mlflow.db').as_posix()
    mlflow.set_tracking_uri(tracking)
    client=mlflow.MlflowClient();experiment_name='BridgePulse-state-grouped'
    experiment=client.get_experiment_by_name(experiment_name)
    experiment_id=experiment.experiment_id if experiment else client.create_experiment(experiment_name,artifact_location=(ROOT/'mlartifacts').as_uri())
    with mlflow.start_run(experiment_id=experiment_id,run_name='RF-fixed89s-robust-MFCC-candidate') as run:
        mlflow.log_params({'n_estimators':200,'max_depth':2,'min_samples_leaf':4,'features':1392,'fs':128,'duration_seconds':89,'frame_seconds':4,'hop_seconds':2,'n_mfcc':8,'n_mels':16,'records':25,'states':8,'grouping':'original experimental state','fit_scope':'all 25 records; CV metrics separate'})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({'status':'exploratory_candidate','independent_test':'none','direct_leakage_audit':'passed','post_hoc_selection_bias':'present','raw_sha256':checksum})
        for file in [report_path,ROOT/'outputs/forest_regularization_report.json',ROOT/'scripts/bridge_model.py',ROOT/'scripts/register_bridge_model.py',ROOT/'requirements-lock.txt']:
            mlflow.log_artifact(str(file),artifact_path='audit_and_reproducibility')
        versions=[]
        for package in ['numpy','pandas','scikit-learn','librosa','mlflow','cloudpickle']:
            from importlib.metadata import version
            versions.append(package+'=='+version(package))
        info=mlflow.sklearn.log_model(final,name='bridge_damage_pipeline',registered_model_name='BridgePulseDamageRF',
            serialization_format='cloudpickle',code_paths=[str(ROOT/'scripts/bridge_model.py')],
            signature=infer_signature(X.iloc[:2],final.predict(X.iloc[:2])),input_example=X.iloc[:1],pip_requirements=versions)
        loaded=mlflow.sklearn.load_model(info.model_uri)
        assert np.allclose(loaded.predict_proba(X),final.predict_proba(X))
        pyfunc=mlflow.pyfunc.load_model(info.model_uri)
        assert np.array_equal(pyfunc.predict(X),final.predict(X))
        mlflow.log_metric('reload_prediction_check_passed',1)
        saved={'run_id':run.info.run_id,'experiment_id':experiment_id,'tracking_uri':tracking,'model_uri':info.model_uri,'registered_model_name':'BridgePulseDamageRF','registered_model_version':info.registered_model_version,'metrics':metrics,'reload_verified':True}
        (ROOT/'outputs/mlflow_registration.json').write_text(json.dumps(saved,indent=2),encoding='utf-8')
        print(json.dumps(saved,indent=2),flush=True)

if __name__=='__main__':main()
