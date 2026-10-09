import base64
import json
import shutil
from datetime import datetime
from pathlib import Path
import h5py
import nbformat
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapezoid

ROOT=Path(__file__).resolve().parents[1]

def metrics(x,fs,nperseg=2048):
    f,p=welch(x,fs=fs,window='hann',nperseg=min(nperseg,len(x)),noverlap=min(nperseg,len(x))//2,detrend=False)
    positive=f>0
    if p[positive].sum()==0:
        return f,p,{'fraction_4_8':None,'power_4_8':0.,'total_power':0.,'spectral_spread':None,'rms':0.,'peak_abs':float(np.max(np.abs(x)))}
    weights=p[positive]/p[positive].sum();centroid=np.sum(f[positive]*weights)
    total=trapezoid(p[positive],f[positive]);band=(f>=4)&(f<=8)
    power=trapezoid(p[band],f[band])
    return f,p,{'fraction_4_8':float(power/total),'power_4_8':float(power),
        'total_power':float(total),'spectral_spread':float(np.sqrt(np.sum(weights*(f[positive]-centroid)**2))),
        'rms':float(np.sqrt(np.mean(x*x))),'peak_abs':float(np.max(np.abs(x)))}

def run_signal_check():
    report=[];images=[]
    with np.load(ROOT/'data/processed/all_acceleration_features.npz') as saved:
        ids=saved['record_ids'].astype(str);names=saved['feature_names'].astype(str);old_X=saved['X']
    with h5py.File(ROOT/'data/raw/Rt345Bridge.h5','r') as raw:
        for sensor,axis,feature in [('A06','y','fraction_4_8'),('A18','z','spectral_spread')]:
            rows=[];fig,axes=plt.subplots(3,1,figsize=(12,12),layout='constrained')
            for index,identity in enumerate(ids):
                state,test=identity.split('/');group=raw[f'{state}/acceleration/{test}/{sensor}']
                x=group[axis][...].reshape(-1).astype(float);fs=float(np.asarray(group.attrs['sample_rate']).item())
                f,p,whole=metrics(x,fs)
                assert np.isclose(whole[feature],old_X[index,np.flatnonzero(names==f'{sensor}_{axis}_{feature}')[0]])
                category='damage_1' if state=='damage_1' else 'baseline' if state.startswith('baseline') else 'other_damage'
                color={'damage_1':'C3','baseline':'C0','other_damage':'gray'}[category]
                axes[1].semilogy(f,p,color=color,alpha=.8 if category=='damage_1' else .25,lw=1.5 if category=='damage_1' else .6,
                    label=category if not any(r['category']==category for r in rows) else None)
                frames=[]
                for start in range(0,len(x)-int(20*fs)+1,int(20*fs)):
                    _,_,local=metrics(x[start:start+int(20*fs)],fs,1024)
                    frames.append(local[feature])
                if category=='damage_1':
                    axes[0].plot(np.arange(len(x))/fs,x,alpha=.6,lw=.5,label=test)
                    axes[2].plot(np.arange(len(frames))*20+10,frames,'o-',label=identity,color=color,alpha=.65)
                trimmed=x[int(5*fs):-int(5*fs)]
                _,_,trim_metric=metrics(trimmed,fs)
                rows.append({'recording':identity,'category':category,**whole,
                    'trim_5s_each_end_feature':trim_metric[feature],'20s_frame_feature_values':frames})
            summary={}
            for category in ['damage_1','baseline','other_damage']:
                selected=[r for r in rows if r['category']==category]
                values=np.array([r[feature] for r in selected])
                trimmed=np.array([r['trim_5s_each_end_feature'] for r in selected])
                all_frame_values=[v for r in selected for v in r['20s_frame_feature_values']]
                frame_values=np.array([v for v in all_frame_values if v is not None])
                summary[category]={'whole_min':float(values.min()),'whole_median':float(np.median(values)),
                    'whole_max':float(values.max()),'trimmed_min':float(trimmed.min()),'trimmed_max':float(trimmed.max()),
                    'undefined_zero_energy_frames':sum(v is None for v in all_frame_values),
                    'frame_min':float(frame_values.min()),'frame_median':float(np.median(frame_values)),'frame_max':float(frame_values.max()),
                    'band_power_median':float(np.median([r['power_4_8'] for r in selected])),
                    'total_power_median':float(np.median([r['total_power'] for r in selected]))}
                if category!='damage_1':axes[2].axhspan(np.quantile(frame_values,.25),np.quantile(frame_values,.75),color='C0' if category=='baseline' else 'gray',alpha=.12,label=category+' frame IQR')
            axes[0].set(title=f'{sensor}/{axis.upper()} — raw damage_1 recordings',xlabel='Time [s]',ylabel='Acceleration [m/s²]');axes[0].legend()
            axes[1].set(title='All recording PSDs; red = damage_1',xlabel='Frequency [Hz]',ylabel='PSD',xlim=(0,64));axes[1].legend()
            axes[2].set(title='20s frames: descriptive stability only (different PSD resolution)',xlabel='Frame center [s]',ylabel=feature);axes[2].legend()
            output=ROOT/'outputs'/f'problem_signal_{sensor}_{axis}.png';fig.savefig(output,dpi=130);plt.close(fig);images.append(output)
            report.append({'sensor':sensor,'axis':axis,'feature':feature,'summary':summary,'recordings':rows})
    result={'analysis':report,'notes':'Whole-recording features reproduce stored values. Trim and frames are diagnostics, not new labeled samples. Frames are not synchronized excitation events across recordings. Post-hoc group comparison; no physical cause is established.'}
    (ROOT/'outputs/problem_signal_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    path=ROOT/'notebooks/experiment.ipynb';nb=nbformat.read(path,as_version=4)
    shutil.copy2(path,ROOT/'archive/recovery'/f'experiment_before_signal_check_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    md=nbformat.v4.new_markdown_cell('## 22. Raw-signal checks: A06/Y and A18/Z\n\nInspect all recording PSDs and the three raw damage_1 traces. Recompute the proposed features, then check sensitivity to trimming five seconds from each end and to non-overlapping 20-second frames. Frames are descriptive, not independently labeled examples; event times across recordings are not assumed aligned. These are post-hoc checks and do not establish a physical damage cause.')
    cell=nbformat.v4.new_code_cell('import sys\nif str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))\nfrom check_problem_signals import run_signal_check\nsignal_check_report=run_signal_check()\nfor analysis in signal_check_report["analysis"]:\n    print(analysis["sensor"],analysis["axis"],analysis["summary"])\nfrom IPython.display import Image,display\nfor name in ["problem_signal_A06_y.png","problem_signal_A18_z.png"]:\n    display(Image(filename=str(ROOT/"outputs"/name)))')
    md['id']='signal-check-00';cell['id']='signal-check-01'
    cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(p.read_bytes()).decode()},metadata={}) for p in images]
    nb.cells=[c for c in nb.cells if not c.get('id','').startswith('signal-check-')];nb.cells[-1:-1]=[md,cell]
    nbformat.validate(nb);nbformat.write(nb,path)
    print(json.dumps([{k:v for k,v in a.items() if k!='recordings'} for a in report],indent=2))
    return result

if __name__=='__main__':run_signal_check()
