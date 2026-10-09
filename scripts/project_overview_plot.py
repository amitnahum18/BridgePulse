from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapezoid

ROOT=Path(__file__).resolve().parents[1]
bands=[(4,8),(8,14),(25,35),(55,64)]
spectra={key:[] for key in ['Baseline','Damage 1','Other damage']}
powers={key:[] for key in spectra}
with np.load(ROOT/'data/processed/fixed89_mfcc_features.npz') as data:ids=data['record_ids'].astype(str)
with h5py.File(ROOT/'data/raw/Rt345Bridge.h5','r') as raw:
    for identity in ids:
        state,test=identity.split('/');x=raw[f'{state}/acceleration/{test}/A15/z'][...].reshape(-1)[:89*128]
        f,p=welch(x,fs=128,window='hann',nperseg=2048,noverlap=1024,detrend=False)
        key='Baseline' if state.startswith('baseline') else 'Damage 1' if state=='damage_1' else 'Other damage'
        spectra[key].append(p)
        total=trapezoid(p[f>0],f[f>0])
        powers[key].append([trapezoid(p[(f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)])/total for lo,hi in bands])
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
colors={'Baseline':'C0','Damage 1':'C3','Other damage':'C1'}
for key,values in spectra.items():
    p=np.asarray(values);axes[0,0].semilogy(f,np.median(p,axis=0),color=colors[key],label=f'{key} (n={len(p)})')
for lo,hi in bands:axes[0,0].axvspan(lo,hi,color='gray',alpha=.1)
axes[0,0].set(title='A15/Z: median recording PSD (first 89s)',xlabel='Frequency [Hz]',ylabel='PSD [(m/s²)²/Hz]',xlim=(0,64));axes[0,0].legend()
rng=np.random.default_rng(42)
for j,key in enumerate(powers):
    values=np.asarray(powers[key])*100
    for i in range(4):
        position=i+(j-1)*.22
        axes[0,1].scatter(position+rng.uniform(-.04,.04,len(values)),values[:,i],s=22,color=colors[key],alpha=.75,label=key if i==0 else None)
        axes[0,1].plot([position-.08,position+.08],[np.median(values[:,i])]*2,color='black')
axes[0,1].set(title='A15/Z: fraction of spectral power per recording',ylabel='Fraction [%]');axes[0,1].set_xticks(range(4),[f'{lo}–{hi} Hz' for lo,hi in bands]);axes[0,1].legend()
axes[1,0].bar(['Baseline (0)','Damage (1)'],[6,19],color=['C0','C1']);axes[1,0].set(title='25 recordings, one bridge, eight experimental states',ylabel='Recordings');axes[1,0].set_ylim(0,22)
for i,v in enumerate([6,19]):axes[1,0].text(i,v+.4,str(v),ha='center')
audit=json.loads((ROOT/'outputs/final_pipeline_audit.json').read_text())['metrics']
hold=json.loads((ROOT/'outputs/group_holdout_report.json').read_text())['metrics']
positions=np.arange(2);width=.35
axes[1,1].bar(positions-width/2,[audit['cv_train_BA_mean'],hold['train']['balanced_accuracy']],width,label='Training',color='C0')
axes[1,1].bar(positions+width/2,[audit['grouped_validation_BA_mean'],hold['holdout']['balanced_accuracy']],width,label='Held-out states',color='C1')
axes[1,1].set_xticks(positions,['Repeated grouped validation','Retrospective holdout']);axes[1,1].set(title='Corrected fixed-89s model | exploratory evaluation',ylabel='Balanced accuracy',ylim=(0,1.08));axes[1,1].axhline(.5,color='gray',ls='--');axes[1,1].legend()
for j,val in enumerate([audit['grouped_validation_BA_mean'],hold['holdout']['balanced_accuracy']]):axes[1,1].text(j+width/2,val+.02,f'{val:.1%}',ha='center')
output=ROOT/'outputs/project_overview.png';fig.savefig(output,dpi=150);plt.close(fig)
print(output)
