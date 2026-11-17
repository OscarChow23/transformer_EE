"""r05-r09 standard performance overlays; reuse verified r05-r07 bootstrap exports.
New r08/r09 intervals use the same indices/seeds as that exact prior bootstrap implementation.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import talk12 as t
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def compute_bin(args):
 b,s,sel,pred,truth=args
 n=len(sel);out={}
 # Identical sample/bin seeds, int32 draw and chunk order as Nov6 perf_vs_trueE.py.
 rng=np.random.default_rng([t.rs.BOOT_SEED,s,b])
 idx=rng.integers(0,n,size=(t.rs.N_BOOT,n),dtype=np.int32)
 for r,x in pred.items():
  x=x/truth-1;q=np.quantile(x,[.16,.5,.84]);bq=np.empty((t.rs.N_BOOT,3))
  for k in range(0,t.rs.N_BOOT,100):bq[k:k+100]=np.quantile(x[idx[k:k+100]],[.16,.5,.84],axis=1).T
  bb=bq[:,1];br=(bq[:,2]-bq[:,0])/2
  out[r]={'bias':float(q[1]),'res':float((q[2]-q[0])/2),'bias_lo':float(np.quantile(bb,.16)),
   'bias_hi':float(np.quantile(bb,.84)),'res_lo':float(np.quantile(br,.16)),'res_hi':float(np.quantile(br,.84))}
 return b,out

def table(h,meta):
 rows=[]
 for r,m in meta['models'].items():
  c=m['card'];e=c['ellipse'];assert not e['provisional'] and e['a_percent']==e['b_percent']==22
  for s in ['1','2','3']:
   a=c['metrics']['trueE'][s];b=c['metrics']['trueLepE'][s]
   rows.append({'horn':h,'model':r,'sample':s,'n_window':a['n'],'nu_bias_pct':100*a['bias'],
    'nu_resolution_pct':100*a['resolution'],'nu_within10_pct':100*a['band_10'],
    'mu_bias_pct':100*b['bias'],'mu_resolution_pct':100*b['resolution'],'mu_within10_pct':100*b['band_10'],
    'frozen22_jointcircle_pct':100*e['by_sample'][s]['fraction'],'model_dir':m['model_dir'],
    'test_fingerprint_sha256':c['test_set']['fingerprint_sha256']})
 out=t.ROOT/h
 with open(out/'standard_metrics.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 t.dump(out/'standard_metrics.json',rows)
 lines=['# Standard metrics, '+h.upper(),'','Unweighted test rows, true neutrino energy 0.5–6 GeV for BOTH targets. All entries in percent. r07 is the original single run, not the four-run mean. Each model is a single training; no training-variation significance claim.','', '| sample | model | N | nu bias | nu res | nu ±10% | mu bias | mu res | mu ±10% | joint 22% |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for z in rows:lines.append('| '+ ' | '.join([z['sample'],z['model'],str(z['n_window'])]+[f'{z[k]:.2f}' for k in ['nu_bias_pct','nu_resolution_pct','nu_within10_pct','mu_bias_pct','mu_resolution_pct','mu_within10_pct','frozen22_jointcircle_pct']])+' |')
 lines+=['','Bias=median r; resolution=(q84−q16)/2, r=pred/true−1. Joint circle: unbinned fraction with (100 r_nu/22)^2+(100 r_mu/22)^2 <= 1. All numbers copied from verified scorecards. Sources and hashes in cache/'+h+'_verified.json. No v2 peak metrics.']
 (out/'standard_metrics.md').write_text('\n'.join(lines)+'\n')

def main(h,s):
 d=t.load(h);meta=t.getmeta(h);bins=np.array(t.rs.PERF_BINS);cent=(bins[:-1]+bins[1:])/2
 oldp=t.OLD/h/'manifest.json';old=json.loads(oldp.read_text())[f'p5_s{s}_bias_vs_trueE']
 for r in ['r05','r06','r07']:
  assert old['models'][r]['result_csv_sha256']==meta['models'][r]['hashes']['result.csv']
 assert old['test_fingerprint_sha256']==meta['test_fingerprint_sha256']
 cp=t.ROOT/'cache'/f'performance_{h}_s{s}.json'
 if cp.exists():stats=json.loads(cp.read_text())
 else:
  stats=[];tasks=[]
  for b,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
   sel=np.flatnonzero((d['s']==s)&(d['t']>=lo)&((d['t']<hi) if b<len(cent)-1 else (d['t']<=hi)))
   ref=old['per_bin'][f'{lo:.2f}-{hi:.2f}'];assert ref['n']==len(sel)
   row={'n':len(sel),'drawn':len(sel)>=50}
   if row['drawn']:
    for r in ['r05','r06','r07']:
     x=d['p_'+r][sel]/d['t'][sel]-1;q=np.quantile(x,[.16,.5,.84])
     row[r]={k:ref[r][k+'_pct']/100 for k in ['bias','res','bias_lo','bias_hi','res_lo','res_hi']}
     assert abs(q[1]-row[r]['bias'])<.00000051 and abs((q[2]-q[0])/2-row[r]['res'])<.00000051
     row[r]['bias']=float(q[1]);row[r]['res']=float((q[2]-q[0])/2)
    tasks.append((b,s,sel,{r:d['p_'+r][sel] for r in ['r08','r09']},d['t'][sel]))
   stats.append(row)
  with ProcessPoolExecutor(max_workers=4) as pool:
   for b,new in pool.map(compute_bin,tasks):stats[b].update(new);print(h,'S'+str(s),'bin',b,'bootstrap done',flush=True)
  t.dump(cp,stats)
 table(h,meta)
 fig,(a,b)=plt.subplots(2,1,figsize=(7.5,6.8),sharex=True,gridspec_kw={'hspace':0})
 offsets=dict(zip(t.RUN_NAMES,np.linspace(-.04,.04,5)))
 for ax,key,ylabel in [(a,'bias','Median fractional residual (%)'),(b,'res','Resolution (%)')]:
  for r in t.RUN_NAMES:
   ok=np.array([z['drawn'] for z in stats]);v=np.array([z.get(r,{}).get(key,0) for z in stats])*100
   lo=np.array([z.get(r,{}).get(key+'_lo',0) for z in stats])*100;hi=np.array([z.get(r,{}).get(key+'_hi',0) for z in stats])*100
   st=t.rs.MODELS[r]
   ax.errorbar(cent[ok]+offsets[r],v[ok],yerr=np.vstack([np.maximum(0,v[ok]-lo[ok]),np.maximum(0,hi[ok]-v[ok])]),fmt=st['marker'],
    color=st['color'],mfc=st['mfc'],mec=st['color'],ms=4,elinewidth=1,label=r)
  t.rs.labels(ax,None,ylabel);ax.set_xlim(.5,6)
 a.axhline(0,color='black',ls=':',lw=1);t.rs.tag(a,f'{h.upper()} · S{s} · test',loc='upper right')
 a.legend(loc='upper center',ncol=5,fontsize=10,bbox_to_anchor=(.5,1.16),columnspacing=1.1)
 b.set_ylim(bottom=0);t.rs.labels(b,t.rs.E_TRUE,None);fig.align_ylabels([a,b])
 cap=(f'Top: median fractional neutrino residual; bottom: (q84-q16)/2, in percent, versus true neutrino energy. r=pred/true−1. Bins 0.25 GeV over 0.5–6, final bin inclusive; N<50 hidden; cosmetic x offsets −0.04 to +0.04 GeV.\n'
  f'{h.upper()} S{s}, SAME ordered unweighted test events for r05/r06/r07/r08/r09; window N='+str(sum(z['n'] for z in stats))+'. Both-target truths and event-ID/sample fingerprint verified against all scorecards. Full paths, input/result SHA256 and configurations in cache/'+h+'_verified.json.\n'
  'Errors are central 68% intervals from 2000 nonparametric bootstrap resamples/bin, seed [1,sample,bin], identical event indices across models. Statistical only, excludes training-seed variation. r05–r07 intervals reused from verified Nov6 manifest (rounding ≤0.00005 percentage point); r08/r09 use the identical index-generation implementation.\n'
  'r07 means the original training, NOT the four-run mean. r08/r09 are each seed 1 and no input noise; older runs had no fixed training seed. Comparisons therefore do not isolate weighting alone. No significance cutoff or v2 peak metric adopted.\n'
  'Standard aggregate metrics (both targets, within10%, unbinned frozen22% joint circle) are in standard_metrics.csv/.json/.md; per-bin counts and intervals in manifest.\n')
 # publish() source normally points to talk12; add the actual performance driver as a second source.
 t.publish(h,f'r05-r09-s{s}-bias-resolution',fig,cap,{'models':meta['models'],'test_fingerprint_sha256':meta['test_fingerprint_sha256'],
 'event_alignment':meta['event_alignment'],'bins_gev':bins.tolist(),'per_bin':stats,'bootstrap':{'n':2000,'seed':[1,s,'bin'],'joint_across_models':True,'old_manifest':t.source(oldp)},
 'performance_driver':t.source(__file__),'sample':s,'n_window':sum(z['n'] for z in stats),'claim':'Standard bias/resolution on identical test events'},'transformeree/reconstruction')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--horn',required=True);p.add_argument('--sample',type=int,default=3);a=p.parse_args();main(a.horn,a.sample)
