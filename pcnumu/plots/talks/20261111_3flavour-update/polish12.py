"""New immutable IDs for layout corrections and full-range weighting backups.
Original images/plot scripts preserved; no inference/training is performed.
"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import talk12 as t
import numpy as np
import matplotlib.pyplot as plt

def pub(h,oldslug,fig,cap,extra,newslug=None):
 oldid=f'F-20261111-{oldslug}-{h}';p=t.ROOT/h/'manifest.json';m=json.loads(p.read_text())
 old=m.get(oldid,{})
 extra={**old,**extra,'script':t.source(__file__),'helper':t.source(t.__file__),
 'command':f'{t.PYTHON} -I {__file__} --horn {h}'}
 for k in ['id','png','pdf','caption']:extra.pop(k,None)
 slug=newslug or oldslug+'-v2';newid=f'F-20261111-{slug}-{h}'
 t.publish(h,slug,fig,cap,extra,thesis='transformeree/reconstruction' if ('bias-resolution' in oldslug or 'r04-r05' in oldslug) else 'transformeree/training-weights')
 if old:
  side=t.FIGS/(oldid+'.md');meta,body=t.figindex.read_sidecar(side);meta.update(status='superseded',superseded_by=newid);t.figindex.write_sidecar(side,meta,body)
  m=json.loads(p.read_text());m[oldid]['superseded_by']=newid;t.dump(p,m)
 # Sidecar actual regeneration script, command, not just helper.
 side=t.FIGS/(newid+'.md');meta,body=t.figindex.read_sidecar(side)
 meta.update(script=str(Path(__file__).resolve()),command=extra['command'],commit='Standalone script SHA256 '+t.sha(__file__))
 t.figindex.write_sidecar(side,meta,body)

def bridge(h):
 if h!='fhc':return
 bins=np.array(t.rs.PERF_BINS);cent=(bins[:-1]+bins[1:])/2
 for target,label in [('trueE',r'E_\nu'),('trueLepE',r'E_\mu')]:
  stats=json.loads((t.ROOT/'cache'/f'bridge_{target}.json').read_text())
  fig,(a,b)=plt.subplots(2,1,figsize=(7.4,6.7),sharex=True,gridspec_kw={'hspace':0})
  for ax,key,ylabel in [(a,'bias',f'Median residual, ${label}$ (%)'),(b,'res',f'Resolution, ${label}$ (%)')]:
   for r,off in [('r04',-.02),('r05',.02)]:
    st=stats[r];ok=np.array([p['n']>=50 for p in st]);v=np.array([p.get(key,0) for p in st])*100
    lo=np.array([p.get(key+'_lo',0) for p in st])*100;hi=np.array([p.get(key+'_hi',0) for p in st])*100
    t.rs.model_points(ax,cent[ok],v[ok],np.vstack([np.maximum(0,v[ok]-lo[ok]),np.maximum(0,hi[ok]-v[ok])]),r,offset=off,ms=4.5)
   t.rs.labels(ax,None,ylabel);ax.set_xlim(.5,6)
  a.axhline(0,color='black',ls=':',lw=1);t.rs.tag(a,'FHC · PC / S3 test',loc='upper left')
  a.legend(loc='upper right',fontsize=11);b.set_ylim(bottom=0);t.rs.labels(b,t.rs.E_TRUE,None);fig.align_ylabels([a,b])
  slug='r04-r05-s3-'+target.lower();cap=(t.ROOT/h/f'F-20261111-{slug}-{h}.txt').read_text()
  pub(h,slug,fig,cap,{'layout_correction':'Legend moved clear of high-energy data; no numerical changes'})

def weights(h):
 d=t.load(h);meta=t.getmeta(h);y=d['ty'];s=d['ts'];ww={}
 for r in ['unw','r05','r06','r08','r09']:
  if r=='unw':ww[r]=np.ones(len(y))
  else:
   o=t.create_weighter(meta['models'][r]['config'],y,sample=s)
   ww[r]=np.asarray(o.getweight(y,s) if getattr(o,'needs_sample',False) else o.getweight(y),float)
 # Verify pooled old-weight diagnostics against recorded independent preview.
 prev={z['option']:z for z in t.csv.DictReader(open(t.RUNS/f'r07_s1-subsample/preview/metrics_{h}.csv')) if z['option'] in ['r05','r06']}
 for r in ['r05','r06']:
  w=ww[r];checks={'ess_frac':w.sum()**2/(w*w).sum()/len(w),'mean_w':w.mean(),'wfrac_gt10':w[y>10].sum()/w.sum()}
  checks.update({f's{k}_w_share':w[s==k].sum()/w.sum() for k in [1,2,3]})
  assert all(np.isclose(a,float(prev[r][k]),rtol=1e-4) for k,a in checks.items())
 bins=np.array(t.rs.SPEC_BINS);labels={'unw':'Unflattened','r05':'Flattened (alpha 0.05)','r06':'Flattened (alpha 0.01)'}
 styles={r:({'color':'#7f7f7f','ls':'--'} if r=='unw' else {k:t.rs.MODELS[r][k] for k in ['color','ls']}) for r in ww}
 for slug,rr,within in [('pooled-weight-components',list(labels),False),('r08-r09-sample-contributions',['r08','r09'],False),('r08-r09-within-sample-shapes',['r08'],True)]:
  fig,axes=plt.subplots(3,1,figsize=(7.5,8),sharex=True,gridspec_kw={'hspace':.10})
  for sm,ax in zip([1,2,3],axes):
   mask=s==sm
   for r in rr:
    vals=np.histogram(y[mask],bins,weights=ww[r][mask])[0]/(ww[r][mask].sum() if within else ww[r].sum())
    kw={'color':t.rs.SAMPLES[sm]['line'],'label':'r08 = r09 (within sample)'} if within else {**styles[r],'label':labels.get(r,r)}
    ax.stairs(vals,bins,baseline=None,**kw)
   ax.set_xlim(0,6);ax.set_ylim(bottom=0);t.rs.labels(ax,None,('Within-sample fraction' if within else 'Weight fraction')+'\n/ 0.2 GeV');t.rs.tag(ax,h.upper()+f' · S{sm}',loc='upper right')
  axes[0].legend(loc='upper center',fontsize=10,bbox_to_anchor=(.59,.99));t.rs.labels(axes[-1],t.rs.E_TRUE,None);fig.align_ylabels(axes)
  cap=(t.ROOT/h/f'F-20261111-{slug}-{h}.txt').read_text()
  pub(h,slug,fig,cap,{'layout_correction':'Short two-line y labels prevent adjacent-panel overlap; same bins, weights and normalization'})
 # Entire observed train energy range, same 0.2 GeV bins, log-y so the tail remains legible.
 maxx=float(np.ceil(y.max()/.2)*.2);fullbins=np.arange(0,maxx+.1,.2)
 fig,ax=t.rs.figure(size=(8,4.8));histograms={}
 for r in labels:
  vals=np.histogram(y,fullbins,weights=ww[r])[0]/ww[r].sum();assert np.isclose(vals.sum(),1,atol=1e-11)
  ax.stairs(vals,fullbins,baseline=None,label=labels[r],**styles[r]);histograms[r]=vals.tolist()
 ax.set_xlim(0,maxx);ax.set_yscale('log');t.rs.labels(ax,t.rs.E_TRUE,t.YWEIGHT);t.rs.tag(ax,h.upper()+' · train',loc='upper right');ax.legend(loc='upper center',fontsize=10)
 cap=(f'Entire observed {h.upper()} training true-neutrino-energy range, 0–{maxx:g} GeV, 0.2 GeV bins, logarithmic y. Fraction of total full-range training weight in each bin; all three full-range histograms sum to one. Zero bins are absent on the logarithmic axis.\n'
  'Unflattened is normalized event counts (weight 1); alpha 0.05/r05 and alpha 0.01/r06 are normalized sums of their actual FlatSpectraWeights, not event fractions.\n'
  f'Train N={len(y)}; no event or training-weight overflow beyond this displayed range. Exact sample counts, ESS, weight concentration and recipe details in weight_diagnostics.csv/.json and cache/{h}_verified.json.\n'
  'Pooled ESS, mean weight, sample shares and >10 GeV weight fractions agree with the independently existing r07 preview metrics (relative tolerance 1e-4, accounting for CSV rounding).\n'
  'Deterministic weighted histograms; no statistical error bars and no claim that one weighting is optimal. Full per-bin sums and provenance in manifest.\n')
 pub(h,'pooled-weights-fullrange',fig,cap,{'models':meta['models'],'bins_gev':fullbins.tolist(),'histograms':histograms,'n_train':len(y),'denominator':'recipe full train sum_w','max_trueE':float(y.max()),'overflow':0},newslug='pooled-weights-fullrange')

def perf(h):
 bins=np.array(t.rs.PERF_BINS);cent=(bins[:-1]+bins[1:])/2
 for s in [1,2,3]:
  cp=t.ROOT/'cache'/f'performance_{h}_s{s}.json'
  if not cp.exists():continue
  stats=json.loads(cp.read_text());slug=f'r05-r09-s{s}-bias-resolution';old=t.ROOT/h/f'F-20261111-{slug}-{h}.txt'
  if not old.exists():continue
  fig,(a,b)=plt.subplots(2,1,figsize=(7.5,6.8),sharex=True,gridspec_kw={'hspace':0})
  for ax,key,ylabel in [(a,'bias','Median fractional\nresidual (%)'),(b,'res','Resolution (%)')]:
   for r,off in zip(t.RUN_NAMES,np.linspace(-.04,.04,5)):
    ok=np.array([z['drawn'] for z in stats]);v=np.array([z.get(r,{}).get(key,0) for z in stats])*100
    lo=np.array([z.get(r,{}).get(key+'_lo',0) for z in stats])*100;hi=np.array([z.get(r,{}).get(key+'_hi',0) for z in stats])*100;st=t.rs.MODELS[r]
    ax.errorbar(cent[ok]+off,v[ok],yerr=np.vstack([np.maximum(0,v[ok]-lo[ok]),np.maximum(0,hi[ok]-v[ok])]),fmt=st['marker'],color=st['color'],mfc=st['mfc'],mec=st['color'],ms=4,elinewidth=1,label=r)
   t.rs.labels(ax,None,ylabel);ax.set_xlim(.5,6)
  a.axhline(0,color='black',ls=':',lw=1);t.rs.tag(a,f'{h.upper()} · S{s} · test',loc='upper right');a.legend(loc='upper center',ncol=5,fontsize=10,bbox_to_anchor=(.5,1.16),columnspacing=1.1)
  b.set_ylim(bottom=0);t.rs.labels(b,t.rs.E_TRUE,None);fig.align_ylabels([a,b])
  pub(h,slug,fig,old.read_text(),{'layout_correction':'Two-line upper y label prevents overlap; no numerical changes'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--horn',required=True);p.add_argument('--group',default='all',choices=['all','bridge','weights','performance']);a=p.parse_args()
 if a.group in ['all','bridge']:bridge(a.horn)
 if a.group in ['all','weights']:weights(a.horn)
 if a.group in ['all','performance']:perf(a.horn)
