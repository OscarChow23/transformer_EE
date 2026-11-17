"""12 November talk: existing predictions/weights only. No training or configuration writes.
Run python -I talk12.py prepare|weights|bridge|performance|index --horn fhc|rhc [--sample 3].
"""
import argparse, csv, hashlib, json, os, re, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent
LOOP=Path('/data/nova/chow/transformerEE-pcnumu-loop')
RUNS=Path('/data/nova/chow/pcnumu-runs')
sys.path[:0]=[str(OLD/'style'),str(LOOP),str(LOOP/'pcnumu/eval')]
import rootstyle as rs
import matplotlib.pyplot as plt
from transformer_ee.utils.weights import create_weighter
import figindex
rs.MODELS.update({
 'r04':dict(color='#7f7f7f',marker='D',mfc='none',ls=':',label='r04 (dropout 0.3)'),
 'r08':dict(color='#7b3294',marker='v',mfc='none',ls=':',label='r08'),
 'r09':dict(color='#2e7d32',marker='D',mfc='none',ls='-',label='r09'),
})
rs.use()
# Labels stay concise; configs are in captions/manifests.
RUN_NAMES={'r05':'r05_v3data-allsamples_v3features','r06':'r06_flat-alpha001','r07':'r07_s1-subsample',
 'r08':'r08_per-sample-flat','r09':'r09_per-sample-flat-r05shares'}
IDS=['run','subrun','cycle','evt','subevt']
YWEIGHT='Fraction of total training weight / 0.2 GeV'
PYTHON='/users/chow/miniforge3/envs/transformer_ee/bin/python'
FIGS=RUNS/'figures'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for ch in iter(lambda:f.read(1<<20),b''):h.update(ch)
 return h.hexdigest()
def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def source(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def cards(h):
 out={}
 for r,name in RUN_NAMES.items():
  p=RUNS/name/h/'scorecard/scorecard.json';c=json.loads(p.read_text());md=Path(c['model_dir'])
  assert 'smoke' not in str(md)
  hashes={f:sha(md/f) for f in ['input.json','result.csv']}
  assert all(hashes[f]==c['provenance']['hashes'][f] for f in hashes),(r,'stale scorecard')
  cfg=json.loads((md/'input.json').read_text())
  ep=md.parent/'effective_config.json'
  if ep.exists(): assert cfg==json.loads(ep.read_text()),(r,'effective config mismatch')
  out[r]={'card':c,'config':cfg,'scorecard':source(p),'model_dir':str(md),'hashes':hashes}
 assert len({v['card']['test_set']['fingerprint_sha256'] for v in out.values()})==1
 assert len({v['config']['data_path'] for v in out.values()})==1
 return out

def getmeta(h):return json.loads((ROOT/'cache'/f'{h}_verified.json').read_text())
def load(h):return np.load(ROOT/'cache'/f'{h}_verified.npz')

def prepare(h):
 c=cards(h);cfg=c['r05']['config'];dp=cfg['data_path'];sc=cfg['split_column']
 print(h,'projecting IDs, sample, split and two targets; no feature payload retained',flush=True)
 df=pl.scan_csv(dp,infer_schema_length=0).select(IDS+['sample',sc,'trueE','trueLepE']).collect()
 test=df.filter(pl.col(sc)=='test');train=df.filter(pl.col(sc)=='train')
 fp=hashlib.sha256(test.select(IDS+['sample']).write_csv().encode()).hexdigest()
 assert fp==c['r05']['card']['test_set']['fingerprint_sha256']
 t=test['trueE'].cast(pl.Float64).to_numpy();tl=test['trueLepE'].cast(pl.Float64).to_numpy()
 sm=test['sample'].cast(pl.Int8).to_numpy()
 arr={'t':t,'tl':tl,'s':sm,'ty':train['trueE'].cast(pl.Float64).to_numpy(),'ts':train['sample'].cast(pl.Int8).to_numpy()}
 for r,v in c.items():
  res=pl.read_csv(Path(v['model_dir'])/'result.csv')
  assert res.height==test.height
  for name,truth,key in [('trueE',t,'p'),('trueLepE',tl,'pl')]:
   rt=res['true_'+name].to_numpy();p=res['pred_'+name].to_numpy().astype(float)
   assert np.all(np.isclose(rt,truth,rtol=1e-5,atol=1e-6))
   assert np.all(np.isfinite(p))
   arr[f'{key}_{r}']=p
   if r!='r05':assert np.array_equal(rt,arr['rt_'+name]),(r,'saved truths differ row for row')
   else:arr['rt_'+name]=rt
  for s in [1,2,3]:
   m=(sm==s)&(t>=.5)&(t<=6)
   for target,pr,tr in [('trueE',arr['p_'+r],t),('trueLepE',arr['pl_'+r],tl)]:
    x=pr[m]/tr[m]-1;q=np.quantile(x,[.16,.5,.84]);ref=v['card']['metrics'][target][str(s)]
    assert m.sum()==ref['n'] and np.isclose(q[1],ref['bias'],atol=1e-10) and np.isclose((q[2]-q[0])/2,ref['resolution'],atol=1e-10)
  log=Path(v['model_dir']).parent/'train.log'
  if r in ['r05','r06','r08','r09']:
   txt=log.read_text();nn=re.findall(r'train indicies size:\s+(\d+)',txt)
   assert nn and int(nn[-1])==train.height
   assert txt.count('No NaN/null values found.')>=3
   assert 'removing them.' not in txt
  print(h,r,'hash, config, test alignment and scorecard metrics verified',flush=True)
 cache=RUNS/f'r07_s1-subsample/preview/rows_{h}.npz'
 z=np.load(cache);print('preview cache schema',z.files,flush=True)
 # Use freshly projected rows; the old lightweight cache is recorded as an inventory reference only.
 np.savez(ROOT/'cache'/f'{h}_verified.npz',**arr)
 meta={'horn':h,'models':c,'data':{'path':dp,'size':Path(dp).stat().st_size,'mtime_ns':Path(dp).stat().st_mtime_ns,
 'read_mode':'projected scalar/ID columns only; no full feature table materialized','n_train':train.height,'n_test':test.height},
 'test_fingerprint_sha256':fp,'event_alignment':'All result.csv truths match the identical ordered data test rows for both targets; IDs/sample fingerprint verified against every scorecard; saved truths identical row-for-row across models.',
 'source_files':{str(p):sha(p) for p in [LOOP/'transformer_ee/utils/weights.py',LOOP/'transformer_ee/dataloader/load.py',LOOP/'transformer_ee/dataloader/pl_dataset.py',LOOP/'pcnumu/eval/ellipse_frozen.json',OLD/'style/rootstyle.py']}}
 dump(ROOT/'cache'/f'{h}_verified.json',meta)


def publish(h,slug,fig,caption,extra,thesis='transformeree/training-weights'):
 name=f'F-20261111-{slug}-{h}';out=ROOT/h
 out.mkdir(exist_ok=True)
 assert not (out/(name+'.png')).exists(),f'immutable output exists: {name}'
 rs.save(fig,str(out),name,caption)
 manifest=out/'manifest.json';m=json.loads(manifest.read_text()) if manifest.exists() else {}
 entry={'id':name,'caption':str(out/(name+'.txt')),'png':source(out/(name+'.png')),'pdf':source(out/(name+'.pdf')),
 'script':source(__file__),'style':source(OLD/'style/rootstyle.py'),'command':f'{PYTHON} -I {__file__} '+ ' '.join(sys.argv[1:]),**extra}
 m[name]=entry;dump(manifest,m)
 fields={'title':slug.replace('-',' '),'claim':extra.get('claim','Descriptive comparison; interpretation in caption'),
 'numbers':extra.get('numbers','See machine-readable manifest and tables'), 'script':str(Path(__file__).resolve()),
 'commit':'Standalone plotting artifact; script sha256 '+sha(__file__), 'command':entry['command'],
 'inputs':str(ROOT/'cache'/f'{h}_verified.json'),'findings':str(ROOT/'HANDOFF.md'), 'thesis':thesis,'caption':caption}
 figindex.cmd_add(SimpleNamespace(src=str(out/(name+'.png')),id=name,dir=str(FIGS),force=False,field=[f'{k}={v}' for k,v in fields.items()]))
 print('READY',out/(name+'.png'),flush=True)


def binned(t,x,seed,sample,binidx):
 n=len(x);q=np.quantile(x,[.16,.5,.84]);rng=np.random.default_rng([seed,sample,binidx])
 bq=np.empty((rs.N_BOOT,3))
 for k in range(0,rs.N_BOOT,50):
  idx=rng.integers(0,n,size=(min(50,rs.N_BOOT-k),n),dtype=np.int32)
  bq[k:k+len(idx)]=np.quantile(x[idx],[.16,.5,.84],axis=1).T
 bb=bq[:,1];br=(bq[:,2]-bq[:,0])/2
 return {'n':n,'bias':float(q[1]),'res':float((q[2]-q[0])/2),
 'bias_lo':float(np.quantile(bb,.16)),'bias_hi':float(np.quantile(bb,.84)),
 'res_lo':float(np.quantile(br,.16)),'res_hi':float(np.quantile(br,.84))}


def bridge(h):
 assert h=='fhc'
 d=load(h);meta=getmeta(h)
 md=Path('/data/nova/chow/transformerEE/runs/pc-numu/fhc/pcnumu_50ep_16head_fhc_v4_drop03_20261022/model_bb9283ee26a144615e84ff7b7a1c25dd')
 cfg=json.loads((md/'input.json').read_text());assert cfg['model']['kwargs']['dropout']==.3
 res=pl.read_csv(md/'result.csv');assert res.height==125468
 sel=d['s']==3
 populations={'r04':{'t':res['true_trueE'].to_numpy(),'tl':res['true_trueLepE'].to_numpy(),
 'p':res['pred_trueE'].to_numpy(),'pl':res['pred_trueLepE'].to_numpy()},
 'r05':{k:d[k][sel] for k in ['t','tl','p_r05','pl_r05']}}
 populations['r05']['p']=populations['r05'].pop('p_r05');populations['r05']['pl']=populations['r05'].pop('pl_r05')
 summary=pl.read_csv('/data/nova/chow/transformerEE/runs/pc-numu/fhc/comparisons/r04-vs-r05_sample3/summary.csv')
 bins=np.array(rs.PERF_BINS);cent=(bins[:-1]+bins[1:])/2
 for target,truth,pred,label in [('trueE','t','p',r'E_\nu'),('trueLepE','tl','pl',r'E_\mu')]:
  cp=ROOT/'cache'/f'bridge_{target}.json'
  if cp.exists():stats=json.loads(cp.read_text())
  else:
   stats={}
   for ri,(r,pop) in enumerate(populations.items()):
    win=(pop['t']>=.5)&(pop['t']<=6);x=pop[pred]/pop[truth]-1
    q=np.quantile(x[win],[.16,.5,.84]);ref=summary.filter((pl.col('target')==target)&(pl.col('window')=='trueE 0.5-6.0 GeV')&pl.col('run').str.starts_with(r)).row(0,named=True)
    assert win.sum()==ref['n'] and np.isclose(q[1],ref['bias'],atol=1e-10) and np.isclose((q[2]-q[0])/2,ref['res'],atol=1e-10)
    st=[]
    for b,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
     m=(pop['t']>=lo)&((pop['t']<hi) if b<len(cent)-1 else (pop['t']<=hi))
     st.append(binned(pop['t'][m],x[m],1+ri,3,b) if m.sum()>=rs.MIN_PER_BIN else {'n':int(m.sum())})
    stats[r]=st;print('bridge bootstrap',target,r,'done',flush=True)
   dump(cp,stats)
  fig,(a,b)=plt.subplots(2,1,figsize=(7.4,6.7),sharex=True,gridspec_kw={'hspace':0})
  for ax,key,ylabel in [(a,'bias',f'Median residual, ${label}$ (%)'),(b,'res',f'Resolution, ${label}$ (%)')]:
   for r,off in [('r04',-.02),('r05',.02)]:
    st=stats[r];ok=np.array([p['n']>=rs.MIN_PER_BIN for p in st]);v=np.array([p.get(key,0) for p in st])*100
    lower=np.array([p.get(key+'_lo',0) for p in st])*100;upper=np.array([p.get(key+'_hi',0) for p in st])*100
    rs.model_points(ax,cent[ok],v[ok],np.vstack([np.maximum(0,v[ok]-lower[ok]),np.maximum(0,upper[ok]-v[ok])]),r,offset=off,ms=4.5)
   rs.labels(ax,None,ylabel);ax.set_xlim(.5,6)
  a.axhline(0,color='black',ls=':',lw=1);rs.tag(a,'FHC · PC test / S3 test',loc='upper right')
  b.set_ylim(bottom=0);b.legend(loc='upper right',fontsize=11);rs.labels(b,rs.E_TRUE,None);fig.align_ylabels([a,b])
  cap=(f'Top: median of r = {target}_pred/{target}_true - 1; bottom: (q84-q16)/2, in percent, versus TRUE NEUTRINO energy for both targets. Bins 0.25 GeV, 0.5–6 GeV; final bin inclusive; N<50 hidden.\n'
   'FHC: r04 uses its v2 pc_only test population (125468 total; 96721 in window); r05 uses v3 S3 test (125420 total; 96543 in window). These are different, unpaired test populations; this is not a controlled matched-event comparison or attribution to one training change.\n'
   f'r04 is the 22 October v4_drop03 model, dropout 0.3, NOT the dropout 0.1 model shown in the earlier October talk: {md}. Input and prediction SHA256 are in the manifest. r05: {meta["models"]["r05"]["model_dir"]}.\n'
   'Error bars: central 68% intervals of 2000 nonparametric bootstrap resamples per bin, independent between populations (seeds 1 and 2); test statistics only, excludes training variation. Cosmetic x shifts ±0.02 GeV.\n'
   'Aggregate statistics independently reproduce the existing comparisons/r04-vs-r05_sample3/summary.csv. Binned values and counts are in the manifest.\n'
   'RHC bridge omitted: the old RHC v2 dataset header lacks sample labels; no S3 membership is inferred.\n')
  publish(h,'r04-r05-s3-'+target.lower(),fig,cap,{'models':meta['models'],'r04':{'model_dir':str(md),'input':source(md/'input.json'),'result':source(md/'result.csv'),'config':cfg},
   'population_alignment':'different unpaired test populations','bins_gev':bins.tolist(),'per_bin':stats,'claim':'Comparison of the r04 dropout 0.3 PC test population with r05 S3','numbers':'Window N: r04 96721; r05 96543'},'transformeree/reconstruction')


def weights(h):
 d=load(h);meta=getmeta(h);y=d['ty'];s=d['ts'];models=meta['models'];rows={};tables=[]
 for r in ['unw','r05','r06','r08','r09']:
  if r=='unw':w=np.ones(len(y))
  else:
   cfg=models[r]['config'];obj=create_weighter(cfg,y,sample=s);w=np.asarray(obj.getweight(y,s) if getattr(obj,'needs_sample',False) else obj.getweight(y),float)
  assert np.isfinite(w).all() and (w>0).all();rows[r]=w
  for sm in [0,1,2,3]:
   mask=np.ones(len(y),bool) if sm==0 else s==sm;a=w[mask];ys=y[mask];peak=a[(ys>1.8)&(ys<2.2)]
   tables.append({'horn':h,'model':r,'sample':'all' if sm==0 else str(sm),'n':len(a),'sum_w':float(a.sum()),'mean_w':float(a.mean()),'weight_share':float(a.sum()/w.sum()),
    'ess_n':float(a.sum()**2/np.square(a).sum()),'ess_fraction':float(a.sum()**2/np.square(a).sum()/len(a)),
    'max_w':float(a.max()),'mean_w_peak_1p8_2p2':float(peak.mean()),'max_to_peak':float(a.max()/peak.mean()),
    'weight_share_above_6':float(a[ys>6].sum()/w.sum()),'weight_share_above_10':float(a[ys>10].sum()/w.sum())})
 for sm in [1,2,3]:
  m=s==sm;ratio=rows['r09'][m]/rows['r08'][m]
  assert np.allclose(ratio,ratio[0],rtol=1e-13)
  assert np.isclose(rows['r08'][m].mean(),1,rtol=1e-12)
  assert np.isclose(rows['r09'][m].sum()/rows['r09'].sum(),models['r09']['config']['weight']['kwargs']['per_sample'][str(sm)]['share']/sum(p['share'] for p in models['r09']['config']['weight']['kwargs']['per_sample'].values()),rtol=1e-10)
 for r in ['r08','r09']:
  log=(Path(models[r]['model_dir']).parent/'train.log').read_text()
  for sm in [1,2,3]:
   row=next(a for a in tables if a['model']==r and a['sample']==str(sm));line=next(ln for ln in log.splitlines() if ln.startswith(f'PerSampleFlatWeights sample {sm}:'))
   assert f"rows {row['n']} " in line and f"ESS {row['ess_fraction']:.1%}" in line and f"share of weight {row['weight_share']:.2%}" in line
 out=ROOT/h;out.mkdir(exist_ok=True)
 with open(out/'weight_diagnostics.csv','w') as f:
  wr=csv.DictWriter(f,fieldnames=list(tables[0]));wr.writeheader();wr.writerows(tables)
 dump(out/'weight_diagnostics.json',tables)
 bins=np.array(rs.SPEC_BINS)
 hist=lambda r,sm=None:np.histogram(y if sm is None else y[s==sm],bins,weights=rows[r] if sm is None else rows[r][s==sm])[0]/rows[r].sum()
 labels={'unw':'Unflattened','r05':'Flattened (alpha 0.05)','r06':'Flattened (alpha 0.01)'}
 styles={'unw':{'color':'#7f7f7f','ls':'--'},**{r:{'color':rs.MODELS[r]['color'],'ls':rs.MODELS[r]['ls']} for r in ['r05','r06','r08','r09']}}
 def caption(kind,rr,maxx):
  tails='; '.join(f"{r}: "+format(next(z for z in tables if z['model']==r and z['sample']=='all')[f'weight_share_above_{maxx}'],'.3%') for r in rr)
  return (f'{kind}. True neutrino energy; displayed 0–{maxx} GeV, 0.2 GeV bins. Each bin is SUM OF EVENT WEIGHTS divided by total training weight over the FULL energy range. With weight 1 this equals event fractions; weighted curves are not event fractions.\n'
   f'{h.upper()} train split, N={len(y):,}; S1/S2/S3 counts: '+', '.join(str(int((s==k).sum())) for k in [1,2,3])+'. No subsampling for these models.\n'
   'Models and exact effective weight blocks, input/result hashes, source code hashes and data path are in cache/'+h+'_verified.json and manifest.json; maps fitted on actual train trueE/sample rows with create_weighter, same function used by loader. Train logs verify no NaN/null removals, matching row counts and rounded r08/r09 weight summaries.\n'
   'Overflow share of full training weight above displayed range: '+tails+'. Denominator includes this overflow; no renormalization to visible range.\n'
   'Deterministic training histograms; no uncertainty bars. ESS=(sum w)^2/sum(w^2), weight-concentration diagnostic, not an independent-event claim. weight_diagnostics.csv/.json give absolute ESS, ESS/N, shares, max w and max w/mean w at 1.8<trueE<2.2 GeV for each sample.\n'
   'r08 and r09 have identical within-sample shapes: all r09/r08 event-weight ratios are constant separately in S1/S2/S3. r09 changes their total contributions. No choice of flattening is declared optimal.\n')
 baseextra={'provenance':str(ROOT/'cache'/f'{h}_verified.json'),'models':models,'weight_diagnostics':tables,'bins_gev':bins.tolist(),'denominator':'each recipe full-range total train sum_w','n_train':len(y)}
 # Main-range overview and component panels; same normalization on all four panels.
 for maxx in [6,10]:
  fig,ax=rs.figure(size=(7.5,4.8))
  for r in labels:ax.stairs(hist(r),bins,baseline=None,label=labels[r],**styles[r])
  ax.set_xlim(0,maxx);ax.set_ylim(bottom=0);rs.labels(ax,rs.E_TRUE,YWEIGHT);rs.tag(ax,h.upper()+' · train',loc='upper right');ax.legend(loc='upper center',fontsize=11,bbox_to_anchor=(.60,.99))
  publish(h,'pooled-weights-'+('main' if maxx==6 else 'backup'),fig,caption('Combined unflattened, pooled alpha 0.05 and pooled alpha 0.01 spectra',list(labels),maxx),baseextra)
 fig,axes=plt.subplots(3,1,figsize=(7.5,8),sharex=True,gridspec_kw={'hspace':.08})
 for sm,ax in zip([1,2,3],axes):
  for r in labels:ax.stairs(hist(r,sm),bins,baseline=None,label=labels[r],**styles[r])
  ax.set_xlim(0,6);ax.set_ylim(bottom=0);rs.labels(ax,None,'Weight fraction / 0.2 GeV');rs.tag(ax,h.upper()+f' · S{sm}',loc='upper right')
 axes[0].legend(loc='upper center',fontsize=10,bbox_to_anchor=(.59,.99));rs.labels(axes[-1],rs.E_TRUE,None);fig.align_ylabels(axes)
 publish(h,'pooled-weight-components',fig,caption('S1/S2/S3 contributions, each divided by the same FULL POOLED total weight of its recipe (not unit-normalized per sample)',list(labels),6),baseextra)
 # r08/r09 per sample contributions preserve total shares.
 fig,axes=plt.subplots(3,1,figsize=(7.5,8),sharex=True,gridspec_kw={'hspace':.08})
 for sm,ax in zip([1,2,3],axes):
  for r in ['r08','r09']:ax.stairs(hist(r,sm),bins,baseline=None,label=r,**styles[r])
  ax.set_xlim(0,6);ax.set_ylim(bottom=0);rs.labels(ax,None,'Weight fraction / 0.2 GeV');rs.tag(ax,h.upper()+f' · S{sm}',loc='upper right')
 axes[0].legend(loc='upper center');rs.labels(axes[-1],rs.E_TRUE,None);fig.align_ylabels(axes)
 publish(h,'r08-r09-sample-contributions',fig,caption('r08/r09 contributions by sample; area over full range equals sample weight share',['r08','r09'],6),baseextra)
 # Within sample, r08 and r09 overlap exactly; draw one curve with explicit legend.
 fig,axes=plt.subplots(3,1,figsize=(7.5,8),sharex=True,gridspec_kw={'hspace':.08})
 for sm,ax in zip([1,2,3],axes):
  m=s==sm;st=rs.SAMPLES[sm]
  for r in ['r08','r09']:assert np.allclose(hist(r,sm)/(rows[r][m].sum()/rows[r].sum()),hist('r08',sm)/(rows['r08'][m].sum()/rows['r08'].sum()))
  vals=np.histogram(y[m],bins,weights=rows['r08'][m])[0]/rows['r08'][m].sum()
  ax.stairs(vals,bins,baseline=None,color=st['line'],label='r08 = r09 (within sample)');ax.set_xlim(0,6);ax.set_ylim(bottom=0)
  rs.labels(ax,None,'Within-sample weight fraction\n/ 0.2 GeV');rs.tag(ax,h.upper()+f' · S{sm}',loc='upper right')
 axes[0].legend(loc='upper center',fontsize=10);rs.labels(axes[-1],rs.E_TRUE,None);fig.align_ylabels(axes)
 cap=caption('r08/r09 within-sample shapes (identical)',['r08','r09'],6)
 cap=cap.replace('Each bin is SUM OF EVENT WEIGHTS divided by total training weight over the FULL energy range.','Each panel is SUM OF EVENT WEIGHTS divided by that SAMPLE\'S full-range sum of weights (not pooled); r08 and r09 overlap exactly.').replace('Denominator includes this overflow; no renormalization to visible range.','The quoted overflow is pooled; sample-specific overflow / sample share is available in weight_diagnostics.csv. No renormalization to visible range.')
 publish(h,'r08-r09-within-sample-shapes',fig,cap,{**baseextra,'denominator':'each sample full-range sum_w'})
 print(h,'weight figures and diagnostics finished',flush=True)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','weights','bridge','performance','index']);ap.add_argument('--horn',choices=['fhc','rhc'],default='fhc');ap.add_argument('--sample',type=int,default=3);a=ap.parse_args()
 if a.action=='index':figindex.cmd_index(SimpleNamespace(dir=str(FIGS)))
 else:globals()[a.action](a.horn) if a.action!='performance' else performance(a.horn,a.sample)
if __name__=='__main__':main()
