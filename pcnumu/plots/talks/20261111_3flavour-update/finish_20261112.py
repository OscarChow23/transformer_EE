"""Finish interrupted Nov11 plots on Nov12; preserve existing scripts and PNG bytes.
New IDs use the actual creation date. Run --horn rhc, or --horn fhc --bridge.
"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import polish12 as p
import talk12 as t

def pub(h,oldslug,fig,cap,extra,newslug=None):
 manifest=t.ROOT/h/'manifest.json';m=json.loads(manifest.read_text())
 base=f'F-20261111-{oldslug}-{h}'
 oldid=base.replace('-'+h,'-v2-'+h) if base.replace('-'+h,'-v2-'+h) in m else base
 old=m.get(oldid,{})
 slug=newslug or oldslug+('-v3' if 'r04-r05' in oldslug else '-v2')
 name=f'F-20261112-{slug}-{h}';out=t.ROOT/h
 assert not (out/(name+'.png')).exists()
 t.rs.save(fig,str(out),name,cap)
 extra={**old,**extra}
 for k in ['id','png','pdf','caption','superseded_by']:extra.pop(k,None)
 entry={**extra,'id':name,'caption':str(out/(name+'.txt')),'png':t.source(out/(name+'.png')),'pdf':t.source(out/(name+'.pdf')),
 'script':t.source(__file__),'layout_helper':t.source(p.__file__),'helper':t.source(t.__file__),
 'command':f'{t.PYTHON} -I {__file__} --horn {h}'+(' --bridge' if h=='fhc' else '')}
 m[name]=entry
 if old:
  m[oldid]['superseded_by']=name
  side=t.FIGS/(oldid+'.md');meta,body=t.figindex.read_sidecar(side);meta.update(status='superseded',superseded_by=name);t.figindex.write_sidecar(side,meta,body)
 t.dump(manifest,m)
 fields={'title':slug.replace('-',' '),'claim':extra.get('claim','Descriptive comparison, see caption'),
 'numbers':extra.get('numbers','See standard_metrics and weight_diagnostics tables'),
 'script':str(Path(__file__).resolve()),'commit':'Standalone plotting script SHA256 '+t.sha(__file__),
 'command':entry['command'],'inputs':str(t.ROOT/'cache'/f'{h}_verified.json'),
 'findings':str(t.ROOT/'HANDOFF.md'),'thesis':'transformeree/reconstruction' if ('bias-resolution' in slug or 'r04-r05' in slug) else 'transformeree/training-weights','caption':cap}
 t.figindex.cmd_add(SimpleNamespace(src=str(out/(name+'.png')),id=name,dir=str(t.FIGS),force=False,field=[f'{k}={v}' for k,v in fields.items()]))
 print('READY',out/(name+'.png'),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--horn',required=True);a.add_argument('--bridge',action='store_true');o=a.parse_args();p.pub=pub
 if o.bridge:
  original_tag=t.rs.tag
  def bridge_tag(ax,text,*args,**kwargs):
   original_tag(ax,text,*args,**kwargs)
   if 'PC / S3' in text:ax.texts[-1].set_position((.29,.95))
  t.rs.tag=bridge_tag;p.bridge(o.horn)
 else:
  p.weights(o.horn);p.perf(o.horn)
