"""Validate curated immutable figures; write their captions, handoff, checksums and retrieval archive.
Run only after finish_20261112.py completes. PNG/PDF bytes are never edited.
"""
import csv,hashlib,json,shutil,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import talk12 as t
from PIL import Image
ROOT=t.ROOT
CANON=Path('/data/nova/chow/talk-plots_20261106/update_20261111')
def canon(p):return str(p).replace('/.data/nova/chow','/data/nova/chow')
def main():
 selected=[];allentries={}
 for h in ['fhc','rhc']:
  path=ROOT/h/'manifest.json';m=json.loads(path.read_text());meta=t.getmeta(h)
  assert set(p.stem for p in (ROOT/h).glob('F-*.png'))==set(m),(h,'manifest incomplete')
  for fid,e in m.items():
   side=t.FIGS/(fid+'.md');sm,body=t.figindex.read_sidecar(side)
   assert sm['id']==fid
   p=ROOT/h/(fid+'.png');pdf=ROOT/h/(fid+'.pdf');cap=ROOT/h/(fid+'.txt')
   assert p.exists() and pdf.exists() and cap.exists()
   assert t.sha(p)==t.sha(t.FIGS/(fid+'.png'))==e['png']['sha256']
   assert t.sha(pdf)==e['pdf']['sha256']
   with Image.open(p) as im:assert im.width>=1000 and im.height>=1000
   assert pdf.read_bytes().startswith(b'%PDF-')
   if sm.get('status')=='superseded':
    assert sm['superseded_by'] in m
    continue
   assert not e.get('superseded_by'),fid
   # Add exact paths to every curated caption and synchronize its editable sidecar body.
   rr=['r05','r06'] if 'pooled-' in fid else ['r08','r09'] if 'r08-r09' in fid else list(t.RUN_NAMES)
   if 'r04-r05' in fid:rr=['r05']
   text=cap.read_text()
   if '\nExact data/model provenance:' not in text:
    text=text.rstrip()+'\nExact data/model provenance: data='+meta['data']['path']+'; '+ '; '.join(r+'='+meta['models'][r]['model_dir'] for r in rr)+'. Full input/result SHA256 and effective configurations in '+canon(ROOT/'cache'/f'{h}_verified.json')+'.\n'
    cap.write_text(text)
   sm.update(script=e['script']['path'],command=e['command'],commit='Standalone plotting artifact SHA256 '+e['script']['sha256'])
   t.figindex.write_sidecar(side,sm,text)
   e['caption_sha256']=t.sha(cap);e['figure_record']=canon(side)
   selected.append({'id':fid,'horn':h,'png':canon(p),'pdf':canon(pdf),'caption':canon(cap),'png_sha256':t.sha(p),'pdf_sha256':t.sha(pdf),
    'caption_sha256':t.sha(cap),'figure_record':canon(side),'provenance':canon(ROOT/'cache'/f'{h}_verified.json')})
   allentries[fid]=e
  t.dump(path,m)
 assert len(selected)==20,('curated count',len(selected))
 # All final driver/helper script hashes are checked; raw historical metadata retained separately.
 for e in allentries.values():
  for key in ['script','helper','layout_helper']:
   if key in e:assert t.sha(e[key]['path'])==e[key]['sha256'],(key,e['id'])
 selected.sort(key=lambda r:(r['horn'],r['id']))
 compact={'created':'2026-11-12','population_checks':'Both horns: r05–r09 aligned to identical ordered event IDs/sample and both-target truths; r04 bridge uses different test populations.',
 'figures':selected,'tables':{h:{x:canon(ROOT/h/x) for x in ['standard_metrics.csv','standard_metrics.json','standard_metrics.md','weight_diagnostics.csv','weight_diagnostics.json']} for h in ['fhc','rhc']},
 'source_manifests':{h:canon(ROOT/h/'manifest.json') for h in ['fhc','rhc']},'script':t.source(__file__)}
 t.dump(ROOT/'curated_manifest.json',compact)
 # Human-readable handoff serves as the figure records' evidence and laptop import instructions.
 lines=['# For Oscar — 12 November talk plots','',
 'Twenty curated PNG/PDF/caption sets, built from existing predictions and configured weights. No training, inference rerun, signed-config edit or new peak metric. Earlier plot files are preserved; superseded versions are excluded from the retrieval bundle.','',
 '## Priority figures','', '| use | exact PNG |','|---|---|']
 purpose=[('FHC October-model bridge, neutrino','r04-r05-s3-truee-v3-fhc'),('FHC October-model bridge, muon','r04-r05-s3-truelepe-v3-fhc'),('FHC pooled-weight overview','pooled-weights-main-fhc'),
 ('FHC pooled sample contributions','pooled-weight-components-v2-fhc'),('FHC full observed energy range','pooled-weights-fullrange-fhc'),('FHC r08/r09 sample shares','r08-r09-sample-contributions-v2-fhc'),
 ('FHC r08/r09 identical within-sample shapes','r08-r09-within-sample-shapes-v2-fhc'),('FHC S3 r05–r09 reconstruction','r05-r09-s3-bias-resolution-v2-fhc'),('RHC S3 r05–r09 reconstruction','r05-r09-s3-bias-resolution-v2-rhc')]
 for title,end in purpose:
  z=next(a for a in selected if a['id'].endswith(end));lines.append('| '+title+' | `'+z['png']+'` |')
 lines+=['','All other S1/S2 checks, RHC weights and 0–10 GeV backups are listed in `curated_manifest.json`. Every PNG has the same-basename vector PDF and `.txt` caption.','',
 '## Source distinction for the opening','',
 'The r04 comparison is **pcnumu_50ep_16head_fhc_v4_drop03_20261022/model_bb9283ee26a144615e84ff7b7a1c25dd**, configured dropout **0.3**. It is not the dropout 0.1 model described in the earlier October talk. The checked input.json and result.csv hashes are in the bridge manifest; checkpoint_metadata.json reports best_model epoch 25 (as stored).',
 'Its 125,468 v2 pc_only test events are different from r05 S3’s 125,420 v3 events. The 0.5–6 GeV counts are 96,721 versus 96,543. Independent bootstrap errors are used; do not call this matched events, or attribute the change to one input/training modification.',
 'The old RHC v2 dataset header has no sample column; no RHC S3 bridge is inferred.','',
 'Verified window statistics, r04 → r05: neutrino bias −0.680 → −0.687%, resolution 15.972 → 15.182%, within10% 49.009 → 51.929%; muon bias +1.859 → −0.243%, resolution 18.051 → 16.859%, within10% 47.150 → 53.058%. These independently reproduce the existing comparison summary.','',
 '## Standard S3 table','',
 'All numbers below are percent, copied from verified scorecards. r07 is the original single run, not the repeat mean. Common unweighted test events within each horn; true neutrino energy 0.5–6 GeV for both targets. Joint circle is the unbinned frozen 22% circle.','',
 '| horn | run | nu bias | nu resolution | nu within10 | mu resolution | joint22 |','|---|---|---:|---:|---:|---:|---:|']
 for h in ['fhc','rhc']:
  rows=json.loads((ROOT/h/'standard_metrics.json').read_text())
  for r in rows:
   if r['sample']=='3':lines.append('| '+h.upper()+' | '+r['model']+' | '+' | '.join(f'{r[k]:.2f}' for k in ['nu_bias_pct','nu_resolution_pct','nu_within10_pct','mu_resolution_pct','frozen22_jointcircle_pct'])+' |')
 lines+=['','Full S1/S2/S3 tables, including both-target biases, resolutions, within10 fractions and counts: `fhc/standard_metrics.csv` and `rhc/standard_metrics.csv` (also JSON/Markdown).',
 'Performance errors are test-event statistics: 2000 bootstrap resamples per bin, central 68% intervals. Paired indices across r05–r09; independent populations for the bridge. Errors do not include training variation. No old 2.8 SD significance cutoff and no v2 peak metrics. r08/r09 are each one seed (1), with no input noise; older runs did not fix the training seed.','',
 '## Weight interpretation and verification','',
 'Overview and sample-contribution y values are SUM OF WEIGHTS per 0.2 GeV divided by the full pooled training sum of weights, including all energies. Unflattened uses weight 1 and equals normalized counts. Sample panels retain their pooled contributions. Only the explicitly labelled within-sample-shape plot divides by each sample’s own full-range sum.',
 'r08/r09 spectra use the actual `create_weighter` implementation and each saved input/effective config, fitted to projected trueE/sample training rows. Train counts/no-removal logs and rounded printed weighting summaries were checked. Ratios r09/r08 are constant within each sample; ESS within each sample is unchanged. Pooled r05/r06 diagnostics agree with the independent pre-existing preview to its printed precision.',
 'FHC S3: r08 → r09 share 21.9147% → 40.3900%; ESS 235,189 (49.31%) in both. RHC S3: 27.2411% → 48.3148%; ESS 399,994 (59.98%) in both. Use the exact CSV for more digits. RHC configured shares sum to 0.9999, and the implementation normalizes their sum; the table reports actual normalized shares.',
 'FHC full training weight above 6 GeV: unflattened 5.282%, r05 25.037%, r06 45.923%. Thus a 0–6 GeV weighted plot has visible total below one. Captions provide the horn/recipe overflow. Entire observed-range backups use 0.2 GeV bins and logarithmic y; their full histograms sum to one.',
 'All per-sample and pooled ESS counts/fractions, sum/mean/max weights, sample shares, max/mean(peak weight at 1.8<trueE<2.2), and overflow shares are in `weight_diagnostics.csv/.json`. No flattening recipe is declared optimal.','',
 '## Retrieval','',
 'Bundle: `/data/nova/chow/talk-plots_20261106/update_20261111/talk12_curated_20261112.tar.gz`.',
 'It contains only the 20 curated PNG/PDF/TXT sets, their immutable figure records, tables, manifests, scalar bootstrap-stat JSON, plot scripts and shared style/fonts. It excludes event arrays, predictions, weights/checkpoints and datasets.',
 'When the laptop SSH alias authenticates:', '',
 '```bash',
 'rsync -av --progress gpu:/data/nova/chow/talk-plots_20261106/update_20261111/talk12_curated_20261112.tar.gz ./',
 'tar -xzf talk12_curated_20261112.tar.gz',
 'cd talk12_curated_20261112',
 'shasum -a 256 -c SHA256SUMS',
 '```','',
 'Alternatively rsync the ready unpacked directory `gpu:/data/nova/chow/talk-plots_20261106/update_20261111/talk12_curated_20261112/`. Do not use --delete for importing figure records. The earlier laptop public-key/password rejection is an authentication blocker; this GPU session has not validated that route or transferred the files. The laptop agent can also read this chat’s exact path list.',
 'The flat curated `figures/` folder holds PNG + editable metadata sidecars for normal laptop import. An existing ID must have the same PNG SHA256; do not overwrite a different image. Regenerated global figure index: `/data/nova/chow/pcnumu-runs/figures/INDEX.md`.','',
 '## Reproduction and boundaries','',
 'Original Nov6 source plots and SPEC/style remain unchanged. New plot scripts live in `scripts/`; new figures use dated immutable IDs. The r04, r08 and r09 styles extend the original palette at runtime; no training code or configurations were edited.',
 'Provenance is `cache/fhc_verified.json` / `cache/rhc_verified.json`: exact model/config/scorecard paths, input/result SHA256, source code hashes, full test fingerprint and data path/stat. The large CSV was only streamed for selected scalar/ID columns; no full feature table was read into memory.',
 'Standalone scripts are uncommitted artifact sources; hashes, commands and helper dependencies are recorded rather than inventing a Git commit. Rendering commands intentionally refuse existing output IDs. To redraw, choose a new ID and retain its provenance; do not rerun into immutable existing names.','']
 (ROOT/'HANDOFF.md').write_text('\n'.join(lines))
 t.figindex.cmd_index(SimpleNamespace(dir=str(t.FIGS)))
 dest=ROOT/'talk12_curated_20261112';assert not dest.exists();dest.mkdir()
 for z in selected:
  for ext in ['png','pdf','txt']:
   src=ROOT/z['horn']/(z['id']+'.'+ext);out=dest/z['horn']/src.name;out.parent.mkdir(exist_ok=True);shutil.copy2(src,out)
  for ext in ['png','md']:
   src=t.FIGS/(z['id']+'.'+ext);out=dest/'figures'/src.name;out.parent.mkdir(exist_ok=True);shutil.copy2(src,out)
 for h in ['fhc','rhc']:
  for f in ['standard_metrics.csv','standard_metrics.json','standard_metrics.md','weight_diagnostics.csv','weight_diagnostics.json']:
   shutil.copy2(ROOT/h/f,dest/h/f)
  t.dump(dest/h/'manifest.json',{k:v for k,v in allentries.items() if k.endswith('-'+h)})
 for f in ['HANDOFF.md','curated_manifest.json']:shutil.copy2(ROOT/f,dest/f)
 for p in (ROOT/'cache').glob('*.json'):
  (dest/'cache').mkdir(exist_ok=True);shutil.copy2(p,dest/'cache'/p.name)
 for p in (ROOT/'scripts').glob('*.py'):
  (dest/'scripts').mkdir(exist_ok=True);shutil.copy2(p,dest/'scripts'/p.name)
 shutil.copytree(t.OLD/'style',dest/'style',ignore=shutil.ignore_patterns('__pycache__'))
 hashes=[]
 for p in sorted(dest.rglob('*')):
  if p.is_file():hashes.append(t.sha(p)+'  '+str(p.relative_to(dest)))
 (dest/'SHA256SUMS').write_text('\n'.join(hashes)+'\n')
 archive=ROOT/'talk12_curated_20261112.tar.gz'
 with tarfile.open(archive,'x:gz') as tf:tf.add(dest,arcname=dest.name)
 with tarfile.open(archive,'r:gz') as tf:
  names={a.name for a in tf.getmembers() if a.isfile()}
  expected={dest.name+'/'+str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}
  assert names==expected
  for p in dest.rglob('*'):
   if p.is_file():assert hashlib.sha256(tf.extractfile(dest.name+'/'+str(p.relative_to(dest))).read()).hexdigest()==t.sha(p)
 report={'curated_count':len(selected),'all_registered_pngs_match':True,'all_pdf_signatures_valid':True,
 'all_recorded_script_hashes_match':True,'archive_files_verified':len(expected),'archive':canon(archive),'archive_sha256':t.sha(archive),'archive_bytes':archive.stat().st_size,
 'laptop_transfer':'not attempted; earlier laptop SSH authentication failure unresolved'}
 t.dump(ROOT/'validation.json',report);print(json.dumps(report,indent=2));print('HANDOFF',canon(ROOT/'HANDOFF.md'))
if __name__=='__main__':main()
