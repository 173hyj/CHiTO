import subprocess,os,re
if len(__import__('sys').argv)==5:
    import sys,pathlib
    exe,fixture,scene,output=sys.argv[1:]
    output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=True)
    config=output/'config.yaml';trace=output/'trace.jsonl'
    trace.write_text('')
    config.write_text(re.sub(r'(?m)^scene:.*$', 'scene: '+str(pathlib.Path(scene).resolve()),pathlib.Path(fixture).read_text()))
    env=dict(os.environ,CHITO_QP_TRACE=str(trace))
    import tempfile
    result=pathlib.Path(tempfile.mkdtemp(prefix='result-',dir=output))
    with (output/'run.log').open('w') as f:
        proc=subprocess.run([exe,'--config',str(config),'--output',str(result),'--trials','1'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    assert proc.returncode in [0,1],proc.returncode
    sys.argv=[sys.argv[0],str(trace)]
import pathlib,json,sys
rows=[json.loads(s) for s in pathlib.Path(sys.argv[1]).read_text().splitlines()]
exits=[r for r in rows if r['event']=='exit' and r['d_m']<0]
assert exits,'Regression must contain failed colliding midpoints'
revisited=[]
for e in exits:
 for r in rows:
  if r['event']=='entry' and r['call']>e['call'] and max(abs(a-b) for a,b in zip(e['m'],r['m']))<1e-12:
   revisited.append((e['call'],r['call']));break
print('Failed colliding midpoints:',len(exits),'revisited:',len(revisited),flush=True)
assert revisited,'Existing colliding repair midpoints were never optimized again'
print('PASS: existing failed collision node participates in later local optimization')
