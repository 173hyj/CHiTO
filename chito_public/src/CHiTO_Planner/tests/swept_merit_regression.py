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
import pathlib,json,math,sys
trace=pathlib.Path(sys.argv[1]);rows=[json.loads(s) for s in trace.read_text().splitlines()]
entry=next(r for r in rows if r['event']=='entry' and r['call']==1)
model=next(r for r in rows if r['event']=='model' and r['call']==1 and r['iter']==0)
# Frozen scene3 case has one active point violation (verified by legacy linear sum).
gap=max(0.,entry['d_safe']-entry['d_m'])
assert abs(entry['point_linear']-entry['mu_global']*gap)<1e-10
expected=entry['smooth']+0.5*entry['mu_local']*gap*gap+entry['edge_squared']
err=abs(entry['merit']-expected)
rel=math.sqrt(sum((x-y)**2 for x,y in zip(model['model_grad'],model['fd_merit_grad'])))/math.sqrt(sum(y*y for y in model['fd_merit_grad']))
print(json.dumps(dict(actual_merit=entry['merit'],expected_squared_merit=expected,cost_error=err,relative_gradient_error=rel)),flush=True)
assert err<1e-10,'Continuous merit does not use QP squared point penalty and local weight'
assert rel<0.03,'Actual geometric merit gradient disagrees with model at frozen regression state'
print('PASS: real scene3 continuous merit value and local gradient match squared QP objective',flush=True)
