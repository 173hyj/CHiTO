#!/usr/bin/env python3
"""Replay frozen optimization inputs; do not regenerate corridors or IK."""
import argparse,csv,json,pathlib,subprocess,tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--executable',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--scene',type=int,choices=[1,2,3])
args=parser.parse_args();root=pathlib.Path(__file__).resolve().parent
args.output.mkdir(parents=True,exist_ok=False);exe=args.executable.resolve()
def path(file):
 with file.open() as f:return [[float(row[f'q{k}_rad']) for k in range(1,7)] for row in csv.DictReader(f)]
rows=[]
for scene in ([args.scene] if args.scene else [1,2,3]):
 for trial in range(5):
  for mode,arm in [('formal','split'),('shared','shared')]:
   dest=args.output/f'scene{scene}/{arm}/run_{trial}';dest.parent.mkdir(parents=True,exist_ok=True)
   with (dest.parent/f'run_{trial}.log').open('w') as log:
    subprocess.run([str(exe),str(root/f'inputs/scene{scene}/run_{trial}'),str(dest),mode],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
   actual=path(dest/'trajectory.csv');reference=path(root/f'reference/scene{scene}/{arm}/run_{trial}/trajectory.csv')
   error=max((abs(x-y) for a,b in zip(actual,reference) for x,y in zip(a,b)),default=0.) if len(actual)==len(reference) else None
   result=json.loads((dest/'result.json').read_text());old=json.loads((root/f'reference/scene{scene}/{arm}/run_{trial}/result.json').read_text())
   rows.append(dict(scene=scene,trial=trial,arm=arm,max_joint_error=error,trajectory_matches=error is not None and error<1e-8,strict_success=result['success'],historical_strict_success=bool(old['success']),status=result['status']))
   print(json.dumps(rows[-1]),flush=True)
(args.output/'verification.json').write_text(json.dumps(rows,indent=2)+'\n')
if not all(r['trajectory_matches'] and r['strict_success']==r['historical_strict_success'] for r in rows):raise SystemExit('Replay differs from historical reference; inspect verification.json')
print(f'Verified {len(rows)} paired public refinement replays')
