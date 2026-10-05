"""CPU review-revision verification; optional installed standalone WGPU parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--natural-spec');p.add_argument('--spectral',action='store_true');p.add_argument('--public-core',action='store_true');a=p.parse_args()
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'reproduction'))
    commands=[('tests',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(output/'pytest-temp')]),('diagnostics',[sys.executable,'-m','jsr_repro.revision_diagnostics','--output',str(output/'diagnostics')]+(['--natural-spec',a.natural_spec] if a.natural_spec else [])),('real',[sys.executable,'-m','jsr_repro.real_reference','--spec','configs/revision_real_missing.json','--output',str(output/'real')])]
    if a.spectral:commands.append(('spectral',[sys.executable,'-m','jsr_repro.validate_spectral','--output',str(output/'spectral')]))
    if a.public_core:commands.append(('public-core',[sys.executable,'scripts/compare_public_core.py','--output',str(output/'public-core.json')]))
    results=[]
    for name,command in commands:
        with (output/(name+'.log')).open('w',encoding='utf8') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        results.append(dict(name=name,command=command,exit_code=result.returncode))
        (output/'commands.json').write_text(json.dumps(results,indent=2))
        if result.returncode:raise SystemExit(f'{name} failed; see {output/(name+".log")}')
        print(name+' passed',flush=True)
if __name__=='__main__':main()
