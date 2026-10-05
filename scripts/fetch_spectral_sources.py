"""Fetch the small public tables pinned by the bundled source manifest.

This explicit command downloads research-licensed numerical data, not images
or model weights. See docs/SPECTRAL_DATA.md for the separate data licenses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(output):
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root/'reproduction/jsr_repro/spectral_assets/source_manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    for name,info in record.items():
        path = output/name
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(info['url'],headers={'User-Agent':'JSR-independent-reproduction/1.0'})
            with urllib.request.urlopen(request,timeout=60) as stream:
                content = stream.read()
        if hashlib.sha256(content).hexdigest()!=info['sha256']:
            raise ValueError(f'{name}: content changed; inspect upstream and provenance before updating the lock')
        if not path.exists():
            path.write_bytes(content)
        print(name,len(content))
    (output/'sources.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/public_spectral_sources'))
    fetch(p.parse_args().output)
