"""Write deterministic hashes of the curated, intentionally public result files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_manifest(root=ROOT):
    results = Path(root) / 'results'
    paths = [p for name in ['cpu', 'mps', 'nvidia-l4', 'provenance'] for p in (results / name).rglob('*') if p.is_file()]
    paths += [results / name for name in ['README.md', 'audit.json', 'mac-dependencies.txt']]
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (results / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    print(f'Wrote hashes for {len(build_manifest())} published result files.')
