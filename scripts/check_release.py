"""Check public files, embedded notebook sources, links, and result checksums."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def public_files(root=ROOT):
    root = Path(root)
    files = [root / name for name in ('README.md', 'LICENSE', 'MANIFEST.in', 'pyproject.toml', '.gitignore', '.gitattributes')]
    for name in ('.github', 'docs', 'examples', 'notebooks', 'scripts', 'tests', 'src'):
        files += [p for p in (root / name).rglob('*') if p.is_file()
                  and not any(part == '__pycache__' or part.endswith('.egg-info') for part in p.parts)]
    manifest = json.loads((root / 'results/manifest.json').read_text())
    files += [root / path for path in manifest] + [root / 'results/manifest.json']
    return sorted(set(files))


def check(root=ROOT):
    root = Path(root)
    hashes = json.loads((root / 'results/manifest.json').read_text())
    for path, expected in hashes.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    suspicious = [r'AKIA[A-Z0-9]{16}', r'gh[pousr]_[A-Za-z0-9]{20,}', r'sk-[A-Za-z0-9_-]{32,}',
                  r'AIza[0-9A-Za-z_-]{35}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                  r'https?://[^/\s:]+:[^@\s]+@', r'/' + r'Users/[^/\s]+/', r'/' + r'home/[^/\s]+/',
                  r'/' + r'content/attention-lab-[a-z0-9]+', r'GPU-[a-f0-9]{8}-[a-f0-9-]{20,}']
    patterns = [re.compile(p) for p in suspicious]
    inspected = 0

    def scan(label, data):
        nonlocal inspected
        if data[:2] == b'PK':
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for item in archive.infolist():
                    if not item.is_dir():scan(label + ':' + item.filename, archive.read(item))
            return
        if data.startswith(b'\x89PNG'):
            return
        text = data.decode('utf-8')
        assert not any(p.search(text) for p in patterns), f'Secret or machine-specific identifier in {label}'
        inspected += 1
        if label.endswith('.ipynb'):
            notebook = json.loads(text)
            first = ''.join(next(c for c in notebook['cells'] if c['cell_type'] == 'code')['source'])
            for node in ast.parse(first).body:
                if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'PAYLOAD':
                    scan(label + ':embedded-source', base64.b64decode(ast.literal_eval(node.value.args[0])))
        if label.endswith('.md') and ':' not in label:
            text = re.sub(r'```.*?```', '', text, flags=re.S)
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if re.match(r'https?://|#', target):continue
                target = target.strip('<>').split('#')[0]
                assert (root / label).parent.joinpath(target).exists(), (label, target)
    files = public_files(root)
    for path in files:scan(str(path.relative_to(root)), path.read_bytes())
    # The result whitelist must survive Git's ignore rules even before init.
    text = (root / '.gitignore').read_text()
    assert all(f'!/results/{name}/' in text for name in ('cpu', 'mps', 'nvidia-l4', 'provenance'))
    return {'status': 'passed', 'public_files': len(files), 'text_payloads_scanned': inspected,
            'result_hashes_verified': len(hashes), 'secret_patterns_detected': 0, 'local_links': 'resolved'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
