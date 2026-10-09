"""Fetch the exact official released SALF checkpoint into ignored artifacts."""
import hashlib,json,zipfile
from pathlib import Path
import gdown
ROOT=Path(__file__).resolve().parents[1]

def main():
    directory=ROOT/'artifacts/pretrained';directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'salf-cbm_models.zip'
    if not archive.exists():
        result=gdown.download('https://drive.google.com/uc?id=15uFkhV37kJjWFK3rjfcD1c9bqjdmzjFG',str(archive),quiet=False)
        if result is None:raise RuntimeError('Official checkpoint download failed')
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path=Path(item.filename)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path')
            if item.is_dir():continue
            if path.parts[:2]!=('salf-cbm_models','imagenet'):continue
            output=directory/'imagenet'/path.name;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(source.read(item))
    expected=json.loads((ROOT/'reports/entry_smoke.json').read_text())['checkpoint_sha256']
    for name,digest in expected.items():
        actual=hashlib.sha256((directory/'imagenet'/name).read_bytes()).hexdigest()
        if actual!=digest:raise RuntimeError(f'Released file changed: {name}')
    print('Official pretrained files match the pinned hashes')

if __name__=='__main__':main()
