#!/usr/bin/env python3
"""Download checksum-verified release assets and optionally unpack sources."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(path, destination):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite source tree: ' + str(destination))
    with tarfile.open(path, 'r:gz') as archive:
        # Source snapshots must not contain links, devices, or escaping paths.
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or ':' in member.name or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe archive member: ' + member.name)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/release_assets.json').read_text())
    download_dir = ROOT / 'downloads'
    download_dir.mkdir(exist_ok=True)
    for asset in manifest['assets']:
        if args.sources_only and asset['kind'] != 'source':
            continue
        path = download_dir / asset['name']
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            print('Downloading', asset['name'], flush=True)
            with urllib.request.urlopen(asset['url'], timeout=300) as response:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if sha256(temporary) != asset['sha256']:
                raise ValueError('SHA256 mismatch: ' + asset['name'])
            temporary.replace(path)
        if sha256(path) != asset['sha256']:
            raise ValueError('SHA256 mismatch: ' + asset['name'])
        print('Verified', asset['name'], flush=True)
        if args.extract and asset['kind'] == 'source':
            unpack(path, ROOT / 'third_party' / asset['extract_to'])
    for assembled in manifest.get('assembled', []):
        parts = [download_dir / name for name in assembled['parts']]
        if args.sources_only or not all(p.exists() for p in parts):
            continue
        path = download_dir / assembled['name']
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            with temporary.open('wb') as output:
                for part in parts:
                    with part.open('rb') as source:
                        shutil.copyfileobj(source, output)
            if sha256(temporary) != assembled['sha256']:
                raise ValueError('Assembled checkpoint SHA256 mismatch')
            temporary.replace(path)
        if sha256(path) != assembled['sha256']:
            raise ValueError('Assembled checkpoint SHA256 mismatch')
        print('Verified assembled checkpoint', path.name)


if __name__ == '__main__':
    main()
