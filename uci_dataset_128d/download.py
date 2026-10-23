"""Fetch UCI 270 from its provider and verify the published experiment input hashes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def verified_files(archive, checksums):
    result = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for filename, expected in checksums.items():
            matches = [i for i in z.infolist() if not i.is_dir() and i.filename.split('/')[-1] == filename]
            if len(matches) != 1 or matches[0].file_size > MAX_ARCHIVE_BYTES:
                raise ValueError(f'Expected exactly one bounded {filename} in official archive')
            data = z.read(matches[0])
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'SHA-256 mismatch for {filename}; no files installed')
            result[filename] = data
    return result


def install_files(files, output):
    output = Path(output)
    # Validate all conflicts before writing any files. Never overwrite differing data.
    for name, data in files.items():
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite different existing data: {target}')
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = output / name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    parser.add_argument('--archive', type=Path, help='Verify/install an already downloaded official ZIP without network access')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'checksums.json').read_text(encoding='utf-8'))
    if args.archive:
        if args.archive.stat().st_size > MAX_ARCHIVE_BYTES:
            parser.error('Archive is too large')
        archive = args.archive.read_bytes()
    else:
        request = urllib.request.Request(manifest['source_url'], headers={'User-Agent': 'GasSensorResearch/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(archive) > MAX_ARCHIVE_BYTES:
            parser.error('Archive is too large')
    files = verified_files(archive, manifest['files'])
    install_files(files, args.output)
    print(f'Verified {len(files)} official batch files. Local directory: {args.output}')
    print('Provider terms apply to the downloaded data; this project does not relicense them.')


if __name__ == '__main__':
    main()
