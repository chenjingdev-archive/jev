"""Download a pinned public Clef release once, without a duplicate HF cache."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='Cloudflare/clef-flash')
    parser.add_argument('--revision', default='17f0b0ad64efb65d273590632833508766b2aae6')
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{args.repo}/tree/{args.revision}?recursive=true'
    with urllib.request.urlopen(url) as response:
        files = json.load(response)
    (directory / 'download-manifest.json').write_text(json.dumps({
        'repo': args.repo, 'revision': args.revision, 'files': files,
    }, indent=2) + '\n')

    def download(entry):
        if entry['type'] != 'file':
            return
        name = entry['path']
        final = directory / name
        partial = directory / (name + '.part')
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() and final.stat().st_size == entry['size']:
            print(f'EXISTS {name}', flush=True)
            return
        print(f'DOWNLOAD {name} {entry["size"]}', flush=True)
        subprocess.run([
            'curl', '--fail', '--location', '--silent', '--show-error',
            '--retry', '5', '--retry-delay', '2', '--continue-at', '-',
            f'https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}',
            '--output', str(partial),
        ], check=True)
        if partial.stat().st_size != entry['size']:
            raise RuntimeError(f'{name}: expected {entry["size"]}, got {partial.stat().st_size}')
        if entry.get('lfs'):
            digest = hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest()
            if digest != entry['lfs']['oid']:
                raise RuntimeError(f'{name}: SHA256 mismatch')
        partial.rename(final)
        print(f'VERIFIED {name}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download, files))
    print('DOWNLOAD COMPLETE', flush=True)


if __name__ == '__main__':
    main()
