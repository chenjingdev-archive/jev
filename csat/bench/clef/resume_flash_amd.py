"""Resume the same pinned public files directly; no duplicate Hub cache."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,subprocess,time,shutil
root=Path(__file__).resolve().parent/'checkpoint'
manifest=json.loads((root/'download-manifest.json').read_text());revision=manifest['revision']
if revision!='17f0b0ad64efb65d273590632833508766b2aae6':raise RuntimeError('Unexpected Flash revision')
files=manifest['files']
remaining=sum(max(0,f['size']-(root/f['path']).stat().st_size) if (root/f['path']).exists() else f['size'] for f in files)
if shutil.disk_usage(root).free < remaining+8*1024**3:raise RuntimeError('Insufficient disk reserve for pinned Flash copy')
print(json.dumps({'phase':'range_resume_started','remaining_bytes':remaining,'free_bytes':shutil.disk_usage(root).free}),flush=True)
def fetch(entry):
    path=root/entry['path'];size=path.stat().st_size if path.exists() else 0
    if size>entry['size']:raise RuntimeError('Oversize existing file preserved: '+entry['path'])
    if size<entry['size']:
        url='https://huggingface.co/Cloudflare/clef-flash/resolve/'+revision+'/'+entry['path']+'?download=true'
        print(json.dumps({'fetch':entry['path'],'resume_from':size,'expected':entry['size']}),flush=True)
        result=subprocess.run(['C:/Windows/System32/curl.exe','--fail','--location','--continue-at','-',
            '--retry','4','--retry-all-errors','--connect-timeout','15','--max-time','1200',
            '--silent','--show-error','--output',str(path),url])
        if result.returncode:raise RuntimeError('Download failed '+entry['path']+' exit '+str(result.returncode))
    if path.stat().st_size!=entry['size']:raise RuntimeError('Size mismatch '+entry['path'])
    expected=(entry.get('lfs') or {}).get('oid')
    if expected:
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(4*1024**2),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:raise RuntimeError('SHA mismatch, file preserved: '+entry['path'])
    print(json.dumps({'verified':entry['path'],'size':entry['size']}),flush=True)
with ThreadPoolExecutor(2) as pool:list(pool.map(fetch,files))
(root/'transfer-verified.json').write_text(json.dumps({'repo':manifest['repo'],'revision':revision,'verified_at_unix':time.time(),'all_pinned_lfs_sha256_match':True,'total_bytes':sum(f['size'] for f in files)},indent=2)+'\n')
print('FLASH_TRANSFER_VERIFIED',flush=True)
