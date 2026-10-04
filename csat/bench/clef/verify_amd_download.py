"""Verify pinned Hugging Face LFS hashes without loading tensors or copying files."""
import hashlib,json,time,urllib.request
from pathlib import Path
base=Path(__file__).resolve().parent; checkpoint=base/'checkpoint'
revision='2f3de3dd85f379784083b0814d997ab627200f0c'
with urllib.request.urlopen('https://huggingface.co/api/models/Cloudflare/clef/revision/'+revision+'?blobs=true',timeout=60) as response:
    metadata=json.load(response)
verified={}
for remote in metadata['siblings']:
    path=checkpoint/remote['rfilename']
    if not path.is_file(): raise RuntimeError('Missing pinned file '+remote['rfilename'])
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024**2),b''): digest.update(chunk)
    actual=digest.hexdigest(); expected=(remote.get('lfs') or {}).get('sha256')
    if expected and actual != expected: raise RuntimeError('Hash mismatch '+remote['rfilename'])
    verified[remote['rfilename']]={'size':path.stat().st_size,'sha256':actual,'hub_lfs_sha256':expected}
    print(json.dumps({'verified':remote['rfilename'],'size':path.stat().st_size}),flush=True)
out={'repo':'Cloudflare/clef','revision':revision,'verified_at_unix':time.time(),'files':verified,'total_bytes':sum(x['size'] for x in verified.values())}
(checkpoint/'download-manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL_PINNED_FILES_VERIFIED',flush=True)
