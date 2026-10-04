"""Download exactly one pinned public model into local_dir, without a duplicate Hub cache."""
import os,json,time,shutil
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from huggingface_hub import snapshot_download
base=Path(__file__).resolve().parent
checkpoint=base/'checkpoint'
required=55_100_000_000
free=shutil.disk_usage(base).free
print(json.dumps({'phase':'starting','free_bytes':free,'estimated_download_bytes':required,'revision':'2f3de3dd85f379784083b0814d997ab627200f0c'}),flush=True)
if free < required + 5*1024**3 and not checkpoint.exists():
    raise RuntimeError('Insufficient space for one model plus a 5GiB reserve; no files removed')
started=time.time()
path=snapshot_download('Cloudflare/clef',revision='2f3de3dd85f379784083b0814d997ab627200f0c',local_dir=checkpoint,max_workers=2,token=False)
print(json.dumps({'phase':'download_complete','path':str(path),'seconds':time.time()-started,'free_bytes':shutil.disk_usage(base).free}),flush=True)
