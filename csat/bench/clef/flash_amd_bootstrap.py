"""Flash-only Windows ROCm host: official weights/forward, bounded file loading."""
import ctypes,gc,hashlib,importlib.util,json,shutil,time,traceback
from ctypes import wintypes
from pathlib import Path
import psutil
import transformers.modeling_utils as modeling_utils
from sdpa_chunked import install
chunker=install(query_chunk_size=512,min_query_tokens=4096)

class Perf(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(x,wintypes.DWORD) for x in ('HandleCount','ProcessCount','ThreadCount')]
def resources(path):
    p=Perf();p.cb=ctypes.sizeof(p);fn=ctypes.windll.psapi.GetPerformanceInfo;fn.argtypes=[ctypes.POINTER(Perf),wintypes.DWORD];fn.restype=wintypes.BOOL
    if not fn(ctypes.byref(p),p.cb):raise RuntimeError('Cannot inspect Windows commit reserve')
    return {'disk_free_bytes':shutil.disk_usage(path).free,'system_committed_bytes':p.CommitTotal*p.PageSize,
            'system_commit_limit_bytes':p.CommitLimit*p.PageSize,'system_commit_available_bytes':(p.CommitLimit-p.CommitTotal)*p.PageSize,
            'physical_available_bytes':p.PhysicalAvailable*p.PageSize,'cpu_rss_bytes':psutil.Process().memory_info().rss}
original_safe_open=modeling_utils.safe_open
class TensorSlice:
    def __init__(self,owner,key):self.owner=owner;self.key=key;self.view=owner.handle.get_slice(key)
    def __getitem__(self,index):
        return self.owner.handle.get_tensor(self.key) if index is Ellipsis else self.view[index]
    def __getattr__(self,name):return getattr(self.view,name)
class PreadHandle:
    def __init__(self,*args,**kwargs):kwargs['backend']='pread';self.handle=original_safe_open(*args,**kwargs)
    def get_slice(self,key):return TensorSlice(self,key)
    def __getattr__(self,name):return getattr(self.handle,name)
    def __enter__(self):self.handle.__enter__();return self
    def __exit__(self,*args):return self.handle.__exit__(*args)
modeling_utils.safe_open=PreadHandle
modeling_utils.caching_allocator_warmup=lambda *args,**kwargs:None
spec=importlib.util.spec_from_file_location('clef_local_server',Path(__file__).with_name('serve.py'))
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
original_init=server.Runner.__init__;original_health=server.Runner.health;original_infer=server.Runner.infer
def init(self,args):
    if args.model!='clef-flash':raise RuntimeError('Only Flash is authorized; large Clef benchmark cancelled')
    print(json.dumps({'event':'resources_before_load',**resources(args.model_path)}),flush=True)
    original_init(self,args)
    self.meta['hip']=server.torch.version.hip
    self.meta['compatibility_changes']=[{'kind':'file_io_only','change':'safetensors official pread/get_tensor materialization avoids COW mmap and double-buffer get_slice full materialization'},
        {'kind':'allocation_only','change':'disabled caching_allocator_warmup'},
        {'kind':'allocation_only','change':'synchronize/gc/empty_cache between completed requests; guard thresholds unchanged'}]
    self.meta['attention_chunking']=chunker.clef_sdpa_chunked_config
    self.meta['sdpa_chunked_sha256']=hashlib.sha256(Path(__file__).with_name('sdpa_chunked.py').read_bytes()).hexdigest()
    self.meta['loader_wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def health(self):
    info=original_health(self);info['resources']=resources(self.args.model_path);info['last_allocator_cleanup']=getattr(self,'last_allocator_cleanup',None);info['attention_chunking_counters']=dict(chunker.clef_sdpa_chunked_counters);return info
def cleanup(self):
    self.synchronize();gc.collect();before=resources(self.args.model_path);reserved_before=server.torch.cuda.memory_reserved()
    server.torch.cuda.empty_cache();after=resources(self.args.model_path)
    self.last_allocator_cleanup={'reserved_before':reserved_before,'reserved_after':server.torch.cuda.memory_reserved(),'commit_available_before':before['system_commit_available_bytes'],'commit_available_after':after['system_commit_available_bytes']}
    print(json.dumps({'event':'allocator_cleanup',**self.last_allocator_cleanup}),flush=True)
def infer(self,body):
    info=resources(self.args.model_path)
    if info['system_commit_available_bytes']<512*1024**2:
        cleanup(self)
        for _ in range(30):
            info=resources(self.args.model_path)
            if info['system_commit_available_bytes']>=512*1024**2:break
            time.sleep(1)
    if info['disk_free_bytes']<8*1024**3 or info['system_commit_available_bytes']<512*1024**2:
        return 507,{'error':'resource_reserve','resources':info}
    try:
        output=original_infer(self,body);cleanup(self);return output
    except (server.torch.OutOfMemoryError,MemoryError) as error:
        traceback.print_exc()
        kind=type(error).__name__;message=str(error)[:500];error.__traceback__=None
        cleanup(self)
        return 507,{'error':kind,'message':message,'resources':resources(self.args.model_path)}
server.Runner.__init__=init;server.Runner.health=health;server.Runner.infer=infer
server.main()
