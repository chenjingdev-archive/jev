"""Bound host staging memory while preserving every loaded weight value.

Only safetensor materialization/allocation changes. The official model forward,
encoder, schema head, attention, dtype and probability conversion are unchanged.
Every chunked GPU copy is compared exactly with its source at the target dtype.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import torch
import transformers.modeling_utils as modeling_utils
import transformers.core_model_loading as loading

CHUNK_BYTES=32*1024**2
COUNTS={'tensors':0,'chunks':0,'bytes':0,'bitwise_verified':True}
original_materialize=loading._materialize_copy

def bounded_materialize(tensor,device=None,dtype=None):
    if device is None or torch.device(device).type!='cuda' or not hasattr(tensor,'get_shape') or dtype is None:
        return original_materialize(tensor,device,dtype)
    shape=tuple(tensor.get_shape())
    element_size=torch.empty((),dtype=dtype).element_size()
    if not shape or math.prod(shape)*element_size < 128*1024**2:
        return original_materialize(tensor,device,dtype)
    destination=torch.empty(shape,device=device,dtype=dtype)
    rows=max(1,CHUNK_BYTES//(math.prod(shape[1:])*element_size))
    for start in range(0,shape[0],rows):
        end=min(start+rows,shape[0])
        source=tensor[start:end]
        destination[start:end].copy_(source,non_blocking=False)
        actual=destination[start:end].to(device='cpu')
        expected=source.to(dtype=dtype)
        if not torch.equal(actual,expected):
            COUNTS['bitwise_verified']=False
            raise RuntimeError('Chunked weight transfer changed tensor values')
        COUNTS['chunks']+=1
        del source,actual,expected
    COUNTS['tensors']+=1
    COUNTS['bytes']+=math.prod(shape)*element_size
    print({'event':'chunked_weight_verified','shape':shape,'counts':dict(COUNTS)},flush=True)
    return destination

modeling_utils.caching_allocator_warmup=lambda *args,**kwargs:None
loading._materialize_copy=bounded_materialize
spec=importlib.util.spec_from_file_location('clef_local_server',Path(__file__).with_name('serve.py'))
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
original_init=server.Runner.__init__
def annotated_init(self,args):
    original_init(self,args)
    self.meta['compatibility_changes'].extend([{'kind':'allocation_only','change':'disabled caching_allocator_warmup after giant HIP allocation failure'},
        {'kind':'weight_transfer_only','change':'32MiB row-wise copy for weights >=128MiB; full destination allocated first',
         'verification':dict(COUNTS),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'preserved':'weight values at original target BF16, model forward, encoder, head, attention, probability conversion'}])
server.Runner.__init__=annotated_init
server.main()
