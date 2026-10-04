"""Use safetensors' official pread backend for Windows UMA weight loading.

Avoids full-checkpoint copy-on-write mmap commit charges. Inference, dtype and
the official encoder/model/head stay unchanged. No attention/autocast changes.
"""
import hashlib,importlib.util,json
from pathlib import Path
import torch
import safetensors
import transformers.modeling_utils as modeling_utils

original_safe_open=modeling_utils.safe_open
def pread_open(*args,**kwargs):
    kwargs['backend']='pread'
    return original_safe_open(*args,**kwargs)
modeling_utils.safe_open=pread_open
modeling_utils.caching_allocator_warmup=lambda *args,**kwargs:None

spec=importlib.util.spec_from_file_location('clef_local_server',Path(__file__).with_name('serve.py'))
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
original_init=server.Runner.__init__
def annotated_init(self,args):
    original_init(self,args)
    root=Path(args.model_path)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    parameters=dict(self.model.backbone.named_parameters())
    names=sorted(set(index)&set(parameters))
    chosen=[names[i] for i in (0,len(names)//3,2*len(names)//3,len(names)-1)]
    verified=[]
    for name in chosen:
        parameter=parameters[name]
        with safetensors.safe_open(root/index[name],framework='pt',device='cpu',backend='pread') as source:
            view=source.get_slice(name)
            for row in sorted({0,parameter.shape[0]//2,parameter.shape[0]-1}):
                expected=view[row:row+1].to(dtype=parameter.dtype)
                actual=parameter[row:row+1].detach().cpu()
                same=torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
                if not same: raise RuntimeError('Loaded backbone sample differs from pinned tensor bytes')
                verified.append({'name':name,'row':row,'values':actual.numel(),'bitwise_equal':same})
    head_parameters=dict(self.model.head.named_parameters())
    with safetensors.safe_open(root/'joint_head.safetensors',framework='pt',device='cpu',backend='pread') as source:
        name=next(k for k in source.keys() if k in head_parameters and head_parameters[k].ndim>0)
        expected=source.get_slice(name)[0:1].to(dtype=head_parameters[name].dtype)
        actual=head_parameters[name][0:1].detach().cpu()
        same=torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
        if not same: raise RuntimeError('Loaded schema-head sample differs from pinned target-dtype values')
        verified.append({'name':'head.'+name,'row':0,'values':actual.numel(),'bitwise_equal':same})
    self.meta['compatibility_changes'].extend([
        {'kind':'allocation_only','change':'disabled transformers caching_allocator_warmup'},
        {'kind':'file_io_only','change':'official safetensors pread instead of mmap for backbone shards',
         'reason':'Windows UMA full-checkpoint COW commit charge; transformers issue 48285',
         'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}])
    self.meta['sampled_weight_readback']=verified
    print({'event':'sampled_weight_readback_verified','samples':len(verified),'values':sum(x['values'] for x in verified)},flush=True)
server.Runner.__init__=annotated_init
server.main()
