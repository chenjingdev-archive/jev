"""Disable only the single huge allocator reservation rejected by Windows ROCm.

The official pinned encoder/model/head and all inference formulas remain intact.
"""
import importlib.util
from pathlib import Path
import transformers.modeling_utils as modeling_utils

modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
spec=importlib.util.spec_from_file_location('clef_local_server',Path(__file__).with_name('serve.py'))
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
original_init=server.Runner.__init__
def annotated_init(self,args):
    original_init(self,args)
    self.meta['compatibility_changes'].append({'kind':'allocation_only','change':'disabled transformers caching_allocator_warmup','reason':'Windows ROCm rejected one 50.96GiB empty allocation despite 53.72GiB reported free; tensor loading and inference are unchanged'})
server.Runner.__init__=annotated_init
server.main()
