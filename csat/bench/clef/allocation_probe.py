import torch,transformers,json,sys,os,ctypes
from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
session=ctypes.c_ulong();ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(),ctypes.byref(session))
in_job=ctypes.c_int();ctypes.windll.kernel32.IsProcessInJob(ctypes.windll.kernel32.GetCurrentProcess(),None,ctypes.byref(in_job))
print(json.dumps({'pid':os.getpid(),'session':session.value,'in_job':bool(in_job.value),'prefix':sys.prefix,'torch':torch.__version__,'transformers':transformers.__version__,'fraction':torch.cuda.get_per_process_memory_fraction() if hasattr(torch.cuda,'get_per_process_memory_fraction') else None,'gpu':torch.cuda.mem_get_info()}),flush=True)
values=[]
for i in range(3):
 try:
  t=torch.empty((248320,5120),device='cuda',dtype=torch.bfloat16);t.fill_(1);torch.cuda.synchronize();values.append(t)
  print(json.dumps({'n':i+1,'allocated':torch.cuda.memory_allocated(),'gpu':torch.cuda.mem_get_info()}),flush=True)
 except Exception as error:
  print(type(error).__name__,str(error),flush=True);break
