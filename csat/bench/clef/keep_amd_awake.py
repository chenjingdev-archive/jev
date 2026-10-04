"""Keep only this inference process from idle sleep; leave display and plan alone."""
import ctypes,json,time,psutil,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);TARGET=parser.parse_args().pid
class Power(ctypes.Structure):
    _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
p=Power();ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))
parent=psutil.Process(TARGET);started=parent.create_time()
previous=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
if not previous:raise RuntimeError('Process-scoped keep-awake request failed')
state={'target_pid':TARGET,'target_create_time':started,'active':True,'system_required':True,'display_required':False,'ACLineStatus':p.ACLineStatus,'BatteryLifePercent':p.BatteryLifePercent}
file=Path(__file__).with_name('keep-awake-status.json');file.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
try:
    while psutil.pid_exists(TARGET) and psutil.Process(TARGET).create_time()==started:time.sleep(15)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);state['active']=False;file.write_text(json.dumps(state,indent=2)+'\n')
