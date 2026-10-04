"""Apply frozen Mac/AMD parity criteria to identical original CSAT requests."""
from datetime import datetime,timezone
from decimal import Decimal
import argparse,hashlib,importlib.util,json,os,statistics,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output-prefix',default='flash-migration-parity');args=parser.parse_args();prefix=args.output_prefix
sys.path.insert(0,str(ROOT/'csat/bench'));import adapters
spec=importlib.util.spec_from_file_location('parity_csat_runner',ROOT/'csat/bench/run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
plan=json.loads((HERE/'flash-migration-parity-plan.json').read_text())
questions={q['id']:q for q in json.loads((ROOT/'csat/bench/canonical-2027-09.json').read_text())['questions']}
os.environ['JEV_SHIM_PORT']='18818';adapter=adapters.get('clef-flash');health=adapter.health()
assert health['metadata']['revision']==plan['revision']
assert health['metadata']['device'].startswith('cuda')
assert health['metadata']['weight_dtype']=='torch.bfloat16'
rows=[]
previous={}
record_path=HERE/(prefix+'-records.jsonl')
if record_path.exists():
    for line in record_path.read_text().splitlines():
        old=json.loads(line)
        if old['status']=='complete':previous[(old['id'],old['rotation'])]=old
with record_path.open('a') as log:
    for i,item in enumerate(plan['requests'],1):
        if (item['id'],item['rotation']) in previous:
            rows.append(previous[(item['id'],item['rotation'])]);continue
        baseline_path=ROOT/item['baseline_path'];raw=baseline_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['baseline_file_sha256']
        baseline=json.loads(raw);mapping,req=runner.make_request(questions[item['id']],item['rotation'])
        digest=hashlib.sha256(json.dumps(req,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        assert digest==item['request_sha256']
        row={'id':item['id'],'rotation':item['rotation'],'request_sha256':digest,'device':'amd-rocm'}
        started=time.perf_counter()
        try:
            probabilities=adapter.choose(**req);meta=adapter.last_response_metadata();reference=baseline['probabilities']
            delta=max(abs(Decimal(str(probabilities[k]))-Decimal(str(reference[k]))) for k in reference)
            row.update(status='complete',probabilities=probabilities,reference_probabilities=reference,
                same_answer=max(probabilities,key=probabilities.get)==max(reference,key=reference.get),
                max_probability_delta=float(delta),tokens_equal=(meta['usage']['input_tokens']==baseline['response_metadata']['usage']['input_tokens']),
                response_metadata=meta,latency_ms=round((time.perf_counter()-started)*1000,2))
        except Exception as error:row.update(status='error',error_type=type(error).__name__,message=str(error)[:700])
        rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        print(i,row['status'],row.get('max_probability_delta'),row.get('latency_ms'),flush=True)
        if row['status']=='error':break
valid=[r for r in rows if r['status']=='complete'];agreement=sum(r['same_answer'] for r in valid)/len(valid) if valid else None;maximum=max((r['max_probability_delta'] for r in valid),default=None)
passed=(len(valid)==plan['n'] and all(r['tokens_equal'] for r in valid) and agreement>=plan['thresholds']['answer_agreement_min'] and maximum<=plan['thresholds']['max_absolute_option_probability_delta_max'])
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':hashlib.sha256((HERE/'flash-migration-parity-plan.json').read_bytes()).hexdigest(),
    'model':plan['model'],'revision':plan['revision'],'thresholds':plan['thresholds'],'expected':plan['n'],'valid':len(valid),'coverage':len(valid)/plan['n'],'completion_status':'complete' if len(valid)==plan['n'] else 'incomplete',
    'agreeing_answers':sum(r['same_answer'] for r in valid),'answer_agreement':agreement,'max_probability_delta':maximum,
    'all_token_counts_equal':all(r['tokens_equal'] for r in valid),'passed':passed,'target_health':health,
    'median_latency_ms':statistics.median(r['latency_ms'] for r in valid) if valid else None,
    'mean_latency_ms':statistics.mean(r['latency_ms'] for r in valid) if valid else None,
    'mixing_rule':'No same-suite device mixing if this test fails; thresholds remain frozen'}
(HERE/(prefix+'-result.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)
