"""Run one local model through the four fixed existing runners, then score each.

This is a sequential command launcher, with no task queue or workers of its own.
Existing per-request cache files provide resumption. Paid APIs are never used.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=['clef', 'clef-flash'], required=True)
    parser.add_argument('--device-label', required=True)
    parser.add_argument('--suites', nargs='+', choices=['csat', 'game', 'lang-en', 'claims'], default=['csat', 'game', 'lang-en', 'claims'])
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    if manifest.get('paid_api_permitted') is not False or manifest.get('execution_mode') != 'local_public_weights':
        raise RuntimeError('This launcher only permits the local public-weights manifest')
    if args.system not in manifest.get('active_systems', manifest['systems']):
        raise RuntimeError(f'{args.system} is not active; its benchmark was cancelled or disabled')
    env = {**os.environ, 'JEV_BENCH_DEVICE': args.device_label, 'PYTHONUNBUFFERED': '1'}
    status_path = HERE / f'{args.system}-run-status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    state = {'system': args.system, 'device_label': args.device_label, 'pid': os.getpid(),
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_suites': previous.get('completed_suites', []),
             'blocked_suites': previous.get('blocked_suites', {}), 'api_cost_usd': 0}
    def save(**update):
        state.update(update, updated_at_utc=datetime.now(timezone.utc).isoformat())
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temp.replace(status_path)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    for suite in args.suites:
        if suite == 'claims' and manifest.get('claims_auto_start_permitted') is False:
            save(status='blocked_claims_verification', suite=suite, reason='Long-input validation has not passed; no claims calls started')
            return 2
        entry = manifest['suites'][suite]
        if shutil.disk_usage(ROOT).free < 8 * 1024**3:
            save(status='stopped_disk_reserve', suite=suite)
            return 2
        command = [sys.executable, '-u', 'csat/bench/run.py' if suite == 'csat' else 'gamebench/run.py',
                   '--system', args.system, '--dataset' if suite == 'csat' else '--items', entry['path'],
                   '--tag', entry['tag'], '--workers', '1']
        save(status='running', suite=suite, command=command)
        with (HERE / 'logs' / f'{args.system}-{suite}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save(status='stopped_execution_error', suite=suite, exit_code=result.returncode)
            return result.returncode
        save(status='scoring', suite=suite)
        with (HERE / 'logs' / f'{args.system}-{suite}-score.log').open('a') as log:
            result = subprocess.run([sys.executable, str(HERE / 'score.py'), '--suite', suite,
                                     '--systems', args.system], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save(status='stopped_scoring_error', suite=suite, exit_code=result.returncode)
            return result.returncode
        if suite not in state['completed_suites']:
            state['completed_suites'].append(suite)
        state['blocked_suites'].pop(suite, None)
        save(status='suite_complete', suite=suite)
    remaining = [suite for suite in manifest['suites'] if suite not in state['completed_suites']]
    save(status='partial_complete' if remaining else 'complete', remaining_suites=remaining)
    return 0


if __name__ == '__main__':
    sys.exit(main())
