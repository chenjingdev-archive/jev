"""시스템 × 문항 × 보기 순환(0~4) 호출. 재시작 가능. 정답 파일은 읽지 않는다. PROTOCOL.md 1~2절."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import adapters  # noqa: E402
from run_exam import INSTRUCTIONS  # noqa: E402

ROTATIONS = range(5)


def make_request(q, rotation):
    order = list(range(1, 6))
    order = order[rotation:] + order[:rotation]
    mapping = dict(zip('ABCDE', order))
    return mapping, {'state': q['state'], 'instructions': INSTRUCTIONS,
                     'criteria': {k: q['options'][n - 1] for k, n in mapping.items()}}


def run_one(adapter, q, rotation, out_dir):
    mapping, req = make_request(q, rotation)
    digest = hashlib.sha256(json.dumps(req, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    target = out_dir / f"{q['id'].replace(':', '-')}-r{rotation}-{digest[:12]}.json"
    if target.exists() and json.loads(target.read_text())['status'] in ('complete', 'unsupported'):
        return 'cached'
    rec = {'id': q['id'], 'system': adapter.name, 'rotation': rotation, 'choice_to_original': mapping,
           'request_sha256': digest, 'input_chars': sum(len(str(v)) for v in req['state'].values()),
           'attempts': [], 'status': 'error'}
    if os.environ.get('JEV_BENCH_DEVICE'):
        rec['device'] = os.environ['JEV_BENCH_DEVICE']
    if hasattr(adapter, 'execution'):
        rec['execution'] = adapter.execution
    for attempt in range(3):
        started = time.perf_counter()
        try:
            probs = adapter.choose(req['state'], req['instructions'], req['criteria'])
            rec.update(status='complete', latency_ms=round((time.perf_counter() - started) * 1000, 2),
                       probabilities=probs, original_probabilities={mapping[k]: v for k, v in probs.items()})
            if hasattr(adapter, 'last_response_metadata'):
                rec['response_metadata'] = adapter.last_response_metadata()
            break
        except adapters.Unsupported as exc:
            rec.update(status='unsupported', reason=str(exc)); break
        except Exception as exc:
            rec['attempts'].append({'error_type': type(exc).__name__, 'message': str(exc)[:200]})
            rec['retryable'] = adapters.retryable(exc) and getattr(adapter, 'fatal_error', None) is None
            if not rec['retryable'] or attempt == 2:
                break
            time.sleep(2 ** attempt)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp'); tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1)); tmp.replace(target)
    return rec['status']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True)
    ap.add_argument('--dataset', type=Path, default=HERE.parent / 'all_subjects' / 'dataset.json')
    ap.add_argument('--tag', default='2026-dev', help='결과 폴더 이름. 데이터셋·정본 버전마다 다르게')
    ap.add_argument('--ids', nargs='*', help='특정 문항만 (스모크 테스트용)')
    ap.add_argument('--rotations', type=int, nargs='*', default=list(ROTATIONS))
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    adapter = adapters.get(a.system)
    if hasattr(adapter, 'health'):
        adapter.execution = adapter.health()
        print('health', json.dumps(adapter.execution, ensure_ascii=False), flush=True)
    qs = json.loads(a.dataset.read_text())['questions']
    if a.ids:
        qs = [q for q in qs if q['id'] in set(a.ids)]
    out_dir = HERE / 'results' / a.tag / a.system
    tasks = [(q, r) for q in qs for r in a.rotations]
    print(f'{len(qs)} questions x {len(a.rotations)} rotations = {len(tasks)} calls', flush=True)
    counts = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_one, adapter, q, r, out_dir) for q, r in tasks]), 1):
            counts[f.result()] = counts.get(f.result(), 0) + 1
            if i % 100 == 0 or i == len(tasks):
                print(i, counts, flush=True)
            if getattr(adapter, 'fatal_error', None) is not None:
                print('Stopped: non-retryable execution condition', str(adapter.fatal_error), flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
    return 1 if counts.get('error') else 0


if __name__ == '__main__':
    sys.exit(main())
