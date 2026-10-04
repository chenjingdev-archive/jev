"""판단 벤치마크 실행기. 시스템 × 문항 × 보기 순환. 정답 파일은 읽지 않는다. PROTOCOL.md.

csat/bench/run.py와 같은 어댑터(choose)를 쓰되, 보기 개수가 문항마다 다르다(2~16).
보기 순환은 min(n, 5)개의 서로 다른 순환 이동이다.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import string
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'csat' / 'bench'))
import adapters  # noqa: E402

MAX_ROTATIONS = 5
KEYS = string.ascii_uppercase


def shifts(n):
    r = min(n, MAX_ROTATIONS)
    return sorted({round(k * n / r) % n for k in range(r)})


def keys(n):
    """보기 키. 26개까지는 A~Z(기존과 같다), 넘으면 A~Z, AA, AB, ... (claims/의 Banking77 72·77개)."""
    out = list(KEYS[:n])
    for a in KEYS:
        for b in KEYS:
            if len(out) >= n:
                return out
            out.append(a + b)
    return out


def make_request(item, shift):
    n = len(item['options'])
    order = list(range(1, n + 1))
    order = order[shift:] + order[:shift]
    mapping = dict(zip(keys(n), order))
    assert len(mapping) == n, (item['id'], n)
    return mapping, {'state': item['state'], 'instructions': item['question'],
                     'criteria': {k: item['options'][i - 1] for k, i in mapping.items()}}


def request_hash(req):
    return hashlib.sha256(json.dumps(req, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run_one(adapter, item, shift, out_dir):
    mapping, req = make_request(item, shift)
    digest = request_hash(req)
    target = out_dir / f"{item['id'].replace(':', '-').replace('/', '-')}-s{shift}-{digest[:12]}.json"
    if target.exists() and json.loads(target.read_text())['status'] in ('complete', 'unsupported'):
        return 'cached'
    rec = {'id': item['id'], 'system': adapter.name, 'rotation': shift, 'choice_to_original': mapping,
           'request_sha256': digest, 'attempts': [], 'status': 'error'}
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
            rec['attempts'].append({'error_type': type(exc).__name__, 'message': str(exc)[:300]})
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
    ap.add_argument('--items', type=Path, required=True, help='items.jsonl (정답 없음)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    adapter = adapters.get(a.system)
    if hasattr(adapter, 'health'):
        adapter.execution = adapter.health()
        print('health', json.dumps(adapter.execution, ensure_ascii=False), flush=True)
    items = [json.loads(l) for l in a.items.read_text().splitlines() if l.strip()]
    if a.ids:
        items = [x for x in items if x['id'] in set(a.ids)]
    out_dir = HERE / 'results' / a.tag / a.system
    tasks = [(x, s) for x in items for s in shifts(len(x['options']))]
    print(f'{len(items)} items, {len(tasks)} calls', flush=True)
    counts = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_one, adapter, x, s, out_dir) for x, s in tasks]), 1):
            counts[f.result()] = counts.get(f.result(), 0) + 1
            if i % 200 == 0 or i == len(tasks):
                print(i, counts, flush=True)
            if getattr(adapter, 'fatal_error', None) is not None:
                print('Stopped: non-retryable execution condition', str(adapter.fatal_error), flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
    return 1 if counts.get('error') else 0


if __name__ == '__main__':
    sys.exit(main())
