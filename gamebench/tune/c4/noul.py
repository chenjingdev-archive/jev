"""보기마다 noul(예/아니오)로 따로 묻고 "예" 확률이 가장 높은 보기를 고르는 방식의 시험 (2026-10-02, 결과 보기 전 고정).

- 문항: coords 시험(test/items-coords.jsonl)에서 게임·단계마다 파일 순서로 앞부분을 뽑은 100문항이다.
  사목과 오목이 50문항씩이고, 단계별로 13·13·12·12문항이다.
- 질문: 원래 질문과 보기 하나를 함께 주고 "제시된 답이 이 질문의 정답인가?"를 묻는다. 상태(state)는 그대로다.
  - Jev는 SDK의 Noul 질문으로 묻는다.
  - Jeff는 shim의 두 보기 choice({"No / false", "Yes / true"})로 묻는다. Jeff의 noul과 같은 프롬프트다(jeff/model.py options()).
- 답: 보기마다 받은 "예" 확률이 가장 큰 보기다. 보기 순환은 없다(보기를 하나씩 따로 묻는다).
- 비교: 같은 시스템, 같은 100문항의 choice 결과(c4test-coords)와 정확도와 McNemar로 비교한다.
  100문항이라 검정력이 낮다. 이 시험은 방향을 보는 확인용이다.
- --mode score (2026-10-03 추가, 결과 보기 전 고정): 같은 100문항에서 보기마다 "제시된 답이 이 질문의 정답일 가능성은?"을
  score로 묻는다. 단계는 LEVELS 네 개(0~3)이고, 기댓값 점수가 가장 큰 보기를 고른다.
  - Jev는 SDK의 Score 질문으로 묻는다.
  - Jeff는 LEVELS를 보기로 한 choice로 묻는다. Jeff의 score와 같은 프롬프트이고, 기댓값은 Σ 단계 × 확률이다.

  op-away run op run --env-file=csat/.env.tpl -- .venv/bin/python gamebench/tune/c4/noul.py --system jev-1.13.0
  .venv/bin/python gamebench/tune/c4/noul.py --system jeff-0.8b-torch   # shim이 떠 있어야 한다
  .venv/bin/python gamebench/tune/c4/noul.py --score            # 둘 다 --mode score로 같은 순서
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'csat' / 'bench'))
import adapters  # noqa: E402

PER_STAGE = [13, 13, 12, 12]
LEVELS = ['확실히 틀렸다', '아마 틀렸다', '아마 맞다', '확실히 맞다']
SYSTEMS = ['jev-1.13.0', 'jeff-0.8b-torch']
JEFF_NOUL = {'No / false': None, 'Yes / true': None}


def sample():
    items = [json.loads(line) for line in (HERE / 'test' / 'items-coords.jsonl').open()]
    stages = sorted({it['stage'] for it in items})
    out = []
    for game in ('connect4', 'gomoku'):
        for st, k in zip(stages, PER_STAGE):
            out += [it for it in items if it['game'] == game and it['stage'] == st][:k]
    return out


def out_dir(mode):
    return ROOT / 'results' / ('c4noul' if mode == 'noul' else 'c4score')


def instructions(item, option, mode='noul'):
    ask_ = '제시된 답이 이 질문의 정답인가?' if mode == 'noul' else '제시된 답이 이 질문의 정답일 가능성은?'
    return f"{item['question']}\n\n제시된 답: {option}\n\n{ask_}"


def ask(system, state, text, mode='noul'):
    """noul이면 "예" 확률, score면 기댓값 점수(0~3)를 돌려준다."""
    if system == 'jev-1.13.0':
        from typesafe_sdk import Noul, RetryPolicy, Score, TypeSafeClient
        q = Noul(instructions=text) if mode == 'noul' else Score(instructions=text, criteria=LEVELS)
        with TypeSafeClient(timeout=60, retry=RetryPolicy(max_retries=0)) as client:
            r = client.system_one(model=system, state=adapters.as_state(state), questions={'answer': q})
        return float(r.answers['answer'].noul if mode == 'noul' else r.answers['answer'].score)
    if mode == 'noul':
        return adapters.get(system).choose(state, text, JEFF_NOUL)['Yes / true']
    probs = adapters.get(system).choose(state, text, {level: None for level in LEVELS})
    return sum(i * probs[level] for i, level in enumerate(LEVELS))


def run_one(system, item, i, mode):
    text = instructions(item, item['options'][i], mode)
    digest = hashlib.sha256(json.dumps([item['state'], text], ensure_ascii=False).encode()).hexdigest()[:12]
    target = out_dir(mode) / system / f"{item['id'].replace(':', '-')}-o{i + 1}-{digest}.json"
    if target.exists():
        return 'cached'
    for attempt in range(3):
        try:
            started = time.perf_counter()
            p = ask(system, item['state'], text, mode)
            target.write_text(json.dumps({'id': item['id'], 'option': i + 1, 'value': p,
                                          'latency_ms': round((time.perf_counter() - started) * 1000, 1)}) + '\n')
            return 'complete'
        except Exception as exc:  # noqa: BLE001
            err = f'{type(exc).__name__}: {str(exc)[:200]}'
            time.sleep(2 ** attempt)
    return 'error ' + err


def score(mode):
    from analyze import mcnemar
    from score import score_system
    items = sample()
    gold = json.loads((HERE / 'test' / 'gold.json').read_text())
    out = {}
    for s in SYSTEMS:
        p = {}
        for f in (out_dir(mode) / s).glob('*.json'):
            r = json.loads(f.read_text())
            p.setdefault(r['id'], {})[r['option']] = r.get('value', r.get('p_yes'))
        choice = {r['id']: r for r in score_system(items, gold, 'c4test-coords', s)}
        rows = []
        for it in items:
            if len(p.get(it['id'], {})) != len(it['options']):
                continue
            pick = max(p[it['id']], key=p[it['id']].get)
            rows.append({'id': it['id'], 'game': it['game'], 'stage': it['stage'], 'correct': pick == gold[it['id']]['answer'],
                         'choice_correct': choice[it['id']]['correct'], 'mean_value': statistics.mean(p[it['id']].values()),
                         'spread': max(p[it['id']].values()) - min(p[it['id']].values())})
        if not rows:
            continue
        o = {}
        groups = {'전체': lambda r: True, '사목': lambda r: r['game'] == 'connect4', '오목': lambda r: r['game'] == 'gomoku',
                  **{st: (lambda r, st=st: r['stage'] == st) for st in sorted({it['stage'] for it in items})}}
        for g, f in groups.items():
            rs = [r for r in rows if f(r)]
            b = sum(r['correct'] and not r['choice_correct'] for r in rs)
            c = sum(r['choice_correct'] and not r['correct'] for r in rs)
            o[g] = {'n': len(rs), 'per_option': round(statistics.mean(r['correct'] for r in rs), 3),
                    'choice': round(statistics.mean(r['choice_correct'] for r in rs), 3),
                    'only_per_option': b, 'only_choice': c, 'mcnemar_p': round(mcnemar(b, c), 3)}
        o['mean_value'] = round(statistics.mean(r['mean_value'] for r in rows), 3)
        o['mean_spread'] = round(statistics.mean(r['spread'] for r in rows), 3)
        out[s] = o
        for g, e in o.items():
            print(s, g, e)
    (HERE / f'results-{mode}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', choices=SYSTEMS)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--mode', choices=['noul', 'score'], default='noul')
    a = ap.parse_args()
    if a.score:
        return score(a.mode)
    (out_dir(a.mode) / a.system).mkdir(parents=True, exist_ok=True)
    jobs = [(it, i) for it in sample() for i in range(len(it['options']))]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: run_one(a.system, *j, a.mode), jobs))
    print(a.system, len(jobs), Counter(r if not r.startswith('error') else 'error' for r in res))
    errs = [r for r in res if r.startswith('error')]
    if errs:
        print(errs[:3])


if __name__ == '__main__':
    main()
