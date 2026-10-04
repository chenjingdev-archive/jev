"""표기를 바꿔 학습한 모델 채점 (c4/PROTOCOL-rep.md 3절). 기준선은 학습 전 Jeff에 같은 표기로 준 결과다.
grid로 학습한 jeff-0.8b-c4-1k(grid 시험)와의 비교는 참고로 같은 문항끼리 적는다.

  .venv/bin/python gamebench/tune/c4/analyze_rep.py --rep stacks
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from analyze import game_mean, mcnemar  # noqa: E402
from score import distribution, load, score_system  # noqa: E402

BASE, JEV, GRID_TUNED = 'jeff-0.8b-torch', 'jev-1.13.0', 'jeff-0.8b-c4-1k'
STAGES = ['1-판정', '2-변형', '3-한수(판)', '4-앞보기(판)']
GAMES = {'T': ('사목 시험', 'connect4'), 'N': ('오목 시험', 'gomoku')}


def compare(A, B, ids):
    both = [i for i in ids if i in A and i in B]
    b = sum(A[i]['correct'] and not B[i]['correct'] for i in both)
    c = sum(B[i]['correct'] and not A[i]['correct'] for i in both)
    return {'only_tuned': b, 'only_other': c, 'mcnemar_p': mcnemar(b, c)}


def summary(R, ids):
    rs = [R[i] for i in ids if i in R]
    return {'scored': len(rs), 'accuracy': round(statistics.mean(r['correct'] for r in rs), 3) if rs else None,
            'score': round(game_mean(rs), 3) if rs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', required=True, choices=['coords', 'stacks'])
    a = ap.parse_args()
    tuned, tag = f'{GRID_TUNED}-{a.rep}', f'c4test-{a.rep}'
    gold = json.loads((HERE / 'test' / 'gold.json').read_text())
    items = [json.loads(line) for line in (HERE / 'test' / f'items-{a.rep}.jsonl').open()]
    grid_items = [json.loads(line) for line in (HERE / 'test' / 'items.jsonl').open()]
    R = {s: {r['id']: r for r in score_system(items, gold, tag, s)} for s in (JEV, BASE, tuned)}
    R['grid-tuned'] = {r['id']: r for r in score_system(grid_items, gold, 'c4test', GRID_TUNED)}
    out = {}
    for k, (name, game) in GAMES.items():
        for st in [None] + STAGES:
            ids = [it['id'] for it in items if it['game'] == game and (st is None or it['stage'] == st)]
            if not ids:
                continue
            o = {'items': len(ids), **{s: summary(R[s], ids) for s in R}}
            o['tuned_vs_base'] = compare(R[tuned], R[BASE], ids)
            o['tuned_vs_grid_tuned'] = compare(R[tuned], R['grid-tuned'], ids)
            out[f'{k} {name}' if st is None else f'{k} {st}'] = o
    # ① 판정에서 고른 답의 분포: 판을 읽지 않고 한 답만 고르는지 본다
    picks = {}
    for s, t, its in [(BASE, tag, items), (tuned, tag, items), (GRID_TUNED, 'c4test', grid_items)]:
        recs = load(t, s)
        for it in its:
            if it['stage'] == '1-판정' and recs.get(it['id']):
                d, _ = distribution(it, recs[it['id']])
                if d:
                    picks.setdefault(s, {}).setdefault(it['game'], Counter())[it['options'][max(d, key=d.get) - 1]] += 1
    out['judge_picks'] = {s: {g: dict(c) for g, c in v.items()} for s, v in picks.items()}
    (HERE / f'results-{tuned}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for g, o in out.items():
        if g == 'judge_picks':
            continue
        print(f"{g} ({o['items']}): " + ', '.join(f"{s} {o[s]['accuracy']}/{o[s]['score']}({o[s]['scored']})" for s in R)
              + f" | vs base {o['tuned_vs_base']} | vs grid-tuned {o['tuned_vs_grid_tuned']}")
    print(json.dumps(out['judge_picks'], ensure_ascii=False))


if __name__ == '__main__':
    main()
