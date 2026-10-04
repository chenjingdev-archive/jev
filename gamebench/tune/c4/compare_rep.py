"""판 표기 비교 채점 (c4/PROTOCOL-rep.md 2절). 같은 시스템에서 grid 대비 각 표기의 정확도와 McNemar.

  .venv/bin/python gamebench/tune/c4/compare_rep.py [--systems jev-1.13.0 jeff-0.8b-torch]
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

TAGS = {'grid': ('c4test', 'items.jsonl'), 'coords': ('c4test-coords', 'items-coords.jsonl'),
        'stacks': ('c4test-stacks', 'items-stacks.jsonl')}
GROUPS = {'T 사목': 'connect4', 'N 오목': 'gomoku'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=['jev-1.13.0', 'jeff-0.8b-torch'])
    ap.add_argument('--out', default='results-rep.json')
    a = ap.parse_args()
    gold = json.loads((HERE / 'test' / 'gold.json').read_text())
    out = {}
    for s in a.systems:
        rows, picks = {}, {}
        for rep, (tag, f) in TAGS.items():
            items = [json.loads(line) for line in (HERE / 'test' / f).open()]
            rows[rep] = {r['id']: r for r in score_system(items, gold, tag, s)}
            recs = load(tag, s)
            # ① 판정에서 고른 답의 분포: 판을 읽지 않고 한 답만 고르는지 본다
            picks[rep] = {}
            for it in items:
                if it['stage'] != '1-판정' or not recs.get(it['id']):
                    continue
                d, st = distribution(it, recs[it['id']])
                if d:
                    picks[rep].setdefault(it['game'], Counter())[it['options'][max(d, key=d.get) - 1]] += 1
        o = {}
        for g, game in GROUPS.items():
            for rep in TAGS:
                ids = [i for i, r in rows[rep].items() if r['game'] == game]
                if not ids:
                    continue
                rs = [rows[rep][i] for i in ids]
                e = {'scored': len(rs), 'accuracy': round(statistics.mean(r['correct'] for r in rs), 3),
                     'score': round(game_mean(rs), 3), 'judge_picks': dict(picks[rep].get(game, {}))}
                if rep != 'grid':
                    both = [i for i in ids if i in rows['grid']]
                    b = sum(rows[rep][i]['correct'] and not rows['grid'][i]['correct'] for i in both)
                    c = sum(rows['grid'][i]['correct'] and not rows[rep][i]['correct'] for i in both)
                    e['vs_grid'] = {'only_rep': b, 'only_grid': c, 'mcnemar_p': mcnemar(b, c)}
                o[f'{g} {rep}'] = e
        out[s] = o
    (HERE / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for s, o in out.items():
        for k, e in o.items():
            print(s, k, e['scored'], e['accuracy'], e['score'], e.get('vs_grid', ''), e['judge_picks'])


if __name__ == '__main__':
    main()
