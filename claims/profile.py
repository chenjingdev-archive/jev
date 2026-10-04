"""모델별 강점·약점 표 (claims/PROFILES.md). 세 벤치 결과를 한 파일로 모은다.

- claims: 세트·세부 유형별 정확도. 받지 못한 문항(413)이 있으면 받은 문항 기준 정확도도 낸다.
- CSAT: 영역별, 그림 있음/글만 정확도 (csat/bench/score.py와 같은 채점).
- 게임 v1: 판단 능력별·단계별 점수, 판만 준 문항과 보기에 사실을 붙인 문항의 정확도 (report.json).

  tar xzf archives/*.tar.gz   # 호출 기록을 푼 뒤
  .venv/bin/python claims/profile.py
"""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ['jev-1.13.0', 'kev', 'semif', 'open-jev', 'jevmlx', 'laya', 'clm', 'julia',
           'jeff-0.8b', 'jeff-2b', 'jeff-gemma4-e2b', 'laya-typed', 'd1', 'nimble', 'tev1-4b', 'tev1-0.8b']


def csat_rows():
    spec = importlib.util.spec_from_file_location('csat_score', ROOT / 'csat/bench/score.py')
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / 'csat/bench'))
    spec.loader.exec_module(mod)
    report = importlib.util.spec_from_file_location('csat_report', ROOT / 'csat/bench/report.py')
    rep = importlib.util.module_from_spec(report)
    sys.modules['score'], argv = mod, sys.argv
    sys.argv = ['report.py']
    report.loader.exec_module(rep)
    sys.argv = argv
    ds = json.loads((ROOT / 'csat/bench/canonical-2027-09.json').read_text())['questions']
    gold = json.loads((ROOT / 'csat/exams/2027-09/gold.json').read_text())
    ds = [q for q in ds if q['id'] in gold]
    out = {}
    for s in SYSTEMS:
        if not (ROOT / 'csat/bench/results/2027-09' / s).exists():
            continue
        per_item = mod.score(ds, gold, '2027-09', s, list(range(5)))[1]
        groups = defaultdict(list)
        for q in ds:
            ok = per_item[q['id']]
            groups[rep.area(q['subject'])].append(ok)
            groups['그림 있음' if q.get('has_visual') else '글만'].append(ok)
        out[s] = {k: {'n': len(v), 'accuracy': round(statistics.mean(v), 4)} for k, v in groups.items()}
    return out


def main():
    claims = json.loads((ROOT / 'gamebench/results/claims/report.json').read_text())
    v1 = json.loads((ROOT / 'gamebench/results/v1/report.json').read_text())
    out = {'claims': {}, 'csat': csat_rows(), 'v1': {}}
    for s in SYSTEMS:
        sets = {}
        for name, r in claims.get(s, {}).items():
            u = r.get('unsupported', 0)
            row = {'n': r['items'], 'accuracy': round(r['accuracy'], 4), 'unsupported': u,
                   'accuracy_received': round(r['correct'] / (r['items'] - u), 4) if r['items'] > u else None}
            if len(r.get('stages', {})) > 1:
                row['stages'] = {k: {'n': v['items'], 'accuracy': round(v['accuracy'], 4)} for k, v in r['stages'].items()}
            sets[name] = row
        out['claims'][s] = sets
        if s in v1:
            r = v1[s]
            out['v1'][s] = {'score': r['score'], 'facets': r['facets'], 'stages': r['stages'],
                            'board_only_accuracy': r['board_raw_acc'], 'board_annotated_accuracy': r['board_annotated_acc'],
                            'ece10': r['ece10']}
    (Path(__file__).resolve().parent / 'profile.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for s in SYSTEMS:
        c = out['csat'].get(s, {})
        print(s, {k: v['accuracy'] for k, v in c.items()})


if __name__ == '__main__':
    main()
