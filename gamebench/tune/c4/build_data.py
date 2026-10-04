"""사목 한 게임 학습 실험 데이터 (c4/PROTOCOL.md 2절). 사목 생성기로 새 시드 문항을 만들고 v1과 겹치는 것을 뺀다.

  .venv/bin/python gamebench/tune/c4/build_data.py --train 1000
출력: c4/data/{train,dev}.jsonl (Jeff 학습 행 형식), c4/test/{items.jsonl,gold.json} (run.py·score.py 형식, 사목 + 오목), manifest.json
시험 → dev → 학습 순서로 뽑아 세 묶음이 서로 겹치지 않는다. 학습 묶음은 같은 시드 순서의 앞부분이라
--train을 늘려도 작은 묶음이 큰 묶음의 부분집합이 된다.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from build_data import collect, key, row  # noqa: E402

GAME = 'connect4'
NEAR = 'gomoku'          # 학습하지 않는 같은 계열 게임(한 줄 잇기). 시험에만 쓴다
NEAR_SEEDS = range(3_300_000, 3_300_100)
TEST_SEEDS = range(3_000_000, 3_000_100)
DEV_SEEDS = range(3_100_000, 3_100_100)
TRAIN_SEEDS = range(3_200_000, 3_201_000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=1000)
    ap.add_argument('--dev', type=int, default=100)
    ap.add_argument('--test', type=int, default=500)
    ap.add_argument('--near', type=int, default=300)
    a = ap.parse_args()
    v1 = HERE.parent.parent / 'sets' / 'v1'
    banned = {key(json.loads(line)) for f in ('items.jsonl', 'dev.jsonl') for line in (v1 / f).open()}
    mod = importlib.import_module(f'games.{GAME}')
    test = collect(mod, TEST_SEEDS, a.test, banned)
    near = collect(importlib.import_module(f'games.{NEAR}'), NEAR_SEEDS, a.near, banned)
    assert len(near) == a.near, len(near)
    dev = collect(mod, DEV_SEEDS, a.dev, banned)
    train = collect(mod, TRAIN_SEEDS, a.train, banned)
    assert (len(test), len(dev), len(train)) == (a.test, a.dev, a.train), (len(test), len(dev), len(train))
    random.Random(20261002).shuffle(train)
    test_all = test + near
    data, tdir = HERE / 'data', HERE / 'test'
    data.mkdir(exist_ok=True); tdir.mkdir(exist_ok=True)
    for name, rows in (('train.jsonl', train), ('dev.jsonl', dev)):
        (data / name).write_text(''.join(json.dumps(row(x), ensure_ascii=False) + '\n' for x in rows))
    hidden = ('answer', 'original_answer', 'twin_of')
    (tdir / 'items.jsonl').write_text(''.join(json.dumps({k: v for k, v in x.items() if k not in hidden}, ensure_ascii=False) + '\n'
                                              for x in test_all))
    gold = {x['id']: {k: x[k] for k in hidden if k in x} | {'n': len(x['options'])} for x in test_all}
    (tdir / 'gold.json').write_text(json.dumps(gold, ensure_ascii=False, indent=1) + '\n')
    count = lambda rows: {k: sum(1 for x in rows if x['stage'] == k) for k in sorted({x['stage'] for x in rows})}
    files = [data / 'train.jsonl', data / 'dev.jsonl', tdir / 'items.jsonl', tdir / 'gold.json']
    manifest = {'game': GAME, 'v1_keys_banned': len({key(json.loads(l)) for f in ('items.jsonl', 'dev.jsonl') for l in (v1 / f).open()}),
                'train': len(train), 'dev': len(dev), 'test': len(test), 'near_test': len(near),
                'train_by_stage': count(train), 'dev_by_stage': count(dev), 'test_by_stage': count(test),
                'near_by_stage': count(near),
                'sha256': {str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
