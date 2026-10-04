"""판 게임 특화 학습 데이터 (tune/PROTOCOL.md 2절). 게임 생성기로 새 시드 문항을 만들고 v1과 겹치는 것을 뺀다.

  .venv/bin/python gamebench/tune/build_data.py
출력: tune/data/{train,dev}.jsonl (Jeff 학습 행 형식) + manifest.json
"""
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TRAIN_GAMES = ['minesweeper', 'sudoku4', 'tictactoe', 'connect4']
PER_GAME = 2000          # 게임당 학습 문항 상한
DEV_PER_GAME = 50        # 게임당 dev 문항 (학습 중 손실 확인용)
TRAIN_SEEDS = range(1_000_000, 1_000_400)   # v1 시드(20260927)와 겹치지 않는 범위
DEV_SEEDS = range(2_000_000, 2_000_020)


def key(x):
    """상황·질문·보기 집합이 같으면 같은 문항으로 본다(보기 순서는 무시)."""
    return hashlib.sha256(json.dumps([x['state'], x['question'], sorted(map(str, x['options']))],
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def row(x):
    codes = [chr(65 + i) for i in range(len(x['options']))]
    return {'id': x['id'], 'game': x['game'], 'stage': x['stage'], 'state': x['state'],
            'question': {'type': 'choice', 'instructions': x['question'], 'criteria': dict(zip(codes, map(str, x['options'])))},
            'label': codes[x['answer'] - 1]}


def collect(mod, seeds, limit, banned):
    out = []
    for s in seeds:
        for x in mod.generate(random.Random(s), 10):
            if '주석' in x['stage']:
                continue
            k = key(x)
            if k in banned:
                continue
            banned.add(k)
            out.append({**x, 'id': f"{x['id']}@{s}"})
            if len(out) >= limit:
                return out
    return out


def main():
    v1 = HERE.parent / 'sets' / 'v1'
    banned = {key(json.loads(line)) for f in ('items.jsonl', 'dev.jsonl') for line in (v1 / f).open()}
    n_v1 = len(banned)
    train, dev = [], []
    for g in TRAIN_GAMES:
        mod = importlib.import_module(f'games.{g}')
        dev += collect(mod, DEV_SEEDS, DEV_PER_GAME, banned)        # dev를 먼저 뽑아 학습과도 겹치지 않게
        train += collect(mod, TRAIN_SEEDS, PER_GAME, banned)
    random.Random(20260930).shuffle(train)
    out = HERE / 'data'
    out.mkdir(exist_ok=True)
    for name, rows in (('train.jsonl', train), ('dev.jsonl', dev)):
        (out / name).write_text(''.join(json.dumps(row(x), ensure_ascii=False) + '\n' for x in rows))
    count = lambda rows, f: {k: sum(1 for x in rows if x[f] == k) for k in sorted({x[f] for x in rows})}
    manifest = {'train_games': TRAIN_GAMES, 'v1_keys_banned': n_v1, 'train': len(train), 'dev': len(dev),
                'train_by_game': count(train, 'game'), 'train_by_stage': count(train, 'stage'),
                'sha256': {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ('train.jsonl', 'dev.jsonl')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
