"""같은 판을 다른 표기로 적는다 (c4/PROTOCOL-rep.md). 판의 내용과 규칙·질문·보기는 그대로 두고 state의 "판"과 "표기"만 바꾼다.

- grid   : 원래 격자 (바꾸지 않음)
- coords : X와 O가 놓인 칸의 좌표 목록 (사목·오목)
- stacks : 열마다 맨 아래부터 쌓인 말 (사목만)

  .venv/bin/python gamebench/tune/c4/represent.py     # test/items-<rep>.jsonl, data-<rep>/{train,dev}.jsonl
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID_NOTE = '로 적고 빈칸은 "."이다.'


def parse(board):
    lines = board.split('\n')
    cols = lines[0].split()
    cells = {}
    rows = []
    for ln in lines[1:]:
        p = ln.split()
        rows.append(p[0])
        for c, v in zip(cols, p[1:]):
            cells[p[0] + c] = v
    return rows, cols, cells


def coords(state):
    rows, cols, cells = parse(state['판'])
    pos = lambda who: ', '.join(r + c for r in rows for c in cols if cells[r + c] == who) or '없음'
    assert GRID_NOTE in state['표기'], state['표기']
    note = state['표기'].replace(GRID_NOTE, '로 된 격자다. 아래 "판"에는 X와 O가 놓인 칸의 좌표를 각각 적는다. 적히지 않은 칸은 빈칸이다.')
    return {**state, '표기': note, '판': f'X: {pos("X")}\nO: {pos("O")}'}


def stacks(state):
    rows, cols, cells = parse(state['판'])
    assert GRID_NOTE in state['표기'], state['표기']
    bottom = rows[-1]
    note = state['표기'].replace(GRID_NOTE, f'로 된 격자다. 아래 "판"에는 열마다 맨 아래({bottom}행)부터 위로 쌓인 말을 차례로 적는다. '
                                            f'한 열에는 {len(rows)}개까지 들어간다.')
    line = lambda c: ' '.join(v for v in (cells[r + c] for r in reversed(rows)) if v != '.') or '비어 있음'
    return {**state, '표기': note, '판': '\n'.join(f'{c}열: {line(c)}' for c in cols)}


REPS = {'coords': coords, 'stacks': stacks}


def convert(x, rep):
    if rep == 'stacks' and x['game'] != 'connect4':
        return None
    return {**x, 'state': REPS[rep](x['state'])}


def main():
    items = [json.loads(line) for line in (HERE / 'test' / 'items.jsonl').open()]
    for rep in REPS:
        out = [y for y in (convert(x, rep) for x in items) if y]
        (HERE / 'test' / f'items-{rep}.jsonl').write_text(''.join(json.dumps(y, ensure_ascii=False) + '\n' for y in out))
        d = HERE / f'data-{rep}'
        d.mkdir(exist_ok=True)
        for name in ('train.jsonl', 'dev.jsonl'):
            rows = [convert(json.loads(line), rep) for line in (HERE / 'data' / name).open()]
            (d / name).write_text(''.join(json.dumps(y, ensure_ascii=False) + '\n' for y in rows))
        print(rep, len(out), 'test items')


if __name__ == '__main__':
    main()
