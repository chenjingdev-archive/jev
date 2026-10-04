"""Jeff-0.8B를 판 게임 문항으로 추가 학습한다 (tune/PROTOCOL.md 3절). Jeff 저장소의 .venv에서 돌린다.

Jeff 원래 학습기(jeff/train.py)는 optimizer 상태를 CPU 메모리에 두는데, amd는 시스템 RAM이 16GB라 그대로 쓸 수 없다.
그래서 같은 모델 클래스와 손실(정답 보기의 교차 엔트로피)로 짧은 학습 루프를 따로 둔다.
가중치와 AdamW 상태는 GPU에 fp32로 두고, forward·backward만 bf16 autocast로 돈다(bf16 가중치에 작은 lr 갱신이
반올림으로 사라지는 것을 막는다).

  JEFF_DEVICE=cuda python train.py --init checkpoints/jeff-0.8b --data tune-data --out checkpoints/jeff-0.8b-board
"""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F

from jeff.models import device_from_environment, load_decision_model


def rows(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf-8')]


def label_index(r):
    return list(r['question']['criteria']).index(r['label'])


def batch_loss(model, batch):
    prepared = model.prepare(batch)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(prepared)
    target = torch.tensor([label_index(r) for r in batch], device=logits.device)
    return F.cross_entropy(logits.float(), target, reduction='sum'), (logits.argmax(-1) == target).sum().item()


@torch.no_grad()
def evaluate(model, dev, micro):
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(dev), micro):
        l, c = batch_loss(model, dev[i:i + micro])
        loss += l.item(); correct += c
    model.train()
    return loss / len(dev), correct / len(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--micro', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--stop-after', type=int)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    train, dev = rows(Path(a.data) / 'train.jsonl'), rows(Path(a.data) / 'dev.jsonl')
    model = load_decision_model(checkpoint=a.init, device=device_from_environment(), train=True)
    model.float()
    # max_options는 모델 가중치가 아니라 저장 때 넘기는 메타데이터라 초기 체크포인트 값을 이어받는다
    init_config = json.loads((Path(a.init) / 'decision_config.json').read_text(encoding='utf-8'))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    order = [i for e in range(a.epochs) for i in random.Random(a.seed + e).sample(range(len(train)), len(train))]
    total = len(order) // a.batch
    if a.stop_after:
        total = min(total, a.stop_after)
    warmup = max(1, int(0.05 * total))
    log = Path(a.out + '.log.jsonl')
    log.parent.mkdir(parents=True, exist_ok=True)
    write = lambda d: (log.open('a', encoding='utf-8').write(json.dumps(d) + '\n'), print(json.dumps(d), flush=True))
    dl, da = evaluate(model, dev, a.micro)
    write({'step': 0, 'dev_loss': dl, 'dev_acc': da})
    best = (dl, 0)
    started = time.monotonic()
    model.train()
    for step in range(1, total + 1):
        group = [train[i] for i in order[(step - 1) * a.batch:step * a.batch]]
        lr = a.lr * (step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad()
        tl = tc = 0.0
        for i in range(0, len(group), a.micro):
            l, c = batch_loss(model, group[i:i + a.micro])
            (l / len(group)).backward()
            tl += l.item(); tc += c
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if step % 10 == 0:
            write({'step': step, 'loss': tl / len(group), 'acc': tc / len(group), 'lr': lr, 'grad_norm': gn,
                   'elapsed_min': round((time.monotonic() - started) / 60, 1)})
        if step % a.eval_every == 0 or step == total:
            dl, da = evaluate(model, dev, a.micro)
            write({'step': step, 'dev_loss': dl, 'dev_acc': da})
            if dl < best[0]:
                best = (dl, step)
                # Jeff의 save는 기존 체크포인트를 덮어쓰지 않으므로 새 폴더에 쓰고 바꿔 넣는다
                tmp = Path(a.out + '.tmp')
                shutil.rmtree(tmp, ignore_errors=True)
                model.save(tmp, temperature=float(model.temperature), max_options=init_config['max_options'],
                           train_step=step, dev_loss=dl, dev_acc=da)
                shutil.rmtree(a.out, ignore_errors=True)
                tmp.rename(a.out)
    write({'done': True, 'best_step': best[1], 'best_dev_loss': best[0],
           'peak_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None})


if __name__ == '__main__':
    main()
