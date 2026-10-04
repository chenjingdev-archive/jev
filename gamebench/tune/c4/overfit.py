"""한 묶음(16문항)을 반복 학습해 외울 수 있는지 본다. 학습 경로가 망가졌는지 가리는 점검."""
import json, sys, time
import torch, torch.nn.functional as F
from jeff.models import load_decision_model
data, lr, steps, dev = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows = [json.loads(l) for l in open(data, encoding='utf-8')][:16]
m = load_decision_model(checkpoint='checkpoints/jeff-0.8b', device=dev, train=True); m.float()
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
lab = lambda r: list(r['question']['criteria']).index(r['label'])
t0 = time.time()
for s in range(steps + 1):
    opt.zero_grad(); tl = tc = 0
    for i in range(0, 16, 4):
        b = rows[i:i+4]; lg = m(m.prepare(b)); y = torch.tensor([lab(r) for r in b], device=lg.device)
        l = F.cross_entropy(lg.float(), y, reduction='sum'); tl += l.item(); tc += (lg.argmax(-1) == y).sum().item()
        if s < steps: (l / 16).backward()
    if s < steps:
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
    print(json.dumps({'step': s, 'loss': round(tl/16, 4), 'acc': tc/16, 'grad_norm': round(gn, 2) if s < steps else None, 'sec': round(time.time()-t0)}), flush=True)
