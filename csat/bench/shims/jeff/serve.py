"""Jeff shim for the Jev-like knowledge benchmark (csat/bench/adapters contract).

Runs in ~/dev/jev-likes/jeff/.venv. Loads a released Jeff checkpoint with Jeff's own backend and answers the
way jeff.server.predict does for a request:
  - Qwen checkpoints: MLX (jeff.mlx_backend.MlxDecisionModel.decide(), what `JEFF_BACKEND=mlx jeff-serve` uses
    on Apple silicon).
  - jeff-gemma4-e2b: torch only (Jeff's MLX backend runs Qwen only), jeff.models.load_decision_model and
    (model(batch) / model.temperature).softmax(-1), on the amd box's ROCm GPU.
The request row is
    {"state": state, "question": {"type": "choice", "instructions": ..., "criteria": {A..: text}}}
No added wording; Jeff builds the prompt with the checkpoint's own chat template. Probabilities are the
checkpoint's temperature-scaled softmax, unchanged.

Refusals (413, counted as wrong by the runner), both taken from Jeff itself:
  - more options than decision_config.json "max_options" (26 for the 2026-09-28 release; jeff.server
    refuses these with 422, since codes past Z were never trained);
  - prompts over 8192 tokens (jeff.model.DecisionModel.prepare's limit; the MLX path does not check it,
    so the shim does, with the same tokenizer and chat template; the torch path raises it itself). Nothing is truncated.

MLX's buffer cache is bounded (MLX_CACHE_GB, default 2); memory housekeeping only, scores are unchanged.
"""
import argparse
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO = Path(os.path.expanduser('~/dev/jev-likes/jeff'))
COMMIT = '2c1bfce27a394ce48869951332b30aedfba3b7b0'  # firelex/jeff main, cloned 2026-09-29
CHECKPOINTS = {  # name -> (Hub repo, revision fetched 2026-09-29, local dir)
    'jeff-0.8b': ('mstrasser/Jeff-Qwen3.5-0.8B', 'd66458d54426fcf52046b896261df8909bbc8b05', 'checkpoints/jeff-0.8b'),
    'jeff-2b': ('mstrasser/Jeff-Qwen3.5-2B', '30824caa5f255df0086fecba5bdfa63374c4f758', 'checkpoints/jeff-2b'),
    'jeff-gemma4-e2b': ('mstrasser/Jeff-Gemma4-E2B', 'afcb75ae269494582aab3ec3cea0d278685a8cb2', 'checkpoints/jeff-gemma4-e2b'),
    # gamebench/tune: 0.8B를 amd torch 경로로 다시 돌린 기준선과, 판 게임으로 추가 학습한 체크포인트
    'jeff-0.8b-torch': ('mstrasser/Jeff-Qwen3.5-0.8B', 'd66458d54426fcf52046b896261df8909bbc8b05', 'checkpoints/jeff-0.8b'),
    'jeff-0.8b-board': ('local: gamebench/tune/train.py', None, 'checkpoints/jeff-0.8b-board'),
    'jeff-0.8b-c4-1k': ('local: gamebench/tune/train.py (c4)', None, 'checkpoints/jeff-0.8b-c4-1k'),
    'jeff-0.8b-c4-1k-coords': ('local: gamebench/tune/train.py (c4, coords)', None, 'checkpoints/jeff-0.8b-c4-1k-coords'),
    'jeff-0.8b-c4-1k-stacks': ('local: gamebench/tune/train.py (c4, stacks)', None, 'checkpoints/jeff-0.8b-c4-1k-stacks'),
}
TORCH = {'jeff-gemma4-e2b', 'jeff-0.8b-torch', 'jeff-0.8b-board', 'jeff-0.8b-c4-1k', 'jeff-0.8b-c4-1k-coords', 'jeff-0.8b-c4-1k-stacks'}
MAX_TOKENS = 8192

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=sorted(CHECKPOINTS), required=True)
ap.add_argument('--port', type=int, required=True)
a = ap.parse_args()

HUB, REVISION, LOCAL = CHECKPOINTS[a.model]
CKPT = REPO / LOCAL
BACKEND = 'torch' if a.model in TORCH else 'mlx'
if BACKEND == 'mlx':
    import mlx.core as mx
    from jeff.mlx_backend import MlxDecisionModel
    mx.set_cache_limit(int(float(os.environ.get('MLX_CACHE_GB', '2')) * 2**30))
    model = MlxDecisionModel(CKPT)
else:
    import torch
    from jeff.models import device_from_environment, load_decision_model
    model = load_decision_model(checkpoint=CKPT, device=device_from_environment())
CONFIG = json.loads((CKPT / 'decision_config.json').read_text())
MAX_OPTIONS = int(CONFIG['max_options'])
LOCK = threading.Lock()

HEALTH = {
    'name': a.model,
    'returns_probabilities': True,
    'max_input_tokens': MAX_TOKENS,
    'detail': {
        'commit': COMMIT, 'checkpoint': HUB, 'revision': REVISION, 'backend': BACKEND,
        'base_model': CONFIG['base_model'], 'temperature': CONFIG['temperature'], 'max_options': MAX_OPTIONS,
        'weights_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(CKPT.glob('model*.safetensors'))},
        'readout_sha256': hashlib.sha256((CKPT / 'readout.safetensors').read_bytes()).hexdigest(),
    },
}


def choose(body):
    criteria = dict(body['criteria'])
    if len(criteria) > MAX_OPTIONS:
        return 413, {'error': 'unsupported', 'reason': f'{len(criteria)} options; this checkpoint handles at most {MAX_OPTIONS}'}
    row = {'state': body['state'], 'question': {'type': 'choice', 'instructions': body['instructions'], 'criteria': criteria}}
    if BACKEND == 'torch':
        with LOCK, torch.inference_mode():
            try:
                batch = model.prepare([row])
            except ValueError as e:  # Jeff's own 8192-token refusal
                return 413, {'error': 'unsupported', 'reason': str(e)}
            probs = (model(batch) / model.temperature).softmax(-1).cpu().tolist()[0][:batch.counts[0]]
        return 200, {'probabilities': dict(zip(criteria, probs)), 'input_tokens': batch.input_tokens}
    with LOCK:
        ids = model.prompt_ids(row)
        if len(ids) > MAX_TOKENS:
            return 413, {'error': 'unsupported', 'reason': f'prompt is {len(ids)} tokens; limit {MAX_TOKENS}'}
        probs, tokens = model.decide([row])[0]
    return 200, {'probabilities': dict(zip(criteria, probs)), 'input_tokens': tokens}


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            return self._send(200, HEALTH)
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/choose':
            return self._send(404, {'error': 'not found'})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            code, obj = choose(body)
        except Exception as e:  # surfaced as 500 so the runner records an error, never a fake answer
            code, obj = 500, {'error': type(e).__name__, 'message': str(e)[:500]}
        self._send(code, obj)

    def log_message(self, fmt, *args):
        pass


if __name__ == '__main__':
    print(a.model, 'shim ready', json.dumps(HEALTH, ensure_ascii=False), flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
