"""Loopback-only Clef inference using the pinned official model and schema head.

Weights, tokenizer, encoder, logits and SystemOne probability conversion remain
unchanged. The preflight rejects overlength text instead of silently truncating.
Only one request is admitted to inference at a time; health remains responsive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import traceback

import psutil
import torch
import transformers


class Runner:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.completed = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_seconds = None
        self.busy = False
        model_path = Path(args.model_path).resolve()
        sys.path.insert(0, str(model_path))
        from joint_schema_model import encode_record, load_release_model, systemone
        self.encode_record = encode_record
        self.systemone = systemone
        manifest_path = model_path / 'download-manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        revision = args.revision or manifest.get('revision')
        if not revision:
            raise ValueError('Use --revision or provide download-manifest.json')
        torch.set_num_threads(args.cpu_threads)
        print(json.dumps({'event': 'loading', 'model': args.model, 'device': args.device,
                          'dtype': args.dtype, 'revision': revision}), flush=True)
        start = time.monotonic()
        self.model, self.processor = load_release_model(
            model_path, device=args.device, dtype=getattr(torch, args.dtype),
            local_files_only=True, attn_implementation=args.attention,
        )
        self.synchronize()
        parameter_dtypes = {}
        for parameter in self.model.parameters():
            key = str(parameter.dtype)
            parameter_dtypes[key] = parameter_dtypes.get(key, 0) + parameter.numel()
        self.meta = {
            'repo': manifest.get('repo', f'Cloudflare/{args.model}'),
            'model': args.model,
            'revision': revision,
            'device': str(next(self.model.parameters()).device),
            'weight_dtype': str(next(self.model.parameters()).dtype),
            'parameter_dtypes': parameter_dtypes,
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'batch_size': 1,
            'attention': args.attention,
            'cpu_threads': torch.get_num_threads(),
            'quantization': 'none',
            'official_code_sha256': hashlib.sha256((model_path / 'joint_schema_model.py').read_bytes()).hexdigest(),
            'compatibility_changes': [],
            'truncation': 'reject before inference',
            'max_input_tokens': args.max_input_tokens,
            'load_seconds': time.monotonic() - start,
            'started_at': self.started_at,
        }

    def synchronize(self):
        if self.args.device.startswith('mps'):
            torch.mps.synchronize()
        elif self.args.device.startswith('cuda'):
            torch.cuda.synchronize()

    def health(self):
        memory = {'rss_bytes': psutil.Process().memory_info().rss}
        if self.args.device.startswith('mps'):
            memory.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                          mps_driver_bytes=torch.mps.driver_allocated_memory(),
                          mps_recommended_max_bytes=torch.mps.recommended_max_memory())
        elif self.args.device.startswith('cuda'):
            memory.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                          cuda_reserved_bytes=torch.cuda.memory_reserved())
        return {'name': self.args.model, 'ready': True, 'returns_probabilities': True,
                'max_input_tokens': self.args.max_input_tokens, 'metadata': self.meta,
                'completed_requests': self.completed, 'busy': self.busy,
                'last_seconds': self.last_seconds, 'memory': memory}

    def infer(self, body):
        if not isinstance(body, dict) or not isinstance(body.get('questions'), dict):
            return 400, {'error': 'bad request: questions must be an object'}
        if body.get('images') or body.get('videos'):
            return 400, {'error': 'this benchmark server accepts text-only records'}
        # Encode once without a limit to measure exactly what the official encoder
        # would receive. Never shorten state, criteria, or instructions to fit.
        encoded = self.encode_record(self.processor.tokenizer, body, max_length=2**31 - 1,
                                     processor=self.processor)
        token_count = len(encoded.input_ids)
        if token_count > self.args.max_input_tokens:
            return 413, {'error': 'unsupported', 'reason': 'full input exceeds server limit',
                         'input_tokens': token_count, 'max_input_tokens': self.args.max_input_tokens}
        with self.lock:
            self.busy = True
            start = time.monotonic()
            try:
                output = self.systemone(self.model, self.processor, body,
                                        max_length=self.args.max_input_tokens)
                self.synchronize()
                if output['usage']['input_tokens'] != token_count:
                    raise RuntimeError('preflight and inference token counts differ')
                self.last_seconds = time.monotonic() - start
                self.completed += 1
                output['metadata'] = {**self.meta, 'inference_seconds': self.last_seconds}
                print(json.dumps({'event': 'inferred', 'n': self.completed,
                                  'input_tokens': token_count, 'seconds': self.last_seconds}), flush=True)
                return 200, output
            finally:
                self.busy = False


def make_handler(runner):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                return self.send_json(200, runner.health())
            self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/v1/systemone':
                return self.send_json(404, {'error': 'not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 32 * 1024 * 1024:
                    return self.send_json(413, {'error': 'invalid request size'})
                body = json.loads(self.rfile.read(length))
                code, result = runner.infer(body)
                self.send_json(code, result)
            except (KeyError, ValueError, TypeError) as error:
                self.send_json(400, {'error': type(error).__name__, 'message': str(error)[:500]})
            except Exception as error:
                traceback.print_exc()
                self.send_json(500, {'error': type(error).__name__, 'message': str(error)[:1000]})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model', default='clef-flash')
    parser.add_argument('--revision')
    parser.add_argument('--port', type=int, default=8818)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--max-input-tokens', type=int, default=16384)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    runner = Runner(args)
    print(json.dumps({'event': 'ready', **runner.health()}), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(runner)).serve_forever()


if __name__ == '__main__':
    main()
