"""Cloudflare-hosted Clef, preserving the existing choose contract and input.

Only tokenizer/reference source files are needed locally, never model weights.
Usage and network latency are returned separately from throttled call latency.
"""
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from . import Unsupported, as_state, check

TOKENIZER = Path(__file__).resolve().parent.parent / 'clef' / 'reference' / 'tokenizer.json'
SYSTEM_PROMPT = ('Read the complete state and schema. Decide every field jointly. Each answer '
                 "must be exactly one of that field's allowed options.")


class FatalProviderError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def input_token_count(tokenizer, state, instructions, criteria):
    """Text-only framing from the pinned public joint_schema_model.encode_record.

    Counts the complete state before truncation. Criteria are sorted by the model,
    just as in its public implementation; the wire request retains original order.
    """
    chunks = ['\n\nSCHEMA FIELDS:\n', '\nFIELD 1\nID: answer\nTYPE: choice\nINSTRUCTION: ',
              render(instructions if instructions not in (None, '') else 'answer'), '\nALLOWED OPTIONS:\n']
    for i, (key, value) in enumerate(sorted(criteria.items()), 1):
        semantics = {'option_id': key}
        if value is not None:
            semantics['description'] = value
        chunks.extend([f'OPTION {i}: ', render(semantics), '\n'])
    chunks.extend(['END FIELD\n',
                   f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n',
                   '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:',
                   render(as_state(state))])
    return sum(len(tokenizer.encode(chunk, add_special_tokens=False).ids) for chunk in chunks)


class CloudflareAdapter:
    returns_probabilities = True

    def __init__(self, name, timeout=90):
        from tokenizers import Tokenizer
        if name not in ('clef', 'clef-flash'):
            raise ValueError('Unknown Cloudflare benchmark model')
        self.name, self.timeout = name, timeout
        self.account = os.environ['CLOUDFLARE_ACCOUNT_ID']
        self.token = os.environ['CLOUDFLARE_AUTH_TOKEN']
        self.url = f'https://api.cloudflare.com/client/v4/accounts/{self.account}/ai/run/@cf/cloudflare/{name}'
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER))
        self.tokenizer_sha256 = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
        self.rps = float(os.environ.get('CLEF_REQUESTS_PER_SECOND', '2.4'))
        if self.rps <= 0:
            raise ValueError('CLEF_REQUESTS_PER_SECOND must be positive')
        self.lock, self.local = threading.Lock(), threading.local()
        self.next_request = 0.0
        self.fatal_error = None

    def health(self):
        return {'name': self.name, 'returns_probabilities': True, 'max_input_tokens': 65536,
                'detail': {'provider': 'Cloudflare Workers AI', 'model': '@cf/cloudflare/' + self.name,
                           'pinned_api_model_revision': None, 'tokenizer_sha256': self.tokenizer_sha256,
                           'requests_per_second': self.rps,
                           'latency_note': 'record latency_ms includes rate limiting; response_metadata.network_latency_ms excludes it'}}

    def last_response_metadata(self):
        return getattr(self.local, 'metadata', {})

    def choose(self, state, instructions, criteria):
        if self.fatal_error is not None:
            raise self.fatal_error
        count = input_token_count(self.tokenizer, state, instructions, criteria)
        if count > 65536:
            raise Unsupported(f'Clef full input has {count} tokens; exceeds documented 65536 limit; not sent or truncated')
        body = json.dumps({'model': self.name, 'state': as_state(state),
                           'questions': {'answer': {'type': 'choice', 'instructions': instructions,
                                                    'criteria': dict(criteria)}}}, ensure_ascii=False).encode()
        request = urllib.request.Request(self.url, body, {'Content-Type': 'application/json',
                                                        'Authorization': 'Bearer ' + self.token})
        with self.lock:
            wait = max(0.0, self.next_request - time.monotonic())
            self.next_request = max(self.next_request, time.monotonic()) + 1 / self.rps
        if wait:
            time.sleep(wait)
        if self.fatal_error is not None:
            raise self.fatal_error
        start = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
                ray = response.headers.get('cf-ray')
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors='replace')[:500]
            if exc.code in (401, 402, 403):
                self.fatal_error = FatalProviderError(exc.code, f'Cloudflare HTTP {exc.code}: {message}')
                raise self.fatal_error from None
            if exc.code == 429:
                try:
                    retry_after = float(exc.headers.get('Retry-After', '10'))
                except ValueError:
                    retry_after = 10
                with self.lock:
                    self.next_request = max(self.next_request, time.monotonic() + retry_after)
            # Authentication, quotas and server faults are execution errors, never wrong answers.
            if exc.code == 413 or (exc.code in (400, 422) and any(s in message.lower() for s in
                    ('context length', 'maximum context', 'max_length', 'too many options', 'too many candidates',
                     'instructions must be nonempty', 'instructions must be non-empty'))):
                raise Unsupported(message) from None
            exc.msg = message
            raise
        if payload.get('success') is False:
            raise RuntimeError('Cloudflare unsuccessful response: ' + str(payload.get('errors'))[:400])
        result = payload.get('result', payload)
        probs = check(result['answers']['answer']['probabilities'], criteria)
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > .02:
            raise ValueError('Invalid Clef probability distribution')
        self.local.metadata = {'model': result.get('model'), 'usage': result.get('usage'),
                               'request_started_at_utc': started_at_utc,
                               'network_latency_ms': round((time.perf_counter() - start) * 1000, 2),
                               'estimated_full_input_tokens': count, 'cf_ray': ray}
        return probs
