"""OpenAI Decisions API(`POST /v1/decisions`, gpt-6-luna). 2026-10-22 정식 출시.

choose 계약 그대로 choice 질문 하나로 보낸다. 보기 키는 `value`, 보기 글은 `description`이다.
state가 객체(또는 객체 목록)면 user 메시지의 `input_json`으로, 문자열이면 `input` 글 그대로 보낸다.
요청 모양은 openai-python 3.37.0의 `types/decision_create_params.py`를 따랐다.
키는 OPENAI_API_KEY(`op://Agent/OpenAI API/credential`, csat/.env.tpl).
"""
from datetime import datetime, timezone
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

from . import Unsupported, as_state, check

URL = 'https://api.openai.com/v1/decisions'
MODEL = 'gpt-6-luna'


class FatalProviderError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def request_body(state, instructions, criteria):
    state = as_state(state)
    content = state if isinstance(state, str) else [
        {'role': 'user', 'content': [{'type': 'input_json', 'data': state}]}]
    choices = [{'value': k} if v is None else {'value': k, 'description': v} for k, v in criteria.items()]
    return {'model': MODEL, 'input': content,
            'questions': [{'type': 'choice', 'name': 'answer', 'instructions': instructions, 'choices': choices}]}


def retry_after(headers):
    for key in ('retry-after', 'x-ratelimit-reset-requests', 'x-ratelimit-reset-tokens'):
        value = (headers.get(key) or '').strip()
        try:
            return min(float(value.rstrip('s')), 60.0) if value else None
        except ValueError:
            continue
    return None


class OpenAIDecisionsAdapter:
    name = 'openai-decisions'
    returns_probabilities = True

    def __init__(self, timeout: float = 60):
        self.timeout = timeout
        self.key = os.environ['OPENAI_API_KEY']
        self.lock, self.local = threading.Lock(), threading.local()
        self.not_before = 0.0  # 429를 받으면 모든 스레드가 이 시각까지 기다린다
        self.fatal_error = None

    def health(self):
        return {'name': self.name, 'returns_probabilities': True, 'max_input_tokens': None,
                'detail': {'provider': 'OpenAI Decisions API', 'endpoint': URL, 'model': MODEL,
                           'pinned_model_revision': None,
                           'input': 'object state -> user input_json part; string state -> input text',
                           'latency_note': 'record latency_ms includes 429 waits; '
                                           'response_metadata.network_latency_ms excludes them'}}

    def last_response_metadata(self):
        return getattr(self.local, 'metadata', {})

    def choose(self, state, instructions, criteria):
        if self.fatal_error is not None:
            raise self.fatal_error
        with self.lock:
            wait = self.not_before - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        body = json.dumps(request_body(state, instructions, criteria), ensure_ascii=False).encode()
        request = urllib.request.Request(URL, body, {'Content-Type': 'application/json',
                                                     'Authorization': 'Bearer ' + self.key})
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
                headers = response.headers
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors='replace')[:500]
            # 키·권한·잔액 문제는 오답이 아니라 실행 조건이다. 더 부르지 않고 멈춘다.
            if exc.code in (401, 403) or (exc.code == 429 and 'insufficient_quota' in message):
                self.fatal_error = FatalProviderError(exc.code, f'OpenAI HTTP {exc.code}: {message}')
                raise self.fatal_error from None
            if exc.code == 429:
                with self.lock:
                    self.not_before = max(self.not_before, time.monotonic() + (retry_after(exc.headers) or 5.0))
            exc.msg = message
            raise
        answer = payload['answers'][0]
        if answer['type'] == 'refusal':  # 시스템이 이 질문에 답하지 않았다. 오답으로 센다
            raise Unsupported('Decisions API returned a refusal for this question')
        probs = check({p['value']: p['probability'] for p in answer['probabilities']}, criteria)
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > .02:
            raise ValueError('Invalid Decisions probability distribution')
        self.local.metadata = {'model': payload.get('model'), 'usage': payload.get('usage'),
                               'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                               'request_id': headers.get('x-request-id'),
                               'processing_ms': headers.get('openai-processing-ms'),
                               'request_started_at_utc': started_at,
                               'network_latency_ms': round((time.perf_counter() - start) * 1000, 2)}
        return probs
