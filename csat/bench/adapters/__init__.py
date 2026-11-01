"""Jev-like 공통 호출 인터페이스. PROTOCOL.md 1절.

모든 시스템은 choose(state, instructions, criteria) -> {키: 확률} 하나로만 호출된다.
어댑터는 정답 파일을 읽지 않는다.

로컬 Jev-like는 각자 전용 가상환경에서 `shim` 서버를 띄우고, 여기서는 HTTP로만 부른다.
shim 계약 (모든 로컬 시스템 공통):
  GET  /health  -> {"name": str, "returns_probabilities": bool, "max_input_tokens": int | null, "detail": {...}}
  POST /choose  {"state": {...}, "instructions": str, "criteria": {"A": str, ...}}
       200 -> {"probabilities": {"A": float, ...}}   # 키 집합이 criteria와 같아야 한다
       413 -> {"error": "unsupported", "reason": str, "input_tokens": int | null}  # 몰래 자르지 않는다
"""
import json
import threading
from typing import Mapping, Protocol
import urllib.error
import urllib.request


class Unsupported(Exception):
    """입력 길이 등으로 시스템이 처리할 수 없는 문항. 어댑터는 몰래 자르지 않고 이걸 던진다."""


def retryable(exc):
    """Only transient failures benefit from repeating an identical request."""
    code = getattr(exc, 'status_code', getattr(exc, 'status', getattr(exc, 'code', None)))
    return not (isinstance(code, int) and 400 <= code < 500 and code not in (408, 429))


class Adapter(Protocol):
    name: str
    returns_probabilities: bool  # False면 고른 답 1.0으로 기록되고 보고표에 표시된다

    def choose(self, state: Mapping, instructions: str, criteria: Mapping[str, str]) -> dict[str, float]:
        ...


def as_state(state):
    """state는 객체 그대로 보낸다. 제작자 벤치마크(claims/)처럼 원본 state가 문자열이나 목록이면 그대로 보낸다."""
    return state if isinstance(state, (str, list)) else dict(state)


def check(probs, criteria):
    if set(probs) != set(criteria):
        raise ValueError(f'Response options do not match request: {sorted(probs)}')
    return {k: float(v) for k, v in probs.items()}


class JevAdapter:
    """버전마다 이름을 고정해 부른다. 응답이 밝힌 모델·토큰·요청 ID는 response_metadata로 남긴다."""
    versions = ('jev-1.13.0', 'jev-1.14.0')
    returns_probabilities = True

    def __init__(self, name: str = 'jev-1.13.0', timeout: float = 45):
        self.name, self.timeout = name, timeout
        self.local = threading.local()

    def last_response_metadata(self):
        return getattr(self.local, 'metadata', {})

    def choose(self, state, instructions, criteria):
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient
        with TypeSafeClient(timeout=self.timeout, retry=RetryPolicy(max_retries=0)) as client:
            r = client.system_one(model=self.name, state=as_state(state),
                                  questions={'answer': Choice(instructions=instructions, criteria=dict(criteria))})
        self.local.metadata = {'model': r.model, 'input_tokens': r.usage.input_tokens,
                               'output_tokens': r.usage.output_tokens, 'request_id': r.request_id}
        return check(dict(r.answers['answer'].probabilities), criteria)


class LiquidAdapter:
    """Liquid AI d1(API 전용). Jev와 같은 TypeSafe SDK로 base_url만 바꿔 부른다. 키는 LIQUID_API_KEY
    (`op://Agent/Liquid API/credential`, csat/.env.tpl)."""
    name = 'd1'
    returns_probabilities = True

    def __init__(self, timeout: float = 45):
        self.timeout = timeout

    def choose(self, state, instructions, criteria):
        import os
        if isinstance(instructions, str) and not instructions.strip():
            raise Unsupported('Liquid d1 requires nonempty instructions; original input preserved')
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient
        from typesafe_sdk import TypeSafeUnprocessableEntityError
        try:
            with TypeSafeClient(api_key=os.environ['LIQUID_API_KEY'], base_url='https://api.liquid.ai/decisions',
                                timeout=self.timeout, retry=RetryPolicy(max_retries=0)) as client:
                r = client.system_one(model='d1:free', state=as_state(state),
                                      questions={'answer': Choice(instructions=instructions, criteria=dict(criteria))})
        except TypeSafeUnprocessableEntityError as e:  # 422: d1이 요청을 거부한다(예: 빈 지시문)
            raise Unsupported(str(e)[:300]) from None
        return check(dict(r.answers['answer'].probabilities), criteria)


class ShimAdapter:
    """로컬 shim 서버(위 계약)를 부르는 어댑터."""

    def __init__(self, name: str, port: int, timeout: float = 600):
        import os
        port = int(os.environ.get('JEV_SHIM_PORT', port))
        self.name, self.url, self.timeout = name, f'http://127.0.0.1:{port}', timeout
        self.returns_probabilities = True

    def health(self):
        with urllib.request.urlopen(self.url + '/health', timeout=10) as r:
            info = json.loads(r.read())
        self.returns_probabilities = info.get('returns_probabilities', True)
        return info

    def choose(self, state, instructions, criteria):
        body = json.dumps({'state': as_state(state), 'instructions': instructions, 'criteria': dict(criteria)},
                          ensure_ascii=False).encode()
        req = urllib.request.Request(self.url + '/choose', body, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return check(json.loads(r.read())['probabilities'], criteria)
        except urllib.error.HTTPError as e:
            if e.code == 413:
                raise Unsupported(e.read().decode()[:300]) from None
            raise


class OllamaAdapter:
    """Ollama 0.35+의 `/v1/systemone`(TypeSafe System One 모양)을 바로 부른다. shim 없이 Ollama 서버가 곧 계약 서버다.

    Ollama는 입력을 자르지 않고 400(컨텍스트 초과)·413(요청 64 KiB 초과)으로 거부한다. 둘 다 Unsupported로 기록한다.
    """

    def __init__(self, name: str, model: str, port: int, timeout: float = 600):
        import os
        port = int(os.environ.get('JEV_OLLAMA_PORT', port))  # 같은 모델을 다른 기기에서 돌릴 때(Mac 11435 / amd 11437)
        self.name, self.model, self.port, self.timeout = name, model, port, timeout
        self.url = f'http://127.0.0.1:{port}'
        self.returns_probabilities = True

    def health(self):
        with urllib.request.urlopen(self.url + '/api/version', timeout=10) as r:
            version = json.loads(r.read())['version']
        with urllib.request.urlopen(self.url + '/api/tags', timeout=10) as r:
            tag = next(m for m in json.loads(r.read())['models'] if m['name'] == self.model)
        return {'name': self.name, 'returns_probabilities': True, 'max_input_tokens': None,
                'detail': {'ollama': version, 'model': self.model, 'port': self.port, 'digest': tag['digest'],
                           'details': tag.get('details')}}

    def choose(self, state, instructions, criteria):
        if isinstance(instructions, str) and not instructions.strip():
            raise Unsupported('Ollama System One requires nonempty instructions; original input preserved')
        if not 2 <= len(criteria) <= 26:
            raise Unsupported('Ollama System One requires 2-26 candidates; original input preserved')
        body = json.dumps({'model': self.model, 'state': as_state(state), 'keep_alive': -1,
                           'questions': {'answer': {'type': 'choice', 'instructions': instructions,
                                                    'criteria': dict(criteria)}}}, ensure_ascii=False).encode()
        req = urllib.request.Request(self.url + '/v1/systemone', body, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return check(json.loads(r.read())['answers']['answer']['probabilities'], criteria)
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors='replace')[:300]
            # 입력 길이 초과, 보기 26개 초과, 빈 지시문은 Ollama가 요청을 거부한다(시스템 자체 제한)
            refused = ('never truncated', 'context', 'candidates', 'nonempty')
            if e.code == 413 or (e.code == 400 and any(k in text for k in refused)):
                raise Unsupported(text) from None
            e.reason = text
            raise


# Ollama 모델: 이름 -> (Ollama 모델 태그, 포트). Mac의 0.35 서버는 11435(기존 0.34 앱과 분리), amd는 11436(Mac 쪽 터널 11437).
OLLAMA = {
    'nimble': ('nimble:latest', 11435),
    'tev1-4b': ('tev1:latest', 11437),
    'tev1-0.8b': ('tev1:0.8b', 11435),
}


# 로컬 시스템별 shim 포트. shim 코드는 csat/bench/shims/<name>/ 에 있다.
SHIMS = {
    'clef': 8817,  # pinned public weights on amd, via SSH tunnel; paid API disabled
    'clef-flash': 8818,  # pinned public weights on Mac; paid API disabled
    'open-jev': 8801,
    'jevmlx': 8802,
    'semif': 8803,
    'kev': 8804,
    'laya': 8805,
    'clm': 8806,  # amd(Windows) 원격 실행. Mac에서는 ssh -L 8806:127.0.0.1:8806 amd 터널로 부른다
    'julia': 8807,
    'laya-typed': 8808,  # claims/ 전용: Laya가 typed-decisions로 추가 학습한 체크포인트
    'jeff-0.8b': 8809,
    'jeff-2b': 8810,
    'jeff-gemma4-e2b': 8811,
    'jeff-0.8b-torch': 8812,
    'jeff-0.8b-board': 8813,
    'jeff-0.8b-c4-1k': 8814,
    'jeff-0.8b-c4-1k-coords': 8815,
    'jeff-0.8b-c4-1k-stacks': 8816,
}


def get(name: str):
    if name in ('clef', 'clef-flash'):
        from .clef_local import LocalClefAdapter
        return LocalClefAdapter(name, SHIMS[name])
    if name in ('clef-api', 'clef-flash-api'):
        raise ValueError('Paid Cloudflare API execution is disabled for this benchmark; use local clef/clef-flash')
    if name in JevAdapter.versions:
        return JevAdapter(name)
    if name == LiquidAdapter.name:
        return LiquidAdapter()
    if name == 'openai-decisions':
        from .openai_decisions import OpenAIDecisionsAdapter
        return OpenAIDecisionsAdapter()
    if name in OLLAMA:
        return OllamaAdapter(name, *OLLAMA[name])
    return ShimAdapter(name, SHIMS[name])
