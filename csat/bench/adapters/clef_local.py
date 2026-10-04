"""Loopback SystemOne adapter for pinned Clef public weights, never a paid API."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request

from . import ShimAdapter, Unsupported, as_state, check


class LocalClefAdapter(ShimAdapter):
    def __init__(self, name, port):
        super().__init__(name, port)
        self.local = threading.local()
        self.fatal_error = None

    def last_response_metadata(self):
        return getattr(self.local, 'metadata', {})

    def choose(self, state, instructions, criteria):
        if self.fatal_error is not None:
            raise self.fatal_error
        free = shutil.disk_usage(Path(__file__).resolve().parent).free
        if free < 8 * 1024**3:
            self.fatal_error = RuntimeError(f'Local result disk below 8 GiB reserve: {free} bytes; no new inference sent')
            raise self.fatal_error
        body = {'model': self.name, 'state': as_state(state), 'questions': {
            'answer': {'type': 'choice', 'instructions': instructions, 'criteria': dict(criteria)}}}
        request = urllib.request.Request(self.url + '/v1/systemone', json.dumps(body, ensure_ascii=False).encode(),
                                         {'Content-Type': 'application/json'})
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors='replace')[:500]
            if exc.code == 413:
                raise Unsupported(message) from None
            if exc.code == 507:
                self.fatal_error = RuntimeError('Remote inference resource reserve reached: ' + message)
                raise self.fatal_error from None
            exc.msg = message
            raise
        probabilities = check(payload['answers']['answer']['probabilities'], criteria)
        self.local.metadata = {'provider': 'publicweights-local', 'model': payload.get('model'),
                               'usage': payload.get('usage'), 'local_inference': payload.get('metadata'),
                               'request_started_at_utc': started_at,
                               'network_latency_ms': round((time.perf_counter() - start) * 1000, 2),
                               'api_cost_usd': 0}
        return probabilities
