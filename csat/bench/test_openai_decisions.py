"""OpenAI Decisions 어댑터: 요청 모양, 보기 키 보존, 거부·실행 오류 구분. 네트워크 없이 돈다."""
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import Unsupported
from adapters.openai_decisions import FatalProviderError, OpenAIDecisionsAdapter


class Response(io.BytesIO):
    headers = {'x-request-id': 'req_test', 'openai-processing-ms': '42'}


def answer(probs, kind='choice'):
    a = {'type': kind, 'name': 'answer'}
    if kind == 'choice':
        a.update(choice=max(probs, key=probs.get), confidence=max(probs.values()),
                 probabilities=[{'value': k, 'probability': v} for k, v in probs.items()])
    return Response(json.dumps({'model': 'gpt-6-luna', 'answers': [a], 'usage': {'input_tokens': 77}}).encode())


class OpenAIDecisionsTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            self.adapter = OpenAIDecisionsAdapter()

    def test_object_state_goes_as_json_and_keys_stay(self):
        state = {'판': '가\n나', 'nested': [1, None]}
        def call(req, timeout):
            self.assertEqual(req.full_url, 'https://api.openai.com/v1/decisions')
            self.assertEqual(json.loads(req.data), {'model': 'gpt-6-luna',
                'input': [{'role': 'user', 'content': [{'type': 'input_json', 'data': state}]}],
                'questions': [{'type': 'choice', 'name': 'answer', 'instructions': '고르세요',
                               'choices': [{'value': 'B', 'description': '둘'}, {'value': 'A'}]}]})
            return answer({'A': .3, 'B': .7})
        with patch('urllib.request.urlopen', call):
            self.assertEqual(self.adapter.choose(state, '고르세요', {'B': '둘', 'A': None}), {'A': .3, 'B': .7})
        meta = self.adapter.last_response_metadata()
        self.assertEqual((meta['usage']['input_tokens'], meta['request_id']), (77, 'req_test'))

    def test_string_state_goes_as_text(self):
        def call(req, timeout):
            self.assertEqual(json.loads(req.data)['input'], 'plain text')
            return answer({'A': .5, 'B': .5})
        with patch('urllib.request.urlopen', call):
            self.adapter.choose('plain text', 'pick', {'A': 'yes', 'B': 'no'})

    def test_refusal_is_unsupported(self):
        with patch('urllib.request.urlopen', return_value=answer({}, 'refusal')):
            with self.assertRaises(Unsupported):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})

    def test_wrong_option_set_rejected(self):
        with patch('urllib.request.urlopen', return_value=answer({'A': .5, 'X': .5})):
            with self.assertRaises(ValueError):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})

    def test_auth_and_quota_stop_the_run(self):
        for code, body in [(401, b'Unauthorized'), (429, b'{"error": {"code": "insufficient_quota"}}')]:
            self.setUp()
            error = HTTPError('test', code, 'x', {}, io.BytesIO(body))
            with patch('urllib.request.urlopen', side_effect=error) as request:
                for _ in range(2):
                    with self.assertRaises(FatalProviderError):
                        self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})
                self.assertEqual(request.call_count, 1)

    def test_rate_limit_backs_off_every_thread(self):
        error = HTTPError('test', 429, 'x', {'retry-after': '3'}, io.BytesIO(b'{"error": {"code": "rate_limit_exceeded"}}'))
        with patch('urllib.request.urlopen', side_effect=error), patch('time.monotonic', return_value=100.0):
            with self.assertRaises(HTTPError):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})
        self.assertEqual(self.adapter.not_before, 103.0)
        self.assertIsNone(self.adapter.fatal_error)


if __name__ == '__main__':
    unittest.main()
