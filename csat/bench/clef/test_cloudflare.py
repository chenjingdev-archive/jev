"""Protocol checks: unchanged payload, option integrity, errors, and token framing."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from adapters import Unsupported
from adapters.cloudflare import CloudflareAdapter, FatalProviderError, input_token_count


class Response(io.BytesIO):
    headers = {'cf-ray': 'test-ray'}


class CloudflareTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {'CLOUDFLARE_ACCOUNT_ID': 'test-account', 'CLOUDFLARE_AUTH_TOKEN': 'test-token'}):
            self.adapter = CloudflareAdapter('clef')

    def test_payload_and_metadata(self):
        state = {'판': '가\n나', 'nested': [1, False]}
        options = {'B': 'two', 'A': 'one'}
        def call(req, timeout):
            self.assertEqual(json.loads(req.data), {'model': 'clef', 'state': state,
                'questions': {'answer': {'type': 'choice', 'instructions': '', 'criteria': options}}})
            return Response(json.dumps({'success': True, 'result': {'model': 'clef',
                'answers': {'answer': {'probabilities': {'A': .2, 'B': .8}}}, 'usage': {'input_tokens': 101}}}).encode())
        with patch('urllib.request.urlopen', call):
            self.assertEqual(self.adapter.choose(state, '', options), {'A': .2, 'B': .8})
        self.assertEqual(self.adapter.last_response_metadata()['usage']['input_tokens'], 101)

    def test_auth_remains_error(self):
        with patch('urllib.request.urlopen', side_effect=HTTPError('test', 401, 'auth', {}, io.BytesIO(b'Unauthorized'))) as request:
            with self.assertRaises(FatalProviderError):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})
            with self.assertRaises(FatalProviderError):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})
            self.assertEqual(request.call_count, 1)

    def test_wrong_option_set_rejected(self):
        response = Response(json.dumps({'result': {'answers': {'answer': {'probabilities': {'X': 1}}}}}).encode())
        with patch('urllib.request.urlopen', return_value=response):
            with self.assertRaises(ValueError):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})

    def test_over_limit_is_not_sent(self):
        with patch('adapters.cloudflare.input_token_count', return_value=65537), patch('urllib.request.urlopen') as request:
            with self.assertRaises(Unsupported):
                self.adapter.choose('x', 'pick', {'A': 'yes', 'B': 'no'})
            request.assert_not_called()

    def test_token_framing_matches_pinned_publisher_code(self):
        path = BENCH / 'clef/reference/joint_schema_model.py'
        spec = importlib.util.spec_from_file_location('clef_publisher_reference', path)
        source = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = source
        spec.loader.exec_module(source)
        tokenizer = self.adapter.tokenizer
        hf = lambda text, add_special_tokens=False: SimpleNamespace(input_ids=tokenizer.encode(text, add_special_tokens=False).ids)
        for state, instructions, criteria in [('hello', 'pick', {'B': 'no', 'A': 'yes'}),
                ({'언어': '한국어', 'list': [1, None]}, '', {'A': '일', 'B': '이'}),
                ('x', 'many', {chr(65+i): str(i) for i in range(26)})]:
            record = {'state': state, 'questions': {'answer': {'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}
            encoded = source.encode_record(hf, record, max_length=65536)
            self.assertEqual(input_token_count(tokenizer, state, instructions, criteria), len(encoded.input_ids))


if __name__ == '__main__':
    unittest.main()
