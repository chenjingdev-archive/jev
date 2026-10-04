import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapters


class LocalAdapterTests(unittest.TestCase):
    def test_default_is_loopback_with_unchanged_input(self):
        adapter = adapters.get('clef-flash')
        self.assertEqual(adapter.url, 'http://127.0.0.1:8818')
        body = {'model': 'clef-flash', 'state': {'한국어': [1, False]}, 'questions': {
            'answer': {'type': 'choice', 'instructions': '', 'criteria': {'B': 'two', 'A': 'one'}}}}
        def call(request, timeout):
            self.assertEqual(json.loads(request.data), body)
            return io.BytesIO(json.dumps({'model': 'clef-flash', 'answers': {'answer': {
                'probabilities': {'A': .2, 'B': .8}}}, 'usage': {'input_tokens': 100},
                'metadata': {'device': 'mps:0', 'weight_dtype': 'torch.bfloat16'}}).encode())
        with patch('urllib.request.urlopen', call), patch('shutil.disk_usage', return_value=SimpleNamespace(free=10*1024**3)):
            self.assertEqual(adapter.choose(body['state'], '', body['questions']['answer']['criteria']), {'A': .2, 'B': .8})
        self.assertEqual(adapter.last_response_metadata()['provider'], 'publicweights-local')

    def test_low_disk_never_sends_an_inference(self):
        adapter = adapters.get('clef')
        with patch('urllib.request.urlopen') as call, patch('shutil.disk_usage', return_value=SimpleNamespace(free=7*1024**3)):
            with self.assertRaises(RuntimeError):
                adapter.choose('state', 'pick', {'A': 'one', 'B': 'two'})
            call.assert_not_called()
        self.assertIsNotNone(adapter.fatal_error)

    def test_paid_api_names_are_blocked(self):
        for name in ('clef-api', 'clef-flash-api'):
            with self.assertRaises(ValueError):
                adapters.get(name)


if __name__ == '__main__':
    unittest.main()
