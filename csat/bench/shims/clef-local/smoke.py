"""Exercise the local release with the public model-card invoice example."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8818')
    parser.add_argument('--model', default='clef-flash')
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = {
        'model': args.model,
        'state': {'invoice': {'vendor': 'Acme', 'total': 1250.0, 'currency': 'USD', 'status': 'overdue'}},
        'questions': {
            'status': {
                'type': 'choice',
                'instructions': 'What is the invoice status?',
                'criteria': {'paid': 'Invoice is paid.', 'overdue': 'Invoice is past due.', 'draft': 'Not sent.'},
            },
            'large': {'type': 'noul', 'instructions': 'Is the total above 1000 USD?'},
        },
    }
    results = []
    for index in range(args.repeat):
        start = time.monotonic()
        req = urllib.request.Request(args.url + '/v1/systemone',
                                     json.dumps(request).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.load(response)
        result['client_seconds'] = time.monotonic() - start
        results.append(result)
        print(json.dumps({'repeat': index + 1, **result}, ensure_ascii=False), flush=True)
    with urllib.request.urlopen(args.url + '/health', timeout=20) as response:
        health = json.load(response)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(),
                               'source': 'public model-card invoice example',
                               'request': request, 'results': results, 'health': health},
                              ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
