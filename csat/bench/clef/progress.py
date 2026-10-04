"""Read-only progress snapshot; does not schedule, run, or change jobs."""
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
manifest = json.loads((HERE / 'manifest.json').read_text())
out = {}
for suite, entry in manifest['suites'].items():
    base = ROOT / ('csat/bench/results' if suite == 'csat' else 'gamebench/results') / entry['tag']
    out[suite] = {}
    for system in manifest['systems']:
        counts, tokens, latencies, usage_known, usage_missing = {}, 0, [], 0, 0
        for path in (base / system).glob('*.json'):
            record = json.loads(path.read_text())
            status = record['status']
            counts[status] = counts.get(status, 0) + 1
            metadata = record.get('response_metadata', {})
            input_tokens = (metadata.get('usage') or {}).get('input_tokens')
            if isinstance(input_tokens, (int, float)):
                tokens += input_tokens
                usage_known += 1
            elif status == 'complete':
                usage_missing += 1
            if metadata.get('network_latency_ms') is not None:
                latencies.append(metadata['network_latency_ms'])
        out[suite][system] = {'expected_calls': entry['calls_per_model'], 'statuses': counts,
                             'observed_input_tokens': tokens,
                             'usage_known_calls': usage_known, 'usage_missing_calls': usage_missing,
                             'cost_status': 'partial' if usage_missing else ('known' if usage_known else ('unknown' if counts else 'not_started')),
                             'estimated_observed_cost_usd': 0 if manifest.get('execution_mode') == 'local_public_weights' else
                                (tokens * manifest['price_per_million_input_tokens_usd'][system] / 1e6 if usage_known and not usage_missing else None),
                             'execution_mode': manifest.get('execution_mode', 'api_plan'),
                             'median_network_latency_ms': statistics.median(latencies) if latencies else None}
print(json.dumps(out, ensure_ascii=False, indent=2))
