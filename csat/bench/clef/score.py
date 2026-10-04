"""Score complete Clef suites with the existing scorers, into new result tags.

Old reference raw records are linked read-only by convention; no old result is
rewritten. Run: .venv/bin/python csat/bench/clef/score.py --suite csat|game|claims|lang-en
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SYSTEMS = ['clef', 'clef-flash']
REFERENCES = ['jev-1.13.0', 'd1', 'nimble']


def link_if_missing(target, source):
    if not target.exists():
        if not source.exists():
            raise FileNotFoundError(source)
        target.symlink_to(source, target_is_directory=True)


def expected_requests(entry, suite):
    source = ROOT / entry['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError(f'{suite} input changed after the manifest was frozen')
    answer_source = ROOT / entry['answer_path']
    if hashlib.sha256(answer_source.read_bytes()).hexdigest() != entry['answer_sha256']:
        raise RuntimeError(f'{suite} answer source changed after the manifest was frozen')
    runner_path = ROOT / ('csat/bench/run.py' if suite == 'csat' else 'gamebench/run.py')
    spec = importlib.util.spec_from_file_location('clef_expected_' + suite.replace('-', '_'), runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    items = json.loads(source.read_text())['questions'] if suite == 'csat' else [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    expected = set()
    for item in items:
        for rotation in (range(5) if suite == 'csat' else runner.shifts(len(item['options']))):
            _, request = runner.make_request(item, rotation)
            digest = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            expected.add((item['id'], rotation, digest))
    if len(expected) != entry['calls_per_model']:
        raise RuntimeError('Request count differs from the frozen manifest')
    return expected


def ensure_complete(base, entry, suite, selected):
    tag = entry['tag']
    expected = expected_requests(entry, suite)
    for system in selected:
        files = list((base / tag / system).glob('*.json'))
        counts, actual = {}, set()
        for path in files:
            record = json.loads(path.read_text())
            if record['system'] != system:
                raise RuntimeError(f'Wrong system identity in {path}')
            status = record['status']
            counts[status] = counts.get(status, 0) + 1
            if status in ('complete', 'unsupported'):
                actual.add((record['id'], record['rotation'], record['request_sha256']))
        if len(files) != len(expected) or actual != expected:
            raise RuntimeError(f'{system} {tag} incomplete or mismatched: {counts}, missing {len(expected-actual)}, extra {len(actual-expected)} requests')


def score(suite, selected):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    entry = manifest['suites'][suite]
    source = ROOT / entry['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError(f'{suite} input changed after the manifest was frozen')
    answer_source = ROOT / entry['answer_path']
    if hashlib.sha256(answer_source.read_bytes()).hexdigest() != entry['answer_sha256']:
        raise RuntimeError(f'{suite} answer source changed after the manifest was frozen')
    tag = entry['tag']
    base = ROOT / ('csat/bench/results' if suite == 'csat' else 'gamebench/results')
    ensure_complete(base, entry, suite, selected)
    # Keep previously finished local models in later regenerated comparisons.
    # A requested model must be complete; an unrequested one is included only
    # after independently passing the same exact-request validation.
    selected = list(selected)
    for other in SYSTEMS:
        if other in selected or not (base / tag / other).exists():
            continue
        try:
            ensure_complete(base, entry, suite, [other])
            if suite == 'lang-en':
                ensure_complete(ROOT / 'gamebench/results', manifest['suites']['game'], 'game', [other])
        except RuntimeError:
            continue
        selected.append(other)
    selected = [system for system in SYSTEMS if system in selected]
    old_tag = {'csat': '2027-09', 'game': 'v1', 'claims': 'claims', 'lang-en': 'lang-en'}[suite]
    for ref in REFERENCES:
        link_if_missing(base / tag / ref, base / old_tag / ref)
    systems = [*REFERENCES, *selected]
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    if suite == 'csat':
        run('csat/bench/score.py', '--tag', tag, '--dataset', entry['path'], '--gold', 'csat/exams/2027-09/gold.json', '--systems', *systems)
        run('csat/bench/report.py', '--tag', tag, '--systems', *systems)
    elif suite == 'game':
        run('gamebench/score.py', '--tag', tag, '--items', entry['path'], '--gold', 'gamebench/sets/v1/gold.json', '--systems', *systems)
        run('gamebench/report.py', '--tag', tag, '--systems', *systems)
    elif suite == 'claims':
        run('claims/report.py', '--tag', tag, '--systems', *systems)
        report = json.loads((base / tag / 'report.json').read_text())
        if any('incomplete' in value for model in report.values() for value in model.values()):
            raise RuntimeError('Claims scorer found incomplete or mismatched requests; report is not a completed result')
    else:
        # Reuse the language analyzer's game-bootstrap and paired comparison.
        # Only result paths for Clef and the output directory are substituted.
        game_entry = manifest['suites']['game']
        ensure_complete(ROOT / 'gamebench/results', game_entry, 'game', selected)
        sys.path.insert(0, str(ROOT / 'gamebench/lang'))
        sys.path.insert(0, str(ROOT / 'gamebench'))
        spec = importlib.util.spec_from_file_location('clef_language_analysis', ROOT / 'gamebench/lang/analyze.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_score, original_load = module.score_system, module.load
        def mapped(tag, system):
            return {'v1': game_entry['tag'], 'lang-en': entry['tag']}.get(tag, tag) if system in SYSTEMS else tag
        module.score_system = lambda items, gold, tag, system: original_score(items, gold, mapped(tag, system), system)
        module.load = lambda tag, system: original_load(mapped(tag, system), system)
        outdir = HERE / 'language-results'
        outdir.mkdir(exist_ok=True)
        link_if_missing(outdir / 'set', ROOT / 'gamebench/lang/set')
        module.HERE = outdir
        sys.argv = [str(ROOT / 'gamebench/lang/analyze.py'), '--systems', *systems]
        module.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['csat', 'game', 'claims', 'lang-en'], required=True)
    parser.add_argument('--systems', nargs='+', choices=SYSTEMS, default=SYSTEMS)
    args = parser.parse_args()
    with (HERE / f'.score-{args.suite}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        score(args.suite, args.systems)
