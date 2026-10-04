"""Nimble(Bespoke Labs)이 docs/PUBLIC_BENCHMARKS.md에서 Jev와 맞댄 공개 벤치 13개를 gamebench/run.py 공통 문항 형식으로 옮긴다.

원천은 Nimble 제작자 코드 그대로다. ~/dev/jev-likes/nimble@62076b4 의 nimble/datasets/public_sources/*.py를
import해서 변환하고(다시 구현하지 않는다), 제작자가 커밋한 docs/assets/public-benchmarks/subsets/<subset>-manifest.json의
ids로 정확히 같은 레코드를 고른다(재추출 아님. 원래 추출은 seed 20260918, 가족 단위).
  - 받은 원본을 Nimble 방식(parquet → JSON Lines, 원래 필드 이름)으로 내보낸 파일의 sha256이 manifest의 source_sha256과 같아야 하고,
  - 고른 레코드를 Nimble의 jsonl_line으로 쓴 바이트의 sha256이 manifest의 dataset_sha256과 같아야 한다.
  둘 중 하나라도 다르면 멈춘다. 즉 Nimble·Jev가 받은 입력과 바이트 단위로 같은 레코드에서 문항을 만든다.
  aegis2·helpsteer2의 source_sha256은 저장소 원본(test.json, validation.jsonl.gz)이 아니라 HF refs/convert/parquet 내보내기와 맞는다.
  길이 필터(Qwen3.5 토크나이저, 2,048토큰)는 ids 선택에 이미 반영돼 있어 토크나이저가 필요 없다. 자르지 않는다.

변환 규칙 (Nimble 레코드의 input.questions.decision을 그대로 옮긴다):
  - choice: 선택지 = criteria 값(설명), 키 순서 그대로. 정답 = 키 위치.
  - noul: [false 설명, true 설명] 두 선택지. record_for가 criteria를 {"false","true"} 순으로 정규화하고
    evaluate_pilot.adapt_input이 choices=[False, True]로 내보내므로 이것이 Nimble 자신의 순서다. 정답 = true면 2, false면 1.
  - score: 5단계 설명(LEVELS) 순서 그대로. 정답 = 기준 단계 + 1. 정확도는 단계가 정확히 맞아야 맞음이다
    (evaluate_pilot.assess: correct = argmax 키 == str(target); evaluate_public_jev.jev_row도 같다).
  - 키 이름(SUPPORTS, alarm, "0".."4", false/true)은 버린다. build_jeff.py·typed-decisions와 같다.
    Nimble 프롬프트(parallel_schema.prepare_prompts)와 Jev 요청({키: 설명})은 키 이름도 보였다는 점이 다르다.
  - state는 그대로 (대부분 객체, civil_comments만 문자열). question = instructions.
  - stage = Nimble 레코드의 domain (vitaminc-real/synthetic, multinli-<genre>, massive-<locale> 등).

  .venv/bin/python claims/build_nimble.py
"""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

import build  # ITEM_KEYS, leak_paths: build.py의 검사를 그대로 쓴다

HERE = Path(__file__).resolve().parent
SETS_DIR = HERE / 'sets'
NIMBLE = Path(os.environ.get('JEV_LIKES', Path.home() / 'dev' / 'jev-likes')) / 'nimble'
NIMBLE_COMMIT = '62076b4f2d365b5879dafcf7f6dd072a1fe76df7'
SUBSET_MANIFESTS = 'docs/assets/public-benchmarks/subsets'
RAW = NIMBLE / 'data' / 'public' / 'raw'   # Nimble 문서의 원본 위치 (Nimble 저장소에서 gitignore)

# 원본 파일. hf: (repo, revision, file, sha256). url: (url, sha256, 압축 안 파일).
# Nimble 문서 "Get the data" 표의 출처·분할이다. HF 리비전은 2026-09-30에 받은 것이고,
# 내보낸 JSON Lines가 Nimble manifest의 source_sha256과 같으므로 Nimble이 받은 것과 같은 내용이다.
UPSTREAM = {
    'vitaminc': {'url': 'https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip',
                 'sha256': '49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337',
                 'member': 'vitaminc/dev.jsonl', 'out': 'vitaminc/dev.jsonl'},
    'massive': {'url': 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz',
                'sha256': '4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577',
                'member': None, 'out': 'massive/amazon-massive-dataset-1.1.tar.gz'},
    'multinli': {'url': 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip',
                 'sha256': '049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822',
                 'member': 'multinli_1.0/multinli_1.0_dev_matched.jsonl',
                 'out': 'multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl'},
    'boolq': {'hf': ('google/boolq', '35b264d03638db9f4ce671b711558bf7ff0f80d5',
                     'data/validation-00000-of-00001.parquet',
                     '52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8'),
              'out': 'boolq/validation.jsonl'},
    'squad2': {'hf': ('rajpurkar/squad_v2', '3ffb306f725f7d2ce8394bc1873b24868140c412',
                      'squad_v2/validation-00000-of-00001.parquet',
                      '0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb'),
               'out': 'squad_v2/validation.jsonl'},
    'paws': {'hf': ('google-research-datasets/paws', '161ece9501cf0a11f3e48bd356eaa82de46d6a09',
                    'labeled_final/test-00000-of-00001.parquet',
                    'ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d'),
             'out': 'paws/test.jsonl'},
    'civil_comments': {'hf': ('google/civil_comments', 'f2970eb3a55777454c94069077cc8d9b5866312d',
                              'data/test-00000-of-00001.parquet',
                              '410f803ff161e7c4ba005db7de5b444cac413f4861bf056b0fd69d670d75aa08'),
                       'out': 'civil_comments/test.jsonl'},
    'aegis2': {'hf': ('nvidia/Aegis-AI-Content-Safety-Dataset-2.0', 'c28676823546b8128ae1d3e7a53e26a4984d5ce8',
                      'default/test/0000.parquet',
                      'e887e19c628e5667fb10e0bf540a2a7bdb038395a3b418b948d2e97846c7758a'),
               'out': 'aegis2/test.jsonl'},          # refs/convert/parquet (main d86bb8b의 test.json은 해시가 다르다)
    'helpsteer2': {'hf': ('nvidia/HelpSteer2', '7315a43cd8a6090323188d2b32c9688df6288fd9',
                          'default/validation/0000.parquet',
                          '0add582523f049c233d4716d57a578b6a7b4ddee51bc8060e1598d9f26832c70'),
                   'out': 'helpsteer2/validation.jsonl'},  # refs/convert/parquet (main 990b271의 validation.jsonl.gz도 같은 dataset_sha256)
    'summeval': {'hf': ('mteb/summeval', 'bfc121155064afa2d81b5505682ffc0d96f4334c',
                        'data/test-00000-of-00001-35901af5f6649399.parquet',
                        '7ff6f3f4d044223bd6922c5434d27bf28473950346fcfc2a3b7352c19163690e'),
                 'out': 'summeval/test.jsonl'},
    'pubmedqa': {'hf': ('qiaojin/PubMedQA', '9001f2853fb87cab8d220904e0de81ac6973b318',
                        'pqa_labeled/train-00000-of-00001.parquet',
                        '3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c'),
                 'out': 'pubmedqa/train.jsonl'},
}

# Nimble subset -> (dataset 모듈, --subset, 우리 세트 이름, 문서의 문항 수)
SUBSETS = {
    'vitaminc-dev': ('vitaminc', '', 'nimble-vitaminc599', 599),
    'massive-en-US': ('massive', 'en-US', 'nimble-massive-en350', 350),
    'massive-de-DE': ('massive', 'de-DE', 'nimble-massive-de350', 350),
    'boolq': ('boolq', '', 'nimble-boolq300', 300),
    'squad2': ('squad2', '', 'nimble-squad2-299', 299),
    'paws': ('paws', '', 'nimble-paws250', 250),
    'multinli': ('multinli', '', 'nimble-multinli299', 299),
    'civil_comments': ('civil_comments', '', 'nimble-civilcomments300', 300),
    'aegis2': ('aegis2', '', 'nimble-aegis2-250', 250),
    'helpsteer2': ('helpsteer2', '', 'nimble-helpsteer2-249', 249),
    'summeval-relevance': ('summeval', 'relevance', 'nimble-summeval-relevance240', 240),
    'summeval-consistency': ('summeval', 'consistency', 'nimble-summeval-consistency144', 144),
    'pubmedqa': ('pubmedqa', '', 'nimble-pubmedqa250', 250),
}
TOTAL = 3880

# Nimble 문서 Results 표 (docs/PUBLIC_BENCHMARKS.md:382-394), 제작자 발표 수치로 manifest에 남긴다.
PUBLISHED = {  # subset: (nimble-9b, jev-1.13.0) 정확도
    'aegis2': (.812, .804), 'boolq': (.860, .897), 'civil_comments': (.703, .810), 'helpsteer2': (.390, .341),
    'massive-de-DE': (.834, .869), 'massive-en-US': (.869, .874), 'multinli': (.853, .829), 'paws': (.828, .892),
    'pubmedqa': (.756, .772), 'squad2': (.806, .829), 'summeval-consistency': (.757, .812),
    'summeval-relevance': (.492, .350), 'vitaminc-dev': (.766, .801),
}

SCORE_NOTE = ('score: accuracy is exact-level match, as in Nimble\'s own table. nimble/evaluation/evaluate_pilot.py:51-62 '
              'assess(): gold = target_key(target) = str(level index), best = argmax over "0".."4", correct = best == gold; '
              'evaluate_public_jev.py:130-131 jev_row(): Jev prediction = student["prediction"] (the same argmax), correct = prediction == target. '
              'No tolerance. Nimble\'s doc (PUBLIC_BENCHMARKS.md:455-459) says to read the expected-level MAE for these '
              'subsets instead; gamebench/run.py gives per-option probabilities, so it can be computed from results.')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url, dest, pinned):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + '.part')
        with urllib.request.urlopen(url, timeout=600) as r:
            tmp.write_bytes(r.read())
        tmp.replace(dest)
    got = sha256_bytes(dest.read_bytes())
    if got != pinned:
        raise SystemExit(f'{url}: sha256 {got} != pinned {pinned}')
    return dest


def raw_file(name):
    """Nimble 문서대로 원본을 받아 data/public/raw/<...>에 둔다. 경로와 원본 메타를 돌려준다."""
    spec = UPSTREAM[name]
    out = RAW / spec['out']
    if 'hf' in spec:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        repo, rev, fname, pinned = spec['hf']
        p = Path(hf_hub_download(repo, fname, repo_type='dataset', revision=rev))
        got = sha256_bytes(p.read_bytes())
        if got != pinned:
            raise SystemExit(f'{repo}/{fname}@{rev}: sha256 {got} != pinned {pinned}')
        # "the parquet export of each repository, written to JSON Lines with the original field names"
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pq.read_table(p).to_pylist()).encode()
        if not out.exists() or out.read_bytes() != payload:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
        meta = {'url': f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{fname}', 'repo': repo,
                'revision': rev, 'file': fname, 'sha256': pinned, 'exported_to': spec['out']}
    else:
        archive = RAW / 'archives' / spec['url'].rsplit('/', 1)[1]
        if spec['member'] is None:           # MASSIVE: Nimble이 tar.gz를 그대로 읽는다
            archive = out
        fetch(spec['url'], archive, spec['sha256'])
        if spec['member'] is not None:
            payload = zipfile.ZipFile(archive).read(spec['member'])
            if not out.exists() or out.read_bytes() != payload:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(payload)
        meta = {'url': spec['url'], 'sha256': spec['sha256'], 'member': spec['member'], 'exported_to': spec['out']}
    return out, meta


def nimble_modules():
    head = subprocess.run(['git', '-C', str(NIMBLE), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != NIMBLE_COMMIT:
        raise SystemExit(f'{NIMBLE}: HEAD {head} != pinned {NIMBLE_COMMIT}')
    dirty = subprocess.run(['git', '-C', str(NIMBLE), 'status', '--porcelain', 'nimble', SUBSET_MANIFESTS],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f'{NIMBLE}: maker code or manifests modified:\n{dirty}')
    sys.path.insert(0, str(NIMBLE))
    from nimble.datasets import public_benchmarks
    return public_benchmarks


def convert(record, set_name):
    """Nimble 레코드 하나 -> (문항, 정답). 레코드의 input만 읽고 reference는 정답에만 쓴다."""
    q = record['input']['questions']['decision']
    kind, crit, target = q['type'], q['criteria'], record['reference']['target']
    if kind == 'choice':
        keys = list(crit)
        options = [crit[k] for k in keys]
        answer = keys.index(target) + 1
    elif kind == 'noul':
        if list(crit) != ['false', 'true'] or type(target) is not bool:
            raise ValueError(f"{record['id']}: noul criteria/target")
        options = [crit['false'], crit['true']]
        answer = 2 if target else 1
    elif kind == 'score':
        options = list(crit)
        if type(target) is not int or not 0 <= target < len(options):
            raise ValueError(f"{record['id']}: score target")
        answer = target + 1
    else:
        raise ValueError(f"{record['id']}: {kind}")
    item = {'id': f"{set_name}:{record['id']}", 'game': set_name, 'stage': record['domain'],
            'state': record['input']['state'], 'question': q['instructions'], 'options': options}
    return item, {'answer': answer, 'n': len(options)}, kind


def check(set_name, items, gold, expected):
    errs = []
    ids = [x['id'] for x in items]
    if len(set(ids)) != len(ids) or set(gold) != set(ids) or len(items) != expected:
        errs.append(f'{set_name}: ids/count ({len(items)} vs {expected})')
    for x in items:
        if tuple(x) != build.ITEM_KEYS or x['game'] != set_name or '|' in x['stage']:
            errs.append(f"{x['id']}: shape")
        if not isinstance(x['question'], str) or not isinstance(x['state'], (str, dict)):
            errs.append(f"{x['id']}: question/state type")
        if len(x['options']) < 2 or not all(isinstance(o, str) and o for o in x['options']):
            errs.append(f"{x['id']}: options")
        errs += [f"{x['id']}: leak key {p}" for p in build.leak_paths(x)]
        g = gold[x['id']]
        if g['n'] != len(x['options']) or not 1 <= g['answer'] <= g['n']:
            errs.append(f"{x['id']}: gold {g}")
    raw = ''.join(json.dumps(x, ensure_ascii=False) for x in items)
    for key in ('"gold":', '"_meta":', '"provenance":', '"target":', '"reference":', '"label":', '"distribution":'):
        if key in raw:
            errs.append(f'{set_name}: {key} in items')
    return errs


def main():
    pb = nimble_modules()
    sources = pb.load_sources()
    raws = {}
    total = 0
    for subset, (ds, sub, set_name, expected) in SUBSETS.items():
        mpath = NIMBLE / SUBSET_MANIFESTS / f'{subset}-manifest.json'
        mbytes = mpath.read_bytes()
        man = json.loads(mbytes)
        if ds not in raws:
            raws[ds] = raw_file(ds)
        path, src_meta = raws[ds]
        src_sha = sha256_bytes(path.read_bytes())
        if src_sha != man['source_sha256']:
            raise SystemExit(f'{subset}: source sha256 {src_sha} != Nimble manifest {man["source_sha256"]}')
        spec = sources[ds]
        converted = [r for r in (spec.record(raw, sub) for raw in spec.rows(path, sub)) if r is not None]
        by_id = {r['id']: r for r in converted}
        missing = [i for i in man['ids'] if i not in by_id]
        if missing or len(set(man['ids'])) != len(man['ids']):
            raise SystemExit(f'{subset}: {len(missing)} manifest ids missing after conversion, e.g. {missing[:3]}')
        selected = sorted((by_id[i] for i in man['ids']), key=lambda r: r['id'])   # build()의 순서
        payload = ''.join(pb.jsonl_line(r) for r in selected).encode()
        if sha256_bytes(payload) != man['dataset_sha256']:
            raise SystemExit(f'{subset}: dataset sha256 {sha256_bytes(payload)} != Nimble {man["dataset_sha256"]}')
        if len(selected) != man['count'] or len(selected) != expected:
            raise SystemExit(f'{subset}: {len(selected)} records, manifest {man["count"]}, doc {expected}')

        entries = [convert(r, set_name) for r in selected]
        items_l = [e[0] for e in entries]
        gold_d = {e[0]['id']: e[1] for e in entries}
        errs = check(set_name, items_l, gold_d, expected)
        if errs:
            raise SystemExit('\n'.join(errs[:20]))
        # 정답 분포가 Nimble manifest의 labels와 같은지 (라벨 -> 선택지 위치)
        kinds = Counter(e[2] for e in entries)
        (kind,) = kinds
        crit = selected[0]['input']['questions']['decision']['criteria']
        keys = list(crit) if kind == 'choice' else ['False', 'True'] if kind == 'noul' else [str(i) for i in range(len(crit))]
        by_pos = Counter(g['answer'] for g in gold_d.values())
        label_pos = {keys[p - 1]: n for p, n in by_pos.items()}
        if label_pos != man['labels']:
            raise SystemExit(f'{subset}: gold distribution {label_pos} != Nimble labels {man["labels"]}')

        d = SETS_DIR / set_name
        d.mkdir(parents=True, exist_ok=True)
        items = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items_l)
        gold = json.dumps(gold_d, indent=1, ensure_ascii=False) + '\n'
        (d / 'items.jsonl').write_text(items)
        (d / 'gold.json').write_text(gold)
        notes = [
            'Records are exactly the ids in Nimble\'s committed subset manifest, converted by Nimble\'s own '
            f'public_sources/{ds}.py; the selected records re-serialize to Nimble\'s dataset_sha256 (verified).',
            'Option text is the criteria description only, in Nimble\'s key order; the key names are dropped (as in '
            'build_jeff.py / typed-decisions). Nimble\'s prompt (nimble/scoring/parallel_schema.py:95-101) and the Jev '
            'request ({key: description}, dataset_io.request_for) also showed the key name.',
            'state and instructions are Nimble\'s input.state / decision.instructions unchanged; stage = Nimble domain.',
        ]
        if kind == 'noul':
            notes.append('noul -> two options [criteria["false"], criteria["true"]]: public_records.record_for (public_records.py:63-64) fixes '
                         'that order and evaluate_pilot.py:29-30 (adapt_input) renders choices [False, True]. answer 2 = true. '
                         'Jev was asked a native noul (P(true)); here every system gets the same two-option choice.')
        if kind == 'score':
            notes.append('score -> choice over the 5 ordered level descriptions (LEVELS), answer = level index + 1. '
                         'Jev and Nimble were asked a native 5-level score with keys "0".."4". ' + SCORE_NOTE)
        if ds in ('aegis2', 'helpsteer2'):
            notes.append('source_sha256 matches the HF refs/convert/parquet export, not the repository\'s own '
                         'json/jsonl.gz on main; both convert to the same dataset_sha256.')
        if ds == 'aegis2':
            notes.append('The "true" option (Nimble\'s twelve unsafe categories) is 69 tokens by build.py\'s tokenizer, over '
                         'OPTION_TOKEN_CAP 48: Julia refuses such items (unsupported, scored wrong) and Laya truncates. '
                         'Kept verbatim; it is the maker\'s criterion.')
        if man.get('dropped_over_length'):
            notes.append(f'Nimble dropped {man["dropped_over_length"]} over-length row(s) (Qwen3.5-9B tokenizer, 2,048 '
                         'tokens) before selection; the id list already reflects that. Nothing is truncated here.')
        if man.get('ids_from'):
            notes.append(f'Nimble built this subset with --ids-from {man["ids_from"]["path"]} (same MASSIVE ids as '
                         'en-US); record ids do not embed the locale, the set prefix keeps ours distinct.')
        manifest = {
            'set': set_name, 'builder': 'claims/build_nimble.py',
            'source': {
                'maker': 'bespokelabs/nimble', 'commit': NIMBLE_COMMIT,
                'subset': subset, 'dataset': ds, 'dataset_subset': sub,
                'subset_manifest': {'path': f'{SUBSET_MANIFESTS}/{subset}-manifest.json',
                                    'sha256': sha256_bytes(mbytes)},
                'upstream': src_meta | {'nimble_source_url': man['source_url'], 'license': man['license'],
                                        'release_year': man['release_year']},
                'source_sha256': man['source_sha256'], 'dataset_sha256': man['dataset_sha256'],
                'seed': man['seed'], 'limit': man['limit'], 'ids_from': man['ids_from'],
                'converted': man['converted'], 'skipped': man['skipped'],
                'dropped_over_length': man['dropped_over_length'], 'families': man['families'],
                'maker_code': [f'nimble/datasets/public_sources/{ds}.py', 'nimble/datasets/public_records.py',
                               'nimble/datasets/public_benchmarks.py', 'nimble/evaluation/evaluate_pilot.py:19-73',
                               'nimble/evaluation/evaluate_public_jev.py'],
                'maker_command': 'docs/PUBLIC_BENCHMARKS.md:150-162',
            },
            'published': {'nimble-9b': PUBLISHED[subset][0], 'jev-1.13.0': PUBLISHED[subset][1],
                          'source': 'docs/PUBLIC_BENCHMARKS.md:380-394 (accuracy vs human label, no option shuffling)'},
            'items': len(items_l),
            'sha256': {'items.jsonl': sha256_bytes(items.encode()), 'gold.json': sha256_bytes(gold.encode())},
            'type_counts': dict(kinds),
            'stats': {'options': dict(Counter(g['n'] for g in gold_d.values())),
                      'answer_position': dict(sorted(by_pos.items())),
                      'label_by_position': {str(p): keys[p - 1] for p in sorted(by_pos)},
                      'nimble_labels': man['labels'],
                      'stages': dict(Counter(x['stage'] for x in items_l))},
            'notes': notes,
        }
        (d / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
        total += len(items_l)
        print(set_name, len(items_l), dict(kinds), dict(sorted(by_pos.items())))
    if total != TOTAL:
        raise SystemExit(f'{total} items, expected {TOTAL}')
    print('total', total)


if __name__ == '__main__':
    main()
