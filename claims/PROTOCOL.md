# 제작자 벤치마크 교차 실행 — 실행 전 조건 (2026-09-28)

목적: Jev-like 제작자가 성능 근거로 내세운 벤치마크를 모아, 8개 시스템(Jev와 Jev-like 7개)에 같은 조건으로 돌린다. 두 가지를 본다.

- 제작자가 발표한 수치가 우리 실행에서도 나오는가
- 각 제작자가 고른 시험에서도 Jev와의 순서가 수능·게임 벤치마크(`csat/bench`, `gamebench`)와 같은가

모델을 부르기 전에 이 문서를 고정한다.

## 1. 세트

세트는 `claims/build.py`와 `claims/build_ext.py`로 만든다. 원본은 sha256으로 고정하고 `claims/sets/<세트>/manifest.json`에 기록한다. 정답(`gold.json`)은 채점할 때만 읽는다.

| 세트 | 문항 | 인용한 제작자 | 비고 |
|---|---:|---|---|
| typed-decisions | 2,000 | Julia-1, Laya (jevmlx도 실행 가능) | HF `LocalLLaMA/typed-decisions`@c76749ec. choice·score·noul을 모두 보기 고르기로 준다(Julia 재현 스크립트와 같다) |
| kev-transfer-v4 | 764 | Kev | test 분할. Kev 발표 수치는 clean 행만 쓴다 |
| semif-authored144 | 144 | SemIf | |
| semif-wanli256 | 256 | SemIf | |
| semif-typesafe102 | 102 | SemIf | Jev가 받은 형태(문서 객체 state)로 준다. SemIf 0.845는 문자열 state 형태에서 잰 값이다 |
| openjev-synthetic400 | 400 | open-jev | 질문 없음(`""`). 지시가 context 안에 있다 |
| jevbench-agnews100 · emotion100 · banking77-100 | 100씩 | Julia-1 (Jev 기준치 0.910 · 0.480 · 0.870) | `AbdelStark/jev-benchmarks`@0d610cc, BTZSC. Banking77은 보기 72개 전부다. Julia의 64%는 모델이 만든 상위 16개 후보로 잰 값이라 같은 조건이 아니다 |
| laya-agnews400 · emotion400 · banking77-400 | 400씩 | Laya | Laya `bench_apps.py`. Banking77은 보기 77개이고, 앞 400행이라 의도 10개만 들어 있다 |
| jevmlx-typesafe45 | 363 | jevmlx | TypeSafe 공개 사례 45건의 필드별 문항. 0~3 척도를 벗어난 2문항은 뺐다. jevmlx 스키마대로 렌더링한다 |

**만들지 못한 것:**
- Julia의 Banking77 상위 16개 후보: 모델이 후보를 고르는 방식이라 모델 없이 재현할 수 없다.
- CLM: 인용한 벤치마크가 보기 고르기 형식의 공개 데이터가 아니다(시뮬레이션, 궤적 best-of-N, 비공개 데이터).
- Laya 앱 테마 7종: 5종이 예/아니오이고 Jev 기준치가 없다.

**보기 수:** 보기가 26개를 넘으면 키를 A~Z 다음 AA, AB, …로 이어 붙인다(`gamebench/run.py keys`). 26개 이하 문항은 요청이 바뀌지 않는다. Julia는 보기 2~20개만 받으므로 Banking77은 처리 못 함으로 기록된다.

## 2. 실행

1. **시스템:** jev-1.13.0, kev, semif, open-jev, jevmlx, laya, clm, julia. 어댑터·설정은 `csat/bench`와 같다. state는 원본이 문자열이나 목록이면 그대로 보낸다.
2. **보조 시스템 `laya-typed`:** Laya가 typed-decisions로 추가 학습한 체크포인트다. Laya의 0.766을 재현하는 데만 쓰고, 순위에는 넣지 않는다. 그 데이터로 학습한 모델이라 공평한 비교가 아니다.
3. **보기 순환:** min(n, 5)개의 서로 다른 순환 이동. 확률을 원래 보기로 되돌려 평균한 최댓값이 답이다. 제작자 수치와 맞대는 용도로 순환 0(원래 순서) 한 번의 정확도도 따로 보고한다.
4. **계산 장치:** 모든 로컬 시스템을 GPU에서 돌린다. MLX는 jevmlx·open-jev·SemIf, MPS는 Laya·Julia, `amd`(ROCm)는 CLM·Kev이다. Kev는 처음 508회를 Mac(MLX)에서 돌리다 속도 때문에 `amd`의 torch 경로로 옮겼다. torch는 Kev README가 발표 수치를 낸 경로다. 옮기기 전에 Mac 결과 36건과 비교해 답이 모두 같았고, 확률 차이는 최대 0.02였다. `amd`에는 fla·causal_conv1d 커널이 없어 transformers의 참조 구현을 쓴다. 느리지만 결과는 같다. Julia는 공식 지원이 cpu·cuda뿐이라 shim이 입력만 mps로 옮긴다. 게임 문항 105개에서 cpu와 답이 모두 같았고, 확률 차이는 최대 1.4e-5였다.
5. **처리 못 한 문항:** 입력 길이 등으로 처리하지 못한 문항은 몰래 자르지 않는다. 정확도에서는 오답으로 센다.

## 3. 지표 (사전 고정)

6. **주 지표:** 세트별 정확도(순환 평균 기준)와 95% bootstrap 구간(문항 단위, 1만 회, 시드 20260928).
7. **우연 보정 점수:** (acc − 1/n)/(1 − 1/n)을 세트 안에서 문항 평균한다. 보기 수가 세트마다 달라서 세트를 가로지르는 비교는 이것으로 한다.
8. **제작자 수치 대조:** manifest에 적힌 발표 수치와 Jev 기준치를 우리 값 옆에 둔다. 모델 크기·체크포인트·분할이 다르면 표에 적는다(예: Kev 발표는 4B·27B이고, 우리는 9B다).
9. **Jev 대비 비교:** 같은 문항끼리 McNemar 검정을 한다.
10. **결과를 본 뒤의 수정:** 세트나 어댑터를 고치면 새 태그로 다시 돌리고 이전 결과와 섞지 않는다.

## 4. 추가: Jeff (2026-09-29 고정, 실행 전)

Jeff(firelex/jeff@2c1bfce)가 README에서 인용한 패널을 세트로 추가하고, Jeff 체크포인트 세 개를 시스템으로 추가한다. 앞의 13세트 결과와 문항·요청이 겹치지 않으므로(요청 해시가 다르다) 태그는 `claims` 그대로 쓴다. 기존 시스템은 새 세트만 추가로 돈다.

**세트** (`claims/build_jeff.py`). 원본은 Jeff 제작자 코드로 만든 두 파일이고 sha256을 고정한다.

| 세트 | 문항 | 원본 |
|---|---:|---|
| jeff-bbh750 | 750 | `python -m jeff.panel` (시드 20260926) |
| jeff-fpb999 | 999 | 〃 |
| jeff-judgebench350 | 350 | 〃 |
| jeff-ragtruth1500 | 1,500 | 〃 (전부 noul) |
| jeff-winogrande1000 | 1,000 | 〃 |
| jeff-jevbenchhard105 | 105 | `python -m jeff.jevbench` (jevbench@d06ee95 public hard, score 6문항 제외. choice 67, noul 38) |

- **noul 변환:** 공통 계약이 보기 고르기뿐이라 noul은 [거짓 문구, 참 문구] 두 보기로 준다. 문구는 criteria 값, 없으면 Jeff 기본값("No / false", "Yes / true")이다. Jeff 평가기도 noul을 같은 두 보기, 같은 순서로 채점하므로 문제는 같다.
- **보기 문구:** criteria 값, 값이 없으면 키(Jeff 렌더링과 같다).

**시스템** (`csat/bench/shims/jeff/serve.py`)

| 시스템 | 체크포인트@리비전 | 실행 경로 |
|---|---|---|
| jeff-0.8b | mstrasser/Jeff-Qwen3.5-0.8B@d66458d | Jeff MLX 백엔드, Mac |
| jeff-2b | mstrasser/Jeff-Qwen3.5-2B@30824ca | Jeff MLX 백엔드, Mac |
| jeff-gemma4-e2b | mstrasser/Jeff-Gemma4-E2B@afcb75a | Jeff torch 경로(`jeff.server.predict`와 같은 계산), `amd` ROCm. Jeff MLX 백엔드는 Qwen만 돈다 |

- **처리 못 함(413):** 보기가 `max_options`(26)를 넘는 문항(Banking77)과 8,192토큰을 넘는 입력. 둘 다 Jeff 자신의 제한이다.
- **같은 분포 학습:** Jeff 학습 데이터에 RAGTruth 학습 split, WinoGrande 학습 split, 금융 뉴스 감정(twitter_financial)이 들어 있다. 평가 문항과 겹치지는 않지만 RAGTruth·WinoGrande·FPB는 같은 분포라 결과 해석에 적는다. Julia의 Emotion(학습에 emotion 재생 포함)도 같은 경우다.
- **제작자 수치 재현:** 제작자 평가기(`jeff.evaluate`의 `predict_local`·`metrics`)로 패널과 JevBench hard를 먼저 돌려 README 수치와 맞대 본다. 결과는 `~/dev/jev-likes/jeff/runs/eval/`에 둔다.

**실행 장치 기록 (실행 중 결정, 2026-09-30)**

- MLX 시스템(open-jev, jevmlx, Jeff)은 버퍼 캐시를 2 GiB로 묶었다(`mx.set_cache_limit`, `MLX_CACHE_GB`). 메모리 압박으로 두 번 중단된 뒤 넣었고 점수에는 영향이 없다.
- Laya·laya-typed의 새 세트는 `amd`(torch 2.9.1+rocm7.2.1, transformers는 Mac과 같은 버전)에서 돌렸다. Mac 기록과 표본 비교에서 Laya 36/36, laya-typed 18/18 같은 답, 확률 차이 최대 0.02였다. laya-typed가 앞 13세트에서 비어 있던 부분도 이때 채웠다.
- Julia도 `amd`에서 돌려 봤으나 표본 255개 중 252개만 같은 답(확률 차이 최대 0.16)이라 Mac(MPS)에서 다시 돌렸다. 점수는 Mac 결과이고, `amd` 결과는 `julia-amd`로 따로 두었다. 새 6세트 정확도 차이는 세트마다 1%p 이내였다.
- Jeff 세 모델은 CSAT(`2027-09`), 게임 v1(`v1`), 언어 시험(`lang-en`)도 같은 장치로 돌렸다.

## 5. 추가: Nimble (2026-09-30 고정, 실행 전)

Nimble(bespokelabs/nimble@62076b4)이 `docs/PUBLIC_BENCHMARKS.md`에서 Bespoke-Nimble-9B와 Jev 1.13.0을 맞댄 공개 벤치 13개(3,880문항)를 세트로 추가한다. 앞 세트와 문항·요청이 겹치지 않으므로 태그는 `claims` 그대로 쓰고, 기존 시스템은 새 세트만 추가로 돈다.

**세트** (`claims/build_nimble.py`). Nimble 변환 코드(`nimble/datasets/public_sources/*.py`)를 그대로 import해 돌리고, Nimble이 커밋한 subset manifest의 ids로 같은 레코드를 고른다. 원본을 내보낸 파일이 manifest의 `source_sha256`과, 고른 레코드가 `dataset_sha256`과 13개 모두 바이트 단위로 같다(다르면 빌드가 멈춘다).

| 세트 | 문항 | Nimble subset | 원래 형식 | Nimble / Jev 발표 정확도 |
|---|---:|---|---|---|
| nimble-vitaminc599 | 599 | vitaminc-dev | choice 3 | 0.766 / 0.801 |
| nimble-massive-en350 | 350 | massive-en-US | choice 18 | 0.869 / 0.874 |
| nimble-massive-de350 | 350 | massive-de-DE (en-US와 같은 id) | choice 18 | 0.834 / 0.869 |
| nimble-boolq300 | 300 | boolq | noul | 0.860 / 0.897 |
| nimble-squad2-299 | 299 | squad2 | noul | 0.806 / 0.829 |
| nimble-paws250 | 250 | paws | noul | 0.828 / 0.892 |
| nimble-multinli299 | 299 | multinli | choice 3 | 0.853 / 0.829 |
| nimble-civilcomments300 | 300 | civil_comments | noul | 0.703 / 0.810 |
| nimble-aegis2-250 | 250 | aegis2 | noul | 0.812 / 0.804 |
| nimble-helpsteer2-249 | 249 | helpsteer2 | score 5 | 0.390 / 0.341 |
| nimble-summeval-relevance240 | 240 | summeval-relevance | score 5 | 0.492 / 0.350 |
| nimble-summeval-consistency144 | 144 | summeval-consistency | score 5 | 0.757 / 0.812 |
| nimble-pubmedqa250 | 250 | pubmedqa | choice 3 | 0.756 / 0.772 |

- **noul 변환:** [false 설명, true 설명] 두 보기. Nimble `record_for`가 criteria를 이 순서로 고정하고 `adapt_input`이 `choices=[False, True]`로 내보낸다. Jev는 원래 noul(P(true))로 받았다.
- **score 변환:** 5단계 설명(LEVELS)을 순서대로 보기로 준다. 정답은 기준 단계. Nimble 표의 정확도도 단계가 정확히 맞아야 맞음이다(`evaluate_pilot.assess`: argmax 키 == 기준 단계, `evaluate_public_jev.jev_row`도 같다). Nimble 문서는 이 세 세트에서 기대 단계 MAE를 보라고 한다.
- **보기 문구:** criteria 설명만 쓰고 키 이름(SUPPORTS, alarm, "0"~"4" 등)은 버린다(Jeff·typed-decisions와 같다). Nimble 프롬프트와 Jev 요청은 키 이름도 보였다는 점이 다르다.
- **state·질문:** Nimble 레코드의 `input.state`(civil_comments만 문자열, 나머지 객체)와 `instructions`를 그대로 쓴다. stage는 Nimble domain이다.
- **원본:** aegis2·helpsteer2의 `source_sha256`은 저장소 원본이 아니라 HF `refs/convert/parquet` 내보내기와 맞는다. 길이 필터(Qwen3.5 토크나이저 2,048토큰)는 ids에 이미 반영돼 있고 자르지 않는다.
- **비교 조건 차이:** Nimble 발표 수치는 원래 순서 한 번, 원래 형식(noul·score) 질문이다. 우리 값은 보기 순환 평균이고 모든 형식이 보기 고르기다.
- **긴 보기:** nimble-aegis2-250의 true 보기(위험 범주 12개 나열)는 69토큰으로 `OPTION_TOKEN_CAP`(48)을 넘는다. Julia는 처리 못 함(오답)으로, Laya는 잘라서 받는다. 제작자 문구라 그대로 둔다.

## 6. 추가: Ollama 판단 모델(Nimble·Tev1)과 Liquid d1 (2026-09-30)

Ollama 0.35.0이 TypeSafe System One과 같은 모양의 `/v1/systemone`을 열면서 Nimble(Bespoke Labs)과 Tev1-4B·0.8B(Together AI)를 배포했다. Liquid AI도 같은 모양의 API로 d1을 냈다. 넷을 기존 시스템과 같은 계약으로 수능(2027-09), 게임 v1, 언어 시험(lang-en), claims 32세트에 돌린다.

| 시스템 | 모델 | 실행 경로 | 기기 | 입력 한도 |
|---|---|---|---|---|
| nimble | `nimble:latest` (Bespoke-Nimble-9B, Q8_0 GGUF, digest `24e550a16a70`) | Ollama 0.35.0 `/v1/systemone` | `amd` (Radeon 8060S, ROCm) | 8,194토큰 (modelfile `num_ctx`) |
| tev1-4b | `tev1:latest` (Tev1-4B-experimental, Q8_0, digest `cef45ef93cf6`) | 같음 | `amd` | 2,050토큰 |
| tev1-0.8b | `tev1:0.8b` (Tev1-0.8B-experimental, Q8_0, digest `d45e875d63fe`) | 같음 | Mac Studio (Metal) | 2,050토큰 |
| d1 | `d1:free` | TypeSafe SDK 0.6.0, `base_url=https://api.liquid.ai/decisions` | Liquid API | 공개 안 됨 |

- **어댑터:** `OllamaAdapter`는 Ollama의 System One에 choice 질문 하나를 바로 보낸다(shim 없음). `LiquidAdapter`는 Jev와 같은 SDK를 쓰고, 키는 `op://Agent/Liquid API/credential`(`csat/.env.tpl`)이다. state·지시문·보기는 다른 시스템과 같다.
- **Ollama 서버:** 사용자의 기존 Ollama 앱(0.34)과 섞지 않으려고 0.35.0 실행 파일을 따로 받아 다른 포트와 모델 폴더로 띄웠다(Mac 11435, `amd` 11436을 Mac 11437로 터널). `ollama pull`이 Cloudflare R2의 IP 하나에 연결하지 못해 멈췄다. 그래서 레이어를 curl로 받아 manifest의 sha256과 맞춘 뒤 넣었다. `amd`에는 Mac에서 복사했고 해시가 같다.
- **처리 못 함(시스템 자체 거부):** 세 경우다. 입력이 modelfile `num_ctx`를 넘을 때(Ollama 400, "input is never truncated"), 보기가 2~26개 밖일 때(Ollama), 지시문이 비었을 때(Ollama 400, d1 422)다. 어댑터는 입력을 고치지 않고 처리 못 함으로 적는다. 세트별 수는 `README.md`에 있다.
- **같은 분포 학습:**
  - Jeff는 Nimble 공개 벤치와 같은 데이터셋 9개의 학습 split으로 학습했다. 근거는 `jeff/src/jeff/data.py`의 `TASKS`·`TRAIN_QUOTAS`다. Nimble 레코드와 그 family는 해시를 확인해 뺐다.
  - Tev1은 MultiNLI·BoolQ·Banking77·AG News·SST-5로 학습했다(tev1 `DATA_SOURCES.md`).
  - Nimble 학습 데이터(2,676문항)에는 공개 데이터셋이 없다.
  - d1은 학습 데이터를 공개하지 않았다.

**실행 중 결정 (2026-09-30 ~ 10-01)**

- **Nimble은 `amd` 하나로 돌렸다.**
  - 처음에는 Mac과 `amd`에 나눠 돌렸다. 같은 v1 40문항(192호출)을 두 기기에서 비교하니 189호출의 답이 같았고, 확률 차이는 최대 0.142였다.
  - Julia를 `amd`에서 Mac으로 되돌렸을 때(252/255, 0.16)와 같은 수준이라 한 기기로 모았다. 기록 대부분(49,175건)이 `amd`에서 나와 `amd`로 정했다.
  - Mac에서 나온 386건은 `gamebench/results/v1/nimble-mac`에 따로 두었다. 비교 기록은 `gamebench/results/parity-mac`·`parity-amd`에 있다.
  - 호출당 속도는 `amd` 0.31초, Mac 0.93초였다.
- **Tev1:** Tev1-4B는 `amd`에서만, Tev1-0.8B는 Mac에서만 돌렸다.
- **어댑터 오류 처리 변경:**
  - Ollama의 거부 응답(토큰 초과, 보기 수, 빈 지시문)이 처음에는 일반 오류로 잡혔다. 실행 중에 이를 처리 못 함으로 분류하도록 고쳤다. 같은 요청을 시스템이 거부한 것을 오류 대신 처리 못 함으로 적는 변경이라, 완료된 답은 하나도 바뀌지 않는다. 3절 10번의 새 태그 대상이 아니다.
  - d1의 일시 오류(429·502·시간 초과) 136호출은 다시 불러 모두 완료했다.
- **다른 세션의 실행:**
  - 2026-09-30 22:37 ~ 10-01 07:00에 Codex 세션이 사용자 요청으로 자체 분배기(`gamebench/dispatch.py`, 나중에 사용자 요청으로 삭제)를 만들어 일부를 돌렸다. 기존 Mac 시스템 6개(Julia, Jeff-0.8B·2B, SemIf, open-jev, jevmlx)의 Nimble 세트와 새 시스템 일부가 여기서 나왔다.
  - 그 기록의 health(체크포인트·revision·가중치 sha256)는 예전 실행과 같다. Jeff는 가중치 해시의 표기 형식만 바뀌었다.
  - 그 세션이 `gamebench/run.py`, `csat/bench/run.py`, `csat/bench/adapters/__init__.py`에 넣은 변경은 남겨 두었다. 새 기록은 `device`(`JEV_BENCH_DEVICE`)와 `execution`(health)을 함께 적는다. 408·429를 뺀 4xx는 다시 부르지 않는다(`retryable`). Ollama·d1 어댑터는 빈 지시문과 2~26개 밖의 보기를 부르기 전에 처리 못 함으로 돌린다. 답에는 영향이 없다.
- **Jev:** Nimble 세트 12,907호출을 2026-09-30 22:11~22:18에 돌렸다(workers 8).
- **5절 정정:** `nimble-aegis2-250`에서 Laya도 Julia처럼 처리 못 함이다. Laya는 보기를 잘라야 받을 수 있는데, shim이 몰래 자르지 않고 413으로 돌려보낸다("Laya would truncate"). 5절의 "Laya는 잘라서 받는다"는 실행 전 예상이었다.

## 7. 추가: OpenAI Decisions API (2026-10-24 고정, 실행 전)

OpenAI가 2026-10-22에 Decisions API를 정식으로 냈다. `POST /v1/decisions` 하나에 모델은 `gpt-6-luna`뿐이다. 질문 형식(predicate·choice·score)과 보기별 확률을 돌려주는 모양이 Jev와 같다. 기존 시스템과 같은 계약으로 수능(2027-09), 게임 v1, claims 32세트에 돌린다.

| 시스템 | 모델 | 실행 경로 | 입력 한도 |
|---|---|---|---|
| openai-decisions | `gpt-6-luna` (버전을 고정할 수단이 없다) | `https://api.openai.com/v1/decisions`를 HTTP로 바로 부른다. 요청 모양은 openai-python 3.37.0의 `types/decision_create_params.py` | 문서에 없음. JSON 입력은 256 KiB·값 10,000개 |

- **어댑터:** `csat/bench/adapters/openai_decisions.py`. choice 질문 하나를 보낸다. 보기 키(A, B, …, AA)는 `value`, 보기 글은 `description`, 지시문은 `instructions`에 그대로 넣는다. state가 객체나 객체 목록이면 user 메시지의 `input_json`으로, 문자열이면 `input` 글로 보낸다. 세 벤치의 state는 모두 이 둘 중 하나이고, 가장 큰 것(36 KiB)도 JSON 한도 안이다. 키는 `op://Agent/OpenAI API/credential`(`csat/.env.tpl`)이다.
- **처리 못 함:** 답 대신 `refusal`이 오면 처리 못 함(오답)으로 적는다. API가 요청을 400으로 거부하면 실행 오류로 남기고, 응답을 읽어 시스템 자체 제한(빈 지시문, 길이 등)으로 확인된 것만 처리 못 함으로 옮긴다. 입력은 고치지 않는다.
- **태그:** `2027-09-openai-20261024`, `v1-openai-20261024`, `claims-openai-20261024`. 비교 시스템(jev-1.13.0, d1, kev, nimble, clef-flash)은 원래 태그 폴더를 가리키는 상대 링크로 둔다. 원래 태그의 요약 파일은 고치지 않는다.
- **지표:** 1~3절과 같다. 같은 문항끼리 Jev·d1과 McNemar 검정을 한다.
- **실행:** Mac에서 workers 8로 시작한다. 429는 응답의 대기 시간만큼 모든 스레드가 쉬고 다시 부른다. 키·권한·잔액 오류(401·403·`insufficient_quota`)는 더 부르지 않고 멈춘다.
- **비용:** 입력 100만 토큰당 $0.10, 출력 무료(2026-10-22 공지). 세 벤치 53,969호출, 입력 약 2,500만 토큰으로 3달러 안팎을 예상한다.
- **오염:** gpt-6-luna의 지식 기준일은 제3자 표기로 2026-05다. 수능 2027-09(2026-09-03 시행)는 그 뒤다. claims 세트는 대부분 공개 데이터셋이라 실력인지 학습 노출인지 가릴 수 없다. d1과 같은 한계다.

### 실행 기록 (2026-10-25)

- **늦어진 이유:** 키를 만들 계정의 platform 로그인이 2026-10-24 01시부터 10-25 08:59 KST까지 실패했다(`POST /v1/dashboard/onboarding/login`이 503). 키는 그 뒤에 만들었다. kakao 계정 Personal 조직의 `jev-decisions-bench`이고, 만료는 30일이다.
- **실행:** 2026-10-25 00:15~01:14 UTC에 Mac에서 workers 8, `JEV_BENCH_DEVICE=mac`으로 돌렸다. 수능 4,420, 게임 4,216, claims 45,333호출로 모두 53,969호출이다. 응답의 `model`은 모두 `gpt-6-luna`였다.
- **다시 부른 호출:** 실행 오류로 남은 호출은 없다. 506호출은 한 번 이상 다시 불렀다. 실패한 시도 511회 중 472회는 연결·TLS 시간 초과였고, 39회는 503 `Decision inference is unavailable.`이었다. 429·401·403은 없었다.
- **거절:** 113호출이 refusal이었다(수능 48, 게임 41, claims 24). 보기 순서 중 하나라도 거절되면 그 문항은 처리 못 함이다. 그래서 수능 32문항, 게임 20문항, claims 16문항이 처리 못 함이다. 수능 32문항 중 23문항은 다섯 순서 중 한 번만 거절됐다. 같은 문항도 보기 순서에 따라 거절 여부가 달라진다는 뜻이다. 그래도 규칙대로 오답으로 셌다.
- **비용:** 답을 받은 호출의 입력은 2,759만 토큰이었고 출력은 0이었다. 공지 가격으로 $2.76이다. 거절 호출의 토큰은 기록에 남지 않아 빠졌다.
- **지연:** 호출 지연 중앙값은 0.23초, 95백분위는 0.28초다.

## 8. 추가: Jev 1.14.0 (2026-10-25 고정, 실행 전)

2026-10-25에 TypeSafe API의 `jev-latest`와 `jev-preview`가 둘 다 `jev-1.14.0`으로 답했다. 지금까지 세 벤치의 Jev 기록은 모두 `jev-1.13.0`이다. 같은 계약으로 `jev-1.14.0`을 수능(2027-09), 게임 v1, claims 32세트에 돌린다.

- **어댑터:** `JevAdapter`가 버전 이름(`jev-1.13.0`, `jev-1.14.0`)을 받는다. 요청 모양은 1.13.0 때와 같다. 응답의 `model`, 토큰 수, 요청 ID는 `response_metadata`로 남긴다. 키는 `op://Agent/typesafe jev key/credential`(`csat/.env.tpl`)이다.
- **태그:** `2027-09-jev114-20261025`, `v1-jev114-20261025`, `claims-jev114-20261025`. 비교 시스템은 원래 태그 폴더를 가리키는 상대 링크로 둔다. 원래 태그의 요약 파일은 고치지 않는다.
- **지표:** 1~3절과 같다. 같은 문항끼리 jev-1.13.0, openai-decisions, d1과 McNemar 검정을 한다.
- **실행:** Mac에서 workers 256으로 돌린다. 실행 전에 수능 요청으로 동시 16~512호출을 시험했다. 429나 오류는 한 번도 없었다. 처리량은 동시 192호출 부근에서 초당 약 300호출에 이른 뒤 더 늘지 않았고, 지연만 늘었다(256호출에서 중앙값 0.7초). 프로세스 두 개로 나눠도 합계가 같았으니 한도는 서버 쪽이다. 실행 오류가 남으면 같은 명령을 다시 돌려 채운다.
- **비용:** 입력 100만 토큰당 $0.042다. OpenAI 실행의 입력 토큰(2,759만)으로 어림하면 1.2달러 안팎이다.
- **오염:** 1.14.0의 출시일과 학습 기준일은 공개되지 않았다. 수능 2027-09(2026-09-03 시행)보다 늦게 나온 판이라 이 시험이 학습에 들어갔을 수 있다. 수능 점수가 1.13.0보다 오르면 이 가능성을 같이 적는다.

### 실행 기록 (2026-10-25)

- **실행:** 2026-10-25 06:31~06:35 UTC에 Mac에서 workers 256, `JEV_BENCH_DEVICE=mac`으로 돌렸다. 수능 4,420호출(15초), 게임 4,216호출(57초), claims 45,333호출(2분 37초)로 모두 53,969호출에 3분 49초가 걸렸다. 응답의 `model`은 모두 `jev-1.14.0`이었다. 수능 한 문항(5호출)은 본 실행 전에 같은 태그로 먼저 돌려 기록을 확인했고, 본 실행은 그 기록을 그대로 썼다.
- **한도 시험:** 위 실행 항목의 시험에 수능 요청 6,288호출을 따로 보냈다. 이 호출은 기록하지 않았다.
- **다시 부른 호출:** 실행 오류로 남은 호출은 없다. 35호출은 한 번씩 다시 불렀다. 28회는 이 Mac의 DNS 조회 실패(`nodename nor servname provided`)였고, 6회는 45초 시간 초과, 1회는 연결 끊김이었다. 429는 없었다.
- **비용:** 입력 4,312만 토큰, 출력 401만 토큰이었다. 입력 가격으로 $1.81이다. 출력은 공지대로면 무료다. 1.14.0의 가격은 중개 사이트(OpenRouter, Infrabase, Cloudprice) 표기로 1.13.0과 같다. 같은 수능 요청 60개를 `jev-1.13.0`에 다시 보내 보니 호출마다 입력 토큰이 1.14.0 기록과 같았고, 출력 토큰도 합계가 같았다(입력 53,120, 출력 3,120). 1.13.0 실행은 토큰을 기록하지 않았지만, 같은 세 벤치의 비용은 두 버전이 같다고 본다.
- **지연:** 호출 지연 중앙값은 0.74초, 95백분위는 0.95초다. 동시 256호출이라 서버 대기 시간이 들어 있다. workers 8로 잰 1.13.0(0.23초)과는 견줄 수 없다.
