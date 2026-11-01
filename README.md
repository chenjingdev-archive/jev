# sif

semantic if. TypeSafe의 판단 모델 Jev를 평범한 제어문처럼 쓴다.

Jev는 참/거짓, 선택, 점수 세 가지만 돌려주는 모델이다. 조건을 코드로 쓸 수 없는 것,
예를 들어 "이 문의가 급한가", "이 명령이 위험한가"를 if문 조건에 넣을 수 있게 된다.

## 결론 (2026-10-04)

**판단 모델은 Jev API를 쓴다.** Jev-like 15개와 Cloudflare Clef-flash를 같은 문항으로 시험했다. 2026-10-25에는 OpenAI Decisions API를 더했고, Jev 1.14.0으로도 다시 돌렸다. 수능, 게임 판단, 제작자 벤치 어디에서도 Jev보다 확실히 나은 시스템은 없었다.

| 시스템 | 수능 884문항 | 게임 판단 | 제작자 벤치 기존 19세트 | Nimble 13세트 |
|---|---:|---:|---:|---:|
| Jev 1.14.0 (TypeSafe API, 10/25) | **70.7%** | **0.414** | **0.826** | 0.759 |
| Jev 1.13.0 (TypeSafe API) | 69.3% | 0.377 | 0.823 | 0.759 |
| d1 (Liquid API) | 69.0% | 0.331 | 0.780 | **0.792** |
| OpenAI Decisions (`gpt-6-luna` API) | 65.8% | 0.295 | 0.784 | 0.711 |
| Clef-flash (공개 가중치, amd ROCm) | 60.9% | 0.224 | 0.798 | 0.768 |
| Nimble-9B (공개 가중치) | 59.8% | 0.241 | 0.607 | 0.761 |
| Kev-9B (공개 가중치) | 54.2% | 0.214 | 0.775 | 0.756 |

- **Jev 1.14.0은 게임에서 1.13.0보다 높다**(0.414 대 0.377, p = 0.015). 수능(70.7% 대 69.3%, p = 0.17)과 제작자 벤치(32세트를 모으면 p = 0.49)에서는 차이를 확인하지 못했다. 세 벤치 53,969호출을 동시 256호출로 3분 49초에 돌렸고 $1.81이 들었다. 가격과 토큰 수가 1.13.0과 같아 비용 차이는 없다(`claims/PROTOCOL.md` 8절). 아래 비교는 1.13.0 기준이다.
- **Jev와 견줄 만한 것은 d1 하나다.** 수능에서는 차이를 확인하지 못했고(p = 0.86), 게임에서는 경계에서 낮다(p = 0.05). Nimble 13세트에서는 Jev보다 높지만, d1은 학습 데이터를 공개하지 않아 실력인지 학습 노출인지 가릴 수 없다. 확률 보정이 거칠고(ECE 수능 0.066 대 0.023, 게임 0.140 대 0.054), 버전을 고정할 수 없는 무료 API다.
- **OpenAI Decisions API도 Jev보다 낮다.** 수능(p = 0.022)과 게임(p = 0.002) 모두 그렇다. 수능 격차는 대부분 답을 거절한 32문항에서 나온다. 그 문항을 빼면 Jev와 차이를 확인하지 못한다(p = 0.34). 게임에서는 확률 보정이 가장 좋다(ECE 0.022). 제작자 벤치 32세트 중 Jev보다 유의하게 낮은 세트는 11개, 높은 세트는 2개다. 세 벤치 53,969호출에 $2.76이 들었다(`claims/PROTOCOL.md` 7절).
- **공개 가중치 모델 중에서는 Clef-flash와 Nimble이 가장 높다.** 그래도 수능과 게임에서는 Jev보다 확실히 낮다(수능 p < 1e-7). Clef-flash는 영어 분류형 세트(금융 감성, WinoGrande, 의도 분류)에서는 Jev를 이긴다. 추론·사실 확인 세트(BBH, RAGTruth, JudgeBench)와 한국어 시험에서는 진다. 영어로 물어도 한국어보다 유의하게 낫지 않았다.
- **직접 돌리는 비용이 크다.** Clef-flash 원본(BF16)은 amd GPU에서 문항당 약 2.3초 걸렸다. Jev API는 0.2초다(기기가 달라 직접 비교는 아니다). 제작자 벤치 45,333회에 약 17시간이 걸렸다. Clef 27B는 amd 메모리 한도에 걸려 올리지 못했다.
- **미세조정으로 따라잡기도 어려웠다.** Jeff-0.8B를 사목 문항으로 학습시켰지만 학습 문항조차 외우지 못했다(`gamebench/tune/c4/`).

모델 가중치, 체크포인트, 가상환경, Jev-like 소스 사본은 지웠다. 남은 것은 결과다. 보고서와 점수는 각 README, `report.json`, `scores.json`에 있고, 호출별 원자료는 `archives/`에 있다. 다시 돌리려면 각 README에 적힌 출처에서 모델을 새로 받아야 한다.

## 저장소 구조

Jev로 하는 실험은 전부 이 저장소에 모은다. 라이브러리 하나, 실험 여럿, 데모 프로젝트 여럿.

| 경로 | 무엇 |
|---|---|
| `src/sif/` | `sif` 라이브러리 (Python). 위 사용법. |
| `tests/` | `sif` 테스트 |
| `examples/` | `sif` 데모 스크립트 |
| `experiments/<이름>/` | 측정 실험 하나당 폴더. 스크립트 + 결과 + `README.md`(질문·방법·수치). `branches`, `limits`, `hangul`, `chess`, `vision`. |
| `tetris/` | Jev가 플레이하는 테트리스 (Node). 자체 `README.md`와 `docs/neurons-spec.md`. |
| `guardrail/` | 튜닙 Safety Check를 흉내낸 한국어 유해성 가드레일 (Python). 문장 → 카테고리 12개 확률 + 심각도 4단계. 자체 `README.md`. |
| `2048-puzzle/` | Jev가 플레이하는 2048, 타일이 그림 조각 (Node). **종료된 실패 실험** — 성적이 전부 엔진과 프롬프트에서 나왔다. 사유와 측정표는 자체 `README.md`. |
| `bug-hunter/` | Python 함수를 주루룩 먹이면 버그 냄새 10개 확률 + 심각도로 위험한 순서로 정렬하는 버그헌터. 자체 `README.md`. |
| `gomoku/` | Jev가 백을 두는 오목 (Python). 엔진이 후보 자리마다 사실 한 줄을 붙인 정답지를 주고 Jev가 고른다. 상대는 사람·엔진 봇·Claude Opus·Gomocup 엔진 Rapfi. 자체 `README.md`. |
| `chess/` | Jev 체스 판독과 엔진 대결 실험. Lichess CC0 국면 판독 화면은 :3461, 엔진 수와 Jev 개입을 분리해 보여주는 3D 아레나는 :3470. 후보 선택에서는 Stockfish 심판 점수로는 무작위보다 나은 수를 골랐지만, 그 수를 끝까지 둬 보면 무작위와 같은 결과였고 대국 점수 개선은 아직 확정되지 않았다. 탐색 깊이를 어디에 쓸지 고르는 자리에서는 코드 기준선을 넘지 못했다. 같은 과제가 질문 문구에 따라 0.507과 0.708로 갈렸다는 것이 그 절의 실제 교훈이다. 자체 `README.md`. |
| `go/` | 9줄 바둑에서 GNU Go 후보 중 Jev가 고르는 실험 (Python). 덤 7.5로 무승부가 없어 체스의 반복 무승부 문제를 규칙으로 없앤 판. 좌표 대신 엔진 라벨(수비·공격·연결·집)을 주면 가려내는지 본 오프라인 구분 시험 포함. 결과와 반사실 재현은 자체 `README.md`. |
| `sliding-puzzle/` | 그림을 5×6으로 자른 슬라이딩 퍼즐 (Python). Jev는 "다음에 벗길 줄"만 고르고 밀기는 코드가 BFS로. 숫자는 정확히 따르지만 어느 숫자가 중요한지는 못 찾고, 지시문에 규칙을 적어 주면 따른다. 3×3 한 칸씩 버전부터의 과정이 자체 `README.md`. |
| `csat/` | 수능 지식 벤치마크 (Python). 2027학년도 9월 모평 객관식 884문항을 Jev와 Jev-like 14개, Clef-flash에 같은 텍스트로 준다. Jev 69.3%, d1 69.0%(차이 확인 못 함), Clef-flash 60.9%, Nimble-9B 59.8%, Kev-9B 54.2%. Clef-flash 결과는 `csat/bench/clef/`. 그림은 글 설명으로 옮긴 입력이라 절대 정확도는 확정치가 아니다. 결과는 `csat/bench/README.md`, 2026 수능 전과목과 ㄱㄴㄷ 형식 진단은 `csat/README.md`, 시험지 출처는 `csat/SOURCES.md`. |
| `gamebench/` | 게임 30개·1,064문항 판단 벤치마크 (Python). 글로 준 규칙대로 판단하는지 잰다. 우연 보정 점수로 Jev 0.377, d1 0.331, Nimble-9B 0.241, Clef-flash 0.224, Kev-9B 0.214이고, 순서는 수능과 거의 같다. 판만 주면 약하고 보기마다 계산된 사실을 붙이면 크게 오른다. 180문항을 영어로 옮겨도 어느 시스템에서도 점수 차이를 확인하지 못했고, Jev와의 격차는 남았다(`gamebench/lang/`). 자체 `README.md`. |
| `claims/` | 제작자 벤치 (Python). 각 Jev-like 제작자가 내세운 공개 벤치 32세트(14,113문항)를 모든 시스템에 같은 조건으로 돌린다. 기존 19세트는 Jev 1위(0.823), Nimble이 고른 13세트는 d1 1위(0.792). 자체 `README.md`. |
| `research/` | Jev 같은 모델이 어떻게 만들어지는지 조사한 자료. 복제본 카탈로그와 원리 정리. |

새 데모 프로젝트는 `tetris/`처럼 루트에 폴더 하나로, 새 측정 실험은 `experiments/` 아래에 둔다.
키는 어디서든 1Password 참조(`op run`)로만 주입하고 디스크에 쓰지 않는다.

## 설치

```sh
uv sync
```

## 사용

```python
import sif

if sif.true(message, "The customer is demanding immediate action"):
    team = sif.switch(message, {"refund": "Wants money back", "shipping": None, "praise": None})
    anger = sif.score(message, ["Calm", "Annoyed", "Furious"], "How angry is the customer?")
```

| 함수 | 역할 | 반환 |
|---|---|---|
| `check(state, question)` | 참일 확률 | float |
| `true(state, question, threshold=0.5)` | 확률을 임계값으로 자른 것 | bool |
| `switch(state, options)` | 갈래 하나 고르기 | str |
| `decide(state, options)` | switch와 같되 확률과 confidence 유지 | Decision |
| `score(state, levels, instructions)` | 순서 있는 단계 위의 위치 | float |
| `ask(state, **questions)` | 여러 질문을 한 요청에 묶기 | dict |

```python
answers = sif.ask(
    message,
    urgent="The message is urgent",
    topic=sif.options(["refund", "shipping", "praise"]),
    anger=sif.scale(["Calm", "Annoyed", "Furious"], "How angry"),
)
answers["urgent"].noul      # 0.987
answers["topic"].choice     # "refund"
answers["anger"].score      # 1.42
```

`ask()` 안의 질문들은 병렬로 평가되고 서로의 답을 보지 못한다.

모든 함수는 `default=`를 받는다. 타임아웃이나 서버 과부하 같은 일시적 오류에는 그 값을
돌려주고 경고를 남긴다. 키가 없거나 요청 형식이 틀린 경우는 default가 있어도 예외를 낸다.

## 실행

API 키는 1Password에 있고 디스크에 쓰지 않는다.

```sh
op run \
  --env-file=<(echo 'TYPESAFE_API_KEY="op://Agent/typesafe jev key/credential"') \
  -- uv run python examples/demo.py
```

데모를 두 번 실행하면 두 번째는 캐시에서 나오고 API 호출이 없다.

## 캐시와 로그

- 캐시: `~/.cache/sif/cache.sqlite`. 키는 model + state + questions. `SIF_CACHE=0` 또는 `sif.configure(cache=False)`로 끈다.
- 로그: `~/.cache/sif/calls.jsonl`. 호출당 한 줄, raw probabilities와 usage, `latency_ms`, `cached`를 남긴다. `SIF_LOG=0` 또는 `sif.configure(log=False)`로 끈다.

`sif.configure(model=..., timeout=..., cache=..., log=...)`로 공유 클라이언트를 재설정한다.

## 테스트

```sh
uv run pytest -q
```

테스트는 클라이언트를 가짜로 바꾸므로 API 키가 필요 없다.
