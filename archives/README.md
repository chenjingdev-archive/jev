# archives

용량이 크거나 파일 수가 많은 원자료를 묶은 것. 저장소 루트에서 풀면 원래 경로에 그대로 풀린다.

```sh
for f in archives/*.tar.gz; do tar xzf "$f"; done
```

| 파일 | 내용 |
|---|---|
| `csat-bench-results.tar.gz` | `csat/bench/results/*/<시스템>/`: 지식 벤치마크(2027 9월 모평) 호출별 기록 40,107개 |
| `gamebench-results.tar.gz` | `gamebench/results/*/<시스템>/`: 판단 벤치마크 호출별 기록과 로그 29,942개 |
| `gamebench-lang-results.tar.gz` | `gamebench/results/lang-en/`, `lang-ko-rerun/`: 언어 시험 호출별 기록 5,552개와 로그 13개 |
| `claims-results-1.tar.gz`, `claims-results-2.tar.gz` | `gamebench/results/claims/<시스템>/`: 제작자 벤치 19세트 호출별 기록(시스템당 32,426개)과 로그. 1은 jev-1.13.0·kev·semif·open-jev·jevmlx·laya, 2는 clm·julia·julia-amd·jeff-0.8b·jeff-2b·jeff-gemma4-e2b·laya-typed·logs |
| `julia-results.tar.gz` | `csat/bench/results/2027-09/julia/`, `gamebench/results/{v1,lang-en,smoke}/julia/`: Julia-1 호출별 기록 |
| `jeff-results.tar.gz` | `csat/bench/results/2027-09/`, `gamebench/results/{v1,lang-en}/`의 jeff-0.8b·jeff-2b·jeff-gemma4-e2b 호출별 기록과 세 태그의 최신 로그 |
| `csat-all_subjects-results.tar.gz` | `csat/all_subjects/results/<모델>/`: 2026 수능 전과목 호출 기록 3,536개 |
| `csat-prompt_study-development.tar.gz` | `csat/prompt_study/development/<변형>/`: 지시문 비교 호출 기록 3,536개 |
| `csat-bench-results-2.tar.gz` | `csat/bench/results/2027-09/`의 d1·nimble·tev1-4b·tev1-0.8b 호출별 기록 17,680개와 로그 |
| `gamebench-results-2.tar.gz` | `gamebench/results/`의 v1·lang-en 추가 시스템(d1·nimble·tev1 두 모델·jeff-0.8b 변형), Nimble 기기 간 대조(`parity-*`, `v1/nimble-mac`), 사목 진단(`c4test*`, `c4noul`, `c4score`) 호출별 기록 58,968개 |
| `claims-results-3.tar.gz` | `gamebench/results/claims/`의 d1·nimble·tev1-4b·tev1-0.8b: 제작자 벤치 32세트 호출별 기록(시스템당 45,333개) |
| `claims-results-4.tar.gz` | `gamebench/results/claims/`의 기존 13개 시스템에 더한 Nimble 13세트 호출별 기록(시스템당 12,907개)과 로그 |
| `clef-results.tar.gz` | Clef-flash 네 태그(`csat/bench/results/2027-09-clef-local-20261003/`, `gamebench/results/{v1,lang-en,claims}-clef-local-amd-20261003/`)의 clef-flash 호출별 기록 54,663개. 같은 태그의 d1·jev-1.13.0·nimble은 원래 태그 폴더를 가리키는 상대 링크로 저장소에 있다. |
| `amd-run-logs.tar.gz` | `amd`에서만 남았던 실행 로그 50개: Jeff 학습·평가 로그, Clef·Clef-flash 서버 로그, shim 로그. 원래 경로가 아니라 `amd-run-logs/` 아래에 풀린다. |
| `chess-matches-full.tar.gz` | `chess/matches/*.json` 중 2MB 넘는 대국 통합 기록 15개(수별 Jev 확률 포함) |

채점 스크립트(`score.py`, `report.py`)는 풀린 폴더를 읽는다. 요약 결과(`report.json`, `scores.json`, PGN, 심판 결과)는 풀지 않아도 저장소에 있다.

체스·바둑의 판별 폴더(`chess/matches/<대국>/`, `go/matches/<대국>/`)는 통합 JSON의 `games`와 같은 내용이라 넣지 않았다.
