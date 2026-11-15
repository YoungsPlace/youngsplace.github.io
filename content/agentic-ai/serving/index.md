---
title: "에이전트의 지연을 나눠 보기 · 모델 서빙, 도구, 재시도"
date: 2026-11-16T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
weight: 110
summary: "가상의 서빙 요청을 구간별로 추적하고, 같은 시간 예산에서 재시도 전후를 비교했다."
tags: ["Agentic AI", "NVIDIA NIM", "OpenTelemetry", "Observability"]
showTableOfContents: true
showAuthor: false
showHero: false
---
사용자 화면에 ‘70ms 만에 시간 초과’가 올라왔다. 큐에서 오래 기다렸는지, 모델이 도구 선택을 끝내지 못했는지, 일정 API에서 시간을 썼는지는 전체 지연만으로 알기 어렵다. 같은 요청 안에서 구간을 나눠 기록했다. [9편의 정확도와 호출 비용](/agentic-ai/evaluation/)

## 앱 실행기와 모델 서버

| 위치 | 맡은 일 | 어떤 값을 볼까 |
|---|---|---|
| 에이전트 앱 | 도구 선택, 정책, 승인, 시간 예산, 최종 응답 | 결정 횟수, 실패 지점, 전체 시간 |
| 모델 서빙 | 모델을 올리고 토큰 생성 요청을 받음 | 큐 길이, 첫 토큰 시간, 토큰 처리량, GPU·메모리 |
| 검색과 업무 도구 | 문서 검색, 일정 읽기, 확인된 예약 처리 | 요청별 지연, 재시도, 오류 상태 |
| 수집기 | 실행 ID로 구간을 연결 | 첫 실패 지점, 경로별 p95, 버전 변화 |

NVIDIA NIM for LLM and VLM은 vLLM 또는 SGLang을 백엔드로 쓰는 모델 서빙을 제공한다. 공식 API에는 채팅 생성, 준비 상태, 버전과 메트릭을 보는 엔드포인트가 있다. `/v1/health/live`가 성공해도 모델이 준비될 때까지 `/v1/health/ready`는 기다릴 수 있다. 에이전트가 도구를 검증하거나 재시도를 결정하는 동작은 모델 서버와 따로 배포할 수 있다. 여기서는 NIM을 띄우지 않고 API 문서와 구간을 대응해 봤다. [API Reference · NVIDIA NIM for LLM and VLM, 2026-11-16 확인](https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html)

동시에 여러 요청을 처리할 때는 큐에서 기다리는 시간과 토큰을 처리하는 시간을 나눈다. 요청을 묶으면 GPU 처리량을 늘릴 수 있지만, 도착량이 늘면 첫 응답의 대기 시간과 꼬리 지연이 달라질 수 있다. 모델과 입력 길이, 동시 요청 조건을 고정하고 별도로 도구 응답 시간까지 봐야 병목을 구분할 수 있다. [Benchmarking · NVIDIA NIM, 2026-11-16 확인](https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html)

{{< mermaid >}}
flowchart TD
    R["같은 요청 ID와 전체 응답 기한"] --> Q["큐 대기 구간"]
    Q --> M["모델 구간"]
    M --> T["도구 조회 구간"]
    T -->|일시 오류와 시간 남음| B["제한된 재시도 대기"]
    B --> T
    T -->|응답 완료 또는 실패| F["초과 지점과 결과 기록"]
    Q -->|기한 소진| F
    M -->|기한 소진| F
    B -->|기한 소진| F
{{< /mermaid >}}

## 같은 총 기한으로 두 실행 설정 비교

[serving.py](serving.py)는 Python 3.13.18에서 합성 시간표를 읽는다. 큐 8ms, 모델 30ms가 기본이고, 모델이 느린 경우에는 77ms다. 첫 도구 타임아웃은 30ms, 재시도 대기는 10ms로 넣었다. `baseline`은 읽기를 한 번 시도하고 `one_retry`는 타임아웃 뒤 한 번 더 시도한다. 아래 시간은 가상 시계에서 나온 값이다.

```bash
python3 serving.py transient baseline
python3 serving.py transient one_retry
python3 serving.py transient one_retry --budget 70
python3 serving.py slow_model one_retry --budget 70
```

| 입력과 총 기한 | 기본 설정 | 재시도 한 번 |
|---|---|---|
| 정상 110ms | `ok` · 53ms | `ok` · 53ms |
| 첫 도구만 타임아웃, 110ms | `timeout` · 68ms | `ok` · 93ms |
| 첫 도구만 타임아웃, 70ms | `timeout` · 68ms | 대기 중 `deadline` · 70ms |
| 모델이 느림, 70ms | 모델 중 `deadline` · 70ms | 모델 중 `deadline` · 70ms |
| 도구가 두 번 모두 타임아웃, 110ms | `timeout` · 68ms | `timeout` · 108ms |

두 70ms 사례는 끝난 시간은 같았다. 첫 사례에서는 도구가 실패한 뒤 재시도 대기 중 시간이 끝났고, 두 번째 사례는 도구에 도달하지 못했다. 서버 배포에서는 새 모델·프롬프트·도구 계약·재시도 설정의 버전을 실행 기록에 함께 두면 비교하기 쉽다. 변경 전후 같은 회귀 사례로 정상 성공률과 마감 초과를 비교하고, 운영 샘플에서는 요청량과 오류 유형도 같이 본다.

OpenTelemetry의 GenAI 에이전트 스팬 문서에는 에이전트 호출, 계획, 도구 실행에 대한 속성이 있다. 문서 상태는 Development다. 예제 JSON은 구간 이름과 상대 시간을 확인할 수 있는 단순 예제이며, 수집기 전송이나 공식 스팬 포맷 구현은 포함하지 않았다. 실서비스에서 원본 프롬프트와 도구 응답을 수집하면 사용자 데이터가 섞일 수 있어, 먼저 식별자·시간·상태만으로 원인을 찾을 수 있는지 본다. [Semantic Conventions for GenAI agent and framework spans · OpenTelemetry, 2026-11-16 확인](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)

## 첫 응답까지 기다린 시간과 생성 속도

LLM 요청은 대부분 입력 전체를 읽는 prefill과 토큰을 하나씩 내는 decode로 나뉜다. 처음 글자가 나올 때까지의 시간(TTFT)은 클라이언트와 서버 간 왕복, 큐 대기, 필요한 로딩, 입력 처리와 첫 출력 전송까지 포함할 수 있다. 그 뒤 토큰이 나오는 간격, 전체 응답 시간, 출력 토큰 처리량은 서로 다른 값이다. 스트리밍을 켜면 첫 응답은 빨리 보이더라도 전체 생성이 끝나는 시점은 별도로 측정한다. [NVIDIA NIM Benchmarking](https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html)

[ollama_timings.py](ollama_timings.py)는 Ollama 스트리밍을 읽으며 최초 비어 있지 않은 응답 조각과 최종 메시지의 시각을 구한다. 서버의 마지막 JSON에는 `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`이 있고 시간 값은 나노초다. TTFT는 클라이언트에서 쟀고 표의 prefill과 decode는 서버가 보고한 값이다. 하나의 응답 조각이 정확히 한 토큰이라는 가정은 두지 않았다. [Generate a response · Ollama API](https://docs.ollama.com/api/generate)

```bash
ollama pull gemma3:4b
python3 ollama_timings.py
python3 ollama_timings.py --distinct-prefix
```

Apple M5 Pro, Ollama 0.40.11의 Gemma 3 4B에서 워밍업 한 번 뒤 짧은 입력과, 같은 내용 앞에 문장 60줄을 붙인 입력을 번갈아 각 5회 실행했다. 출력 상한은 48토큰이다. 두 번째 실행에서는 요청 시작에 숫자 표식을 바꿔 이전 요청과 긴 입력 시작부를 공유하지 않도록 했다. 모델은 로컬에서 계속 로드된 상태로 요청했다.

| 조건 | 입력 토큰 | 출력 토큰 | 첫 응답 중앙값 | prefill 중앙값 | decode 중앙값 | 전체 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| 같은 입력 반복 · 짧음 | 29 | 48 | 39.52ms | 31.62ms | 624.22ms | 660.59ms |
| 같은 입력 반복 · 김 | 689 | 48 | 41.17ms | 35.51ms | 642.25ms | 692.04ms |
| 시작부 변경 · 짧음 | 32 | 48 | 86.17ms | 72.12ms | 614.56ms | 701.68ms |
| 시작부 변경 · 김 | 692 | 48 | 326.88ms | 323.15ms | 626.83ms | 951.88ms |

입력 시작을 바꾼 조건에서는 660개 토큰을 더 넣었을 때 prefill 중앙값이 72.12ms에서 323.15ms로 늘었다. 출력 토큰 수가 같아 decode 중앙값은 더 가깝게 나왔다. 같은 입력을 반복한 조건에서는 긴 입력과 짧은 입력의 차이가 작았다. 이 실험에서는 캐시 적중률을 서버에서 직접 받지 않았다. 반복 입력과 바뀌는 입력의 지연을 모두 기록해야 같은 길이의 프롬프트에서도 시간 차이를 해석할 수 있다. 각 열을 따로 중앙값으로 냈으므로 표의 단계 시간 합과 전체 중앙값은 일치할 필요가 없다.

## 캐시와 동시성에서 달라지는 병목

앞부분이 같은 요청들은 이미 처리한 입력의 KV 상태를 재사용할 수 있다. vLLM의 Automatic Prefix Caching은 앞 토큰들이 같은 블록을 다시 사용해 prefill을 줄이는 기능으로 소개된다. 첫 입력 처리가 줄어도 새로 생성할 토큰의 decode가 그대로 남을 수 있다. 헤더에 요청마다 다른 긴 값이 붙으면 이후 텍스트가 같아도 처음부터 같은 토큰이 이어지는 길이는 짧아진다. 답변 정확도에 필요한 사용자 자료는 보존하고, 바뀌지 않는 도구 정의를 입력 앞부분에 배치할 수 있다. [Automatic Prefix Caching · vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/)

PagedAttention은 길이가 다른 여러 요청의 KV를 블록 단위로 다뤄 장래의 최대 길이만큼 연속 공간을 먼저 예약할 때 생기는 낭비를 줄인다. 연속 배칭은 요청이 끝나기를 한 묶음 전체에서 기다리는 대신 토큰 처리 단계 사이에 다른 요청을 합류시킬 수 있다. 실제 처리량과 지연은 모델 크기, GPU, 요청의 입력·출력 길이, 스케줄러와 동시성에 따라 달라진다. 이 글의 표는 Mac의 단일 실행 측정이며 GPU 배포의 지연 또는 처리량 수치로 환산하지 않았다. [Efficient Memory Management for LLM Serving with PagedAttention · Woosuk Kwon 외, SOSP 2023](https://arxiv.org/abs/2309.06180)

## 운영 지표를 오류 유형과 같이 보기

요청 한 건의 추적에는 모델 버전, 요청 ID, 도구와 정책 버전, 각 구간 시간, 중단 사유가 들어갈 수 있다. 처리량 집계에서는 요청 ID를 각 시계열의 라벨로 달기보다 모델과 결과 유형처럼 가능한 값이 제한된 필드를 쓴다. 개별 요청의 상세 경로는 트레이스 ID에서 찾을 수 있다. 정상 요청의 TTFT만 보지 말고 도구 타임아웃과 승인 대기가 얼마나 섞였는지도 구분한다. 사람의 승인 응답에 걸린 시간은 답변 생성이나 예약 API의 지연과 같은 서비스 구간으로 해석하기 어렵다.

예를 들어 ‘읽기 요청의 99%가 기한 안에 유효한 결과를 얻는다’를 목표로 두면 일만 건 중 100건이 허용 실패량이다. 제때 반환한 `no_slot`을 유효 응답으로 세는지, 기한이 다 되도록 기다린 결과도 포함하는지부터 정한다. 느린 모델을 바꾸는 조치와 도구 재시도를 조절하는 조치는 다른 추적 구간에 나타난다. 같은 입력으로 배포 전후 회귀를 비교하고, 실제 운영에서는 요청 종류와 도구 실패 비중이 바뀌었는지도 함께 살핀다. [9편의 평가 채점표](/agentic-ai/evaluation/)

GTC Berlin 세션의 공개 영어 자막 중 15:10–15:46에서 실행 경로, 각 단계의 시작·종료와 타이밍을 추적하는 설명을 확인했다. 해당 구간은 ATOF 소개이며, 위 코드는 그 규격을 사용하지 않는다. [Beyond Deployment: Engineering Agents That Learn on the Job · Adel El Hallak, Matt Penn, NVIDIA, 2026-10](https://www.nvidia.com/en-us/on-demand/session/gtcberlin26-sb61117/)
