---
title: "챗봇에서 에이전트까지, 다음 행동은 누가 고를까"
date: 2026-11-16T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
weight: 10
summary: "가상의 실험실 일정을 같은 데이터로 조회하며 챗봇, 고정 워크플로, 에이전트 루프의 차이를 확인했다."
tags: ["Agentic AI", "Ollama", "Python", "Workflow"]
showTableOfContents: true
showAuthor: false
showHero: false
---

“실험실 A에서 가장 이른 빈 시간을 알려 줘. 닫혀 있으면 안내에 나온 대체 실험실을 확인해 줘.”

대답하려면 먼저 A의 운영 상태를 알아야 한다. A가 열려 있으면 그곳의 일정을 조회한다. 닫혀 있으면 그때 받은 안내문에서 대체 실험실 이름을 꺼내고, 그곳의 상태를 다시 확인한다. 대체 장소를 처음부터 알 필요는 없다. 첫 도구 응답을 읽고 나서야 두 번째로 할 일이 정해진다.

NCA-AMAD에서 정리했던 에이전트 구조를 이 문제에 대입했다. 가상의 실험실 데이터에 고정 워크플로와 에이전트를 각각 연결하고, 정상 응답과 빈 결과, 대체 장소, 반복을 비교했다. [NCA-AMAD의 Agentic AI 노트](/nca-amad/chapter-01-agentic-ai/) · [Agentic AI 학습 노트](/agentic-ai/)

## 챗봇, 워크플로, 에이전트

챗봇은 사람이 메시지를 주고받는 인터페이스다. 그 뒤에서 모델을 한 번 부를 수도 있고, 고정 워크플로를 실행하거나 에이전트에 일을 넘길 수도 있다. 이 예제에서는 실행 경로를 어디서 정하는지에 따라 세 구성을 비교했다. NVIDIA는 목표, 메모리, 계획, 도구로 에이전트 구성 요소를 설명한다. Anthropic은 실행 경로를 코드가 정하는 경우와 모델이 실행 중에 고르는 경우를 구분한다. [NVIDIA의 LLM 에이전트 입문](https://developer.nvidia.com/blog/introduction-to-llm-agents/) · [Anthropic의 워크플로와 에이전트 구분](https://www.anthropic.com/research/building-effective-agents#what-are-agents)

| 구성 | 다음 행동을 누가 정하나 | A가 닫힌 경우 |
|---|---|---|
| 일정 도구 없는 단일 응답 | 모델이 현재 받은 텍스트로 답한다. | 현재 상태나 빈 시간을 확인할 데이터가 없다. 예제의 `chat` 모드는 이 경계를 `no_live_data`로 표시하는 모형이다. |
| 고정 워크플로 | Python의 순서와 조건문이 다음 호출을 정한다. | A를 확인한 뒤 응답에 있는 대체 장소를 한 번 조회한다. |
| 에이전트 루프 | 모델이 목표와 이전 결과를 보고 행동 하나를 제안한다. Python이 허용 여부를 결정하고 실행한다. | 모델이 대체 장소의 상태 확인을 제안하고, 확인된 상태에 따라 일정 조회를 제안할 수 있다. |

고정 워크플로도 `if`로 경로를 바꿀 수 있다. 여기서는 대체 장소가 한 곳이고 순서도 짧아서, 워크플로가 필요한 분기를 모두 코드로 표현할 수 있다. 도구가 늘고 어떤 자료부터 확인할지가 요청마다 달라지는 경우에는 모델에게 일부 선택을 맡길 여지가 생긴다. 선택 후보가 늘면 모델 호출, 오류 처리, 경로 검증에 드는 비용도 늘어난다.

## 에이전트 루프에 들어가는 것

| 요소 | 이 예제에서 맡는 일 |
|---|---|
| 목표 | `LAB-A`의 첫 빈 시간을 구한다. 닫혔으면 안내된 대체 실험실을 찾는다. |
| 도구 | `room_status(room)`이 운영 상태와 대체 장소를 읽고, `first_slot(room)`이 첫 빈 시간을 읽는다. |
| 상태 | 조회한 장소, 관찰 결과, 시도한 호출을 현재 실행의 메모리에 둔다. |
| 모델 | 현재 관찰을 받아 다음 행동과 대상 장소를 JSON으로 제안한다. |
| 런타임 | 제안이 허용 목록과 현재 상태에 맞는지 확인한다. 도구를 실행하고 결과를 기록하며 종료를 처리한다. |

{{< mermaid >}}
flowchart TD
    G["목표 · LAB-A의 첫 빈 시간 확인"] --> M["모델 · 관찰을 읽고 행동 하나 제안"]
    M --> V["Python · 실행 조건 검사"]
    V -->|"허용된 조회"| T["실험실 상태 또는 빈 시간 조회"]
    T --> S["상태에 관찰 추가"]
    S -->|"횟수가 남음"| M
    S -->|"한도에 도달"| E["step_limit"]
    V -->|"근거가 있는 종료"| F["도구 결과로 시간 또는 빈 일정 반환"]
    V -->|"근거 부족 또는 반복"| R["거부 이유를 기록하고 종료"]
{{< /mermaid >}}

코드에서는 모델이 `{"action": "first_slot", "room": "LAB-B"}`처럼 다음 행동을 보낸다. `LAB-B`는 유효한 장소 ID다. 그래도 아직 열려 있다는 응답이 없으면 실행하지 않는다. JSON 스키마는 값의 모양을 좁히고, 그 값이 현재 단계에서 가능한지는 Python이 확인한다. 이 예제에서 JSON은 행동을 요청하기 위한 형식이다. Ollama의 `format`에 스키마를 전달하고, 실제 함수 호출은 Python에서 한다. [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)

```python
# agent_loop.py에서 조건 검사와 관찰 기록에 해당하는 부분
if action == "first_slot" and observations.get(
    ("room_status", room), {}
).get("status") != "open":
    return rejected(action, room, "room_not_confirmed_open", trace)

key = (action, room)
if key in observations:
    return rejected(action, room, "repeat", trace)
result = tools[action](data, room)
observations[key] = result
trace.append({"action": action, "room": room, "result": result})
```

`finish`도 같은 방식으로 검증한다. 해당 장소에서 `first_slot`을 완료한 기록이 있으면 그 결과에 있는 시간을 돌려준다. 도구가 `null`을 반환했다면 장소는 확인했고 빈 시간이 없다는 뜻이다. 일정 조회 기록이 없이 `finish`를 요청하면 `missing_evidence`에서 멈춘다. 모델 출력 스키마에는 행동 이름과 장소만 들어 있다.

## 한 번의 모델 호출에서 실제로 오가는 것

대화 API에는 메시지 배열과 모델 식별자를 보낸다. 도구를 붙인 에이전트에서는 여기에 호출 가능한 도구의 이름, 인자 정의와 간단한 설명이 들어간다. 모델은 그 입력을 보고 다음 메시지를 생성한다. 메시지에 도구 호출이 있으면 앱이 목록에서 함수를 찾아 인자를 검사하고 실행한다. 이어서 반환값을 도구 결과 메시지로 붙여 다시 모델을 부른다. Ollama의 네이티브 도구 호출 예제에서는 모델 응답의 `tool_calls`를 꺼낸 뒤 함수의 결과를 `tool` 역할의 메시지로 보낸다. 이 글의 예제는 같은 왕복을 `observations`라는 JSON 배열로 구현한다. 모델 출력은 스키마에 맞춘 단일 제안이고, 조회는 로컬 Python 함수에서 일어난다. [Tool calling · Ollama](https://docs.ollama.com/capabilities/tool-calling)

| 요청 단계 | 모델에게 보이는 정보 | 앱이 보관하는 값 |
|---|---|---|
| 처음 | A에서 시간을 찾으라는 목표와 행동 종류 | 호출 예산, 초기 대상 A, 빈 관찰 목록 |
| A 상태 확인 뒤 | A가 닫혔고 B가 대체 후보라는 응답 | A의 상태와 B를 조회할 수 있다는 사실 |
| B 상태 확인 뒤 | B가 열렸다는 응답 | B의 일정 조회를 허용하는 선행 조건 |
| B 일정 확인 뒤 | B의 첫 빈 시간이 14:00이라는 응답 | 최종 답을 구성할 근거와 그 출처 |

한 번의 모델 호출이 내부에서 여러 토큰을 생성하는 것과, 앱이 도구 결과를 받아 여러 번 모델을 부르는 것은 비용이 쌓이는 지점이 다르다. B까지 가는 루프에서 모델 결정 네 번은 상태 조회 두 번, 일정 조회 한 번, 종료 제안 한 번에 해당한다. 로컬 앱은 종료 제안 뒤에 일정 함수가 준 값을 그대로 반환한다.

## 관찰로 상태를 갱신하고 행동 가능 범위를 좁히기

모델에 알려진 A와 B의 상태는 지금까지 도구가 돌려준 관찰에 한정된다. 부분 관측 의사결정에서는 실제 환경 상태와 관찰을 구분하고, 이전 행동과 새 관찰을 바탕으로 상태에 대한 믿음(belief)을 갱신한다. 이 예제는 관찰된 값과 아직 확인하지 않은 값을 구별하는 결정적 상태표를 둔다. 그래서 ‘B가 대체 장소’와 ‘B가 열림’이 별개 사실이다. [Planning and Acting in Partially Observable Stochastic Domains · Leslie P. Kaelbling, Michael L. Littman, Anthony R. Cassandra, 1998](https://cs.brown.edu/research/pubs/techreports/reports/96/cs96-08.pdf)

```text
관찰 이력 hₜ = 지금까지의 (행동, 반환값)
앱 상태 xₜ = 그 이력에서 읽은 대상, 조회 결과, 남은 예산
허용 행동 A(xₜ) = 범위와 선행 조건을 충족하는 행동–대상 쌍
xₜ₊₁ = update(xₜ, 실행한 행동, 새로 받은 관찰)
```

같은 `first_slot` 문자열도 A가 열린 상태에서는 유효하고 A가 닫혔을 때는 유효하지 않다. JSON 스키마의 `enum`은 문자열 종류를 제한한다. 어느 장소가 실제로 열렸는지, 다른 사용자의 정보가 섞였는지, 근거가 어떤 버전인지는 실행 상태를 읽어 검사한다. 둘을 분리하면 아래의 오류가 서로 다른 위치에서 잡힌다.

| 오류 형태 | 예 | 검사 위치 |
|---|---|---|
| 표현 위반 | `room`이 없거나 숫자임 | 출력 형식 |
| 상태 위반 | 열림을 확인하기 전에 일정 조회 | 실행 직전 선행 조건 |
| 반복 위반 | 같은 관찰을 다시 요청하고 정보가 늘지 않음 | 호출 이력과 예산 |
| 근거 위반 | 관찰한 일정 없이 완료 제안 | 종료 후조건 |
| 외부 상태가 변함 | 조회 후 예약 전 다른 사람이 선점 | 예약 서비스의 재확인과 원자적 쓰기 |

이력 전체를 매번 프롬프트에 되보내면 n번째 결정까지 보내는 관찰의 수는 대략 `0 + 1 + … + (n−1)`로 증가한다. 관찰 크기가 비슷하다는 가정에서 누적 입력은 O(n²)이다. 모델 컨텍스트에는 현재 후보와 최신 상태만 보내고, 상세 호출 기록은 실행 로그에 남기는 방법이 있다. 이때 압축에서 빠진 근거가 이후 판단에 필요한지 확인해야 한다. [3편의 컨텍스트와 실행 상태](/agentic-ai/memory-layers/)

## 같은 일정 조회를 세 방식으로 실행하기

대체 장소까지 조회되는 순서가 달라지는지 보기 위해 합성 데이터 네 가지를 넣었다. `workflow`는 대체 장소를 최대 한 번 확인한다. `scripted`는 관찰 결과에 따라 미리 적어 둔 행동을 반환해서 런타임 검사를 재현한다. `ollama`는 같은 런타임에 Gemma 3 4B를 연결해 실제 다음 행동을 고른다.

[실행 코드 다운로드](agent_loop.py)

```bash
python3 agent_loop.py chat fallback
python3 agent_loop.py workflow fallback
python3 agent_loop.py scripted fallback
python3 agent_loop.py scripted cycle
python3 agent_loop.py scripted fallback --max-steps 2

# 별도 터미널에서 OLLAMA_NO_CLOUD=1 ollama serve를 시작한 뒤
ollama pull gemma3:4b
python3 agent_loop.py ollama fallback
```

Apple M5 Pro의 Python 3.13.18과 Ollama 0.40.11에서 실행했다. Gemma 3 4B의 로컬 모델 ID는 `68ceab7e468a`였다. 모델 요청에는 `temperature=0`, `seed=42`, `num_ctx=4096`, 출력 상한 80토큰을 적용했다. 아래는 케이스별 한 번의 실행이다.

| 입력 | 고정 워크플로 | 루프 · 정해 둔 판단 | 루프 · Gemma 3 4B |
|---|---|---|---|
| `regular` · A가 열려 있고 10:30이 비어 있음 | A 10:30 | A 10:30 | A 10:30 |
| `fallback` · A는 닫혔고 B의 14:00이 비어 있음 | B 14:00 | B 14:00 | `repeat` |
| `empty` · A는 열려 있고 빈 시간이 없음 | `no_slot` | `no_slot` | `no_slot` |
| `cycle` · A는 B를, B는 A를 안내하며 둘 다 닫힘 | `unavailable` | `repeat` | `repeat` |

대체 장소가 열린 사례에서 두 루프가 어디서 갈라지는지 실행 기록으로 확인했다.

| 순서 | 정해 둔 판단 | Gemma 3 4B |
|---|---|---|
| 1 | A 상태 조회 → 닫힘, 대체 장소 B | A 상태 조회 → 닫힘, 대체 장소 B |
| 2 | B 상태 조회 → 열림 | B 상태 조회 → 열림 |
| 3 | B 빈 시간 조회 → 14:00 | B 상태 조회를 다시 제안 → 반복으로 거부 |
| 4 | B로 종료 → 도구 응답의 14:00 반환 | |

둘 다 B가 열려 있다는 관찰을 받았다. Gemma는 B의 상태를 다시 물었다. 런타임은 세 번째 제안을 반복으로 거부했고, 응답의 `slot`은 `null`이었다. 이때 `repeat`는 제한 조건이 작동했다는 뜻이며, 빈 시간을 확인했다는 뜻의 `found`와 구분된다.

## 넓은 선택지와 상태로 제한한 선택지 비교

기본 모드의 스키마에는 행동 이름 세 개와 장소 두 곳이 항상 들어간다. 이전 호출 이력을 프롬프트에 보내더라도, 코드상 허용되는 다음 행동은 훨씬 적을 수 있다. 같은 Gemma에 현재 단계의 `action`과 `room`을 각각 한 개만 허용하는 `masked` 모드를 추가했다. 이 모드는 정해 둔 선택기에서 다음 이동을 얻은 뒤 그 한 쌍으로 스키마를 만든다. 모델을 고정한 채 허용 범위가 바뀌었을 때 도구 경로를 비교하기 위한 예제다.

```bash
python3 agent_loop.py ollama fallback
python3 agent_loop.py masked fallback
python3 agent_loop.py masked cycle
```

| 조건 | 전체 열거형을 보낸 Gemma | 현재 단계의 한 쌍만 보낸 Gemma |
|---|---|---|
| 정상 A | A 10:30 | A 10:30 |
| A 닫힘, B 열림 | B 상태 재조회로 `repeat` | B 일정 조회를 거쳐 14:00 반환 |
| 빈 일정 | `no_slot` | `no_slot` |
| 대체 장소 순환 | `repeat` | A와 B를 확인한 뒤 A 재조회가 거부돼 `repeat` |

추가 실행에서도 모델 ID와 추론 설정은 위 표와 같았다. `fallback`은 네 번, `cycle`은 세 번의 결정 기록을 남겼다. 제한된 모드가 `fallback`을 끝낸 것은 올바른 행동을 알고 있는 코드가 스키마 선택지를 정했기 때문이다. 가능한 행동이 이렇게 한 개로 정해지면 모델 호출을 건너뛰고 고정 워크플로에서 바로 실행할 수 있다. 문서 후보가 많거나 사용자 요구가 모호한 구간에서는 모델에게 여러 후보를 순위화하게 하고, 실행기는 권한과 근거 조건을 고정하는 구성이 가능하다. 여러 행동–대상 쌍을 허용할 때 두 필드의 목록을 따로 열면 허용하지 않은 조합도 만들어질 수 있다. 행동 ID 하나로 쌍을 표현하거나 쌍 자체를 검증한다.

도구를 더 붙일 때는 허용 행동 외에 종료의 진전 조건도 정한다. 읽기 성공으로 상태가 바뀌면 다음 단계로 넘어가고, 변화가 없으면 원인을 분류한다. 네트워크 타임아웃 뒤의 제한된 재시도는 새로운 시도다. 동일한 성공 응답을 소비하지 않고 계속 요청하는 반복과 분리해 센다. [2편의 타임아웃과 재계획](/agentic-ai/replanning/)

## 언제 종료할지 정하기

빈 일정, 닫힌 장소, 반복, 예산 소진을 같은 실패로 처리하면 다음에 할 일을 판단하기 어렵다. 이 예제는 `found`와 `no_slot`을 조회 결과로 두고, `repeat`와 `step_limit`을 실행 중단 사유로 뒀다. `scripted fallback --max-steps 2`로 실행하면 A와 B의 상태만 확인한 뒤 `step_limit`로 끝난다. 아직 B의 일정을 조회하지 않은 시점이다.

단계 상한은 모델이 한 번 제안할 때 한 단계로 센다. 조회 세 번과 종료 제안 한 번을 거치는 대체 사례에는 네 단계가 필요하다. Ollama 요청 한 번에는 30초 제한을 뒀다. 단계 상한은 전체 처리 시간과 별개의 제약이다. 서비스에서는 사용자에게 응답해야 하는 시간도 별도로 두고, 오래 걸리는 도구를 어떻게 취소하거나 이어 갈지 정하게 된다.

실험실 데이터는 실행 중에 바뀌지 않는다. 실제 일정이라면 운영 상태를 본 시각과 예약 가능 시간을 본 시각 사이에 값이 바뀔 수 있다. 조회 결과에는 시점이 필요하고 예약 같은 쓰기에는 실행 직전 재확인과 별도 권한이 필요하다. 예제의 도구는 실험실 상태와 일정만 읽는다.

## 결과를 평가하는 법

성공 조건을 `status`, `room`, `slot`으로 적어 두면 말투나 설명 길이와 무관하게 확인할 수 있다. 같은 14:00이라도 `LAB-A`의 시간으로 보고하면 실패다. 도구 호출마다 이름, 대상, 반환값, 거부 사유를 남기면 판단 문제인지 런타임 문제인지 좁힐 수 있다.

- **목표 달성**은 요청한 장소와 시간이 맞는지, 빈 일정이면 그 상태를 제대로 반환하는지 본다.
- **호출 정확성**은 올바른 도구와 장소를 골랐는지, 선행 확인을 건너뛰지 않았는지 본다.
- **운영 비용**은 모델 호출 수, 전체 지연, 실패 후 복구 시간을 본다. 이번에는 호출 경로를 비교했다.

현재 예제의 합성 데이터에서는 고정 워크플로가 필요한 대체 분기를 처리했다. 과정이 더 길어질 때도 같은 워크플로로 충분한지는 새로운 분기와 과업을 넣어 다시 비교할 문제다. 서로 다른 에이전트 모델의 일반 성능을 비교하려면 같은 조건의 반복 실행과 예제 작성에 쓰지 않은 케이스가 필요하다. [AI Agents That Matter](https://arxiv.org/abs/2407.01502)

## 다시 생각해 볼 질문

1. A가 닫힌 뒤 B가 열려 있다는 응답을 받았다. B에서 `finish`를 바로 제안하면 어떤 기록이 빠져 있나?
2. 일정 도구가 `null`을 반환한 경우와 단계 제한으로 일정 도구를 부르지 못한 경우를 화면에서 각각 어떻게 보여 줄까?
3. 대체 장소 세 곳을 응답으로 받으면 어떤 기준으로 하나를 고를까? 고정 조건으로 정할 수 있는 부분은 어디까지일까?

## 참고 자료와 영상

- **Introduction to LLM Agents** · Tanay Varshney, NVIDIA, 2023-11-30. 목표와 상태, 계획, 도구가 하나의 실행 구조로 연결되는 배경을 읽었다. [NVIDIA 기술 블로그](https://developer.nvidia.com/blog/introduction-to-llm-agents/)
- **Building effective agents** · Anthropic Engineering, 2024-12-19. 고정된 실행 경로와 모델이 경로를 고르는 구조를 비교할 때 참고했다. [Anthropic](https://www.anthropic.com/research/building-effective-agents)
- **Structured Outputs** · Ollama, 2026-11-16 확인. 이 글의 실행 코드에서 JSON 스키마를 전달하는 API 형식을 확인했다. [Ollama 문서](https://docs.ollama.com/capabilities/structured-outputs)
- **AI Agents That Matter** · Sayash Kapoor 외, 2024-07-01 초판. 정확도와 비용, 별도 평가 사례를 함께 보는 근거다. [논문](https://arxiv.org/abs/2407.01502)
- **Agentic AI 101** · Erik Pounds, NVIDIA Developer, YouTube 게시 2026-03-31. 영어 자막에서 03:16–03:37은 문제 분해와 계획·검토, 11:26–12:13은 기술과 도구 및 API를 구분하는 설명을 확인했다. [계획부터 보기](https://www.youtube.com/watch?v=ETq3ZTqxFlo&t=196s) · [도구 설명부터 보기](https://www.youtube.com/watch?v=ETq3ZTqxFlo&t=686s)
