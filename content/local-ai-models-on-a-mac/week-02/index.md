---
title: "[2주차] Python으로 로컬 AI 챗봇 만들기: 모델, 가상환경, 스트리밍과 대화 기록"
date: 2026-11-01T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
summary: "모델 크기와 양자화, Python 가상환경부터 Ollama 호출, 스트리밍과 대화 기록까지. 2주차 공부 기록."
tags: ["로컬 AI", "Mac", "Ollama", "Python", "가상환경", "챗봇"]
series: ["Local AI Models on a Mac"]
showTableOfContents: true
showAuthor: false
---

1주차에는 Mac에 Ollama를 설치하고 터미널에서 모델과 대화했다. 2주차에는 Python으로 질문을 보내고, 답변을 스트리밍으로 출력하고, 대화 기록을 관리하는 코드를 공부했다.

범위는 『Build Applications with Local AI Models on a Mac』 CH4~7이다. 모델 크기와 양자화, Python 가상환경, Ollama Python 라이브러리 사용법을 정리했다.

## 1. Python에서 모델까지

{{< mermaid >}}
%%{init: {"flowchart": {"rankSpacing": 24, "padding": 10}}}%%
flowchart TB
    subgraph MAC["내 Mac"]
        T["터미널<br>Python 파일 실행"]
        P["Python 프로그램<br>질문·기록·출력 관리"]
        L["Ollama 라이브러리<br>요청·응답 처리"]
        S["Ollama 서버<br>127.0.0.1:11434"]
        M["모델 실행<br>Gemma 3 가중치로 추론"]

        T -->|"실행"| P
        P --> L
        L -->|"HTTP 요청"| S
        S --> M
        M -->|"생성 결과"| S
        S -->|"HTTP 응답"| L
        L --> P
    end
{{< /mermaid >}}

터미널에서 Python 파일을 실행하면 코드가 Ollama 라이브러리를 호출한다. 라이브러리는 로컬 서버에 HTTP 요청을 보내고, 서버는 모델을 실행해 응답을 돌려준다. Python에서는 이 응답을 출력하거나 파일에 저장한다.

| 구성 요소 | 역할 |
|---|---|
| 터미널 | Python 파일 실행과 출력 확인 |
| Python 프로그램 | 질문, 대화 기록, 출력 관리 |
| Ollama Python 라이브러리 | 서버 요청과 응답 처리 |
| Ollama 서버 | 모델 로딩과 추론 관리 |
| 모델 가중치 | 응답 계산에 사용하는 학습된 수치 |

실습에는 Ollama 실행 도구, `gemma3:4b` 모델 파일, Python의 `ollama` 패키지를 사용한다. [Ollama Python 라이브러리](https://github.com/ollama/ollama-python)

## 2. 모델 크기와 양자화

`gemma3:4b`의 `gemma3`는 모델 계열이고, `4b`는 약 40억 개 규모의 파라미터를 가진 버전을 뜻한다.

[3Blue1Brown의 LLM 설명 영상](https://www.youtube.com/watch?v=HnvitMTkXro)에서는 파라미터를 수많은 다이얼에 비유한다. 다이얼의 값이 바뀌면 다음 토큰에 부여하는 확률도 달라진다. 학습은 예측이 정답에 가까워지도록 이 값들을 조정하는 과정이다.

추론에서는 학습된 가중치를 사용한다. 입력 문맥을 토큰과 벡터로 바꾸고, 여러 층의 계산을 거쳐 다음 토큰의 확률을 구한다. 선택한 토큰을 문맥에 붙여 같은 과정을 반복하면 답변이 만들어진다.

파라미터 수가 늘면 더 복잡한 패턴을 표현할 여지가 커진다. 답변 품질에는 학습 데이터와 학습 방식, 모델 구조, 질문의 종류도 영향을 준다. 내 Mac에서 사용할 모델은 답변 품질, 생성 속도, 메모리 사용량을 함께 보며 고르려고 한다.

수업의 기준 모델은 `gemma3:4b`다. [Gemma 3 모델 정보](https://ollama.com/library/gemma3:4b)

### 양자화: 다이얼의 눈금을 줄이기

미팅에서는 양자화를 숫자 표현을 줄이는 압축으로 설명했다. 가중치 하나를 16비트로 저장하던 모델을 8비트로 표현하면, 가중치 저장에 쓰는 공간이 대략 절반으로 줄어든다. Mac의 메모리에 들어갈 모델을 고르는 데 이 차이가 영향을 준다.

다이얼에 아주 촘촘한 눈금이 있다고 생각해보면, 양자화는 그 값을 더 적은 수의 눈금으로 옮기는 과정이다. 원래 값과 가까운 눈금에 맞추면서 수치에 근사 오차가 생긴다. 더 낮은 비트 수를 사용하면 저장 공간은 줄고, 표현할 수 있는 값의 종류도 줄어든다.

| 정수 표현에 쓰는 비트 수 | 가능한 코드 값의 수 |
|---|---:|
| 8비트 | 256개 |
| 4비트 | 16개 |
| 2비트 | 4개 |

실제 가중치는 그룹별 범위에 맞춰 스케일 등을 정하고 이 코드 값에 대응시킨다. [양자화 개념](https://huggingface.co/docs/transformers/quantization/concept_guide)

{{< mermaid >}}
%%{init: {"flowchart": {"rankSpacing": 24, "padding": 10}}}%%
flowchart LR
    A["가중치 범위 확인"] --> B["스케일과<br>낮은 비트의 값 저장"]
    B --> C["근삿값으로 연산"]
{{< /mermaid >}}

### 눈금 간격 0.25로 계산해보기

간단한 반올림으로 보면 수치가 어떻게 달라지는지 알 수 있다. 눈금 간격을 0.25로 정하고, 원래 값을 가장 가까운 눈금에 옮겨본다.

```text
정수 코드 q = round(원래 값 / 0.25)
복원한 근삿값 = q × 0.25
```

| 원래 값 | 정수 코드 q | 근삿값 | 근삿값 − 원래 값 |
|---:|---:|---:|---:|
| -0.83 | -3 | -0.75 | +0.08 |
| -0.27 | -1 | -0.25 | +0.02 |
| 0.34 | 1 | 0.25 | -0.09 |
| 0.91 | 4 | 1.00 | +0.09 |

스케일과 작은 정수 값으로 원래 가중치에 가까운 수를 표현한 것이다. 모델에서는 많은 가중치에 이런 근사가 적용되며, 기법에 따라 그룹의 크기와 스케일, 중요한 값에 배정하는 정밀도가 달라진다.

### 메모리에는 얼마나 차이가 날까?

가중치 40억 개를 모두 같은 비트 수로 저장하는 조건으로 계산했다. 아래 단위는 십진 GB다.

> 가중치 저장량 = 40억 × 가중치당 비트 수 ÷ 8

| 가중치당 비트 수 | 계산한 저장량 | 16비트 대비 |
|---|---:|---:|
| 32비트 | 16GB | 2배 |
| 16비트 | 8GB | 기준 |
| 8비트 | 4GB | 1/2 |
| 4비트 | 2GB | 1/4 |

실제 모델 파일에는 양자화용 스케일과 메타데이터, 다른 정밀도로 저장한 구성 요소가 함께 들어간다. 실행할 때는 가중치에 더해 KV 캐시와 연산용 버퍼도 메모리를 사용한다.

미팅에서는 macOS와 다른 앱이 쓰는 통합 메모리까지 살펴보며 모델 크기를 고르는 흐름을 설명했다. 활동 상태 보기의 메모리 압력, 모델 로딩 후의 사용량을 함께 기록할 수 있다.

### 낮은 정밀도에서 답변 품질 유지하기

가중치를 근사하면 다음 토큰의 점수에도 변화가 생길 수 있다. 후보 토큰의 점수가 가까운 곳에서는 작은 차이가 이후 문장을 바꾸기도 한다.

양자화는 이 오차를 줄이면서 저장량과 메모리 이동량을 낮추는 방향으로 설계한다. 학습을 마친 뒤 적용하는 PTQ와, 학습 중 양자화 효과를 반영하는 QAT가 있다. 추론 속도에는 실행 엔진과 하드웨어의 저정밀 연산 지원, 변환 비용도 영향을 준다. [Hugging Face 양자화 기법](https://huggingface.co/docs/transformers/quantization/concept_guide#quantization-techniques)

이번 실습에서는 같은 모델의 정밀도를 바꿨을 때 한국어 설명, 계산 문제, 짧은 코드의 결과가 얼마나 유지되는지 비교해보고 싶다.

### 내 모델의 Q4_K_M

`/show info`에서 확인한 양자화 이름은 `Q4_K_M`이다.

| 이름의 부분 | 의미 |
|---|---|
| `Q4` | 주로 4비트 계열로 가중치를 표현 |
| `K` | 가중치를 블록으로 묶어 스케일 등을 저장하는 K 계열 방식 |
| `M` | 같은 계열의 S/M 구분 중 중간 크기의 양자화 구성 |

이 구성은 텐서에 따라 더 높은 정밀도를 섞어 사용한다. llama.cpp 구현에서도 `Q4_K_M`을 선택하면 일부 텐서에 `Q6_K` 등을 배정하는 경로를 볼 수 있다. [llama.cpp 양자화 구현](https://github.com/ggml-org/llama.cpp/blob/dde879bc4f954b809892fd0d660515f42536394a/src/llama-quant.cpp#L631)

내 모델은 `3.9B`, `Q4_K_M`으로 표시됐고 다운로드된 패키지 크기는 `3.4 GB`였다. `/show info`에는 이미지 입력을 처리하는 Projector 정보도 함께 나온다.

### /show info로 본 모델 정보

대화 모드에서 `/show info`를 실행했다.

```text
/show info
```

{{< figure src="ollama-show-info.png" alt="Gemma 3의 모델 구조와 생성 설정을 보여주는 show info 출력" figureClass="study-capture study-capture--portrait" >}}

| 항목 | 출력 값 |
|---|---|
| 모델 구조 | `gemma3` |
| 파라미터 수 | `3.9B` |
| 모델의 컨텍스트 길이 | `131072` 토큰 |
| 임베딩 길이 | `2560` |
| 양자화 | `Q4_K_M` |
| 기능 | `completion`, `vision` |
| Temperature | `1` |
| Top-k | `64` |
| Top-p | `0.95` |

`Projector`에는 `clip`, `419.82M` 파라미터가 표시됐다. 이미지 입력을 처리하는 구성 요소의 정보도 함께 볼 수 있었다.

## 3. 작업으로 살펴본 모델 계열

모델을 고를 때 내가 먼저 비교해보고 싶은 작업을 정리했다. 공부 노트 요약, 한국어 설명, 짧은 코드 작성처럼 같은 질문을 두 모델에 보내 결과를 모아볼 생각이다.

| 해보고 싶은 작업 | 비교할 후보 | 기록할 항목 |
|---|---|---|
| 공부 노트 요약·설명 | Gemma 3 · Llama 3.2 | 빠진 내용, 설명 순서, 응답 길이 |
| 한국어 질의·번역 | Gemma 3 · Qwen 3.5 | 자연스러운 표현, 용어와 형식 유지 |
| 작은 앱의 대화 기능 | Phi-4-mini · Ministral 3 | 메모리 사용량, 첫 응답 시간, 지시 준수 |
| 수학·코드 문제 풀이 | Qwen · DeepSeek-R1-Distill | 답의 정확성, 생성 토큰 수, 실행 가능한 코드 |
| 이미지와 텍스트 함께 입력 | Gemma 3 · Qwen 3.5 | 이미지 내용 파악, 글자 읽기, 답변 근거 |

### Gemma — 현재 실습의 기준

Google DeepMind의 공개 가중치 모델 계열이다. 이 글에서는 `gemma3:4b`를 기준으로 사용한다. Gemma 3의 4B·12B·27B 모델은 텍스트와 이미지를 입력받아 텍스트를 생성하고, 1B 모델은 텍스트 입력을 다룬다. 공식 모델 카드는 질문 응답, 요약, 추론과 이미지 이해를 사용 사례로 제시한다. [Gemma 3 모델 카드](https://huggingface.co/google/gemma-3-4b-it)

먼저 지금 모델로 결과를 모아두고 다른 계열의 답변과 비교할 예정이다.

### Qwen — 같은 4B 크기에서 비교

Alibaba의 Qwen 계열은 다국어와 코드, 추론, 시각 입력을 다루는 모델을 제공한다. 수업의 두 번째 모델은 `qwen3.5:4b`다. Qwen 3.5는 언어·시각 입력을 함께 학습하고 다국어 지원을 넓힌 계열이다. [Qwen 3.5 4B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-4B)

Gemma와 같은 4B 규모에서 한국어 설명의 구성과 길이, 코드의 실행 결과를 비교해볼 수 있다.

### Llama — 작은 텍스트 챗봇 후보

Meta의 Llama 계열에는 세대별로 크기와 구조가 다른 모델이 있다. Mac에서 작은 모델부터 비교하는 후보로 Llama 3.2의 1B·3B 텍스트 모델을 골랐다. Instruct 버전은 대화, 요약, 검색 기반 응용을 위한 지시 학습을 거쳤다. [Llama 3.2 3B Instruct 모델 카드](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

공부 노트를 짧게 요약하는 작업에 써보고 싶다. 한국어 출력도 별도 항목으로 기록할 예정이다.

### Mistral / Ministral — 작은 앱에 넣을 모델

Mistral AI는 범용 Mistral과 소형 Ministral, 코드용 Codestral 등 작업별 모델을 제공한다. 여기서는 로컬·엣지 배포를 지향하는 Ministral 3를 비교 후보로 잡았다. Ministral 3 3B Instruct는 텍스트와 이미지를 처리하고, 함수 호출과 JSON 출력을 지원한다. [Ministral 3 3B Instruct 모델 카드](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)

짧은 대화와 정해진 JSON 형식으로 답하게 하는 앱을 만들어볼 때 살펴볼 계열이다.

### Phi — 메모리와 대기 시간을 줄여보기

Microsoft의 Phi 계열은 소형 모델에 집중한다. Phi-4-mini-instruct는 추론에 밀도 높은 합성 데이터와 필터링한 웹 데이터를 활용하고, 메모리·연산 자원이 제한된 환경과 수학·논리 작업을 사용 사례로 제시한다. [Phi-4-mini 모델 카드](https://huggingface.co/microsoft/Phi-4-mini-instruct)

수업 연습 문제에는 `phi3:mini`가 등장한다. 추가 비교에서는 Phi-4-mini도 후보로 두고 버전과 태그를 따로 기록하려 한다.

### DeepSeek — 풀이 과정이 긴 작업

DeepSeek 계열에서는 추론용 R1과 소형 Distill 모델을 살펴봤다. `DeepSeek-R1-Distill-Qwen-7B`는 Qwen2.5-Math-7B를 바탕으로 R1이 생성한 예제를 사용해 학습한 모델이다. [DeepSeek-R1-Distill 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

증류는 큰 모델이 만든 예제로 작은 모델을 학습하는 과정이고, 양자화는 가중치 등의 수치를 낮은 정밀도로 표현하는 과정이다. Distill 모델에도 양자화를 적용해 배포할 수 있다. 수학 문제나 코드 오류를 설명하게 한 뒤 답의 정확성과 출력 길이를 함께 비교해보고 싶다.

## 4. 모델 다운로드와 실행

내려받은 모델 파일은 디스크에 보관된다. 대화를 요청하면 모델을 메모리에 준비하고 입력을 처리한다. 응답 후에는 설정에 따라 모델을 메모리에 유지하다가 해제한다.

{{< mermaid >}}
%%{init: {"flowchart": {"rankSpacing": 24, "padding": 10}}}%%
flowchart TD
    A["모델 다운로드"] --> B["디스크에 모델 파일 보관"]
    B --> C["대화 요청"]
    C --> D{"모델이 메모리에<br>준비되어 있는가?"}
    D -->|"아니요"| E["모델 로딩·실행 준비"]
    D -->|"예"| F["입력 처리·응답 생성"]
    E --> F
    F --> G["응답 반환"]
    G --> H["설정에 따라 메모리에 유지"]
    H --> I["언로드 후에도<br>디스크의 모델 파일은 유지"]
{{< /mermaid >}}

| 명령어 | 동작 |
|---|---|
| `ollama pull gemma3:4b` | 모델 다운로드 |
| `ollama list` | 로컬 모델 목록 |
| `ollama show gemma3:4b` | 파라미터 수, 양자화 등 상세 정보 |
| `ollama run gemma3:4b` | 대화 시작 |
| `ollama ps` | 현재 로드된 모델 확인 |
| `ollama stop gemma3:4b` | 모델 언로드 |
| `ollama rm gemma3:4b` | 로컬 모델 삭제 |

`list`는 디스크에 보관된 모델을, `ps`는 현재 메모리에 로드된 모델을 보여준다. 처음 요청할 때는 모델 로딩 시간이 포함될 수 있다. [Ollama CLI 문서](https://docs.ollama.com/cli)

`ollama ps`의 `PROCESSOR` 열에는 모델의 CPU·GPU 배치 상태가 표시된다. 실제 연산 중 GPU 이용률은 활동 상태 보기 등의 도구로 확인한다.

### 서버 실행과 모델 목록

한 터미널에서 서버를 실행하고, 다른 터미널에서 모델 목록을 조회했다.

```sh
ollama serve
```

```sh
ollama list
```

{{< figure src="ollama-serve-list.png" alt="Ollama 서버 실행 로그와 gemma3:4b 모델 목록" figureClass="study-capture" >}}

Ollama `0.40.11`이 `127.0.0.1:11434`에서 실행 중이다. 서버 로그에 `Apple M5 Pro`와 `Metal`이 표시됐고, 모델 목록에는 `gemma3:4b`가 `3.4 GB`로 나왔다. 모델 ID는 `68ceab7e468a`다.

### 같은 질문, 다른 답변

자료의 비결정성(nondeterministic) 설명을 보고 `gemma3:4b`에 `Hello! who are you?`를 두 번 입력했다. 첫 답변은 텍스트와 이미지를 입력받을 수 있다고 소개했고, 두 번째는 챗봇이라는 설명과 함께 이모지를 붙였다. 같은 질문 문구에 소개 방식과 표현이 달라졌다.

{{< figure src="ollama-repeated-prompt.png" alt="Gemma 3에 같은 질문을 두 번 입력했을 때의 응답" figureClass="study-capture" >}}

샘플링을 사용하는 생성에서는 다음 토큰을 확률에 따라 선택하며, `temperature`와 `seed` 등의 설정이 결과에 영향을 준다. [Ollama 생성 설정](https://docs.ollama.com/modelfile#valid-parameters-and-values)

### 신경망을 10살 아이에게 설명해 달라고 요청하기

```text
Explain what a neural network is in simple terms, like I'm 10 years old.
```

{{< figure src="ollama-neural-network-answer.png" alt="신경망을 10살 아이에게 설명해 달라는 질문에 대한 Gemma 3의 응답" figureClass="study-capture" >}}

고양이 사진을 구분하도록 강아지를 가르치는 비유로 답했다. 뉴런을 작은 일꾼과 전구에 빗대고, 층을 거치는 신호 전달과 패턴 인식, 실수를 통한 학습을 여섯 항목으로 풀어 설명했다.

### 두 번째 모델 다운로드

비교할 두 번째 모델로 `qwen3.5:4b`를 내려받았다.

```sh
ollama pull qwen3.5:4b
ollama list
```

{{< figure src="qwen-download-list.png" alt="Qwen 3.5 4B 다운로드 완료와 Qwen·Gemma 모델 목록" figureClass="study-capture" >}}

다운로드가 `success`로 끝난 뒤 목록에 두 모델이 함께 표시됐다.

| 모델 | ID | 다운로드 크기 |
|---|---|---:|
| `gemma3:4b` | `68ceab7e468a` | 3.4 GB |
| `qwen3.5:4b` | `7f3251fa6878` | 4.0 GB |

### 간단한 모델 비교 테스트

두 모델에 같은 질문을 보냈다.

```text
What is the meaning of life?
```

{{< figure src="gemma-qwen-response-comparison.png" alt="삶의 의미를 묻는 질문에 대한 Gemma 3와 Qwen 3.5의 터미널 응답 비교" caption="왼쪽: Gemma 3 · 오른쪽: Qwen 3.5" figureClass="study-capture" >}}

Gemma는 본문 답변이 바로 나와 Qwen보다 간결하고 빠르게 느껴졌다. 철학, 종교, 인간적인 관점으로 나눠 삶의 의미를 설명했다.

Qwen은 `Thinking...`이 표시된 뒤 질문 분석, 답변 방향, 내용 구성과 검토 항목을 순서대로 길게 출력했다. 답변을 준비하는 과정이 더 세분되어 보였다.

### 두 모델의 show 결과 비교

```sh
ollama show gemma3:4b
ollama show qwen3.5:4b
```

<div class="study-capture-pair">

{{< figure src="gemma-show.png" alt="ollama show gemma3:4b의 모델 정보와 생성 설정" caption="Gemma 3" figureClass="study-capture study-capture--portrait" >}}

{{< figure src="qwen-show.png" alt="ollama show qwen3.5:4b의 모델 정보와 thinking 기본 설정" caption="Qwen 3.5" figureClass="study-capture study-capture--portrait" >}}

</div>

| 항목 | Gemma 3 | Qwen 3.5 |
|---|---|---|
| 모델 태그 | `gemma3:4b` | `qwen3.5:4b` |
| Architecture | `gemma3` | `qwen3_5` |
| Parameters | `3.9B` | `4.5B` |
| Context length | `131072` | `262144` |
| Embedding length | `2560` | `2560` |
| Quantization | `Q4_K_M` | `nvfp4` |
| Capabilities | `completion`, `vision` | `completion`, `vision`, `thinking`, `tools` |
| Temperature | `1` | `1` |
| Top-k | `64` | `20` |
| Top-p | `0.95` | `0.95` |

Qwen의 `thinking` 항목에는 `levels: false, true`, `default: true`가 표시됐다. 양자화 방식과 Top-k 설정에서도 차이를 확인했다.

### 연습문제: 세 모델의 중력 설명 비교

연습문제에서 비교할 `phi3:mini`를 추가로 내려받았다.

```sh
ollama pull phi3:mini
```

{{< figure src="phi-download.png" alt="Phi 3 Mini 모델 다운로드가 success로 완료된 터미널" figureClass="study-capture" >}}

세 모델에 “다섯 살 아이에게 중력을 설명해 달라”는 같은 질문을 보냈다.

```sh
ollama run gemma3:4b "Explain gravity to a five-year-old."
ollama run qwen3.5:4b "Explain gravity to a five-year-old."
ollama run phi3:mini "Explain gravity to a five-year-old."
```

| 모델 | 응답에서 본 점 |
|---|---|
| Gemma 3 | 공을 던졌다가 떨어지는 예시와 지구의 보이지 않는 포옹에 비유해 설명했다. |
| Qwen 3.5 | 긴 `Thinking...` 출력 뒤 미끄럼틀, 사과, 장난감의 예시를 들고 목록과 이모지를 사용했다. |
| Phi 3 Mini | 슈퍼히어로의 힘, 자석, 보이지 않는 줄을 비유로 사용해 한 문단으로 답했다. 점프 후 내려온다는 표현이 반복됐다. |

{{< figure src="gravity-gemma.png" alt="다섯 살 아이에게 중력을 설명해 달라는 질문에 대한 Gemma 3의 응답" caption="Gemma 3" figureClass="study-capture" >}}

{{< figure src="gravity-qwen-answer.png" alt="Thinking 출력 이후 Qwen 3.5가 작성한 중력 설명" caption="Qwen 3.5 — 최종 답변" figureClass="study-capture" >}}

<details>
<summary>Qwen 3.5 — Thinking 출력</summary>

<div class="study-capture-pair">

{{< figure src="gravity-qwen-thinking-1.png" alt="Qwen 3.5의 중력 설명에 앞서 나온 Thinking 출력 앞부분" caption="앞부분" figureClass="study-capture study-capture--portrait" >}}

{{< figure src="gravity-qwen-thinking-2.png" alt="Qwen 3.5의 Thinking 출력 뒷부분과 done thinking 표시" caption="뒷부분" figureClass="study-capture study-capture--portrait" >}}

</div>
</details>

{{< figure src="gravity-phi.png" alt="다섯 살 아이에게 중력을 설명해 달라는 질문에 Phi 3 Mini가 한 문단으로 답한 결과" caption="Phi 3 Mini" figureClass="study-capture" >}}

세 모델 중에서는 Phi 3 Mini가 가장 마음에 들었다. 한 문단으로 짧게 설명해주는 방식이 좋았다.

### 메모리 탐정

```sh
ollama list
ollama ps
ollama show phi3:mini
```

{{< figure src="ollama-three-models.png" alt="Phi 3 Mini, Qwen 3.5, Gemma 3 세 모델의 로컬 목록과 다운로드 크기" figureClass="study-capture" >}}

| 질문 | 내 기록 |
|---|---|
| 내려받은 모델은 몇 개인가? | 3개: `gemma3:4b`, `qwen3.5:4b`, `phi3:mini` |
| 전체 디스크 공간은 얼마나 차지하는가? | `ollama list`에 표시된 크기의 합계는 약 9.6 GB다. Gemma 3.4 GB + Qwen 4.0 GB + Phi 2.2 GB. |
| 메모리에 로드된 모델이 있는가? | `ollama ps` 조회 당시 `phi3:mini`가 로드돼 있었다. `SIZE`는 `15 GB`, `PROCESSOR`는 `100% GPU`로 표시됐다. |
| 가장 마음에 드는 모델의 컨텍스트 길이는? | Phi 3 Mini. `ollama show phi3:mini`의 `context length`는 `131072`토큰이었다. |

Phi의 `show`에 표시된 모델 컨텍스트 길이는 `131072`토큰이고, 조회 당시 `ps`의 실행 컨텍스트는 `32768`토큰이었다.

### 집 정리하기

연습용 모델로 `smollm2:135m`을 내려받고 한 문장으로 자기소개를 요청했다.

```sh
ollama pull smollm2:135m
ollama run smollm2:135m
```

```text
Introduce yourself in one sentence.
```

{{< figure src="smollm2-cleanup.png" alt="SmolLM2 다운로드, 자기소개 응답, 채팅 종료와 모델 삭제까지의 터미널 기록" figureClass="study-capture" >}}

여러 분야의 일을 돕는 AI 도우미라고 한 문장으로 답했다.

`/bye`로 채팅을 종료한 뒤 모델을 삭제했다.

```sh
ollama rm smollm2:135m
ollama list
```

{{< figure src="model-list-after-cleanup.png" alt="SmolLM2 삭제 후 Phi, Qwen, Gemma 세 모델이 남은 ollama list 결과" figureClass="study-capture" >}}

| 단계 | 기록 |
|---|---|
| 모델 다운로드 | `success` |
| 질문과 응답 | 자기소개를 한 문장으로 요청하고 응답을 받았다. |
| 모델 삭제 | `deleted 'smollm2:135m'` |
| 삭제 후 목록 확인 | SmolLM2가 목록에서 사라지고 Phi, Qwen, Gemma가 남았다. |

### 도전: 원샷 모드

명령어 끝에 질문을 붙여 Gemma 3에 서로 다른 질문 다섯 개를 보냈다.

```sh
ollama run gemma3:4b "Summarize the plot of Romeo and Juliet in three sentences."
ollama run gemma3:4b "Explain photosynthesis in two sentences."
ollama run gemma3:4b "Translate 'I enjoy learning new things' into Korean."
ollama run gemma3:4b "Give three tips for remembering English words."
ollama run gemma3:4b "Write a Python function that returns the larger of two numbers."
```

| 요청 | 출력에서 본 점 |
|---|---|
| 로미오와 줄리엣 줄거리 세 문장 | 짧은 안내 문구 뒤 줄거리를 세 문장으로 출력했다. |
| 광합성 두 문장 | 광합성 과정과 산소 방출을 두 문장으로 설명했다. |
| 한국어 번역 | 번역 후보 세 가지와 로마자 표기, 단어와 표현에 관한 설명을 덧붙였다. |
| 영어 단어 암기 팁 세 가지 | 연상법, 문장 속 활용, 간격 반복과 능동 회상을 세 항목으로 설명했다. |
| 두 수 중 큰 값을 반환하는 Python 함수 | `find_larger(num1, num2)` 함수와 사용 예시, 코드 설명을 작성했다. |

<details>
<summary>원샷 응답 캡처 5개</summary>

{{< figure src="oneshot-romeo-juliet.png" alt="Gemma 3의 로미오와 줄리엣 세 문장 요약 응답" caption="1. 줄거리 요약" figureClass="study-capture" >}}

{{< figure src="oneshot-photosynthesis.png" alt="Gemma 3가 광합성을 두 문장으로 설명한 원샷 응답" caption="2. 광합성 설명" figureClass="study-capture" >}}

{{< figure src="oneshot-korean-translation.png" alt="Gemma 3의 한국어 번역 후보 세 가지와 표현 설명" caption="3. 한국어 번역" figureClass="study-capture" >}}

{{< figure src="oneshot-vocabulary-tips.png" alt="Gemma 3의 영어 단어 암기 방법 세 가지에 대한 원샷 응답" caption="4. 단어 암기 방법" figureClass="study-capture" >}}

{{< figure src="oneshot-python-function.png" alt="Gemma 3가 작성한 두 수 중 큰 값을 반환하는 Python 함수와 설명" caption="5. Python 함수 작성" figureClass="study-capture" >}}

</details>

짧은 번역이나 요약, 함수 작성처럼 한 번 묻고 끝나는 작업에 원샷 모드를 사용할 수 있다. 질문이 포함된 명령어를 다시 실행하거나 스크립트에 넣어 반복하기도 편하다.

## 5. Python 가상환경

Python을 설치하고 터미널에서 `Python 3.13.18`로 표시되는 것을 확인했다.

```sh
python3.13 --version
```

{{< figure src="python-version.png" alt="python3.13 --version 명령으로 Python 3.13.18 설치를 확인한 터미널" figureClass="study-capture" >}}

프로젝트마다 필요한 패키지와 버전이 다르다. 가상환경을 만들면 각 프로젝트의 Python 패키지를 나누어 관리할 수 있다.

{{< mermaid >}}
%%{init: {"flowchart": {"rankSpacing": 24, "padding": 10}}}%%
flowchart TB
    subgraph SHARED["패키지 환경을 공유하는 경우"]
        A["프로젝트 A<br>패키지 1.x 필요"] --> X["공유 설치 환경<br>버전 충돌 가능"]
        B["프로젝트 B<br>패키지 2.x 필요"] --> X
    end

    subgraph ISOLATED["가상환경으로 분리하는 경우"]
        C["챗봇 프로젝트"] --> E["챗봇용 .venv<br>필요한 패키지와 버전"]
        D["다른 프로젝트"] --> F["별도의 .venv<br>다른 패키지와 버전"]
    end
{{< /mermaid >}}

수업 자료는 Python 3.13의 `venv`와 `pip`를 기본으로 사용한다.

```sh
mkdir -p ~/local-ai-study/week-02
cd ~/local-ai-study/week-02

python3.13 -m venv .venv
source .venv/bin/activate

python --version
which python
python -c "import sys; print(sys.executable)"
python -m pip --version
```

{{< figure src="venv-activated.png" alt="Python 3.13으로 venv를 만들고 활성화한 뒤 Python 버전과 실행 경로를 확인한 터미널" figureClass="study-capture" >}}

프롬프트에 `(.venv)`가 붙었고, `which python`은 `…/week-02/.venv/bin/python`으로 표시됐다.

`venv`는 환경을 만드는 모듈이고, `.venv`는 생성할 폴더 이름이다. `source .venv/bin/activate`를 실행하면 현재 셸의 `PATH`가 바뀌어 해당 환경의 Python을 우선 사용한다.

프로젝트에서 사용하는 Python의 위치는 `sys.executable`로 확인한다. 활성화를 해제할 때는 `deactivate`, 실행 파일을 직접 지정할 때는 `.venv/bin/python`을 쓴다. [Python 가상환경 문서](https://docs.python.org/3.13/library/venv.html)

영상에서는 `uv`로 Python 3.13 환경을 만드는 방법도 소개했다. 아래는 환경 생성의 대안이다.

```sh
uv venv --python 3.13 --seed
source .venv/bin/activate
```

`--seed`는 `pip` 같은 기본 패키지 관리 도구를 환경에 준비하는 옵션이다. [uv 환경 안내](https://docs.astral.sh/uv/pip/environments/)

### 실습 도구 버전 확인

`brew --version`, `ollama --version`, `python3 --version`으로 버전을 확인했다.

{{< figure src="tool-versions.png" alt="가상환경에서 Homebrew 7.0.11, Ollama 0.40.11, Python 3.13.18 버전을 확인한 터미널" figureClass="study-capture" >}}

| 도구 | 버전 |
|---|---|
| Homebrew | `7.0.11` |
| Ollama | `0.40.11` |
| Python | `3.13.18` |

### 패키지 설치와 환경 기록

`ollama`와 `streamlit` 패키지를 설치했다. Streamlit은 이후 웹 화면을 만들 때 사용한다.

```sh
python -m pip install ollama
python -m pip install streamlit
```

`pip list`로 설치된 패키지와 버전을 확인하고, `requirements.txt`에 저장한 뒤 파일 내용을 확인했다.

```sh
pip list
pip freeze > requirements.txt
cat requirements.txt
```

<div class="study-capture-pair">

{{< figure src="pip-list.png" alt="가상환경에서 pip list로 확인한 설치 패키지와 버전 목록" caption="pip list" figureClass="study-capture study-capture--portrait" >}}

{{< figure src="requirements-saved.png" alt="pip freeze로 requirements.txt를 저장하고 cat으로 파일 내용을 확인한 터미널" caption="requirements.txt" figureClass="study-capture study-capture--portrait" >}}

</div>

| 패키지 | 확인한 버전 |
|---|---|
| `ollama` | `0.6.3` |
| `streamlit` | `1.66.1` |
| `pip` | `26.2.2` |

새 환경에서는 이 파일을 읽어 패키지를 설치한다.

```sh
python -m pip install -r requirements.txt
```

프로젝트를 공유할 때는 소스 코드, `requirements.txt`, Python 버전을 함께 남기고 `.venv`는 각 환경에서 다시 만든다. [pip freeze 문서](https://pip.pypa.io/en/stable/cli/pip_freeze/)

## 6. Python으로 질문 보내기

### VS Code에서 Python 파일 작성

터미널에서 `code --version`을 실행해 `1.143.1`, `arm64`를 확인했다.

{{< figure src="vscode-version.png" alt="code --version 명령으로 VS Code 1.143.1과 arm64를 확인한 터미널" figureClass="study-capture" >}}

VS Code에서 프로젝트 폴더 `~/local-ai-study/week-02`를 열고 Microsoft의 Python 확장을 설치한다. 명령 팔레트의 `Python: Select Interpreter`에서 앞서 만든 `.venv`의 Python을 선택한다. [VS Code의 Python 환경 설정](https://code.visualstudio.com/docs/python/environments)

`my-ai-chatbot` 폴더에서 `hello.py`를 실행했다.

```sh
python3.13 hello.py
```

{{< figure src="hello-python.png" alt="hello.py를 실행해 Hello, World!와 My AI chatbot project starts here!를 출력한 터미널" figureClass="study-capture" >}}

### 첫 질문: 프랑스의 수도

`chat_basic.py`에서 `gemma3:4b`에 프랑스의 수도를 물었다.

```python
import ollama

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)

print(response["message"]["content"])
```

<details>
<summary>chat_basic.py 코드 화면</summary>

{{< figure src="chat-basic-code.png" alt="VS Code에서 프랑스의 수도를 묻는 chat_basic.py를 작성한 화면" figureClass="study-capture" >}}

</details>

가상환경의 터미널에서 실행했다.

```sh
python3 chat_basic.py
```

{{< figure src="chat-basic-result.png" alt="chat_basic.py를 실행해 프랑스의 수도가 Paris라는 응답을 받은 터미널" figureClass="study-capture" >}}

프랑스의 수도는 Paris라고 답했고, 예술·패션·미식·문화의 중심지라는 설명을 덧붙였다.

| 코드 | 역할 |
|---|---|
| `ollama.chat(...)` | 모델에 대화 목록을 보내고 응답 받기 |
| `model` | 사용할 모델 |
| `messages` | 전달할 대화 목록 |
| `response["message"]["content"]` | 응답에서 답변 텍스트 꺼내기 |

응답을 Python 값으로 받으면 화면 출력, 파일 저장, 다른 프로그램과의 연결에 사용할 수 있다. [Ollama Python 사용법](https://github.com/ollama/ollama-python)

### 메시지의 역할

채팅 메시지는 `role`과 `content`를 묶어 표현한다. `role`은 메시지의 종류를, `content`는 실제 내용을 담는다. 이번 실습에서는 세 가지 role을 사용했다. [채팅 메시지 구조](https://huggingface.co/docs/transformers/main/en/chat_templating#using-apply_chat_template)

```python
{"role": "user", "content": "How does gravity work?"}
```

| role | 담는 내용 | 이번 실습의 예 |
|---|---|---|
| `system` | 프로그램에서 정하는 응답 지침. 말투, 설명 방식, 답변 길이 등을 지정한다. | 친근한 과학 선생님처럼 비유를 사용하고 100단어 미만으로 답하기 |
| `user` | 사용자가 보낸 질문이나 요청 | `How does gravity work?` |
| `assistant` | 모델이 생성한 답변. 이전 답변도 이 role로 대화 기록에 넣는다. | 트램펄린과 볼링공으로 중력을 설명한 답변 |

과학 선생님이나 셰익스피어 배우 같은 설정은 `role="system"` 메시지의 `content`에 작성한다. `user` 메시지에는 실제 질문을 넣고, 모델이 돌려준 답변은 `assistant` 메시지로 이어진다.

{{< mermaid >}}
%%{init: {"flowchart": {"rankSpacing": 24, "padding": 10}}}%%
flowchart LR
    S["system<br>응답 지침"] --> H["messages<br>순서대로 쌓인 대화"]
    U["user<br>사용자의 질문"] --> H
    A["assistant<br>이전 모델 답변"] --> H
    H --> M["Ollama·모델"]
    M --> R["새 assistant 답변"]
    R -. "대화 기록에 추가" .-> H
{{< /mermaid >}}

첫 요청은 `system` 지침과 `user` 질문으로 시작할 수 있다. 응답을 받은 뒤에는 답변을 `assistant`로 저장하고, 다음 `user` 질문을 붙여 메시지 목록을 다시 보낸다. 이전 답변을 가리키는 “방금 추천한 책을 더 설명해줘” 같은 질문에도 이 기록이 사용된다. [Ollama 채팅 API](https://docs.ollama.com/api/chat)

메시지 목록은 모델에 맞는 채팅 템플릿을 거쳐 역할 구분 정보가 포함된 토큰 시퀀스로 변환된다. 모델은 이 시퀀스를 이어서 다음 출력을 생성한다. [Hugging Face 채팅 템플릿 설명](https://huggingface.co/docs/transformers/main/en/chat_templating)

### system 메시지로 역할 지정

`chat_system.py`에는 친근한 과학 선생님처럼 재미있는 비유를 사용하고, 답변은 100단어 미만으로 작성하라는 `system` 메시지를 넣었다. 질문은 `How does gravity work?`다.

```python
import ollama

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a friendly science teacher who explains "
                "concepts using fun analogies. Keep your answers "
                "under 100 words."
            ),
        },
        {
            "role": "user",
            "content": "How does gravity work?",
        },
    ],
)

print(response["message"]["content"])
```

<details>
<summary>chat_system.py 코드 화면</summary>

{{< figure src="chat-system-code.png" alt="과학 선생님 역할과 100단어 미만 응답을 지정한 chat_system.py 코드" figureClass="study-capture" >}}

</details>

```sh
python3 chat_system.py
```

{{< figure src="chat-system-result.png" alt="chat_system.py 실행 후 트램펄린과 볼링공의 비유로 중력을 설명한 응답" figureClass="study-capture" >}}

트램펄린 가운데 볼링공을 놓으면 천이 움푹 들어가는 모습에 빗대어 중력을 설명했다. 지구처럼 질량을 가진 물체와 그쪽으로 움직이는 다른 물체를 이 비유에 연결했다.

#### 셰익스피어 배우 역할

같은 `chat_system.py`에서 `system` 메시지를 셰익스피어 배우 역할로 바꿨다. 모델은 `gemma3:4b`, 질문은 `How does gravity work?`로 유지했다.

```python
{
    "role": "system",
    "content": (
        "You are a dramatic Shakespearean actor. Explain everything "
        "as if you're performing on stage."
    ),
}
```

<details>
<summary>셰익스피어 배우 역할의 코드 화면</summary>

{{< figure src="chat-shakespeare-code.png" alt="system 메시지를 셰익스피어 배우 역할로 바꾼 chat_system.py 코드" figureClass="study-capture" >}}

</details>

```sh
python3 chat_system.py
```

{{< figure src="chat-shakespeare-result.png" alt="셰익스피어 배우 역할로 중력을 설명하며 옛 영어 표현과 무대 동작을 출력한 응답" figureClass="study-capture" >}}

`thee`, `'tis`, `doth` 같은 옛 영어 표현을 사용해 긴 독백처럼 답했다. 지구본을 들고 몸짓하거나 목소리를 낮추고 무대를 걷는 동작도 괄호 안에 함께 출력했다.

| 항목 | 과학 선생님 역할 | 셰익스피어 배우 역할 |
|---|---|---|
| 요청한 방식 | 재미있는 비유, 100단어 미만 | 무대에서 공연하듯 설명 |
| 설명에 사용한 소재 | 트램펄린과 볼링공 | 지구본과 보이지 않는 손 |
| 말투와 구성 | 일상적인 문장, 짧은 두 문단 | 옛 영어 표현, 긴 독백과 무대 동작 지문 |

## 7. 스트리밍 응답

비스트리밍은 전체 답변이 완성된 뒤 결과를 받는다. 스트리밍은 응답 조각을 도착하는 대로 출력한다. 영상에서도 두 설정을 바꾸며 화면에 답변이 나타나는 차이를 보여줬다.

{{< mermaid >}}
sequenceDiagram
    participant P as Python 프로그램
    participant O as Ollama 서버·모델

    rect rgba(128, 128, 128, 0.12)
        Note over P,O: 비스트리밍
        P->>O: 질문 전송
        Note over O: 답변 생성
        O-->>P: 완성된 응답 반환
        Note over P: 전체 답변 표시
    end

    rect rgba(128, 128, 128, 0.12)
        Note over P,O: 스트리밍
        P->>O: 질문 전송
        Note over O: 입력 처리 후 생성 시작
        O-->>P: 응답 조각
        O-->>P: 다음 응답 조각
        O-->>P: 이후 조각·종료 정보
        Note over P: 도착하는 대로 표시
    end
{{< /mermaid >}}

생성 과정은 입력 문맥을 처리하는 Prefill과, 다음 토큰을 순서대로 생성하는 Decode로 나눌 수 있다.

| 단계 | 처리 내용 |
|---|---|
| Prefill | 입력 토큰을 처리하고 이후 생성에 사용할 상태 준비 |
| Decode | 앞선 문맥과 생성 결과를 바탕으로 다음 토큰 계산 |

스트리밍에서는 응답이 완성되기 전부터 내용을 읽을 수 있다. 첫 출력까지의 대기 시간에는 모델 로딩과 입력 처리 시간이 영향을 준다. 전체 완료 시간과 첫 출력 시간은 각각 측정한다. [Ollama 스트리밍 설명](https://docs.ollama.com/api/streaming)

### chat_stream.py 실행

`chat_stream.py`에서 인터넷이 어떻게 작동하는지 쉽게 설명해 달라고 요청했다. `stream=True`를 설정하고 응답 조각을 받을 때마다 출력했다.

```python
import ollama

stream = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Provide detailed, "
                "thoughtful answers."
            ),
        },
        {
            "role": "user",
            "content": "Explain how the internet works in simple terms.",
        },
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

print()
```

<details>
<summary>chat_stream.py 코드 화면</summary>

{{< figure src="chat-stream-code.png" alt="stream=True와 for 반복문으로 응답 조각을 출력하는 chat_stream.py 코드" figureClass="study-capture" >}}

</details>

`end=""`는 조각마다 줄바꿈을 붙이지 않는 설정이고, `flush=True`는 출력 버퍼를 즉시 비우도록 요청하는 설정이다. 마지막 `print()`는 답변이 끝난 뒤 줄바꿈을 추가한다.

```sh
python3 chat_stream.py
```

{{< figure src="chat-stream-result-1.png" alt="chat_stream.py 실행 후 패킷, IP 주소, 라우터, TCP/IP와 DNS를 설명한 응답 앞부분" caption="응답 앞부분" figureClass="study-capture" >}}

{{< figure src="chat-stream-result-2.png" alt="인터넷 동작을 레시피 조각을 친구에게 보내는 과정에 비유한 응답 뒷부분" caption="응답 뒷부분" figureClass="study-capture" >}}

인터넷을 우편 서비스에 비유하면서 패킷, IP 주소, 라우터, TCP/IP, DNS를 단계별로 설명했다. 뒤에서는 레시피를 여러 조각으로 나눠 친구에게 보내는 비유도 덧붙였다.

### 응답 시간 측정

요청 후 첫 번째 비어 있지 않은 응답 조각을 화면에 출력하기까지의 시간과 전체 응답이 끝날 때까지의 시간을 각각 잰다. Python의 `time.perf_counter()`를 요청 전후에 호출해 경과 시간을 구할 수 있다. [Python 시간 측정 함수](https://docs.python.org/3.13/library/time.html#time.perf_counter)

Ollama 응답의 `load_duration`, `eval_count`, `eval_duration`으로 모델 로딩 시간과 출력 토큰 수, 생성 시간을 살펴볼 수 있다. 시간 필드는 나노초 단위이며, 스트리밍의 사용량 정보는 마지막 조각에 담긴다. [Ollama 측정 지표](https://docs.ollama.com/api/usage)

## 8. 대화 기록 유지하기

`chat_history.py`를 실행해 이름을 알려준 뒤 다시 물었다. 답변에 앞서 입력한 이름인 `빵주야`가 다시 나왔다.

{{< mermaid >}}
%%{init: {"flowchart": {"rankSpacing": 24, "padding": 10}}}%%
flowchart TD
    S["system 메시지로<br>history 초기화"] --> I["사용자 입력"]
    I --> Q{"종료 명령인가?"}
    Q -->|"예"| E["프로그램 종료"]
    Q -->|"아니요"| U["history에<br>user 메시지 추가"]
    U --> M["messages=history로<br>Ollama에 요청"]
    M --> R["응답 조각 출력<br>response_text에 누적"]
    R --> A["완성된 답변을<br>assistant 메시지로 추가"]
    A --> I
{{< /mermaid >}}

두 번째 질문을 보낼 때 `history`에는 다음 순서로 메시지가 들어간다.

| 순서 | role | 내용 |
|---|---|---|
| 1 | `system` | 친절하게 돕고, 간결하면서도 꼼꼼하게 답하라는 지침 |
| 2 | `user` | 이름 소개 |
| 3 | `assistant` | 이름을 부르며 인사한 답변 |
| 4 | `user` | 이름을 다시 묻는 질문 |

`messages=history`로 이 목록 전체를 보내면서 이전 소개와 답변도 다음 요청에 함께 전달했다.

### 스트리밍과 대화 기록을 합친 코드

실행한 `chat_history.py` 코드다.

```python
import ollama

history = [
    {
        "role": "system",
        "content": (
            "You are a friendly and helpful assistant. "
            "Be concise but thorough in your answers."
        ),
    }
]

print("Chat with AI! (type 'quit', 'exit', or 'bye' to exit)")
print("-" * 40)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ("quit", "exit", "bye"):
        print("\nGoodbye!")
        break

    history.append({
        "role": "user",
        "content": user_input,
    })

    stream = ollama.chat(
        model="gemma3:4b",
        messages=history,
        stream=True,
    )

    print("\nAI: ", end="")
    response_text = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        response_text += piece

    print()

    history.append({
        "role": "assistant",
        "content": response_text,
    })
```

<details>
<summary>chat_history.py 코드 화면</summary>

{{< figure src="chat-history-code.png" alt="user 입력과 assistant 응답을 history에 추가하며 스트리밍으로 대화하는 chat_history.py 코드" figureClass="study-capture" >}}

</details>

`response_text`에는 화면에 출력한 조각을 계속 이어 붙인다. 응답이 끝나면 이 문자열을 `assistant` 메시지로 `history`에 추가하고 다음 입력을 받는다.

```sh
python3 chat_history.py
```

<details>
<summary>chat_history.py 실행 시작 화면</summary>

{{< figure src="chat-history-start.png" alt="chat_history.py를 시작하고 이름을 소개하는 첫 메시지를 입력한 터미널" figureClass="study-capture" >}}

</details>

{{< figure src="chat-history-result.png" alt="이름을 다시 묻자 앞서 소개한 이름으로 답하고 책을 추천한 뒤 quit으로 종료한 대화 기록" figureClass="study-capture" >}}

| 입력 | 응답·결과 |
|---|---|
| 이름 소개 | 이름을 부르며 인사했다. |
| 이름을 다시 묻기 | `You already told me your name is 빵주야!`라고 답했다. |
| 초보 프로그래머에게 책 추천 요청 | Al Sweigart의 *Automate the Boring Stuff with Python*을 추천했다. |
| `quit` | `Goodbye!`를 출력하고 종료했다. |

코드에서는 `quit`, `exit`, `bye`를 입력하면 `while` 반복문을 끝내도록 작성했다.

## 9. 대화 저장과 컨텍스트

대화에 사용하는 정보는 여러 곳에 나뉘어 있다.

| 구분 | 보관하는 내용 |
|---|---|
| 모델 가중치 | 학습 과정에서 조정된 수치 |
| Python의 `history` | 현재 프로그램의 메시지 목록 |
| 대화 로그 파일 | 다음 실행에서도 읽을 수 있도록 저장한 메시지 |
| 추론 캐시 | 계산 재사용을 위한 실행 상태 |

일반적인 대화 요청은 준비된 가중치로 응답을 계산한다. 대화의 문맥은 프로그램이 `messages`에 구성해 전달한다.

앞선 코드 아래에 다음을 추가하면 `history`를 JSON 파일로 저장할 수 있다.

```python
import json
from pathlib import Path

log_path = Path(__file__).with_name("chat_log.json")
log_path.write_text(
    json.dumps(history, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

파일은 스크립트와 같은 폴더에 저장되며, 이 예시는 기존 파일을 덮어쓴다. 다음 실행에서 대화를 이어가려면 파일을 읽어 `history`를 복원한다.

### 컨텍스트 윈도우

컨텍스트 윈도우는 모델이 한 번에 다룰 수 있는 토큰 범위다. 역할 지시, 이전 대화, 새 질문이 문맥을 구성하고 생성할 응답도 이 범위 안에서 다뤄진다.

`ollama show`에서는 모델 정보를, `ollama ps`에서는 실행 중인 모델의 상태를 확인한다.

```sh
ollama show gemma3:4b
ollama ps
```

모델이 지원하는 최대 길이와 현재 실행에 할당된 길이를 각각 살펴본다. 큰 컨텍스트는 더 많은 메모리를 요구할 수 있다. 긴 대화에서는 최근 메시지를 유지하거나 이전 대화를 요약해 전달하는 방식으로 입력을 구성한다. [Ollama 컨텍스트 설명](https://docs.ollama.com/context-length)

## 10. 두 모델의 토론

수업 후반에는 서로 다른 입장을 가진 두 모델을 연결하는 연습도 소개됐다. 각 모델에 별도의 대화 기록을 두고, 상대의 응답을 다음 입력으로 전달한다.

| 발언 | 모델 A의 기록 | 모델 B의 기록 |
|---|---|---|
| A가 한 말 | `assistant` | `user` |
| B가 한 말 | `user` | `assistant` |

각 모델은 자기 발언을 `assistant`, 상대 발언을 `user`로 받는다. 정해진 횟수만큼 반복하면 토론 루프가 된다.

실행하면서 입장 유지, 상대 논점에 대한 답변, 답변 길이를 살펴보려 한다. 모델 전환에 따른 로딩과 메모리 상태도 관찰할 항목이다.

## 11. 실행 중 확인할 것

| 증상 | 확인할 항목 |
|---|---|
| 서버 연결 실패 | 서버 실행 여부, 주소와 포트 |
| 모델을 찾을 수 없음 | `ollama list`의 태그와 코드의 모델명 |
| `ModuleNotFoundError: ollama` | 패키지 설치 환경과 Python 실행 경로 |
| VS Code에서 패키지 인식 실패 | 선택된 인터프리터 |
| 첫 응답 지연 | 모델 로딩 시간 |
| 이전 대화가 반영되지 않음 | 요청에 포함된 `history` |
| 긴 대화에서 속도 저하 | 입력 토큰 수, 컨텍스트 설정, 메모리 상태 |
| 재시작 후 대화 초기화 | 로그 저장·복원 코드 |
