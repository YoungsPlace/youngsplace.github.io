---
title: "NVIDIA NCP-AAI 7장 · NVIDIA 플랫폼 적용"
date: 2026-11-16T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
weight: 70
summary: "NIM, TensorRT-LLM, Triton, NeMo 도구의 역할을 실행 경로에 놓고, 모의 HTTP 서버에서 준비 상태와 호출을 확인했다."
tags: ["NCP-AAI", "NVIDIA NIM", "TensorRT-LLM", "Triton", "NeMo"]
categories: ["NVIDIA NCP-AAI"]
showTableOfContents: true
showAuthor: false
showHero: false
---

Mac에서 실행하던 검증 도우미의 모델 호출을 NVIDIA GPU 서버로 옮긴다고 가정했다. 모델의 응답을 읽고 시험 자료를 조회할 앱, 모델을 올리고 토큰을 내보내는 서빙, 검색과 도구 호출을 검사하는 부분이 있다. 제품 이름을 하나씩 외우기보다, 이 경로에서 어느 경계를 맡는지 정리했다.

공식 학습 가이드 7.1–7.5는 NeMo Guardrails, NIM, NeMo Agent Toolkit, TensorRT-LLM과 Triton, NVIDIA 하드웨어에서의 멀티모달 입력을 다룬다. 공식 웹 비중은 7%다. [공식 출제표](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/#exam-blueprint) · [공식 학습 가이드 7장](https://dam-cdn.nvd.orangelogic.com/AssetLink/64tei188l3tt132l265u1ipjoexdl1p5.pdf#page=10)

## 1. 에이전트 실행기와 모델 서빙의 경계

| 구성 | 맡는 일 | 검증 도우미에서 보는 것 |
|---|---|---|
| NVIDIA NIM | 지원되는 모델을 컨테이너의 추론 API로 제공 | 모델 로딩, 준비 상태, 채팅 요청과 추론 응답 |
| TensorRT-LLM | NVIDIA GPU에서 LLM 추론을 실행하고 최적화 | 지원 모델·GPU, 정밀도, 병렬화, 캐시와 배칭 |
| Triton Inference Server | 모델과 백엔드를 배포하고 요청을 스케줄링 | 모델 구성, 백엔드에 맞는 배칭과 메트릭 |
| NeMo Guardrails | 입력과 검색, 도구 실행, 출력 경계에 규칙 적용 | 자료 전달 범위, 도구 인자, 결과와 사용자 응답 검사 |
| NeMo Agent Toolkit | 에이전트 워크플로 구성, 관측과 평가에 사용하는 도구 | 단계, 모델과 도구의 연결, 실행 기록 |

NIM의 LLM·VLM 공식 API 문서에는 vLLM 및 SGLang 기반 컨테이너와 백엔드에 따른 API 차이가 있다. 제품 이름만 보고 실행 엔진을 가정하기보다 배포한 NIM의 이미지, 모델 프로파일과 API 문서를 함께 본다. Triton을 직접 구성할 때는 모델별 설정과 선택한 백엔드를 확인한다. 동적 배칭이 처리량을 높이는지 보려면 동시에 들어오는 요청의 수와 허용 대기를 기록한다. 모든 LLM 서빙 경로에서 같은 배처가 쓰이는 것으로 설명할 수 없으므로, 일반 배칭과 실제 토큰 생성 단계의 스케줄링을 나눠 본다. [NVIDIA NIM API Reference](https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html) · [TensorRT-LLM 공식 문서](https://nvidia.github.io/TensorRT-LLM/) · [Triton의 배처](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)

{{< mermaid >}}
flowchart TD
    U["검증 요청"] --> A["에이전트 실행기"]
    A --> G["모델과 도구의 요청 검사"]
    G --> N["NIM · 모델 추론 API"]
    G --> T["시험과 문서 조회 도구"]
    N --> R["응답과 실행 상태 검사"]
    T --> R
    R --> Q["근거가 붙은 결과와 종료 상태"]
{{< /mermaid >}}

## 2. NeMo Guardrails를 붙이는 위치

시험 자료에 대한 접근 검사와 모델 응답의 표현 검사는 다른 시점에서 실행한다. 검색 결과를 모델에 보내기 전에 접근 범위를 확인하고, 모델이 시험 ID를 제안하면 도구를 호출하는 시점에 사용자 범위와 인자를 다시 검사한다. 최종 답변에 숫자와 출처가 들어가면 출처가 실제로 조회한 자료인지 확인한다.

NeMo Guardrails는 입력, 검색, 대화, 실행, 출력 단계에 적용할 수 있는 레일을 설명한다. 예를 들어 검색 레일에서 접근 허용된 자료를 구성하고, 실행 레일에서 도구의 이름과 인자를 확인하며, 출력 레일에서 사용자에게 표시할 응답을 검사할 수 있다. 인증과 권한의 원본은 애플리케이션 및 업무 서비스에서 관리한다. [NeMo Guardrails의 레일 유형](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/rail-types) · [9장의 정책 검사](/ncp-aai/chapter-09-safety-ethics-compliance/)

## 3. NeMo Agent Toolkit에서 읽은 구성과 버전

학습 가이드에는 NeMo Agent Toolkit을 이용한 워크플로 최적화가 들어 있다. Toolkit 예제를 읽을 때는 모델, 도구, 단계, 관측 값과 평가 지표가 어디에 정의되는지 본다. 실행 경로를 바꿨다면 동일한 시험 요청을 실행하고, 모델 호출 수와 도구 인자, 결과가 달라졌는지 비교한다. 아래 실습의 앱에는 Toolkit을 설치하지 않았다.

NVIDIA의 NeMo Agent Toolkit 1.9 릴리스 노트는 2026년 10월부터 deprecated, 2026년 말 EOL로 안내한다. 시험 가이드의 제품명과 예시는 해당 시점의 기술 설명으로 읽고, 새 배포에 적용할 땐 릴리스 노트와 마이그레이션 문서를 함께 확인한다. [NeMo Agent Toolkit 릴리스 노트](https://docs.nvidia.com/nemo/agent-toolkit/latest/release-notes.html)

## 4. NIM을 배포할 때 확인할 순서

먼저 사용할 모델과 그 컨테이너가 지원하는 GPU, 실행 환경을 확인한다. NVIDIA Quickstart에 따라 이미지와 모델을 준비하고, 모델 저장 경로, 필요한 자격 증명과 API 주소를 정한다. 컨테이너의 생존 상태와 모델의 준비 상태를 각각 확인한 뒤 추론 호출을 보낸다. 모델이 처음 올라올 때 준비가 늦다면, 생존 검사 실패로 처리하는 구성을 피한다. [NVIDIA NIM Quickstart](https://docs.nvidia.com/nim/large-language-models/latest/get-started/quickstart.html) · [4장의 Kubernetes 프로브](/ncp-aai/chapter-04-deployment-and-scaling/)

[nim_probe_lab.py](nim_probe_lab.py)는 로컬에 임시 HTTP 서버를 열어 문서에 나온 경로와 유사한 모의 응답을 보낸다. Python 클라이언트로 두 상태에서 세 경로를 각각 조회했다.

```bash
python3 nim_probe_lab.py
```

| 모의 서버 상태 | `/v1/health/live` | `/v1/health/ready` | `/v1/chat/completions` |
|---|---:|---:|---:|
| 로딩 중 | 200 | 503 | 503 |
| 로딩 완료 | 200 | 200 | 200 |

로딩 완료 뒤 채팅 응답의 `choices[0].message.content`에서 `mock result`가 나왔다. 503은 준비 전의 분기를 확인하려고 모의 서버에 지정한 응답이다. 실제 NVIDIA NIM 컨테이너 시작이나 GPU 추론 성능은 이 실험의 범위에 포함되지 않는다. [공식 API의 추론·관리 엔드포인트](https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html)

## 5. 멀티모달 입력에서 모델을 고르는 순서

텍스트·이미지·음성 입력이 있다면 선택한 모델과 배포 프로파일이 어떤 형식을 지원하는지 확인한다. 2장의 장비 화면은 시각 모델에서 시험 ID를 읽는다. 음성은 전사 모델을 따로 두고 그 텍스트를 같은 도구 입력으로 넘길 수 있다. 이미지 해상도나 한 요청에 넣는 장수가 늘면 전송 시간과 전처리, 모델에 들어가는 입력 분량도 달라진다. 음성에서는 전사 완료까지의 시간과 이후 생성 시간을 분리한다.

실제 GPU 성능 비교에서는 모델과 입력 형식, 입력 길이, 출력 길이, 동시성을 고정한 뒤 준비 시간, 첫 답변 지연과 전체 처리량을 잰다. 안전 검사나 문서 검색이 병목이면 서빙 엔진의 설정을 바꿔도 전체 시간이 크게 달라지지 않을 수 있다. 텍스트와 멀티모달 요청을 나눠 어느 구간이 느려지는지 본다. [NVIDIA NIM Benchmarking](https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html) · [2장의 합성 화면과 음성 입력](/ncp-aai/chapter-02-agent-development/)
