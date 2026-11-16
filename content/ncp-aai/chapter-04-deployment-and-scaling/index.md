---
title: "NVIDIA NCP-AAI 4장 · 배포와 확장"
date: 2026-11-16T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
weight: 40
summary: "에이전트 실행기와 모델 서빙을 배포하는 위치를 나누고, 가상 시계에서 복제 수와 대기 제한을 비교했다."
tags: ["NCP-AAI", "Agentic AI", "Kubernetes", "Deployment", "MLOps"]
categories: ["NVIDIA NCP-AAI"]
showTableOfContents: true
showAuthor: false
showHero: false
---

정기 보고에는 여러 자료를 동시에 읽는 일이 많고, 원인 조사는 새 근거가 올 때마다 다음 도구를 고른다. 두 요청이 동시에 들어오면 모델 서버의 대기, 문서 검색, 업무 API 중 어느 곳이 먼저 밀리는지 알기 어렵다. 배포 구조를 짤 때는 처리 단위와 대기 열을 구분하고, 기한이 지난 요청이 어느 시점까지 자원을 쓰는지 본다.

학습 가이드 4.1–4.5는 운영 규모의 멀티에이전트 조율, CI/CD·모니터링과 거버넌스, 분산 부하 프로파일링, 컨테이너와 Kubernetes 확장, 비용과 가용성이다. 비중은 공식 웹 출제표의 13%를 사용했다. [공식 출제표](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/#exam-blueprint) · [공식 학습 가이드 4장](https://dam-cdn.nvd.orangelogic.com/AssetLink/64tei188l3tt132l265u1ipjoexdl1p5.pdf#page=7)

## 1. 실행기, 작업자, 모델을 어디에 띄울까

검증 도우미의 실행기는 작업을 받고 단계를 고르며, 긴 조회는 작업자에게 보낸다. 여러 실행기 복제본에서 같은 조사를 이어 가려면 조사 ID와 완료 단계가 공용 저장소에 있어야 한다. 모델 서버는 GPU 자원, 도구 서비스는 데이터 권한과 호출량에 따라 별도로 확장할 수 있다.

{{< mermaid >}}
flowchart TD
    U["사용자 요청"] --> G["게이트웨이 · 인증과 부하 분산"]
    G --> A["에이전트 실행기 복제본"]
    A --> S["공용 상태 · 단계와 근거"]
    A --> Q["작업 큐 · 대기 제한"]
    Q --> W["도구 작업자"]
    A --> M["모델 서빙"]
    W --> T["검색 · 업무 API"]
    W --> S
{{< /mermaid >}}

작업을 배정할 때 전체 대화를 복제하면 각 작업자가 같은 자료를 중복 조회하기 쉽다. 작업에는 조사 ID, 맡은 단계, 읽어도 되는 입력, 산출물의 형태와 기한을 둔다. 결과가 늦거나 일부 작업자가 종료되면 실행기는 같은 작업을 다시 받을 수 있다. 이미 완료된 일을 확인할 ID와 상태가 남아 있어야 한다. 보고서 등록처럼 외부 상태를 바꾸는 단계에서는 요청 ID와 반영 결과를 연결해 중복 효과를 줄인다. [1장의 역할 분리](/ncp-aai/chapter-01-agent-architecture/) · [원자적 쓰기와 아웃박스 실습](/agentic-ai/lab-assistant/)

## 2. 컨테이너와 Kubernetes에서 맡는 역할

컨테이너 이미지에는 실행 코드, 런타임, 고정한 의존성을 담고, 환경별 비밀 값은 외부 설정으로 주입한다. Kubernetes의 **Deployment**는 복제본과 업데이트를 관리하고, **Service**는 요청을 받을 수 있는 Pod에 연결한다. **HPA**는 구성한 관측 지표를 보고 원하는 복제 수를 바꾼다. 실행기는 CPU가 여유로워도 모델 서버를 오래 기다리고 있을 수 있으므로, 큐 길이와 대기 시간처럼 병목과 가까운 지표도 살핀다. [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) · [Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)

| 검사 | 질문 | 실패 시 주로 찾는 조치 |
|---|---|---|
| 시작 검사 | 초기 로딩이나 초기화가 완료됐나 | 느린 시작을 기다릴 시간과 한도를 정한다. |
| 준비 검사 | 이 인스턴스가 지금 새 요청을 받을 수 있나 | 새 트래픽을 보내는 대상에서 제외한다. |
| 생존 검사 | 프로세스가 살아서 정상 동작을 계속할 수 있나 | 지속 실패 시 컨테이너 재시작을 검토한다. |

모델 로딩 중에는 프로세스가 살아 있어도 새 요청을 받을 준비가 안 될 수 있다. 준비와 생존을 같은 기준으로 판단하면 초기화 중인 인스턴스를 반복 재시작할 수 있다. 실제 헬스 체크는 쓰는 이미지와 엔드포인트의 계약을 기준으로 설정한다. [Kubernetes의 세 가지 프로브](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/) · [7장의 NIM API](/ncp-aai/chapter-07-nvidia-platform/)

## 3. 가상 시계에서 복제와 대기 제한 비교

[scaling_lab.py](scaling_lab.py)는 합성 요청 일곱 건의 도착 시각과 작업 시간을 고정한 뒤 가장 먼저 사용 가능해지는 작업자에 배정한다. 기한은 각 요청 도착 후 150ms다. 대기 제한을 켠 경우 시작까지 40ms를 초과할 것으로 예상한 요청을 거부한다. 이미 배정된 작업은 예제에서 끝까지 계산한다. Python으로 실행한 결정적 시뮬레이션이고, Kubernetes 성능을 측정한 값은 아니다.

```bash
python3 scaling_lab.py
```

| 구성 | 기한 내 성공 | 기한 초과 | 입장 거부 | 배정된 요청의 p95 |
|---|---:|---:|---:|---:|
| 작업자 1개 | 2 | 5 | 0 | 315ms |
| 작업자 2개 | 7 | 0 | 0 | 150ms |
| 작업자 2개, 대기 상한 40ms | 5 | 0 | 2 | 135ms |

마지막 구성은 배정된 요청의 p95가 낮아졌다. 두 건을 큐에 넣지 않았으므로 전체 성공 수는 7건에서 5건으로 줄었다. 배정된 요청의 지연만 보면 그 차이를 놓친다. 서비스에서는 완료 수, 거부 수, 기한 내 성공률을 같이 둔다. 일곱 건에서 최근접 순위 p95는 최대값이므로, 실제 성능을 해석하려면 요청 조건을 넓히고 충분한 표본을 모아야 한다. [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)

## 4. CI/CD에서 버전과 검증을 묶기

에이전트 변경에는 앱 이미지, 모델, 프롬프트, 도구 스키마, 정책이 같이 움직일 수 있다. 배포 버전에 이 참조와 평가셋 버전을 연결한다. 빌드 작업에서는 단위 검사와 도구 계약 검사를 돌린다. 배포 전에는 조회·비교·보류·권한 경계를 같은 조건에서 재검증한다. 제한된 트래픽으로 새 버전을 실행하면 이전 버전과 지연·성공률·거부 상태를 비교할 수 있다.

롤백에서는 앱 이미지만 되돌려도 되는 범위를 먼저 확인한다. 저장한 체크포인트의 필드나 도구 반환 계약이 바뀌었다면, 이전 버전이 그 상태를 이어 읽을 수 있어야 한다. 모델 응답에 포함된 출처 기록은 버전 비교에 남길 수 있지만, 권한 없는 팀의 원문이나 요청 비밀 값은 운영 로그에 넣을 필요가 없다. [8장의 버전별 회귀와 운영 지표](/ncp-aai/chapter-08-run-monitor-maintain/)

## 5. 비용과 가용성을 함께 계산하기

복제본을 더 두면 일부 노드에서 문제가 생겼을 때 나머지가 요청을 받기 쉽다. 다만 같은 저장소와 모델 서버를 공유한다면 그 공통 의존성에서 다시 막힐 수 있다. 각 계층의 복제 수, 준비 가능한 인스턴스, 상태 복구 경로를 같이 본다. 모델 복제는 가중치와 실행 메모리를 인스턴스마다 확보해야 하므로 요청 처리량뿐 아니라 모델 준비 시간과 유휴 자원도 비용에 포함한다.

분산 부하 시험에서는 요청 도착률, 동시 요청 수, 입력·출력 길이와 도구 경로를 고정한 뒤 비교한다. 큐 대기, 추론, 도구 대기, 최종 종합을 따로 재면 어느 계층을 확장할지 결정할 수 있다. 실제 서비스에서 전체 지연은 동시 호출의 임계 경로와 공유 자원 상태에 따라 달라진다. [NVIDIA NIM Benchmarking](https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html) · [블로그의 병렬 임계 경로](/agentic-ai/orchestration/)
