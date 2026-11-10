---
title: "NCA-AMAD 7장 GPU 자원 산정과 서비스 운영"
date: 2026-11-10T00:00:00+09:00
draft: false
weight: 70
author: "빵주(kbbangzoo)"
description: "LLM 메모리, 동시 요청과 Replica 수를 계산하고, SLO·관측 가능성·Drift·Canary·Rollback을 연결해 운영 기준을 정리한다."
summary: "모델이 메모리에 들어가는 것과 서비스가 목표를 만족하는 것은 다르다. 용량, 성능, 품질, 복구를 함께 본다."
categories: ["NVIDIA NCA-AMAD"]
tags: ["Sizing", "GPU", "MLOps", "Observability", "Model Monitoring"]
series: ["NVIDIA NCA-AMAD"]
math: true
showTableOfContents: true
showAuthor: false
showHero: false
---

[NVIDIA NCA-AMAD 시리즈 전체 목차](/nca-amad/)

모델을 한 번 실행하는 데 성공했다고 서비스 운영에 필요한 자원까지 확인한 것은 아니다. 긴 입력, 여러 동시 요청, 재시도, 장애 상황에서는 필요한 메모리와 처리 능력이 달라진다.

Sizing and Configuration을 복습할 때는 세 질문을 순서대로 정리했다. **메모리에 들어가는가? 목표 지연시간 안에서 요청량을 감당하는가? 배포 이후에도 품질과 장애를 확인하고 복구할 수 있는가?** 앞의 두 질문이 용량 산정이라면, 마지막 질문은 그 산정이 실제 서비스에서 유효한지 확인하는 과정이다.

## 1. GPU를 고르기 전에 업무 조건부터 정한다

자원 산정(Sizing)은 GPU 이름을 고르는 일에서 시작하지 않는다. 먼저 어떤 요청을 얼마나 처리해야 하는지 정의한다.

| 항목 | 기록할 조건 | 빠졌을 때 생기는 오해 |
| --- | --- | --- |
| 모델 | 모델·버전·정밀도·서빙 엔진 | 같은 파라미터 수면 성능도 같다고 생각한다. |
| 입력·출력 | 토큰 길이의 분포와 상한 | 짧은 질문 결과로 긴 문서 요청을 예측한다. |
| 트래픽 | 평균·피크 요청률, 동시 진행 요청 | 가입자 수와 동시 추론 수를 혼동한다. |
| 서비스 목표 | 첫 토큰·전체 응답의 지연, 오류율 | 최대 처리량만 보고 사용자 대기를 놓친다. |
| 운영 조건 | 가용성, 유지보수, 데이터 위치, 비용 | 정상 상태에서 겨우 맞는 구성을 운영한다. |

여기서 SLO(Service Level Objective)는 서비스가 달성하려는 측정 가능한 목표다. SLA(Service Level Agreement)는 제공자와 사용자 사이의 서비스 수준 합의다. 둘을 같은 뜻으로 사용하기보다, 내부 운영 목표와 외부 합의의 관계를 구분하는 것이 좋다.

예를 들어 "빠른 응답"보다 "정의한 요청 집합에서 p95 첫 토큰 지연을 일정 값 이하로 유지"하는 형태가 측정 가능하다. p95는 관측값의 약 95%가 그 값 이하인 분위수다. 평균이 낮아도 일부 요청이 오래 대기할 수 있으므로 평균과 p95를 함께 본다.

## 2. 가중치 메모리와 전체 추론 메모리는 다르다

가중치의 이론적인 저장량은 다음과 같이 계산한다.

$$
M_{\mathrm{weights}} = N_{\mathrm{params}} \times \frac{b}{8}
$$

여기서 \(N_{\mathrm{params}}\)는 파라미터 수, \(b\)는 파라미터당 비트 수다. 이 식의 결과 단위는 Byte다.

파라미터가 80억 개인 모델을 가정하면 다음과 같다. 이 표에서는 **1 GB = 10⁹ Byte**를 사용한다.

| 저장 정밀도 | 파라미터당 저장량 | 8B 가중치의 단순 계산 |
| --- | --- | --- |
| FP32 | 4 Byte | 32 GB |
| FP16 / BF16 | 2 Byte | 16 GB |
| INT8 | 1 Byte | 8 GB |
| INT4 | 0.5 Byte | 4 GB |

실제 양자화 모델에는 스케일 등 메타데이터가 추가될 수 있고, 일부 텐서는 다른 정밀도로 남을 수 있다. 16 GB는 약 14.9 GiB이며, GB와 GiB를 혼용하면 계산이 어긋난다.

더 중요한 것은 추론에 가중치만 필요한 것이 아니라는 점이다.

$$
M_{\mathrm{inference}} \approx
M_{\mathrm{weights}} + M_{\mathrm{KV}} + M_{\mathrm{runtime}}
$$

마지막 항에는 활성값, 작업공간, 통신 버퍼, 실행 엔진의 추가 메모리 등이 들어간다. 따라서 **8B FP16 가중치가 16 GB라는 사실로 16 GB GPU에서 정상 서빙을 보장할 수 없다.** [NVIDIA NIM의 메모리 진단 문서](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html)도 가중치와 KV cache, 실행 시 추가 메모리를 구분한다.

이 식을 학습 전체 메모리 공식으로 사용해서도 안 된다. 학습에서는 gradient, optimizer state, 역전파를 위한 활성값 등 추가 항목과 학습 방식에 따른 차이가 생긴다.

### KV cache는 요청 조건에 따라 커진다

일반적인 Transformer에서 캐시에 저장하는 토큰이 늘고 동시에 진행하는 요청이 많아지면 KV 저장 요구도 커진다. 다만 모델의 KV head 수, 캐시 정밀도, 실행 방식에 따라 크기는 달라진다.

모든 요청의 캐시 길이가 같다고 단순화하면 다음처럼 근사할 수 있다.

$$
M_{\mathrm{KV}} \approx 2 L H_{\mathrm{KV}} D T B s
$$

- \(L\): layer 수, \(H_{\mathrm{KV}}\): KV head 수, \(D\): head 차원
- \(T\): 요청당 캐시된 토큰 수, \(B\): 동시 활성 요청 수
- \(s\): 캐시 원소당 Byte 수, 앞의 2: Key와 Value

이 식은 분할 전 모델 인스턴스 전체의 캐시량을 이해하기 위한 근사이며, TP 등으로 분산한 뒤 GPU 한 장에 할당되는 메모리와는 다르다. 실제로는 요청마다 길이가 다르고, paging·공유 prefix·분산 방식·모델 구조가 반영되므로 엔진의 측정값이 필요하다. 입력 최대 길이를 낮췄다고 엔진이 미리 예약한 캐시 메모리가 반드시 줄어드는 것도 아니다.

## 3. Replica와 Tensor Parallelism을 나누어 계산한다

두 개념은 목적부터 다르다.

- **Replica:** 독립적인 모델 서비스 복제본을 늘려 요청을 분산한다.
- **Tensor Parallelism(TP):** 한 모델 인스턴스의 텐서 연산을 여러 GPU에 나누어 실행한다.

한 Replica가 여러 GPU를 사용할 수 있다. 따라서 "Replica가 4개"와 "GPU가 4개"는 같은 말이 아니다. 가중치를 TP로 나누더라도 통신과 일부 중복 메모리 등이 있으므로 실제 메모리와 처리량이 이상적으로 비례한다고 가정하지 않는다.

다음은 **설명을 위한 가정**이다. 실제 모델이나 GPU 벤치마크 수치가 아니다.

- 목표 피크 요청률: 초당 70개
- 목표 지연시간을 만족하는 Replica당 실측 처리 능력: 초당 20개
- 각 Replica의 구성: TP 2, 즉 GPU 2개
- 요청을 균등하게 분산할 수 있다고 가정

$$
N_{\mathrm{replicas}} = \left\lceil \frac{70}{20} \right\rceil = 4
$$

$$
N_{\mathrm{GPUs}} = 4 \times 2 = 8
$$

이것은 정상 상태의 최소 산정 예다. 한 Replica가 장애로 빠지면 나머지 세 개의 합산 처리 능력은 가정상 초당 60개다. 한 Replica 장애 중에도 초당 70개를 처리해야 한다면, 이 단순 조건에서는 최소 다섯 Replica를 검토해야 한다. 실제 운영에서는 여기에 변동성과 유지보수 조건도 더해 검증한다.

Autoscaling도 즉시 자원을 제공하는 마법은 아니다. 모델 다운로드·로딩·warm-up에 시간이 걸리고, GPU 확보나 스케줄링이 지연될 수 있다. 피크가 짧고 갑작스럽다면 대기 자원과 요청 제한, 큐 정책도 함께 필요하다.

여기서 **Horizontal scaling**은 Replica처럼 실행 단위를 늘리는 방향이고, **Vertical scaling**은 기존 실행 단위에 더 많은 메모리나 연산 자원을 할당하는 방향이다. TP는 한 모델의 연산을 나누는 방식이므로 수평 확장과 동의어가 아니다. 어떤 구성이든 변경 뒤에는 지연시간과 비용을 다시 측정한다.

### 요청률과 동시 요청 수는 다르다

안정 상태에서 같은 시스템 경계를 사용하면 Little's Law로 평균 동시 진행 요청 수를 추정할 수 있다.

$$
\bar{N} = \lambda \bar{W}
$$

평균 유입이 초당 20개이고 평균 체류시간이 2초라면 평균 동시 진행 요청은 40개다. 여기에는 정의한 경계 안의 대기와 처리가 포함된다. 평균 관계에 p95 시간을 그대로 대입하거나, 이 값을 곧바로 GPU의 batch size로 해석하면 안 된다.

## 4. 벤치마크는 같은 조건에서 읽는다

[NVIDIA의 LLM Benchmarking 문서](https://docs.nvidia.com/nim/benchmarking/llm/overview)는 모델 성능 측정과 서비스 부하 시험을 구분한다. 둘은 서로 보완적이다.

성능 측정에서는 모델, 정밀도, 입력·출력 길이, 동시성 등 조건을 통제한다. 부하 시험에서는 실제와 비슷한 트래픽, 네트워크, 큐와 확장 동작까지 확인한다.

특히 **시스템 전체 tokens/s가 높다는 사실이 개별 사용자의 생성 속도가 빠르다는 뜻은 아니다.** 동시 요청을 늘리면 전체 처리량이 증가하는 동안 각 요청의 대기나 토큰 간 지연도 길어질 수 있다. 지연 목표를 넘긴 최대 처리량을 정상 운영 용량으로 사용하지 않는다.

측정 결과에는 적어도 다음을 함께 남긴다.

- 모델·엔진·설정 버전과 사용한 하드웨어
- 입력·출력 길이의 실제 분포, concurrency와 요청률
- TTFT, 토큰 간 지연, 전체 응답시간의 평균·분위수
- 성공한 요청 수, timeout·오류율, 메모리 사용량
- warm-up과 cache 사용 여부, 측정 구간

지표의 정의도 맞춰야 한다. 토큰 간 지연 계산에 첫 토큰 대기를 포함하는지 등은 도구마다 차이가 있을 수 있다. 세부 정의는 [공식 Metrics 설명](https://docs.nvidia.com/nim/benchmarking/llm/metrics)에서 확인한다.

## 5. 서버의 정상 상태와 답변 품질을 함께 본다

관측 가능성(Observability)을 위한 대표 수단은 metrics, logs, traces다.

| 수단 | 답하려는 질문 | AI 서비스의 예 |
| --- | --- | --- |
| Metrics | 얼마나 자주, 얼마나 오래 발생하는가? | 지연 분위수, 오류율, 큐 길이, GPU 메모리 |
| Logs | 어떤 사건이 어떤 조건에서 발생했는가? | 오류 종류, 모델 버전, 요청 식별자 |
| Traces | 한 요청이 어디를 거쳐 어디서 지연됐는가? | 검색, reranking, 생성, 도구 호출의 구간별 시간 |

Dashboard는 이 정보를 모아 보여 주고, Alert는 정한 조건을 만족했을 때 대응을 시작하도록 알린다. 대시보드만 있다고 대응 체계가 완성되는 것은 아니다.

예를 들어 문서 질의 서비스가 느려졌다면 GPU 사용률만 보는 대신 검색·reranker·생성 단계의 trace를 비교한다. 검색이 대부분의 시간을 쓰고 있다면 생성 모델용 GPU를 추가하는 것이 우선 해법이 아닐 수 있다.

반대로 HTTP 요청은 모두 성공하고 지연도 정상인데, 오래된 문서를 인용해 틀린 답을 줄 수 있다. 이때는 답변 정확성, 근거 충실도, 문서 최신성, 업무 완료율 같은 품질 지표가 필요하다. [NVIDIA의 AI 플랫폼 관측 가이드](https://docs.nvidia.com/ai-enterprise/planning-resource/ai-factory-reference-design-for-government-white-paper/latest/observability.html)도 인프라 지표와 애플리케이션·에이전트 지표를 함께 다룬다.

운영 정답이 즉시 얻어지지 않는 경우에는 표본 검토와 대리 지표를 활용하되, 대리 지표를 정답률과 동일시하지 않는다. 로그와 trace에는 원문 개인정보·민감정보가 불필요하게 쌓이지 않도록 접근 권한, 마스킹, 보존 기간도 설정한다.

## 6. Drift가 보이면 무엇을 바꿀 것인가

Data drift는 입력 데이터의 분포 변화이고, Concept drift는 입력과 목표 사이의 관계 변화다. 모델 가중치를 바꾸지 않았더라도 환경이 달라지면 예측 품질이 나빠질 수 있다.

하지만 분포 변화만 감지했다고 재학습부터 시작하지 않는다. 먼저 수집 오류, 스키마 변경, 전처리 차이, 검색 데이터의 노후화, 사용자 요청의 변화를 구분한다.

문서 질의 서비스라면 다음과 같이 판단할 수 있다.

| 관측 | 먼저 확인할 부분 |
| --- | --- |
| 새 문서가 검색되지 않는다. | 수집·인덱싱 상태, 권한 필터, 문서 갱신 주기 |
| 관련 문서는 있는데 답변이 근거를 따르지 않는다. | 문맥 구성, prompt, 생성 모델과 평가 기준 |
| 특정 언어의 요청만 품질이 낮아졌다. | 언어별 데이터·검색·모델 성능 |
| 새 배포 이후 지연만 급증했다. | 엔진 설정, 길이 분포, cache, 배포 구성 |

이는 가능한 진단 순서이지 원인을 단정하는 규칙은 아니다. [운영 모델 모니터링 가이드](https://developer.nvidia.com/blog/a-guide-to-monitoring-machine-learning-models-in-production/)의 핵심처럼, 데이터·모델·코드를 함께 추적해야 변화의 원인을 좁힐 수 있다.

모델만 버전 관리해서는 재현이 어려울 수 있다. 전처리, tokenizer, prompt, adapter, embedding 모델, RAG 인덱스, 서빙 설정도 결과를 바꾼다. 특히 embedding 모델을 바꾸면 기존 벡터와의 호환성을 확인하고 재색인이 필요한지 검토한다.

## 7. 배포와 복구를 설계에 포함한다

새 버전은 독립 평가를 통과한 뒤 제한된 트래픽으로 Canary 배포하고, 품질·지연·오류를 관측하며 확대할 수 있다. 통과 조건과 중단 조건, 이전 버전으로 돌아갈 방법을 배포 전에 정한다. [Google의 Productionization 가이드](https://developers.google.com/machine-learning/managing-ml-projects/production)도 배포 승인, 단계적 환경, 실패 대응과 rollback 절차를 문서화하도록 설명한다.

Rollback은 모델 파일만 교체하는 문제로 끝나지 않을 수 있다. 모델과 prompt, 검색 인덱스, 전처리가 서로 맞는 버전인지 확인해야 한다. 외부 시스템에 이미 수행한 쓰기 작업은 모델을 되돌린다고 자동 취소되지 않으므로 별도의 복구 절차가 필요하다.

- **Runbook:** 특정 문제를 확인하고 처리하는 구체적인 실행 절차
- **Playbook:** 상황 분류, 담당자, 판단 분기, 소통과 대응을 포함하는 체계
- **MTTD:** 장애 발생부터 탐지까지 걸린 평균시간
- **MTTR:** 합의한 시작점부터 복구까지 걸린 평균시간. Repair·Recovery·Resolution 등 조직별 정의를 먼저 맞춘다.

예를 들어 두 장애의 탐지 지연이 2분과 6분이면 MTTD는 4분이다. 복구시간을 "탐지부터 서비스 복원까지"로 정의하고 각각 8분과 12분이었다면 MTTR는 10분이다. 평균만으로 긴 장애의 영향을 모두 설명할 수는 없으므로 개별 사고의 범위와 심각도도 남긴다.

## 8. 이 장의 핵심

자원 산정은 가중치 용량 계산으로 시작할 수 있지만 거기서 끝나지 않는다. KV cache와 실행 메모리, 목표 지연시간을 만족하는 처리량, 장애 여유와 확장 준비시간을 함께 확인해야 한다.

운영에서도 자원 효율만 최적화하면 충분하지 않다. 서버가 정상인지, 결과가 유효한지, 문제가 생겼을 때 원인을 찾고 되돌릴 수 있는지를 함께 봐야 한다.

앞선 1~6장의 내용도 여기에서 다시 연결된다. 에이전트의 도구 호출은 지연과 비용을 늘리고, 데이터의 변화는 검색·모델 품질을 바꾸며, 맞춤화는 배포 버전과 검증 항목을 늘린다. **좋은 모델을 선택하는 일과 좋은 서비스를 운영하는 일 사이에는 이 연결을 관리하는 과정이 있다.** 다음 장에서는 시간과 자원 제약이 실제 움직임으로 이어지는 Physical AI를 살펴본다.

## 참고 자료

- [NVIDIA NIM — GPU 메모리와 OOM 진단](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html)
- [NVIDIA NIM — Benchmarking Overview](https://docs.nvidia.com/nim/benchmarking/llm/overview)
- [NVIDIA NIM — LLM 성능 지표의 정의](https://docs.nvidia.com/nim/benchmarking/llm/metrics)
- [NVIDIA — AI 플랫폼 Observability](https://docs.nvidia.com/ai-enterprise/planning-resource/ai-factory-reference-design-for-government-white-paper/latest/observability.html)
- [NVIDIA — A Guide to Monitoring Machine Learning Models in Production](https://developer.nvidia.com/blog/a-guide-to-monitoring-machine-learning-models-in-production/)
- [Google for Developers — Productionization](https://developers.google.com/machine-learning/managing-ml-projects/production)

---

[← 6장. NVIDIA AI Enterprise와 Blueprints](/nca-amad/chapter-06-enterprise-blueprints/) · [전체 목차](/nca-amad/) · [8장. Physical AI →](/nca-amad/chapter-08-physical-ai/)
