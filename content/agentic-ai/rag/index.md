---
title: "문서 검색을 모델 답변에 연결하기 · RAG와 출처"
date: 2026-11-16T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
weight: 70
summary: "같은 예약 질문을 문서와 절 단위로 검색하고, 현행 안내를 Gemma에 전달해 출처와 답변을 확인했다."
tags: ["Agentic AI", "RAG", "Ollama", "Embeddings"]
showTableOfContents: true
showAuthor: false
showHero: false
---
실험실 예약 안내의 첫 버전에는 ‘24시간 전’, 새 버전에는 ‘48시간 전’이라고 되어 있다. 예제의 예약 대상은 공용 실험 장비로 고정했다. 질문에서 쓴 ‘방문’과 ‘예약’은 첫 버전의 표현과 더 가깝다. 문서 검색이 24시간을 올렸다면, 모델은 화면에 올라온 그 근거를 그대로 읽을 수 있다. 검색 순위, 현행 버전, 답변이 근거를 따르는지를 나눠 확인했다. [5편의 사용자 기억 검색](/agentic-ai/memory-retrieval/)

## 문서를 고르고 답을 만들기

RAG는 검색 결과를 모델 생성에 쓰는 구조다. 원 논문은 학습된 검색기와 BART 생성기를 연결했다. RAG-Sequence는 문서 하나를 생성 시퀀스의 잠재 변수로 두고, RAG-Token에서는 토큰마다 다른 문서로부터 확률을 얻을 수 있다. 이번 구현에서는 이미 학습된 임베딩 모델과 채팅 모델을 따로 쓴다. 검색한 문장 하나를 프롬프트에 넣고 답을 요청했다. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · Patrick Lewis 외, NeurIPS 2020, arXiv v4 2021-04-12, §2](https://arxiv.org/html/2005.11401#S2)

{{< mermaid >}}
flowchart TD
    D["문서 · 버전과 섹션 ID"] --> C["문서 전체 또는 절로 나누기"]
    C --> V["텍스트 임베딩"]
    Q["예약 질문"] --> E["질문 임베딩"]
    V --> R["유사도 순위"]
    E --> R
    R --> F["현행 버전만 남기기"]
    F --> G["가장 가까운 절을 출처와 함께 모델에 전달"]
    G --> A["답과 인용 또는 근거 없음"]
{{< /mermaid >}}

절 단위로 자르면 검색어와 가까운 문장을 직접 비교할 수 있고, 어떤 제목과 버전에서 왔는지 붙여 둬야 문맥을 잃지 않는다. 조각이 너무 짧으면 조건과 예외가 떨어질 수 있다. 예를 들어 ‘48시간’이 장비에만 적용되고 방에는 다른 규칙이 있다면, 제목과 본문을 같은 단위로 저장하거나 양쪽을 같이 읽어야 한다.

1차 검색은 질문과 비슷한 후보를 빠르게 찾는다. 2차 순위에서는 질문과 조각을 함께 보는 재정렬 모델을 쓸 수 있다. 이번에는 두 점수 모델을 두는 대신 현행 버전 조건을 적용했다. 각 문서의 모든 조각을 점수화하므로 구버전을 뺀 뒤에도 현행 1순위가 남는다. 큰 인덱스에서는 버전이나 권한 같은 필터를 검색 전으로 당겨야 후보가 상위 K 밖으로 밀리지 않는다. 버전 조건은 유사도 가산점보다 강한 제약이다. [NVIDIA NCP-AAI · Certification Learning Path, 2026-11-16 확인](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/#certification-learning-path)

## 예약 질문과 근거가 없는 질문

[rag.py](rag.py)와 [retrieve.py](/agentic-ai/memory-retrieval/retrieve.py)를 같은 폴더에 두고 실행했다. Python 3.13.18, Ollama 0.40.11, 임베딩 `all-minilm:22m`과 생성 `gemma3:4b`를 사용했다. 생성은 `temperature=0`, `seed=42`, 출력 상한 80토큰이다. 모델의 입력에는 질문, 검색된 절 한 개와 ID만 들어간다.

```bash
ollama pull all-minilm:22m
ollama pull gemma3:4b
python3 rag.py booking --generate
python3 rag.py parking --generate
```

| 예약 질문을 검색한 범위 | 첫 결과 | 나머지 관련 후보 |
|---|---|---|
| 문서 전체 | v1 · 0.8948 | v2 · 0.5168 |
| 절 단위 | v1#book · 0.8948 | v2#book · 0.6111 |
| 현행 절만 | v2#book · 0.6111 | v2#access · 0.2539 |

최신 안내에서 예약 문장만 떼니 유사도는 0.5168에서 0.6111로 올랐다. 구버전의 예약 문장 점수는 0.8948이었다. 현행 버전만 남긴 뒤 Gemma는 `48 hours [v2#book]`를 출력했다.

주말 주차 요금 질문에서는 셔틀 안내 `v2#transport`가 0.1352로 1순위였다. 가장 가까운 절이어도 답을 포함하지 않을 수 있다. 이 입력에서 Gemma는 `NO_EVIDENCE`를 출력했다. 검색 점수에 보류 임계값은 두지 않았고, 모델에는 근거가 있는 경우에만 제공한 ID를 인용하도록 요청했다. 반복 실행과 정답 없는 입력을 더 늘려야 출처 표기의 안정성을 확인할 수 있다.

## 질문과 조각을 함께 읽는 재정렬 모델

임베딩 검색기는 질문과 각 조각을 따로 숫자 벡터로 바꿀 수 있다. 문서가 바뀌기 전까지는 미리 계산한 문서 벡터를 재사용한다. BERT 기반 교차 인코더(cross-encoder)는 질문과 조각을 한 입력으로 받아 둘 사이의 토큰 관계를 직접 본다. 조각 하나에 대해 질문과 함께 추론해야 하므로 전체 문서보다 1차 상위 K개에 적용하는 경우가 많다. Nogueira와 Cho는 MS MARCO에서 먼저 검색한 구절의 순위를 BERT로 다시 매겨 첫 정답의 순위로 평가했다. [Passage Re-ranking with BERT · Rodrigo Nogueira, Kyunghyun Cho, arXiv 2019](https://arxiv.org/abs/1901.04085)

동일한 예약 문서에 비슷한 문장 두 개를 대조군으로 붙였다. 첫 문장은 예약과 도착을 언급하지만 알려 주는 값은 도착 10분 전 체크인이다. 두 번째 문장은 장비와 시간 조건을 함께 쓰지만 대여 *후* 24시간 안에 반납하는 규칙이다. 둘 다 현행 버전에 들어 있으므로 앞의 버전 필터만으로는 제외되지 않는다. 이처럼 질문과 같은 표현을 쓰면서 답을 직접 포함하지 않는 조각을 hard negative로 넣으면, 검색기가 키워드와 실제 조건 중 어느 쪽에 반응하는지 비교할 수 있다.

`all-minilm:22m`의 코사인으로 현행 기록 상위 3개를 고른 뒤, `cross-encoder/ms-marco-MiniLM-L6-v2`를 ONNX Runtime 1.32.0 CPU 실행기로 돌렸다. `rerank.py`는 질문과 후보 조각 쌍의 로짓으로 내림차순 정렬한다. 로짓과 코사인 점수는 서로 단위가 다르다. 정답 측에는 사전에 예약 기한이 들어 있는 `v2#book`을 지정했다. [MiniLM-L6-v2 모델 카드와 고정 리비전](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a)

| 질문 | 벡터 1위 | 교차 인코더 1위 | 정답 절의 순위 |
|---|---|---|---|
| 방문 전에 몇 시간 여유를 두고 예약하는가 | v2#checkin · 0.7807 | v2#checkin · 3.6910 | 2 → 2 |
| 장비 대여 준비에 필요한 최소 신청 기한 | v2#book · 0.6459 | v2#returns · −0.1125 | 1 → 2 |
| 주말 주차 비용 | v2#transport · 0.1352 | v2#storage · −11.2374 | 정답 없음 |

답이 있는 두 질문에서 첫 3개 후보 안에 정답 절이 모두 있었다. 첫 정답 순위의 역수를 평균한 MRR은 벡터만 썼을 때 `(1/2+1)/2=0.75`, 재정렬 뒤 `(1/2+1/2)/2=0.50`이었다. 방문 질문은 재정렬 뒤에도 체크인이 위였고, 대여 질문은 반납 조건이 1위로 올랐다. 주차 질문에도 순위는 생겼다. 점수와 순위를 얻어도 “신청하기 전”과 “빌린 뒤”를 갈라서 정답을 지지하는지 확인할 단계가 필요하다. 여기에는 합성 질문 두 개와 보류 사례 하나를 사용했다.

[rerank.py](rerank.py), [rag.py](rag.py), [retrieve.py](/agentic-ai/memory-retrieval/retrieve.py)를 같은 폴더에 둔다. 로컬 Ollama의 임베딩 모델 외에 고정 리비전의 `tokenizer.json`과 `onnx/model.onnx`를 같은 모델 폴더에 받는다. 마지막 파일명은 `model.onnx`로 둔다.

```bash
python3 -m pip install onnxruntime==1.32.0 tokenizers==0.23.3
ollama pull all-minilm:22m
python3 rerank.py ./model-directory --top-k 3
```

1차에 K=1을 주면 방문 질문의 정답 조각은 재정렬 입력에 들어가지 않는다. 반대로 K를 넓히면 답을 포함할 가능성은 높아지지만, 쌍마다 비용이 들고 비슷한 잡음도 함께 들어온다. 정답을 모를 때는 후보 순위와 별도로 “질문한 값이 이 조각에 직접 있는지”를 검사하거나 보류한다. 후보를 여러 개 모델에 전달할 때는 각 주장과 해당 문단을 연결해 평가한다. [9편의 검색·생성 채점](/agentic-ai/evaluation/)

## 검색 실패와 답변 실패를 나누기

검색 평가에서는 현행 정답 절이 상위 K개에 드는지 본다. 답변에서는 제시한 수치가 맞는지, 출처 ID가 실제로 제공된 목록 안에 있는지, 그 절이 주장을 뒷받침하는지도 읽어 본다. 출처 ID가 맞아도 답변이 틀릴 수 있다. 예를 들어 장비 예약 규칙을 방 예약 전체에 적용하면 확인할 문장 하나가 더 필요하다.

사용자 기억은 누가 언제 말했는지가 주된 필터다. 운영 문서는 적용 버전, 배포 대상, 효력 기간이 우선이다. 두 자료가 충돌하면 사용자 선호는 선택 가능한 장소를 정할 때 쓰고, 운영 문서는 예약 가능 조건에 쓴다.
