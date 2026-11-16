---
title: "NVIDIA NCP-AAI 10장 · 사람과 에이전트의 협업"
date: 2026-11-16T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
weight: 100
summary: "사람이 기준 시험을 고르고 결과를 수정·승인·취소하는 흐름을 판본과 이벤트로 검증했다."
tags: ["NCP-AAI", "Human-in-the-loop", "User Experience", "Oversight"]
categories: ["NVIDIA NCP-AAI"]
showTableOfContents: true
showAuthor: false
---

R42의 검증 결과를 비교하려는데 비교 기준 후보가 R38과 R41 두 개로 남았다. 사용자는 왜 두 후보가 나왔는지 보고 하나를 고른다. 검증 도우미는 근거가 달린 초안을 만든다. 초안이 수정되거나 승인 후 게시를 취소하면 화면과 실행 상태도 바뀌어야 한다. 비교 기준 선택, 승인, 수정은 상태를 바꾸는 이벤트로 다뤘다.

공식 학습 가이드 10.1–10.4는 사용자 인터페이스와 사람 개입, 구조화된 피드백, 결정의 추적성과 설명, 수정·취소·승인 같은 개입을 다룬다. 웹 출제표에서 비중은 5%다. [공식 출제표](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/#exam-blueprint) · [공식 학습 가이드 10장](https://dam-cdn.nvd.orangelogic.com/AssetLink/64tei188l3tt132l265u1ipjoexdl1p5.pdf#page=13)

## 1. 어떤 결정에서 사람에게 넘길지 정하기

비교 기준을 고르는 일은 모호한 입력을 해소한다. 보고서 검토에서는 근거와 판단을 확인한다. 게시는 외부 상태를 바꾸는 작업이다. 인터페이스에 현재 상태, 가능한 선택과 결과, 선택자가 볼 수 있는 근거를 함께 둔다.

| 상태 | 화면에서 확인하는 정보 | 허용할 행동 |
|---|---|---|
| 기준 선택 대기 | 후보별 시험 목적, 측정 조건, 판본과 출처 | 후보 선택, 추가 정보 입력 |
| 초안 검토 | 기준·대상, 계산식, 인용 근거, 보고서 변경 내용 | 수정, 승인, 취소 |
| 승인 완료 | 승인자가 확인한 판본과 작업 범위 | 승인 취소 또는 실행 요청 |
| 실행 결과 | 게시 성공 여부와 게시된 대상 또는 실패 사유 | 권한에 따른 후속 조치 |

오래 걸리는 작업에서는 화면 연결이 끊겼다가 재접속해도 마지막 상태를 다시 보여 줄 수 있어야 한다. 중단 지점을 저장하는 워크플로는 실행 ID로 상태를 찾아 승인이나 수정을 적용하고 다음 단계로 진행한다. [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [5장의 영속 상태 실습](/ncp-aai/chapter-05-cognition-planning-memory/)

## 2. 같은 승인 버튼을 두 번 눌렀을 때

{{< mermaid >}}
stateDiagram-v2
    [*] --> NeedsInput
    state "기준 선택 대기" as NeedsInput
    state "검토 대기" as InReview
    state "승인 완료" as Approved
    state "취소" as Cancelled
    NeedsInput --> InReview : 비교 기준 선택
    InReview --> Approved : 현재 판본 승인
    Approved --> Approved : 같은 판본 승인 재전송
    Approved --> InReview : 초안 수정과 판본 증가
    InReview --> InReview : 초안 수정과 판본 증가
    InReview --> Cancelled : 취소
    Approved --> Cancelled : 취소
    Cancelled --> [*]
{{< /mermaid >}}

[oversight_lab.py](oversight_lab.py)에 모의 사용자 이벤트 여덟 건을 주고 상태 전이와 승인 판본을 한 줄씩 출력했다. `approved`는 검토를 마친 상태다. 스크립트는 게시 전의 상태 전이를 다룬다.

```bash
python3 oversight_lab.py
```

| 순서 | 이벤트 | 처리 | 최종 상태 | 승인 판본 |
|---:|---|---|---|---:|
| 1 | 비교 기준 R38 선택 | `applied` | `in_review`, 1판 | — |
| 2 | 1판 승인 | `applied` | `approved`, 1판 | 1 |
| 3 | 1판 승인 재전송 | `already_applied` | `approved`, 1판 | 1 |
| 4 | 1판 초안 수정 | `applied` | `in_review`, 2판 | — |
| 5 | 뒤늦게 도착한 1판 승인 | `ignored_stale_or_invalid` | `in_review`, 2판 | — |
| 6 | 2판 승인 | `applied` | `approved`, 2판 | 2 |
| 7 | 취소 | `applied` | `cancelled`, 2판 | — |
| 8 | 취소 후 도착한 2판 승인 | `ignored_stale_or_invalid` | `cancelled`, 2판 | — |

실제 서비스에서는 상태를 바꾸는 API에 확인 중이던 판본과 이벤트 ID를 넣는다. 서버는 저장된 현재 판본과 상태를 같은 트랜잭션 안에서 검사한다. 브라우저가 응답을 받기 전에 연결을 잃으면 같은 이벤트 ID로 상태를 조회하거나 재전송할 수 있다. 게시 요청은 별도의 실행 ID와 결과를 남긴다. 게시가 끝난 뒤에는 이미 발생한 변경을 되돌릴 수 있는지, 그 권한이 누구에게 있는지 확인한다.

## 3. 선택한 근거와 사용자 피드백 남기기

비교 기준 선택 화면에는 R38과 R41의 실험 목적, 측정 조건, 사용한 근거를 표시할 수 있다. 초안의 84.0 같은 수치에는 시험 ID와 자료 판본, 적용 조건을 연결한다. 사용자가 왜 특정 후보가 제안됐는지 확인할 수 있도록 선택 규칙과 근거 링크를 보여 준다. 운영자가 같은 실행을 조사할 때에는 요청 ID로 도구 호출, 문서 판본, 정책 결정과 사용자 승인 이벤트를 연결한다. 사용자의 자료 접근 권한은 상세 화면에도 적용된다. [6장의 근거 연결](/ncp-aai/chapter-06-knowledge-integration/) · [8장의 운영 트레이스](/ncp-aai/chapter-08-run-monitor-maintain/)

피드백에는 `run_id`, `revision`, `reason_code`, `target_field`를 두고 선택적으로 설명과 정정 값을 받을 수 있다. 예를 들어 “비교 기준 오류”와 “원본 수치 오류”는 수정할 구성요소가 다르다. 한 요청에서 R38을 선택한 이벤트와, 앞으로 유사한 작업에서 R38을 선호한다는 설정도 적용 범위가 다르다. 검토한 실패 사례를 평가셋에 넣고 변경 전후 과업 성공률을 비교할 수 있다. [3장의 구조화된 피드백과 회귀 검토](/ncp-aai/chapter-03-evaluation-and-tuning/)

검토 요청이 너무 잦으면 사용자가 빠르게 승인하거나 중간에 떠날 수 있다. 기준 선택에 걸린 시간, 수정·취소 사유, 승인 후 다시 검토한 비율을 기록하면 화면에서 빠진 정보를 찾는 데 도움이 된다. 반복되는 모호성은 입력 안내나 업무 규칙을 개선할 후보로 남긴다.
