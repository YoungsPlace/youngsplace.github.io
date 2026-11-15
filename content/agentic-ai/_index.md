---
title: "Agentic AI 학습 노트"
description: "에이전트 루프에서 시작해 계획, 메모리, 검색, 도구와 평가까지 작은 구현으로 확인한 기록."
summary: "다음 행동을 고르는 에이전트부터 메모리, 검색, 평가, 운영까지."
draft: false
showTableOfContents: false
showHero: false
orderByWeight: true
---

NCA-AMAD에서 에이전트의 구성 요소를 살펴봤다. 이번에는 도구의 반환값을 받아 다음 행동을 선택하는 가장 작은 예제부터 만들어 본다. NVIDIA NCP-AAI의 공개 시험 안내에서 설계, 계획, 메모리, 배포와 평가 주제를 가져와 개인 학습 순서로 정리했다. [NCA-AMAD 학습 노트](/nca-amad/) · [NVIDIA NCP-AAI 공개 안내](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/)

## 읽는 순서

처음에는 [에이전트 루프](/agentic-ai/agent-loop/)에서 도구 호출을 한 단계씩 보고, [계획](/agentic-ai/replanning/)과 [메모리](/agentic-ai/memory-layers/)에서 한 번의 응답이 어떻게 이어지는지 따라갈 수 있다. 같은 예제는 [메모리 검색](/agentic-ai/memory-retrieval/)과 [RAG](/agentic-ai/rag/)에서 문서를 고르는 문제로 이어진다.

시스템을 붙여 본 독자라면 [일정 조회의 상태별 행동 제한](/agentic-ai/agent-loop/#넓은-선택지와-상태로-제한한-선택지-비교), [메모리 정정의 두 시간 축](/agentic-ai/memory-writing/#소급-정정은-두-시간-축에서-다르게-보인다), [WAL에서의 삭제 시점](/agentic-ai/memory-lifecycle/#삭제가-끝난-시점에-진행-중이던-읽기), [동시 예약과 아웃박스](/agentic-ai/lab-assistant/#동시에-두-사람이-같은-시간을-확정하면)를 예제와 함께 볼 수 있다.

## 글 목록
