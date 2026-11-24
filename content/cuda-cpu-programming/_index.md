---
title: "CUDA 프로그래밍"
description: "CUDA for LLMs를 읽으며 GPU 프로그래밍과 트랜스포머 커널을 공부한 기록."
showTableOfContents: false
showHero: false
orderByWeight: true
---

『CUDA for LLMs』를 읽으며 파이토치 연산이 GPU에서 어떻게 실행되는지 공부한다. 1주차의 출발점은 벡터 덧셈과 ReLU였다. 각각의 스레드가 입력의 어느 원소를 읽고 결과를 어디에 쓰는지 따라가면, 하나의 연산을 커널로 옮기는 과정이 보인다. [스터디 일정](https://gasidaseo.notion.site/CUDA-GPU-41d50aec5edf80a2bf1efdc3e5e7d24b) · [1주차 발표](https://gasidaseo.notion.site/When-PyTorch-just-isn-t-enough-42150aec5edf80eebff3de9b0a2b87d3)

12월 8일 발표에서는 트랜스포머가 토큰을 하나 생성할 때부터 시작한다. 5장에서는 어떤 행렬을 읽고 무엇을 캐시에 남기는지 추적한다. 6장에서는 그중 GEMV, softmax, LayerNorm, top-k, GEMM을 골라 메모리 접근과 스레드 배치를 바꿨을 때의 효과를 살펴본다. [5장 공부 노트](chapter-05/) · [5장 발표 화면](chapter-05/talk/)

75분 기준으로 5장에 33분, 6장에 35분, 질의응답에 7분을 배정했다. 60분이면 5장의 메모리 배치와 바인딩은 코드 한 줄씩 짚고 넘어가 25분으로 줄이고, 6장 30분과 질문 5분을 확보한다. 90분이면 5장에 35분, 6장에 45분을 쓰고 손계산과 프로파일 비교 뒤 질문을 10분 받는다.
