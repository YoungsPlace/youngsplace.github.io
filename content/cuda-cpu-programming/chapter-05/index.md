---
title: "CUDA 5장 · 트랜스포머 추론"
date: 2026-11-24T20:00:00+09:00
draft: false
weight: 50
author: "빵주(kbbangzoo)"
description: "CUDA for LLMs 5장. KV 캐시, GEMV, MoE의 top-k와 Mac에서 실행한 PyTorch 예제."
summary: "프리필과 디코드에서 계산하는 텐서를 비교했다. 저자의 어텐션 헤드로 K/V를 재사용했을 때의 출력도 확인했다."
showHero: false
showTableOfContents: true
---

## 5.1 트랜스포머 아키텍처와 환경 준비

### 5.1.1 기준 모델과 실행 환경

교재의 작은 PyTorch 모델은 《오즈의 마법사》를 문자 단위로 읽는다. 문자 ID로 임베딩 테이블에서 128차원 벡터를 꺼내고, 해당 위치의 임베딩을 더한다. 같은 문자라도 위치가 다르면 입력 벡터가 달라진다.

| 기호 | 뜻 | 값 |
|---|---|---:|
| B | 한 번에 처리하는 시퀀스 수 | 학습 예시 16 |
| T | 이번 입력에서 한 시퀀스의 토큰 수 | 최대 64 |
| C | 한 토큰의 은닉 벡터 너비 | 128 |
| H, D | 어텐션 헤드 수, 헤드 한 개의 너비 | 4, 32 |
| L, V | 블록 수, 예측할 문자 종류 수 | 8, 65 |

문자 세 개를 넣으면 임베딩 출력의 모양은 `[1, 3, 128]`이다. 블록을 모두 통과한 뒤에는 각 위치에서 어휘 65개에 대한 점수가 나온다.

교재 코드 5.14의 블록은 각 위치의 벡터를 LayerNorm으로 정규화한 뒤 어텐션에 넣는다. 서로 다른 가중치를 곱해 Q, K, V를 만든다. 현재 위치의 Q를 각 위치의 K와 내적해 점수를 구하고, softmax로 얻은 가중치로 V를 섞는다. 내적은 헤드 너비의 제곱근인 √32로 나눈다. 성분이 대체로 독립이고 분산이 비슷할 때, 성분을 많이 더할수록 내적이 크게 출렁이는 것을 이 스케일이 줄여 준다. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

프롬프트가 A, B, C라면 B는 A와 B까지만 볼 수 있다. B 행에서 스케일과 마스크를 적용한 점수를 `[ln 2, 0, −∞]`로 두면 softmax 결과는 `[2/3, 1/3, 0]`이다. 따라서 B의 출력은 `(2/3) × V_A + (1/3) × V_B`다. 마지막 위치인 C는 세 위치의 V를 모두 섞을 수 있다.

네 헤드의 출력을 이어 128차원으로 만들고 출력 투영을 거쳐 블록 입력에 더한다. 이어서 각 위치에 LayerNorm과 MLP를 적용한다. MLP는 128차원을 512차원으로 넓힌 뒤 GELU와 선형층을 거쳐 128차원으로 되돌린다. 이 결과도 잔차에 더한다. 두 연산 모두 정규화가 앞에 오는 pre-LN 구조다. 마지막 출력층이 내는 어휘별 점수를 logits라고 부른다.

#### Python에서 CUDA 커널을 호출하는 경로

교재의 Python 래퍼는 텐서를 받아 C++ 확장의 함수를 호출한다. 빌드할 때 `CUDAExtension`이 C++ 바인딩과 CUDA 소스를 한 모듈로 연결한다. 저자 코드의 GEMV 래퍼는 행렬과 벡터가 CUDA 텐서인지, 둘 다 FP32인지, 곱할 차원이 맞는지 확인한 뒤 출력 텐서를 만든다. 커널에는 포인터와 크기만 넘긴다. 커널이 행 주소를 낼 때는 입력이 행 우선으로 연속 저장돼 있다고 가정한다. 같은 Python 모델을 두고 한 연산씩 교체해 기준 출력과 비교할 수 있다. [저자 GEMV 래퍼와 커널 호출](https://github.com/Infatoshi/book.cu/blob/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer/csrc/inference/kernels/gemv_fwd.cu#L45-L80) · [PyTorch C++·CUDA 확장](https://docs.pytorch.org/docs/stable/cpp_extension.html)

원본 GEMV는 device guard로 행렬이 놓인 GPU를 선택한다. 커널 호출에서 스트림 인자는 생략한다. PyTorch 텐서를 별도 스트림에서 만들었다면, 커널도 같은 스트림으로 보내거나 이벤트로 대기 관계를 걸어야 텐서가 준비된 뒤 읽는다. [PyTorch CUDA 실행 의미론](https://docs.pytorch.org/docs/stable/notes/cuda.html)

#### 저자 코드로 로컬 실습 환경 준비하기

Mac에 만든 실습 폴더는 `cuda-study/chapter-05/`다. `book.cu/3_transformer/`에 원본을 두고, 옆에 둔 실행 스크립트에서 저자의 함수를 불러왔다. [저자 코드](https://github.com/Infatoshi/book.cu/tree/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer) · [실행 스크립트](run_book.py) · [패키지 버전](requirements.txt)

실습 장비는 Apple M5 Pro다. 새 폴더에서 재실행할 경우 `run_book.py`와 `requirements.txt`를 같은 곳에 두고 Python 3.13 가상환경을 만든다. 로그에 찍힌 버전은 Python 3.13.18, PyTorch 2.15.0이다.

설치 출력은 체크아웃, 패키지 설치 완료, 버전 확인 줄을 발췌했다. [설치 로그 전문](setup-output.txt)

```console
$ git clone --filter=blob:none --sparse https://github.com/Infatoshi/book.cu.git
Cloning into 'book.cu'...
$ git -C book.cu sparse-checkout set 3_transformer
$ git -C book.cu checkout --detach 08b406c47fb46f7b2d76bd77b2eaa08958a68f69
HEAD is now at 08b406c Clean up for print: drop agent notes and multi-node leftovers, fix READMEs
$ python3.13 -m venv .venv
$ .venv/bin/python -m pip install -r requirements.txt
Successfully installed MarkupSafe-3.0.4 filelock-4.3.1 fsspec-2026.9.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.7 numpy-2.5.5 setuptools-84.2.1 sympy-1.14.0 torch-2.15.0 typing-extensions-4.16.0
$ git -C book.cu rev-parse HEAD
08b406c47fb46f7b2d76bd77b2eaa08958a68f69
$ .venv/bin/python --version
Python 3.13.18
$ .venv/bin/python -c 'import numpy, torch; print(f"PyTorch {torch.__version__}, NumPy {numpy.__version__}"); print(f"CUDA={torch.cuda.is_available()} MPS={torch.backends.mps.is_available()}")'
PyTorch 2.15.0, NumPy 2.5.5
CUDA=False MPS=True
```

## 5.2 추론 이론과 최적화

### 5.2.1 자기회귀 생성 최적화

프롬프트 A, B, C를 한 번에 처리하는 단계가 프리필(prefill)이다. C 위치의 logits로 D를 고른다. 이어지는 디코드(decode)에서는 D를 모델에 넣어 E를 고른다.

| 호출 | 넣은 ID와 위치 | 호출 후 유효한 K/V 위치 | 다음 토큰을 고르는 출력 |
|---|---|---|---|
| 프리필 | A@0, B@1, C@2 | 0, 1, 2 | C 행의 logits에서 D 선택 |
| 첫 디코드 | D@3 | 0, 1, 2, 3 | D 행에서 E 선택 |
| 둘째 디코드 | E@4 | 0, 1, 2, 3, 4 | E 행에서 F 선택 |

새 토큰 세 개를 요청했다면 F를 고른 시점에 끝난다. 마지막으로 캐시에 추가된 위치는 E다. [Hugging Face의 캐시와 위치 설명](https://huggingface.co/docs/transformers/cache_explanation)

#### 캐시에 K와 V를 저장하는 이유

모델 가중치와 기존 토큰의 위치가 그대로인 인과적 추론에서 A, B, C의 은닉 상태는 뒤에 D를 추가해도 바뀌지 않는다. 그로부터 계산한 각 층의 K와 V를 다시 쓸 수 있다. 위치 3에서는 새 Q, K, V를 계산한다. 새 K와 V를 캐시에 추가하고 Q 하나로 K 네 개와 내적한 뒤, 확률에 따라 V 네 개를 섞는다. 다음 차례에 필요한 과거 정보는 K와 V이므로 앞선 Q는 보관할 필요가 없다.

| 텐서 | 단일 요청에서 프롬프트 길이 P를 처음 읽을 때 | 다음 위치 1개를 처리할 때 |
|---|---|---|
| Q | `[1, H, P, D]` | `[1, H, 1, D]` |
| 이번 호출의 K와 V 각각 | `[1, H, P, D]` | `[1, H, 1, D]` |
| 유효한 K와 V 각각 | `[1, H, P, D]` | `[1, H, P+1, D]` |
| 어텐션 점수의 논리적 모양 | `[1, H, P, P]` | `[1, H, 1, P+1]` |

새로 투영하는 K/V는 토큰 한 개분이다. 그래도 새 Q는 과거의 K를 모두 읽고, 가중합에 과거의 V를 모두 쓴다. 전체 문맥을 보는 어텐션에서는 프롬프트 길이가 P일 때 첫 디코드의 헤드당 Q·K 내적이 P+1개이고, 다음 스텝에는 P+2개로 늘어난다.

#### KV 캐시 크기를 바이트로 계산하기

각 층에 K와 V 두 배열이 있다. 배치에 패딩이 없고, 같은 길이 S에서 표준 다중 헤드 어텐션을 쓴다면 유효 데이터 크기는 다음과 같다. GQA/MQA에 적용할 때는 **실제로 K/V를 보관하는 헤드 수**를 넣는다.

```text
2 × 층 수 × 배치 크기 × 저장 길이 × KV 헤드 수 × 헤드 너비 × 원소당 바이트
```

작은 예제의 L=8, B=1, S=64, H=4, D=32에 FP32(4바이트)를 가정하면 `2 × 8 × 1 × 64 × 4 × 32 × 4 = 524,288`바이트, 즉 512 KiB다. B=1에서 한 위치를 추가할 때 드는 유효 데이터는 8 KiB다. 최대 길이 64를 미리 할당한 구현은 처음부터 512 KiB의 버퍼를 잡고 유효 길이만 늘릴 수 있다. 모델 가중치, 임시 텐서, 패딩, 할당기 여유분은 이 계산의 범위 밖이다. [Hugging Face의 캐시 구현과 메모리 선택](https://huggingface.co/docs/transformers/kv_cache)

#### 어텐션 헤드의 캐시 검산

저자 코드에서 `Head(96)` 하나를 꺼내 실행했다. 입력은 −1에서 1까지 균등하게 만든 FP32 값 3,072개를 `[1, 4, 768]`로 바꿨다. 네 위치를 한 번에 계산한 뒤 마지막 위치의 출력을 꺼냈다. 같은 헤드에 앞 세 위치의 K/V와 넷째 위치를 넘겨 계산한 결과를 이 출력과 비교했다. [저자의 Head.forward](https://github.com/Infatoshi/book.cu/blob/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer/inference.py#L141-L177)

```console
$ .venv/bin/python run_book.py head --device cpu
source=3_transformer/inference.py@08b406c47fb4
python=3.13.18 torch=2.15.0 arch=arm64
device=cpu cuda_available=False mps_available=True
head_input=[1, 4, 768] full_output=[1, 4, 96]
prefill_key=[1, 3, 96] prefill_value=[1, 3, 96]
decode_key=[1, 4, 96] decode_value=[1, 4, 96]
max_abs_diff=2.291e-07
assert_close=passed (rtol=1e-5, atol=1e-6)

$ .venv/bin/python run_book.py head --device mps
source=3_transformer/inference.py@08b406c47fb4
python=3.13.18 torch=2.15.0 arch=arm64
device=mps cuda_available=False mps_available=True
head_input=[1, 4, 768] full_output=[1, 4, 96]
prefill_key=[1, 3, 96] prefill_value=[1, 3, 96]
decode_key=[1, 4, 96] decode_value=[1, 4, 96]
max_abs_diff=1.490e-07
assert_close=passed (rtol=1e-5, atol=1e-6)
```

| 장치 | 앞 세 위치의 K, V 각각 | 넷째 위치 추가 후 각각 | 두 경로의 마지막 출력 최대 절대차 |
|---|---|---|---:|
| Mac CPU | `[1, 3, 96]` | `[1, 4, 96]` | `2.291e-07` |
| Apple GPU · MPS | `[1, 3, 96]` | `[1, 4, 96]` | `1.490e-07` |

K와 V에서 위치를 나타내는 가운데 축이 3에서 4로 늘었다. 각 장치에서 마지막 위치의 출력 96개를 대조했고, 둘 다 `torch.testing.assert_close(rtol=1e-5, atol=1e-6)`를 통과했다. Apple GPU는 PyTorch의 MPS 백엔드로 실행했다. [CPU 실행 로그](head-cpu.txt) · [MPS 실행 로그](head-mps.txt)

#### 작은 예제로 투영과 내적 횟수 세기

[decode_trace.py](decode_trace.py)는 문자 ID 4종, 특성 2개, 헤드 하나로 만든 합성 예제다. 훈련된 체크포인트 없이 고정 가중치와 간단한 위치 벡터를 사용한다. 한 번은 매번 전체 접두사를 다시 계산하고, 다른 한 번은 K/V를 보관한다. 두 경로 모두 가장 큰 logit의 토큰을 고른다. 이 예제는 어텐션과 출력 투영에 집중해 MLP와 LayerNorm은 포함하지 않는다.

같은 Mac의 가상환경에서 입력 `[1, 3, 0]`으로 토큰 세 개를 생성했다.

```console
$ .venv/bin/python decode_trace.py
prefix=3 cache_slots=3 next_id=3
prefix=4 cache_slots=4 next_id=3
prefix=5 cache_slots=5 next_id=2
full   kv_rows=12 qk_pairs=31
cached kv_rows=5 qk_pairs=15
max_abs_logit_diff=0.00e+00
generated [3, 3, 2]
```

`kv_rows`는 한 위치의 K와 V를 한 쌍으로 세었다. 전체 재계산은 3+4+5=12개, 캐시는 3+1+1=5개다. `qk_pairs`는 인과 마스크 안에서 실제로 계산한 Q·K 내적 수다. 전체 재계산은 길이 3, 4, 5의 삼각형을 매번 그려 6+10+15=31개다. 캐시는 최초 삼각형 6개에 새 행 4개, 5개를 더해 15개다. 세 번 비교한 출력 logits는 모두 일치했다. 이 횟수는 CPU 예제에서 센 값이다. [토큰 생성과 연산 횟수 로그](decode-trace-output.txt)

별도 NumPy 구현에서도 `Q @ K.T`를 구해 마스크를 적용하고, 길이 1–4인 입력 340개의 logits를 대조했다. Python 3.13.18과 NumPy 2.5.5에서 실행했다. [검산 코드](verify_trace.py)

```console
$ .venv/bin/python verify_trace.py
prefixes=340 max_abs_logit_diff=2.220e-16
```

모든 위치의 logits가 `rtol=1e-12, atol=1e-12`를 통과했다. [NumPy 검산 로그](verify-trace-output.txt) · [PyTorch의 수치 정확도](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

#### MoE에서 토큰이 사용할 전문가 고르기

밀집 모델에서는 각 토큰이 블록의 같은 MLP를 통과한다. MoE(Mixture of Experts) 블록에는 가중치가 다른 여러 MLP와 라우터가 있다. 교재 방식에서는 라우터가 전문가마다 점수를 계산하고, 상위 k개의 전문가를 골라 가중치를 다시 정규화한 뒤 각 전문가의 출력 벡터를 가중합한다. 교재의 큰 모델 설정은 전문가 8개 중 2개를 고른다. 선택된 전문가에게만 토큰을 보내면 전문가 계산을 줄일 수 있다. 이 소스의 PyTorch 경로는 전문가 8개를 모두 실행하고 선택한 두 결과를 섞는다. [저자 코드의 전문가 계산과 가중합](https://github.com/Infatoshi/book.cu/blob/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer/inference.py#L230-L257) · [Sparsely-Gated Mixture-of-Experts 원 논문](https://arxiv.org/abs/1701.06538)

손계산 예로 전문가 4개의 확률이 0번부터 `[0.1, 0.3, 0.2, 0.4]`라면 top-2 인덱스는 3, 1이다. 두 확률의 합 0.7로 나누면 최종 혼합은 `(4/7) × expert_3(x) + (3/7) × expert_1(x)`다. 여기서 고른 인덱스는 블록 안의 전문가 ID다. 디코딩에서 다음 문자를 고를 때는 모델 끝의 어휘별 logits를 사용한다.

### 5.2.2 추론 실행과 성능 검증

저자의 `run_pytorch_baseline()`을 Mac CPU에서 실행했다. 원본대로 폭 768, 헤드 8개, 블록 24개인 모델을 쓰고 시드 42로 가중치를 무작위 초기화했다. 원본 프롬프트를 넣고 새 토큰을 세 개만 생성했다. [저자의 원본 함수](https://github.com/Infatoshi/book.cu/blob/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer/inference.py#L282-L328)

```console
$ .venv/bin/python run_book.py generate --device cpu
source=3_transformer/inference.py@08b406c47fb4
python=3.13.18 torch=2.15.0 arch=arm64
device=cpu cuda_available=False mps_available=True
seed=42 width=768 heads=8 layers=24
prompt='Once upon a time' max_new_tokens=3 weights=random_init
=== PyTorch Baseline ===
Prompt: 'Once upon a time'
Prompt tokens: [[47, 78, 67, 69, 0, 85, 80, 79, 78, 0, 65, 0, 84, 73, 77, 69]]

Warming up...
Warm-up complete.

DTZGenerated text: Once upon a timeDTZ

generated_ids=[36, 52, 58]
```

| 원본에 넣은 프롬프트 | 입력 길이 | 생성 ID | 생성 문자 |
|---|---:|---|---|
| `Once upon a time` | 16자 | `[36, 52, 58]` | `DTZ` |

원본의 어휘표는 공백부터 시작하는 ASCII 문자 95개다. 출력 ID 36에 32를 더하면 ASCII 68인 `D`가 된다. [CPU 생성 로그](generate-cpu.txt)

이 버전의 `run_pytorch_baseline()`은 프리필 logits를 버리고 프롬프트의 마지막 문자를 디코드의 첫 입력으로 다시 넣는다. `Block.forward()`의 프리필 경로는 K/V를 `None`으로 돌려준다. 위의 생성 결과는 이 루프의 출력이고, 앞에서 수행한 캐시 검산은 `Head.forward()` 하나의 결과다. [블록의 반환값과 실행 루프](https://github.com/Infatoshi/book.cu/blob/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer/inference.py#L268-L319)

#### 첫 토큰 지연과 디코드 처리량

1주차 벡터 덧셈 예제의 마지막 줄은 CPU 기준 배열과 맞았는지 보여 준다. 생성에서는 먼저 토큰이 갈라진 위치를 찾는다. 양쪽에 같은 접두사를 넣고 그 위치의 logits와 MoE 전문가 ID를 비교하면, 첫 오차와 이후 다른 입력에서 생긴 차이를 나눠 볼 수 있다. [1주차 벡터 덧셈 실행 결과](https://gasidaseo.notion.site/Your-first-CUDA-program-42150aec5edf80aa921cc48ab0ad9763) · [PyTorch assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)

TTFT는 요청을 보낸 뒤 첫 토큰이 돌아올 때까지의 시간이다. ITL은 토큰과 다음 토큰 사이의 간격이다. 출력 3개를 받았다면 첫 토큰과 마지막 토큰 사이에는 간격이 2개 있다. 이 구간의 평균 디코드 토큰 속도는 `2 / (마지막 도착 시각 − 첫 도착 시각)`이다. 동시 요청 전체의 처리량을 보고 싶다면 관측 구간에 보낸 총 출력 토큰 수를 그 구간의 시간으로 나눈다. 입력 길이, 출력 길이, 배치, 정밀도, 하드웨어와 분모를 고정해야 서로 비교할 수 있다. [NVIDIA의 LLM 추론 성능 지표](https://docs.nvidia.com/nim/benchmarking/llm/metrics)

CUDA 커널은 일반적으로 비동기 제출된다. Python에서 호출 직후 시계를 읽으면 아직 GPU에서 일을 끝내지 않았을 수 있다. 워밍업한 뒤 같은 스트림의 CUDA 이벤트로 GPU 구간을 재고, 사용자 지연을 알고 싶을 때는 토큰이 실제로 돌아올 때까지의 시간을 따로 잰다. 교재 그림 5.12에 실린 값은 PyTorch가 첫 토큰까지 12 ms, 디코드 95 tok/s이고, 기본 CUDA 구현이 16 ms, 85 tok/s다.

## 5.3 추론 커널

### 5.3.1 메모리 배치와 연속성

CUDA의 스레드는 블록으로 묶이고, 각 블록은 GPU의 SM에 배정된다. 메모리 접근을 볼 때는 한 워프의 스레드 32개가 같은 명령에서 읽는 주소를 함께 본다. [별준의 GPU와 CUDA](https://junstar92.tistory.com/243) · [행렬곱의 메모리 접근](https://junstar92.tistory.com/248)

행 우선으로 저장한 2×3 행렬의 물리적 원소 순서를 `[a,b,c,d,e,f]`라고 하자. 이 행렬의 원소 단위 stride는 `(3,1)`이다. 전치한 뷰는 모양이 3×2이고 stride가 `(1,3)`이 된다. 이 뷰의 `[0,1]`은 원본에서 오프셋 3인 d다. 같은 주소에 대해 연속 행렬을 가정해 `row * 2 + col`을 적용하면 오프셋 1인 b를 읽는다. 커널은 stride를 인자로 받아 색인하거나, 연속 입력만 받도록 계약을 정하고 래퍼에서 필요할 때 복사할 수 있다. 이미 연속인 텐서에서 `.contiguous()`는 자기 자신을 돌려준다. [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)

1주차 ReLU에서 스레드 0, 1, 2는 각각 `x[0]`, `x[1]`, `x[2]`를 읽어 출력 한 칸씩을 채웠다. FP32라면 시작 주소에서 0, 4, 8바이트 떨어진 곳이다. GEMV에서 스레드 하나가 행 하나를 맡고 행이 128열이라면, 같은 세 스레드가 첫 곱셈에서 읽는 가중치 주소가 달라진다. [1주차 발표](https://gasidaseo.notion.site/When-PyTorch-just-isn-t-enough-42150aec5edf80eebff3de9b0a2b87d3)

| 첫 곱셈 | 스레드 0 | 스레드 1 | 스레드 2 |
|---|---:|---:|---:|
| 서로 다른 행의 0번 열 | 0B | 512B | 1,024B |
| 같은 행의 이웃 열 | 0B | 4B | 8B |

위 오프셋은 `W`의 시작 주소부터 센 값이다. 행을 맡은 스레드 하나는 다음 반복에서 바로 옆 원소로 가지만, 워프는 같은 순간에 512바이트씩 떨어진 주소를 읽는다. 같은 행을 여러 스레드가 나눠 읽으면 주소가 모이는 대신, 각 스레드가 낸 부분합을 더해야 출력 한 칸이 완성된다. [NVIDIA의 워프 메모리 접근 설명](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

### 5.3.2 디코드의 행렬·벡터 곱(GEMV)

출력 채널 M개, 입력 특성 N개인 선형층에 대해 행 우선 가중치를 `W[M,N]`으로 둔다. 한 입력 벡터 `x[N]`의 출력 `y[M]`에서는 가중치의 행 하나와 x를 내적한 값이 출력 한 칸이다. 여러 위치를 한꺼번에 계산하면 `X[T,N] @ W.T[N,M]`가 되어 같은 W를 T개 위치가 이용한다.

```cpp
// FP32 연속 행렬, 입력 벡터 하나, M개 출력의 기본 GEMV
__global__ void project_one(const float* w, const float* x, float* y,
                            int m, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        acc += w[row * n + j] * x[j];
    }
    y[row] = acc;
}
```

M=N=128인 FP32 가중치는 64 KiB다. 곱과 덧셈을 2 FLOP으로 세면 산술량은 약 32,768 FLOP이다. 가중치를 한 번 읽는 바이트만 분모에 두면 약 0.5 FLOP/byte다. 입력 읽기와 출력 쓰기까지 더하면 이 근삿값은 작아진다. 여러 위치가 가중치를 재사용하면 같은 수의 가중치 바이트로 더 많은 산술 연산을 할 수 있다. 디코드에서도 여러 요청의 새 토큰을 모아 GEMM으로 처리하면 가중치를 재사용할 수 있다. [NVIDIA의 행렬곱과 연산 집약도](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

#### 2행 3열 GEMV를 실행해 보기

가중치의 두 행을 `[2, −1, 3]`, `[0, 4, 1]`로 두고 입력을 `[1, 2, 1]`로 잡았다. 첫 행의 내적은 `2−2+3=3`, 둘째 행은 `0+8+1=9`다. [gemv.cu](gemv.cu)에는 위 커널과 같은 행 단위 계산을 CPU에서 실행하는 경로와 CUDA에서 실행하는 경로가 들어 있다. Mac의 CPU 경로에서는 다음 결과가 나왔다.

```console
$ clang++ -std=c++17 -O2 -x c++ gemv.cu -o gemv_cpu
$ ./gemv_cpu
CPU: [3, 9]
```

같은 소스의 CUDA 경로는 W와 x를 한 번 복사하고 블록 하나에 스레드 128개를 띄운다. 이 입력에서는 행을 맡은 스레드 0과 1만 결과를 쓰고 나머지는 범위 검사에서 빠진다. 위 출력은 Mac에서 컴파일한 C++ 경로에서 얻었다. [GEMV 실행 로그](gemv-cpu-output.txt)

#### MoE 라우팅의 top-k 커널

저자의 top-k 커널은 토큰마다 블록 하나를 띄우고, 그 안의 스레드 하나가 전문가 점수와 인덱스를 순서대로 읽는다. 상위 k개를 정렬해 유지하므로 전문가 E개에 대해 최악 O(Ek)번 비교하고 옮긴다. 2위와 3위처럼 선택 경계에 놓인 점수가 가까우면 반올림으로 선택이 달라질 수 있다. [PyTorch topk의 동점 처리](https://docs.pytorch.org/docs/stable/generated/torch.topk.html) · [저자 top-k 커널](https://github.com/Infatoshi/book.cu/blob/08b406c47fb46f7b2d76bd77b2eaa08958a68f69/3_transformer/csrc/inference/kernels/topk_fwd.cu)

## 정리

프리필은 프롬프트를 한 번에 처리하고, 디코드는 새 토큰 한 위치의 Q·K·V를 계산한다. 단일 헤드로 확인한 K/V 모양은 각각 `[1, 3, 96]`에서 `[1, 4, 96]`으로 늘었다. 앞의 K/V 투영을 다시 할 필요는 줄어도, 다음 토큰을 계산할 때 읽을 문맥은 계속 길어진다.

배치 1에서 새 위치 하나를 투영하면 행렬과 벡터의 곱이 된다. 행마다 스레드 하나를 배정한 GEMV에서는 행렬이 연속이어도 같은 워프의 스레드가 떨어진 주소를 읽는다. 6장에서는 한 행의 이웃 열을 여러 스레드에 나누고, 그 부분합을 합치는 비용까지 비교한다.
