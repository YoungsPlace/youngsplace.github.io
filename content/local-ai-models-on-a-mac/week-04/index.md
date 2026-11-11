---
title: "[4주차] 로컬 LLM 비교, 샘플링 설정, 오프라인 챗봇"
date: 2026-11-11T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
summary: "Mac에서 소형 모델 세 개를 비교하고, 시스템 프롬프트와 생성 옵션을 바꿨다. Streamlit에서 모델을 전환하고 외부 연결을 차단한 상태에서도 질문을 보냈다."
tags: ["로컬 AI", "Mac", "Ollama", "MLX", "Streamlit", "LLM"]
series: ["Local AI Models on a Mac"]
showTableOfContents: true
showAuthor: false
---

3주차에 만든 음성 챗봇에 모델 선택기와 생성 설정을 붙였다. 이번 범위는 CH12~14다. 같은 질문을 세 모델에 보내 보고, Gemma 3의 temperature와 컨텍스트 길이를 바꿨다. 실습 환경은 Apple M5 Pro, 통합 메모리 48 GiB, Ollama 0.40.11, Python 3.13이다. [3주차 음성 챗봇](../week-03/) · [4주차 강의 노트](https://gasidaseo.notion.site/CH12-14-LLM-LLM-41450aec5edf805a91d4da78b7e370eb)

## 1. CH12 · 모델을 고르는 기준

### 계열, 모델 크기, 양자화

| 계열 | 버전을 고를 때 살펴본 것 |
|---|---|
| Llama | 3.2의 1B·3B는 경량 텍스트 모델이다. 같은 3.2 계열에도 이미지 입력을 받는 별도 모델이 있다. [Meta 모델 카드](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| Gemma | 이번 비교는 Gemma 2 2B를 쓰고, 음성 챗봇과 프롬프트 실습에는 이전에 쓰던 Gemma 3 4B를 사용했다. Gemma 3 4B는 이미지도 입력으로 받을 수 있다. [Gemma 3 모델 카드](https://ai.google.dev/gemma/docs/core/model_card_3) |
| Qwen | Qwen2.5 3B Instruct는 일본어를 포함한 다국어 대화용 후보로, Qwen3-Coder-Next는 코딩용 MoE 사례로 봤다. [Qwen2.5 모델 카드](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) · [Qwen3-Coder-Next 보고서](https://arxiv.org/abs/2603.00729) |
| Mistral | Small 3.2 24B는 다국어·이미지 입력과 함수 호출을 지원하는 예다. [Mistral 모델 카드](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| DeepSeek | R1 계열을 볼 때는 원본과 Qwen·Llama 기반 Distill 모델의 이름, 크기, 기반 모델을 같이 확인한다. [DeepSeek-R1 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-R1) |

모델 이름의 `B`는 파라미터 10억 개를 뜻한다. 8B 모델의 가중치를 모두 16비트로 표현하면 약 16 GB, 4비트로 표현하면 약 4 GB다. 실제 파일에는 스케일과 메타데이터가 들어가고, 텐서에 따라 정밀도가 섞인다. 실행할 때는 가중치에 더해 KV 캐시와 작업 버퍼도 필요하다. `ollama list`의 파일 크기와 `ollama ps`의 실행 중 크기를 따로 적었다. [Ollama 실행 모델 API](https://docs.ollama.com/api/ps)

### 연습문제 · 같은 질문을 세 모델에 보내기

```bash
ollama pull gemma2:2b
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama list

python lab.py compare compare.json
```

모델마다 한 번 예열한 뒤 양자 컴퓨팅 설명, 바다를 소재로 한 시, 공부 팁 세 가지, 일본 문화에 대한 일본어 질문을 각각 세 번 보냈다. 대화 기록 없이 `num_ctx=4096`, `num_predict=200`, `temperature=0.3`, `seed=42`로 실행했다. 표의 속도는 응답 12개의 중앙값이다. `eval_count ÷ (eval_duration / 1e9)`로 계산해 모델 로딩과 입력 처리 시간은 따로 뒀다. [비교 스크립트](lab.py) · [Ollama Chat API](https://docs.ollama.com/api/chat)

| 모델 | 파일 크기 / 양자화 | 생성 속도 중앙값 | 전체 요청 중앙값 |
|---|---|---:|---:|
| `gemma2:2b` | 1.6 GB / Q4_0 | 69.1 토큰/초 | 2.97초 |
| `qwen2.5:3b` | 1.9 GB / Q4_K_M | 65.0 토큰/초 | 2.73초 |
| `llama3.2:3b` | 2.0 GB / Q4_K_M | 74.6 토큰/초 | 2.71초 |

시는 세 모델 모두 200토큰 안에 끝났다. 나머지 세 질문은 출력 상한에 닿아 420토큰으로 한 번씩 다시 보냈다.

| 질문 | 답변을 비교한 결과 |
|---|---|
| 양자 컴퓨팅 | Qwen은 중첩과 간섭을 따로 설명해서 구조가 가장 읽기 쉬웠다. 세 모델 모두 큐비트를 ‘0과 1이 동시에 된다’는 식으로 단순화했다. 큐비트는 두 기저 상태의 선형 결합으로 표현되고, 측정하면 결과 하나를 얻는다. |
| 짧은 바다 시 | Gemma와 Llama는 8행, Qwen은 16행이었다. 파도 소리, 소금기와 해초를 넣은 Llama의 8행이 ‘짧은 시’ 요청에 가장 잘 맞았다. |
| 공부 팁 세 가지 | Gemma는 인출·간격 반복을 한 항목에 넣고 학습 환경과 스터디 그룹을 덧붙였다. Llama는 간격 반복의 예시 날짜를 줬다. Qwen의 ‘25분 공부하고 5분 쉬기’가 바로 적용하기 가장 쉬웠다. |
| 일본 문화 | Qwen은 `和食`·`和装`·다도·만화와 애니메이션·새해로 나눴다. Gemma의 전통과 현대 구분도 읽기 쉬웠지만 마지막에 비슷한 내용이 반복됐다. Llama는 `kimono`, `sushi`처럼 로마자를 자주 섞었고 출력 상한까지 이어졌다. 일본어 표현과 길이를 함께 보면 Qwen이 가장 자연스러웠다. |

일상용은 `qwen2.5:3b`로 정했다. 일본어 답변이 읽기 쉬웠고 공부 팁에 구체적인 행동이 들어갔다. 1.9 GB라서 Mac에서 가볍게 쓸 수 있고, 짧은 영문 글을 쓸 때는 Llama 3.2 3B도 선택지로 남겨 뒀다.

같은 문장도 모델마다 토크나이저가 달라 서로 다른 개수의 토큰이 될 수 있다. 속도를 비교할 때는 토큰/초와 함께 답변이 나오는 데 걸린 시간, 답변 길이, 질문에 맞는 내용인지도 본다. 공개 벤치마크는 데이터셋·채점 기준·도구 사용 조건이 같을 때 비교하기 쉽다. 이번에는 설명의 정확성, 시의 구체적인 이미지, 공부 팁을 바로 실행할 수 있는지, 일본어 표현을 따로 봤다.

### 연습문제 · 2B와 8B 실행 중 메모리 보기

```bash
ollama pull llama3.1:8b
python lab.py resource resource.json
```

두 모델에 같은 등대 이야기 요청을 보내고 `num_ctx=4096`, 최대 출력 320토큰을 적용했다. 실행 중 모델 할당량은 `/api/ps`에서, CPU는 생성 중 러너 프로세스의 `ps` 값을 0.2초 간격으로 읽었다. 이 CPU 수치는 프로세스 기준이라 코어 하나를 전부 쓰면 100%다.

| 모델 | 할당량 | 320토큰 응답 시간 | 러너 CPU 최고 관측값 |
|---|---:|---:|---:|
| `gemma2:2b` | 1.79 GiB | 3.03초 | 220.7% |
| `llama3.1:8b` | 4.92 GiB | 6.56초 | 119.2% |

{{< figure src="resource-chart.png" alt="Gemma 2 2B는 1.79 GiB와 3.03초, Llama 3.1 8B는 4.92 GiB와 6.56초" caption="같은 요청에 대한 모델 할당량과 응답 시간" figureClass="study-capture" >}}

두 모델 모두 `/api/ps`에서 전체 할당량과 `size_vram`이 같은 값으로 보고됐다. 8B 요청의 응답 시간은 약 2.2배였고, 이때 CPU의 최고 관측값은 2B가 더 높았다.

### MoE에서 전체 파라미터와 활성 파라미터 읽기

Dense 모델은 각 토큰을 계산할 때 일반적인 피드포워드 층을 통과한다. MoE는 그 층에 여러 전문가를 두고 라우터가 토큰마다 일부를 선택한다. 토큰이 바뀌면 선택되는 전문가도 달라질 수 있다. 활성 파라미터는 한 토큰의 계산량을 이해하는 데 쓰고, 가중치 파일과 메모리 배치는 전체 전문가를 기준으로 살핀다.

{{< mermaid >}}
flowchart TD
    A["토큰의 은닉 벡터"] --> R["라우터가 전문가에 점수 부여"]
    R --> B["선택된 전문가 1"]
    R --> C["선택된 전문가 2"]
    B --> S["전문가 출력을 가중합"]
    C --> S
    S --> T["다음 층"]
{{< /mermaid >}}

| 강의에 나온 모델 | 전체 | 활성 |
|---|---:|---:|
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80B | 3B |
| [GLM-5](https://huggingface.co/zai-org/GLM-5) | 744B | 40B |
| [MiniMax-M2.5](https://arxiv.org/abs/2605.26494) | 약 229B | 약 10B |

### 도전과제 · 48 GiB Mac에서 Qwen3-Coder-Next 로드하기

`qwen3-coder-next:latest`의 Q4_K_M 패키지를 내려받았다. Ollama에 표시된 크기는 51,741,611,823바이트(48.19 GiB)였다. 이 Mac의 물리 메모리는 51,539,607,552바이트(48 GiB)다. 활성 파라미터가 3B인 모델도 전체 전문가가 패키지에 들어간다. [Ollama 모델 태그](https://ollama.com/library/qwen3-coder-next:latest)

`num_ctx=4096`, 최대 출력 4토큰으로 `Reply only OK.`를 보냈다. `memory_pressure -Q`에서 읽은 시스템 여유 메모리 비율은 로드 전 68%였고, 실행 중 44%, 22%, 6% 순서로 떨어졌다. 14% 아래에서 멈추도록 설정해 3.25초에 요청을 중단했다. 텍스트 출력은 받지 못했다.

같은 Mac에서 Llama 3.1 8B는 320토큰 생성까지 끝났다. 모델을 고를 때는 활성 파라미터 외에 내려받을 양자화 파일, 컨텍스트 길이, 실행 중 메모리 여유를 함께 본다.

## 2. CH13 · 시스템 프롬프트와 생성 옵션

### 연습문제 · 하늘이 파란 이유를 다섯 역할로 설명하기

`system`에는 답변의 역할과 형식, `user`에는 지금 할 질문, `assistant`에는 앞선 답변을 넣는다. 같은 질문을 유지하고 시스템 프롬프트만 바꾸면 말투와 설명 방법의 차이를 볼 수 있다.

Gemma 3 4B에 `Why is the sky blue?`를 보내고 역할마다 새 대화를 시작했다. Temperature는 모두 0.3으로 맞췄다. [페르소나 시스템 프롬프트](lab.py)

| 역할 | 나온 답변 |
|---|---|
| 역사 교수 | 안경을 고쳐 쓰고 청중을 부르는 무대 지시로 시작했다. 과학자들이 오랫동안 붙잡았던 질문이라는 식으로 풀었다. |
| 피트니스 코치 | 레일리 산란을 먼저 말하고 햇빛의 색을 항목으로 나눴다. 운동 비유를 요청했지만 답변 앞부분은 일반적인 과학 설명에 가까웠다. |
| 동화 구연가 | 햇빛을 `Sunny`라는 등장인물로 만들어 여러 색의 빛을 보내는 이야기로 설명했다. |
| 과학 커뮤니케이터 | 흔한 설명이 부정확하다는 건조한 말투로 시작한 뒤 레일리 산란을 설명했다. |
| 선승 | 잠시 자세를 가다듬고 조용히 질문을 받는 장면으로 시작했다. |

형식 제어는 세 문장으로 따로 시험했다. 첫 문장은 정의, 둘째는 ‘마치’로 시작하는 비유, 셋째는 물음표로 끝나는 질문으로 지정했다. 캐시 답변은 세 문장에 비유까지 들어갔지만 마지막이 평서문이었다. 양자화 답변은 세 조건을 모두 따랐고, TCP 재전송 답변은 질문으로 끝났지만 여섯 문장으로 길어졌다. 문장 수와 문장 끝 기호처럼 출력에서 바로 확인할 수 있는 조건은 생성 뒤에도 검사하기 쉽다.

### Temperature, top-k, top-p, 반복 패널티

모델은 다음 토큰 후보별로 로짓을 출력한다. Temperature가 양수이면 로짓을 `T`로 나눈 뒤 확률을 계산한다. 낮추면 큰 로짓에 확률이 더 모이고, 올리면 낮은 순위 후보도 뽑힐 가능성이 커진다. `T=0`은 Ollama에서 가장 점수가 높은 토큰을 고르는 방식으로 쓴다. 정답 여부는 질문, 모델, 입력 정보에도 달려 있다. [Ollama 생성 파라미터](https://docs.ollama.com/modelfile#valid-parameters-and-values)

| 설정 | 다음 토큰을 고를 때 바꾸는 것 |
|---|---|
| `top_k` | 점수가 높은 후보를 최대 k개 남긴다. |
| `top_p` | 확률 순으로 더해 p에 처음 도달하는 후보 집합을 남긴다. |
| `repeat_penalty` | 최근 토큰의 반복을 덜 선택하도록 점수를 조정한다. |
| `seed` | 같은 실행 조건에서 난수 흐름을 비교하는 데 쓴다. |

강의의 후보 선택 예제를 숫자로 바꿔 봤다. 확률이 `0.55, 0.25, 0.13, 0.07`이면 `top_k=2`는 두 개, `top_p=0.9`는 누적 확률 0.93까지 세 개를 남긴다. 후보 제한은 각각 적용했을 때의 예다. [Nucleus sampling 논문](https://arxiv.org/abs/1904.09751)

### 연습문제 · Temperature 0.0, 0.5, 1.0, 1.5

“고장 난 달 기지에서 정비사가 처음 발견한 것을 한국어로 두 문장만 써 줘”를 Gemma 3 4B에 보냈다. `top_k=40`, `top_p=0.9`를 유지하고 각 온도에서 seed 42·43·44로 세 번 실행했다.

| Temperature | 처음 발견한 것과 표현 |
|---:|---|
| 0.0 | 세 번 모두 환풍구의 녹슨 금속 상자에서 같은 물건을 발견했다. 답변 전체가 같았다. |
| 0.5 | 작은 인형이 든 상자와 빛바랜 데이터 패드가 나왔다. 패드에서 목소리가 나거나 ‘지구’라는 기록을 찾았다. |
| 1.0 | 유아용 인형, 지구의 풍경이 담긴 데이터 패드, 빛나는 결정 조각이 나왔다. |
| 1.5 | 금속 케이스, 마지막 메시지가 담긴 데이터 패드, 작은 인형이 나왔다. ‘메시지는 마치 외로운 심장처럼’처럼 비유가 늘어난 응답도 있었다. |

이번 12개 출력은 모두 두 문장이었다. 1.5에서도 이야기의 앞뒤가 끊기는 지점은 보이지 않았다. 인물 시점과 소재는 0.5부터 바뀌었다. [페르소나·온도 실행 코드](lab.py)

짧은 챗봇에서 답이 비면 `done_reason`과 `message.thinking`을 같이 확인할 수 있다. Qwen 3.5 4B에 닭과 토끼의 수를 묻고 `think=true`, `num_predict=256`을 주자 `done_reason=length`였고, `thinking`만 채워졌다. `think=false`에서는 같은 상한에서 본문이 나왔지만 계산 도중에 끝났다. 이 예에서는 답변 길이와 추론 설정을 함께 조절할 필요가 있었다. [Ollama Thinking](https://docs.ollama.com/capabilities/thinking)

### Motif 2.6B에서 한국어 질문 비교

Motif 2.6B는 `mlx-motif 0.1.1`에서 4비트 MLX 변환본을 불러왔다. 서버를 루프백에 띄우고 같은 질문을 운영 엔지니어와 스터디 친구에게 각각 보냈다. “데이터베이스 이중화를 한국어로 두 문장으로 설명해 줘”라는 질문에 temperature 0.0과 0.7을 적용했다. 출력 상한은 160토큰이다. 이 버전의 HTTP 서버는 요청의 temperature를 샘플러에 넘긴다. [Motif 기술 보고서](https://arxiv.org/abs/2508.09148) · [MLX 변환본](https://huggingface.co/junhoyeo/Motif-2.6B-MLX-q4) · [서버의 샘플러 설정](https://github.com/junhoyeo/mlx-motif/blob/255efd45f4bd25e0bf0886b3dffedfdc4129dbb1/src/mlx_motif/server.py#L214-L255)

```bash
mlx-motif serve --model junhoyeo/Motif-2.6B-MLX-q4 --port 18080
```

| 역할 | Temperature 0.0 | Temperature 0.7 |
|---|---|---|
| 운영 엔지니어 | 복사본의 동기화와 장애 시 서비스 유지를 설명하고 비슷한 문단을 반복했다. | 무결성과 가용성을 언급했다. 장애 조치와 재해 복구를 혼용했고, 두 출력 모두 160토큰에서 끝났다. |
| 스터디 친구 | 두 서버에 같은 데이터 사본을 유지한다는 설명을 했다. | 두 개의 우산을 비유로 들었다. 두 온도 모두 두 문장보다 길었다. |

프롬프트를 바꾸자 기술 용어와 비유 사용이 달라졌다. 출력에서 장애 조치(failover)는 대기 시스템으로 역할을 넘기는 동작이고, 재해 복구(disaster recovery)는 장애 뒤 복구하는 계획과 절차 전반을 가리키므로 두 용어를 나눠 읽었다.

### 연습문제 · Modelfile로 용도별 설정 저장하기

`FROM`으로 기반 모델을 고르고, `SYSTEM`과 `PARAMETER`에 기본값을 저장한다. 가중치 블롭은 그대로 공유하고, Ollama가 각 이름으로 시스템 메시지와 기본 옵션을 읽는다. 앱에서 요청할 때 값을 전달하면 해당 요청에 맞게 조절할 수 있다. [Ollama Modelfile](https://docs.ollama.com/modelfile)

```text
FROM gemma3:4b
PARAMETER temperature 0.5
PARAMETER num_ctx 4096
SYSTEM """Explain technical topics in Korean with an example and one quiz question."""
```

```bash
ollama create week4-study -f Modelfile
ollama show --modelfile week4-study
ollama list
```

| 이름 | Temperature | 질문과 결과 |
|---|---:|---|
| `week4-writer` | 1.1 | 달 기지 장면에서 닳은 금속 침대와 끊어진 전선을 묘사했다. 두 문장 요청에 앞뒤 안내도 붙었다. |
| `week4-facts` | 0.1 | 물의 화학식을 `H₂O`로 답하고 원자 구성을 덧붙였다. |
| `week4-study` | 0.5 | FFT를 주파수 성분으로 나누는 계산으로 설명했다. 320토큰 상한에서 피아노 예시가 잘려 확인 질문까지 이어지지 않았다. |

세 이름은 `ollama list`에 각각 3.4 GB로 표시됐다. `ollama show --modelfile`에서 같은 기반 가중치 경로를 확인했다.

### 연습문제 · 512와 4096 컨텍스트 비교

컨텍스트는 한 번의 생성에서 모델이 읽을 수 있는 토큰 범위다. KV 캐시는 이미 읽은 토큰의 키·값을 저장해 다음 토큰을 계산할 때 재사용한다. 같은 모델이라도 `num_ctx`를 늘리면 KV 캐시에 필요한 메모리가 커진다. 길게 남긴 대화도 실제 요청 컨텍스트에 들어가야 앞부분을 다시 참고할 수 있다. Llama 3.1 8B에 같은 짧은 요청을 보내고 매번 모델을 내려 다시 올렸더니, 4,096에서 4.92 GiB, 8,192에서 5.54 GiB, 32,768에서 8.51 GiB가 잡혔다. [Ollama 컨텍스트 길이](https://docs.ollama.com/context-length)

`week4-context-512`와 `week4-context-4096`를 같은 Gemma 3 4B에서 만들었다. 첫 턴에 프로젝트 코드 `ORANGE-731`을 알려 주고, 이후 레이다 실험 기록을 한 번에 한 항목씩 보냈다. 중간에 코드를 물은 질문과 답은 기록에 다시 넣지 않았다. 표의 메시지 수에는 마지막 확인 질문을 포함했고, 시스템 프롬프트는 제외했다.

| 보낸 메시지 | 512 설정 · 입력 토큰 / 답변 | 4096 설정 · 입력 토큰 / 답변 |
|---:|---|---|
| 3 | 74 / `ORANGE-731` | 74 / `ORANGE-731` |
| 9 | 255 / `ORANGE-731` | 255 / `ORANGE-731` |
| 15 | 435 / `ORANGE-731` | 435 / `ORANGE-731` |
| 21 | 472 / `UNKNOWN` | 615 / `ORANGE-731` |
| 27 | 475 / `UNKNOWN` | 798 / `ORANGE-731` |

512 설정에서는 15개에서 기억했고, 다음으로 확인한 21개에서 `UNKNOWN`이 나왔다. 4096 설정에서는 마지막 27개까지 코드가 나왔다. 대화가 쌓였을 때 모델로 보내는 메시지 수와 Ollama가 보고한 `prompt_eval_count`를 함께 보면, 요청에 넣은 기록과 실제로 처리된 범위를 비교할 수 있다.

### 연습문제 · Streamlit에서 모델과 설정 바꾸기

음성 입력은 설치된 Whisper 체크포인트와 FFmpeg를 사용한다. 전사문 편집과 최근 대화 전송을 넣었다. 사이드바에는 설치된 모델 목록, temperature 슬라이더, ‘레이다 스터디 파트너’와 ‘짧은 실험 노트’ 프리셋을 추가했다. 대화 중 모델을 바꿔도 최근 메시지를 다음 요청에 보내고, 화면에는 답변마다 사용한 모델을 표시했다. [Streamlit 실행 코드](model_lab.py)

{{< mermaid >}}
flowchart LR
    A["텍스트 또는 Whisper 전사문"] --> B["새 user 메시지"]
    P["프리셋과 최근 대화"] --> C["messages"]
    B --> C
    C --> D["Ollama chat"]
    S["선택 모델과 temperature"] --> D
    D --> E["스트리밍 출력과 모델명 기록"]
{{< /mermaid >}}

```bash
python -m pip install streamlit==1.66.1 ollama==0.6.3 mlx-whisper==0.4.3
HF_HUB_OFFLINE=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  python -m streamlit run model_lab.py --server.address 127.0.0.1
```

| 동작 | 화면에서 확인한 결과 |
|---|---|
| 대화 중 모델 전환 | Qwen에서 테스트 이름 `BLUE-42`를 알려 준 뒤 Llama로 바꿨다. 이전 대화가 남아 Llama도 `BLUE-42`라고 답했다. |
| 프리셋 변경 | ‘짧은 실험 노트’에서 FFT 샘플 수를 두 배로 늘리는 질문에 “빈 간격은 절반으로 줄어들어요”라고 답했다. 프리셋에 적은 첫 줄 ‘관찰’은 붙지 않았다. |
| Temperature 슬라이더 | 대화를 비우고 같은 달 기지 질문을 보냈다. 0.0에서는 빛바랜 사진이, 1.5에서는 홀로그램 화면이 나왔다. 1.5 답변에는 어색한 한국어 표현도 있었다. |
| 음성 입력 | 3주차 합성 WAV를 올리자 감지 언어가 `ko`였고, 전사문을 보내 Gemma 답변을 받았다. |
| 대화 초기화 | 질문과 답변이 화면에서 지워지고 세션의 대화 목록이 비었다. |

{{< figure src="model-switch.png" alt="Qwen에서 정한 프로젝트 코드 BLUE-42를 모델을 바꾼 뒤 Llama가 답한 화면" caption="Qwen에서 Llama로 바꿔 같은 대화 이어 가기" figureClass="study-capture" >}}

## 3. CH14 · 오프라인 동작과 데이터 흐름

### 연습문제 · 외부 연결을 끊고 다섯 질문 보내기

Ollama와 Streamlit을 실행하는 프로세스에는 루프백 연결만 허용했다. 같은 제한에서 외부 도메인과 외부 IP의 HTTPS 연결은 실패하고, 로컬 Ollama와 Streamlit은 응답했다. Ollama는 `OLLAMA_NO_CLOUD=1`, Whisper는 모델을 미리 받은 뒤 `HF_HUB_OFFLINE=1`로 실행했다. [Ollama 로컬 전용 설정](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)

| Qwen 2.5 3B에 보낸 질문 | 연결 차단 전·후에 받은 답변 |
|---|---|
| 2의 10제곱 | `1024` |
| “Thank you for your time.” 번역 | ‘시간을 내주셔서’ 뒤에 감사 표현이 빠졌다. |
| Python 정수 리스트의 최댓값 | `max_value = max(my_list)` |
| 도쿄가 어느 나라의 수도인지 | `일본` |
| 비 오는 날을 소재로 두 행의 시 | “비가 내리면 / 길은 더욱 아름다워” |

다섯 답변은 temperature 0, seed 42에서 연결 차단 전후에 모두 일치했다. 차단 상태에서 Llama 3.2 3B로 바꿔 첫 질문을 다시 보내도 `1024`가 나왔다. 브라우저에서 합성 WAV를 올리고 전사문을 보내는 과정도 같은 제한의 Streamlit과 Ollama 서버에서 실행했다. [비교 실행 코드](lab.py)

{{< mermaid >}}
flowchart TD
    B["브라우저"] --> S["Streamlit · 127.0.0.1"]
    S --> W["로컬 Whisper · 음성일 때"]
    W --> S
    S --> O["로컬 Ollama · 127.0.0.1"]
    O --> S
    S --> B
    C["클라우드 AI를 쓸 때"] --> H["HTTPS 요청"]
    H --> I["서비스 제공자 서버"]
{{< /mermaid >}}

오프라인 환경을 준비할 때 모델 목록은 `ollama list`, 실행 중인 모델과 컨텍스트는 `ollama ps`로 확인했다. 모델을 바로 내릴 때는 `ollama stop 모델명`을 쓰고, 한 번의 API 요청 뒤에 내리려면 `keep_alive: 0`을 보낸다. [Ollama 모델 유지 시간](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)

### 연습문제 · 데이터 흐름을 열 가지 질문으로 분류하기

실습용 질문 열 가지를 만들었다. 학생·회사 정보가 들어가는 항목은 가상 예시로만 분류했다. 로컬 처리는 이번 모델과 도구 구성에서 작업을 시작할 수 있는지로 판단했다.

| 질문 | 개인정보·비공개 정보 | 처리 경로 |
|---|---|---|
| FFT를 레이다 거리 측정으로 설명해 줘 | 없음 | 로컬 |
| `ollama ps`의 열을 설명해 줘 | 없음 | 로컬 |
| 영문 감사 인사를 한국어로 바꿔 줘 | 없음 | 로컬 |
| 합성 음성 파일을 전사해 줘 | 없음 | 로컬 Whisper |
| 예제 Python 함수의 입력 조건을 정리해 줘 | 없음 | 로컬 |
| 학생 이름이 든 과제 메모를 항목별로 묶어 줘 | 이름·평가 | 승인된 로컬 환경 |
| 사내 레이다 실험 로그에서 측정 조건을 찾아 줘 | 내부 실험 정보 | 승인된 로컬 환경 |
| 내부 회의 초안을 세 줄로 요약해 줘 | 참석자·내부 안건 | 승인된 로컬 환경 |
| 오늘 공개된 모델 가격과 지원 지역을 찾아 줘 | 없음 | 외부 검색 후 공식 문서 확인 |
| 다음 주 실제 항공편 운항 시간을 알려 줘 | 입력에 따라 여행 정보 | 항공사 웹사이트 조회 |

열 가지 중 여덟 가지는 이미 가진 입력과 로컬 도구로 처리할 수 있다. 마지막 두 가지는 날짜에 따라 바뀌는 정보라 외부 조회가 필요하다. 웹에서 자료만 가져와 로컬 모델로 정리하는 방식도 가능하다.

### 연습문제 · 소비자용 AI 서비스의 보관·접근 조건

ChatGPT 개인용은 일반 대화를 삭제할 때까지 계정에 보관하고, 삭제 시 통상 30일 이내에 시스템에서 지운다고 안내한다. 비식별화되었거나 보안·법적 의무가 있는 경우에는 예외가 있다. 개인용 콘텐츠를 모델 개선에 쓰는 설정은 끌 수 있고, 임시 채팅은 모델 개선에 사용하지 않으며 안전 목적으로 최대 30일 보관할 수 있다. [대화 보관](https://help.openai.com/en/articles/8983778-chat-and-file-retention-in-chatgpt) · [데이터 제어](https://help.openai.com/en/articles/7039943-how-openai-handles-data-in-consumer-services)

보안 사고 조사, 지원, 법적 업무 등 필요한 경우 권한을 받은 담당자와 서비스 제공자가 콘텐츠에 접근할 수 있다. 콘텐츠는 미국 및 다른 국가의 시스템에 보관될 수 있다. 한국 부속 문서에는 수탁자별 이전 국가와 목적이 나와 있다. 인수·구조조정·파산 등에서는 거래 실사 과정에서 공개되거나 승계자에게 이전될 수 있다는 조항이 있다. [개인용 데이터 이용 FAQ](https://help.openai.com/en/articles/7039943-openai-privacy-policy) · [개인정보 처리방침](https://openai.com/policies/row-privacy-policy/) · [한국 부속 문서](https://openai.com/ko-KR/policies/kr-privacy-policy-addendum/)

### 연습문제 · 교육 자료를 어디에서 처리할까

학생 피드백 초안에는 이름, 학번, 성적이 섞일 수 있다. 국내 개인정보 보호법 제29조의 안전조치 의무와, 국외 서비스로 보내는 경우 제28조의8의 국외 이전 요건을 확인할 항목으로 적었다. 로컬에서도 학급별 접근 권한과 저장 위치, 보관 기간을 관리해야 한다. [제29조 안전조치의무](https://www.law.go.kr/법령/개인정보보호법/제29조) · [제28조의8 국외 이전](https://www.law.go.kr/법령/개인정보보호법/제28조의8)

과제 피드백의 초안을 만들 때는 승인된 기기에서 식별자를 줄인 텍스트를 쓰고, 교사가 원본과 답변을 대조하는 흐름을 생각했다. 공개 교육 자료의 최신 출처를 찾는 작업은 외부 검색으로 돌릴 수 있다. 검색어에는 학생 답안이나 이름을 넣을 이유가 없다.

### 연습문제 · 비용과 하이브리드 구성

구독 예시는 ChatGPT Plus 한 개, 월 20달러로 잡았다. 연 240달러다. 비교용 하드웨어는 통합 메모리 16 GB인 M6 Mac mini 기본형으로, 국내 시작가 1,499,000원이다. 8B Q4의 실행 할당량과 OS 여유를 함께 보는 구성이다. 환율을 계산용으로 1달러당 1,400원으로 고정하면 연 336,000원이고, 구독을 전부 대체할 때 기기값만의 손익분기점은 약 4.46년이다. 이미 가진 Mac에서 실행할 때는 기기 추가 구매비가 0원이다. 계산에는 전기, 세금, 유지 시간과 실제 사용량의 차이를 넣지 않았다. [ChatGPT Plus 가격](https://help.openai.com/en/articles/6950777-what-is-chatgpt-plus) · [Apple 국내 가격과 사양](https://www.apple.com/kr/newsroom/2026/09/the-new-mac-mini-and-mac-studio-are-available-today/)

| 작업 | 사용할 곳 | 기준 |
|---|---|---|
| 레이다 실습 메모 요약 | 로컬 | 비공개 로그를 외부로 보내지 않고 초안을 만든다. |
| 번역·개념 설명·짧은 코드 | 로컬부터 | 반복하기 쉽고 원문과 답을 직접 비교할 수 있다. |
| 공개 기술의 최신 정보 | 검색 + 로컬 | 공식 문서를 찾아온 뒤 자료를 정리한다. |
| 긴 문서나 복잡한 코드 검토 | 자료 성격에 따라 둘 다 | 공개 자료는 도구와 응답 품질로 고르고, 비공개 자료는 승인된 환경에서 다룬다. |

## 정리

이번 Mac에서는 Qwen 2.5 3B를 번역과 짧은 학습 질문의 기본 모델로 골랐다. 프롬프트는 말투와 설명 방법에 영향을 줬고, 출력 형식을 얼마나 따르는지는 모델마다 달랐다. 길어진 대화에서 첫 정보를 유지하는 데에는 `num_ctx` 차이가 가장 직접적으로 드러났다. 로컬 챗봇은 외부 연결을 차단한 상태에서도 이미 받은 모델로 답했고, 최신 가격처럼 시간이 지나면 바뀌는 정보는 웹 조회가 따로 필요했다.
