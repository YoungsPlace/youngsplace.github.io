---
title: "[3주차] Streamlit으로 로컬 음성 챗봇 만들기"
date: 2026-11-08T00:00:00+09:00
draft: false
author: "빵주(kbbangzoo)"
summary: "Streamlit에서 WAV를 받아 MLX Whisper로 전사하고, Ollama에 질문을 보냈다. 전사 속도와 소음의 영향, 최근 대화만 보냈을 때의 응답을 비교했다."
tags: ["로컬 AI", "Mac", "Ollama", "Python", "Streamlit", "MLX Whisper"]
series: ["Local AI Models on a Mac"]
showTableOfContents: true
showAuthor: false
---

2주차에는 터미널에서 Ollama에 질문을 보내고 답변을 한 조각씩 출력했다. 3주차에는 그 코드에 웹 화면을 붙이고, 음성으로도 질문을 보낼 수 있게 만드는 과정을 살펴봤다. 교재 CH8~11의 범위다. [Build Applications with Local AI Models on a Mac](https://www.manning.com/books/build-applications-with-local-ai-models-on-a-mac)

음성 입력은 macOS에서 만든 한국어·영어 합성 WAV로 테스트했다.

## 1. Streamlit에서 위젯과 재실행 살펴보기

`streamlit run`으로 서버를 띄우면 브라우저에서 Python 코드로 만든 화면을 열 수 있다. `st.number_input`은 입력값을 숫자로 돌려주고, `st.selectbox`는 선택값을 반환한다. 계산기에는 숫자 입력 두 개를 `st.columns`로 나란히 두고, `st.metric`으로 계산 결과를 표시했다. 색상과 숫자 설정은 사이드바에 뒀다. [위젯 실습 코드](widgets_demo.py) · [Streamlit 위젯 동작](https://docs.streamlit.io/develop/concepts/architecture/widget-behavior)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install streamlit==1.66.1 ollama==0.6.3 mlx-whisper==0.4.3

python -m streamlit run widgets_demo.py --server.address 127.0.0.1
```

일반 위젯의 값이 바뀌면 스크립트가 위에서 아래로 다시 실행된다. 계산 버튼은 클릭으로 실행된 회차에만 `True`가 되고, 폼 안의 할 일 입력값은 제출할 때 한 번에 전달된다. 할 일 목록은 `st.session_state.todos`에 두어 다음 실행에서도 이어서 수정할 수 있게 했다. [Streamlit 폼](https://docs.streamlit.io/develop/concepts/architecture/forms)

| 입력과 동작 | 확인한 결과 |
|---|---|
| `12 ÷ 4`를 선택하고 계산 | 결과 `3` |
| 사이드바에서 `Blue`, `7` 선택 | 두 값이 본문에 표시됨 |
| 할 일 두 개 추가 후 마지막 항목 삭제 | 첫 번째 항목이 남음 |
| 전체 삭제 | 목록이 비워짐 |
| 나누는 수를 `0`으로 입력 | 오류 메시지 표시 |

녹화에서는 카운터를 여러 번 눌러 숫자가 누적되는 화면을 보여줬다(1:13 부근). 할 일 목록도 같은 방식으로 재실행 사이에 데이터를 유지한다.

## 2. MLX Whisper로 WAV 전사하기

16 kHz는 1초에 16,000번 소리를 샘플링한다는 뜻이다. 모노 PCM 16비트 WAV라면 오디오 데이터가 초당 약 32 kB다. Whisper는 파형을 log-Mel 스펙트로그램으로 바꿔 시간에 따른 주파수 성분을 읽고, 이를 바탕으로 텍스트 토큰을 생성한다. MLX 구현에서는 파일 경로나 배열을 입력으로 줄 수 있고, 반환값에서 전사문과 감지 언어를 꺼낼 수 있다. [Whisper 모델 설명](https://github.com/openai/whisper#approach) · [MLX Whisper API](https://github.com/ml-explore/mlx-examples/tree/main/whisper)

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "sample-ko.wav",
    path_or_hf_repo="mlx-community/whisper-small-mlx",
)
print(result["language"], result["text"].strip())
```

합성 음성은 `say`로 만든 뒤 FFmpeg로 16 kHz 모노 WAV로 변환했다. 한국어 문장은 “음성 인식과 음성 합성의 차이를 한 문장으로 설명해 주세요.”, 영어 문장은 “What is the difference between speech recognition and text to speech? Explain in one sentence.”로 정했다. 한국어 파일에는 같은 길이의 핑크 노이즈를 섞은 사본도 만들었다. [한국어 테스트 WAV](sample-ko.wav)

MLX Whisper는 파일을 읽을 때 `ffmpeg`를 사용한다. 설치되어 있지 않다면 `brew install ffmpeg`로 준비한다.

```bash
say -v Yuna -o sample-ko.aiff '음성 인식과 음성 합성의 차이를 한 문장으로 설명해 주세요.'
ffmpeg -i sample-ko.aiff -ar 16000 -ac 1 -c:a pcm_s16le sample-ko.wav
```

Apple M5 Pro에서 `mlx-whisper 0.4.3`의 실행 장치는 `Device(gpu, 0)`로 표시됐다. 모델마다 한국어 파일로 한 번 예열한 뒤, 각 파일을 세 번 전사한 시간의 중앙값을 기록했다. 측정은 모델을 내려받은 뒤 `HF_HUB_OFFLINE=1`로 실행했다. MLX는 Apple Silicon의 통합 메모리를 쓰는 배열 프레임워크다. [Apple의 MLX 설명](https://developer.apple.com/videos/play/wwdc2025/315/)

| 오디오 | 감지 언어 | tiny | small |
|---|---|---|---|
| 한국어 4.18초 | `ko` | 0.045초 · 원문과 일치 | 0.133초 · 원문과 일치 |
| 한국어 4.18초 + 소음 | `ko` | 0.046초 · ‘인식’을 ‘민식’으로 전사 | 0.132초 · ‘설명해 주세요’의 띄어쓰기만 달라짐 |
| 영어 5.52초 | `en` | 0.045초 · 원문과 일치 | 0.130초 · 원문과 일치 |

깨끗한 샘플에서는 두 모델의 전사문이 같았다. 소음이 섞였을 때는 small이 ‘인식’을 유지해, 음성 채팅 앱에는 small을 사용했다. 시간과 전사 차이는 위 합성 음성 세 개에서 관찰한 결과다.

## 3. 음성 입력과 Ollama를 연결하기

녹화에서는 브라우저에서 음성을 입력한 뒤 전사문과 Gemma 답변이 한 화면에 나오는 흐름을 시연했다(1:03 부근). 앱에는 마이크와 WAV 파일 입력을 두고, 전사문을 확인하거나 고친 뒤 전송하는 단계를 추가했다. [음성 채팅 실행 코드](voice_chat.py)

{{< mermaid >}}
flowchart TD
    A["마이크 녹음 또는 WAV 파일"] --> B["Streamlit에서 WAV 바이트 받기"]
    B --> C["임시 WAV 파일"]
    C --> D["MLX Whisper 전사"]
    D --> E["전사문 확인·편집"]
    E --> F["user 메시지"]
    T["채팅창의 텍스트 입력"] --> F
    F --> G["시스템 프롬프트 + 최근 대화 + 새 질문"]
    G --> H["로컬 Ollama · Gemma 3"]
    H --> I["응답 조각을 브라우저에 출력"]
    I --> J["질문과 완성된 답변을 세션에 저장"]
{{< /mermaid >}}

`st.audio_input`은 녹음이 끝나면 WAV 데이터를 파일처럼 읽을 수 있는 객체로 반환한다. 파일 업로드와 녹음 양쪽에서 `getvalue()`로 바이트를 읽고, 이 구현에서는 임시 파일 경로를 Whisper에 넘겼다. `with` 블록이 끝나면 임시 파일이 닫히면서 삭제된다. [Streamlit 오디오 입력](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input) · [Python 임시 파일](https://docs.python.org/3.13/library/tempfile.html#tempfile.NamedTemporaryFile)

```python
with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
    wav.write(audio.getvalue())
    wav.flush()
    result = mlx_whisper.transcribe(
        wav.name, path_or_hf_repo=WHISPER, verbose=None
    )
state.voice_draft = result["text"].strip()
state.language = result["language"]
```

전사문과 직접 입력한 문장은 모두 `user` 메시지가 된다. Ollama에서 `stream=True`로 응답 조각을 받고, `st.write_stream`에 텍스트를 하나씩 넘겨 화면에 표시했다. 스트림이 끝나면 완성된 답변을 `assistant` 메시지로 저장한다. [Ollama 스트리밍](https://docs.ollama.com/api/streaming) · [Streamlit 스트림 출력](https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream)

```bash
# Ollama 서버가 꺼져 있다면 별도 터미널에서 시작
OLLAMA_HOST=127.0.0.1:11434 ollama serve

# 모델이 준비된 뒤 앱을 실행
ollama pull gemma3:4b
python -m streamlit run voice_chat.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

앱에서 4.18초짜리 한국어 WAV를 열자 감지 언어가 `ko`로 나왔다. 전사문을 보내니 `gemma3:4b`가 “음성 인식은 사람의 말을 텍스트로 바꾸는 기술이고, 음성 합성은 텍스트를 사람의 말로 바꾸는 기술입니다.”라고 답했다.

{{< figure src="voice-chat-result.png" alt="합성 WAV의 한국어 전사문과 Gemma 3의 답변이 표시된 Streamlit 화면" caption="합성 WAV 전사와 첫 답변" figureClass="study-capture" >}}

브라우저 앱에서 첫 전사에는 0.43초, 질문을 보낸 뒤 첫 텍스트 조각을 받는 데는 4.19초가 걸렸다. 후속 텍스트 질문의 첫 조각은 0.08초 만에 도착했다. 전사 시간은 파일 처리와 전사 호출을 포함하고, 첫 조각 시간은 Python에서 Ollama 요청을 시작한 시점부터 잰 값이다. 녹음 길이와 전사문 확인에 쓴 시간은 별도다.

## 4. 화면에 남길 기록과 모델에 보낼 기록

“방금 질문에서 언급한 두 기술 이름만 쉼표로 구분해 줘.”라고 채팅창에 입력하자 “음성 인식, 음성 합성”이라고 답했다. `chat.txt`를 내려받아 두 번의 질문과 답변이 들어 있는 것을 확인했다. 대화 초기화를 누르면 채팅 목록과 전사문이 비워졌다.

{{< figure src="history-result.png" alt="음성 질문과 이어진 텍스트 질문에 대한 답변이 한 대화에 남은 화면" caption="음성 질문 뒤에 텍스트로 이어서 묻기" figureClass="study-capture" >}}

`st.session_state.history`는 브라우저 세션이 이어지는 동안 재실행 사이에 유지된다. 화면에는 이 목록 전체를 표시하고, 모델에는 시스템 메시지와 최근 대화 쌍, 새 질문을 보내도록 했다. 새 연결이 만들어지면 세션 상태가 초기화될 수 있어서 파일로 남길 때는 다운로드 버튼을 사용했다. [Streamlit 세션 상태](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)

```python
messages = [
    {"role": "system", "content": system_prompt},
    *state.history[-2 * turns:],
    {"role": "user", "content": prompt},
]
```

### 최근 세 쌍으로 줄였을 때

별도 Ollama 요청에서 첫 대화에 `My name is Alex. Say hello.`를 넣고, 이름과 무관한 세 쌍을 더 쌓았다. 동일한 `What is my name?`을 전체 기록과 최근 세 쌍으로 나눠 보냈다. 모델은 `gemma3:4b`, 설정은 `temperature=0`, `num_ctx=4096`을 사용했다.

| 요청에 넣은 기록 | Ollama의 입력 토큰 수 | 답변 |
|---|---:|---|
| 이전 네 쌍 전체 | 139 | `Alex.` |
| 최근 세 쌍 | 116 | `I do not know.` |
| 첫 대화의 요약을 시스템 메시지에 붙인 뒤 최근 세 쌍 | 132 | `Alex.` |

요약은 같은 모델에 첫 대화를 넘겨 이름을 포함한 한 문장으로 만들었다. 이름이 담긴 요약을 붙이자 이번 요청에서는 답이 돌아왔다. 요약 과정에서 빠진 세부 내용은 이후 요청에도 전달되지 않는다.

### 대화 쌍 수와 토큰 수

최근 세 쌍으로 제한해도 각 메시지가 길면 입력은 커진다. 앱 사이드바에는 메시지 길이로 추정한 숫자 대신, 응답 마지막 조각의 `prompt_eval_count`를 표시했다. 입력 예산을 잡을 때는 시스템 프롬프트와 새 질문, 답변을 생성할 공간까지 포함한다. [Ollama 사용량 필드](https://docs.ollama.com/api/usage) · [Ollama 컨텍스트 길이](https://docs.ollama.com/context-length)

이름 소개 뒤에 자전거 문장을 반복한 메모 여덟 쌍을 붙인 별도 요청도 보냈다. 같은 기록에서 `num_ctx=4096`일 때 입력은 1,405토큰, `num_ctx=512`일 때는 389토큰으로 보고됐다. 512에서 마지막 한 쌍만 골라 보내면 209토큰이었다. 세 경우 모두 모델은 이름을 모른다고 답했다. 긴 입력을 받아들인 경우에도 첫머리의 정보를 답에 쓰는지는 따로 확인할 필요가 있었다.

## 5. 정리

| 공부한 내용 | 실습에서 본 동작 |
|---|---|
| Streamlit | 위젯 이벤트로 코드가 다시 실행되고, 할 일 목록은 세션에 남았다. |
| MLX Whisper | 같은 WAV를 두 모델에 넣어 전사문을 비교했다. 소음이 섞인 파일에서 한 단어의 차이가 났다. |
| 음성 채팅 | WAV를 전사한 뒤 확인한 문장을 Ollama에 보냈다. 텍스트 후속 질문도 같은 대화에 이어졌다. |
| 대화 기록 | 화면에 보이는 기록과 모델 입력에 넣는 범위를 따로 관리했다. 처음의 이름을 빼면 답이 달라졌다. |
