import tempfile
import time

import mlx_whisper
import streamlit as st
from ollama import Client, ResponseError

MODEL = "gemma3:4b"
WHISPER = "mlx-community/whisper-small-mlx"
CONTEXT = 4096
client = Client(host="http://127.0.0.1:11434")

st.set_page_config(page_title="로컬 음성 채팅", page_icon="🎙️")
st.title("로컬 음성 채팅")
st.caption("MLX Whisper · Ollama Gemma 3 4B")

state = st.session_state
if "history" not in state:
    state.history = []


def reset_chat():
    state.history = []
    for key in ("voice_draft", "language", "asr_seconds", "last_request"):
        state.pop(key, None)


def stream_reply(messages):
    start = time.perf_counter()
    first_text = None
    last_chunk = None

    def response_chunks():
        nonlocal first_text, last_chunk
        for chunk in client.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options={"num_ctx": CONTEXT, "num_predict": 160, "temperature": 0},
        ):
            last_chunk = chunk
            if chunk.message.content:
                if first_text is None:
                    first_text = time.perf_counter() - start
                yield chunk.message.content

    reply = st.write_stream(response_chunks)
    tokens = last_chunk.prompt_eval_count if last_chunk else None
    return reply, tokens, first_text


with st.sidebar:
    st.header("채팅 설정")
    system_prompt = st.text_area(
        "시스템 프롬프트",
        "한국어로 간결하게 답해 줘. 대화에 없는 정보는 모른다고 말해 줘.",
    )
    turns = st.slider("모델에 보낼 최근 대화 쌍", 1, 12, 3)
    st.button("대화 초기화", on_click=reset_chat)
    if state.history:
        chat_text = "\n\n".join(
            f"{item['role']}: {item['content']}" for item in state.history
        )
        st.download_button("대화 다운로드", chat_text, "chat.txt", "text/plain", on_click="ignore")
    if "last_request" in state:
        tokens, elapsed = state.last_request
        st.caption(f"직전 요청 입력 {tokens}토큰 · 첫 출력 {elapsed:.2f}초")


with st.expander("음성 입력", expanded=True):
    recording = st.audio_input("마이크로 녹음", sample_rate=16000)
    upload = st.file_uploader("WAV 파일로 시험", type="wav")
    audio = upload if upload is not None else recording
    if st.button("전사하기", disabled=audio is None):
        state.pop("voice_draft", None)
        state.pop("language", None)
        state.pop("asr_seconds", None)
        try:
            with st.spinner("음성을 텍스트로 변환하는 중"):
                start = time.perf_counter()
                with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                    wav.write(audio.getvalue())
                    wav.flush()
                    result = mlx_whisper.transcribe(
                        wav.name, path_or_hf_repo=WHISPER, verbose=None
                    )
                state.asr_seconds = time.perf_counter() - start
            state.voice_draft = result["text"].strip()
            state.language = result["language"]
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"전사 실패 · {exc}")

    draft = st.text_area("전사문 확인", key="voice_draft")
    if "language" in state:
        st.caption(f"감지 언어 {state.language} · 전사 {state.asr_seconds:.2f}초")
    send_voice = st.button("전사문 보내기", disabled=not draft.strip())


for item in state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

typed = st.chat_input("텍스트로 질문", max_chars=2000)
prompt = (draft if send_voice else typed or "").strip()

if prompt:
    user_message = {"role": "user", "content": prompt}
    messages = [
        {"role": "system", "content": system_prompt},
        *state.history[-2 * turns:],
        user_message,
    ]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply, tokens, first_text = stream_reply(messages)
            if isinstance(reply, str) and reply.strip():
                state.history.extend([user_message, {"role": "assistant", "content": reply}])
                state.last_request = (tokens, first_text)
                st.rerun()
            else:
                st.warning("모델 응답이 비어 있습니다.")
        except (ConnectionError, ResponseError) as exc:
            st.error(f"Ollama 요청 실패 · {exc}")
