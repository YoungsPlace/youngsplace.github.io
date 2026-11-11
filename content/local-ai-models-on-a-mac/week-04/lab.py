"""Run the Week 4 Ollama exercises and write their raw JSON results."""
import argparse
import json
import os
import statistics
import subprocess
import tempfile
import threading
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

BASE = os.environ.get("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434")
GEMMA = "gemma3:4b"
SMALL_MODELS = ("gemma2:2b", "qwen2.5:3b", "llama3.2:3b")
PRESETS = {
    "역사 교수": "You are a history professor. Tell the story dramatically, while explaining the real science.",
    "피트니스 코치": "You are a motivational fitness coach. Explain with an exercise analogy and accurate science.",
    "동화 구연가": "You tell stories for children aged 5–8. Use simple words and a short story.",
    "과학 커뮤니케이터": "You are a sarcastic science communicator. Be dry, precise, and accurate.",
    "선승": "You are a Zen Buddhist monk. Give a calm, reflective explanation grounded in science.",
}
MODELS = {
    "week4-writer": ('Write vivid short fiction in Korean.', 1.1, 4096),
    "week4-facts": ('Answer in Korean. State facts briefly and say when unsure.', 0.1, 4096),
    "week4-study": ('Explain technical topics in Korean with an example and one quiz question.', 0.5, 4096),
    "week4-context-512": ('Follow the user instructions. Answer briefly.', 0, 512),
    "week4-context-4096": ('Follow the user instructions. Answer briefly.', 0, 4096),
}
COMPARE = (
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are three tips for studying effectively?",
    "日本の文化で最も興味深いことは何ですか？",
)
OFFLINE = (
    "2의 10제곱은? 숫자만 답해.",
    "한글로 'Thank you for your time.'을 번역해.",
    "Python에서 정수 리스트의 최댓값을 구하는 표현을 한 줄로 써 줘.",
    "도쿄는 어느 나라의 수도인지 한국어로 한 단어만 답해.",
    "비가 오는 날을 소재로 한국어 두 행의 짧은 시를 써 줘.",
)


def api(path, payload=None, base=BASE):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def chat(model, prompt=None, *, system=None, messages=None, options=None, keep_alive="5m"):
    if messages is None:
        messages = [{"role": "system", "content": system}] if system else []
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
    settings = {"num_predict": 200, "seed": 42}
    settings.update(options or {})
    start = perf_counter()
    result = api("/api/chat", {
        "model": model, "messages": messages, "stream": False,
        "options": settings, "keep_alive": keep_alive,
    })
    result["wall_seconds"] = perf_counter() - start
    return result


def rate(reply):
    return reply["eval_count"] * 1e9 / reply["eval_duration"]


def compare():
    records = {}
    for model in SMALL_MODELS:
        chat(model, "Reply OK.", options={"num_predict": 6, "num_ctx": 4096})
        trials = [
            [chat(model, prompt, options={"temperature": 0.3, "num_ctx": 4096}) for _ in range(3)]
            for prompt in COMPARE
        ]
        expanded = {
            prompt: chat(model, prompt, options={"temperature": 0.3, "num_ctx": 4096, "num_predict": 420})
            for prompt, runs in zip(COMPARE, trials)
            if runs[0]["done_reason"] == "length"
        }
        records[model] = {
            "trials": trials,
            "expanded": expanded,
            "median_tps": statistics.median(rate(reply) for runs in trials for reply in runs),
        }
        api("/api/chat", {"model": model, "messages": [], "keep_alive": 0})
    return {"models": api("/api/tags")["models"], "prompts": COMPARE, "results": records}


def resource():
    result = {}
    for model in ("gemma2:2b", "llama3.1:8b"):
        chat(model, "Reply OK.", options={"num_predict": 6, "num_ctx": 4096})
        state = next(item for item in api("/api/ps")["models"] if item["name"] == model)
        show = api("/api/show", {"model": model})
        model_path = next(
            line.removeprefix("FROM ") for line in show["modelfile"].splitlines()
            if line.startswith("FROM ")
        )
        stop = threading.Event()
        samples = []

        def sample():
            while not stop.wait(0.2):
                processes = subprocess.check_output(
                    ["ps", "-axo", "pcpu,rss,command"], text=True,
                ).splitlines()
                for line in processes[1:]:
                    cpu, rss, command = line.strip().split(maxsplit=2)
                    if f"--model {model_path} " in command:
                        samples.append({"cpu_percent": float(cpu), "rss_kib": int(rss)})

        thread = threading.Thread(target=sample)
        thread.start()
        try:
            reply = chat(
                model, "Write a 700-word science fiction story about a lighthouse.",
                options={"num_predict": 320, "num_ctx": 4096, "temperature": 0.3},
            )
        finally:
            stop.set()
            thread.join()
        result[model] = {
            "loaded": state, "reply": reply, "samples": samples,
            "cpu_peak_percent": max((x["cpu_percent"] for x in samples), default=None),
        }
        api("/api/chat", {"model": model, "messages": [], "keep_alive": 0})
    return result


def create_models():
    output = {}
    for name, (system, temperature, context) in MODELS.items():
        definition = (
            f'FROM {GEMMA}\nPARAMETER temperature {temperature}\n'
            f'PARAMETER num_ctx {context}\nSYSTEM """{system}"""\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            modelfile = Path(directory) / "Modelfile"
            modelfile.write_text(definition, encoding="utf-8")
            subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile)],
                env={**os.environ, "OLLAMA_HOST": BASE},
                check=True,
            )
        output[name] = definition
    return output


def tuning():
    presets = {
        name: chat(GEMMA, "Why is the sky blue?", system=system, options={"temperature": 0.3})
        for name, system in PRESETS.items()
    }
    prompt = "고장 난 달 기지에서 정비사가 처음 발견한 것을 한국어로 두 문장만 써 줘."
    temperatures = {
        str(t): [
            chat(GEMMA, prompt, system="한국어로 짧은 SF를 쓴다.",
                 options={"temperature": t, "top_k": 40, "top_p": 0.9, "seed": seed})
            for seed in (42, 43, 44)
        ]
        for t in (0.0, 0.5, 1.0, 1.5)
    }
    prompt_tasks = (
        "캐시가 무엇인지 설명해 줘.",
        "양자화가 무엇인지 설명해 줘.",
        "TCP 재전송을 설명해 줘.",
    )
    format_system = (
        "한국어 평문으로 정확히 세 문장만 답해. "
        "첫 문장은 정의, 둘째 문장은 '마치'로 시작하는 비유, "
        "셋째 문장은 독자가 생각할 질문으로 쓰고 물음표로 끝내. "
        "앞 두 문장은 마침표로 끝내. 제목, 목록, 약어의 마침표는 쓰지 마."
    )
    return {
        "presets": presets, "temperature_prompt": prompt, "temperatures": temperatures,
        "format_system": format_system,
        "format_test": {q: chat(GEMMA, q, system=format_system, options={"temperature": 0}) for q in prompt_tasks},
    }


def custom():
    definitions = create_models()
    prompts = {
        "week4-writer": "고장 난 달 기지를 배경으로 한국어 두 문장으로 시작해 줘.",
        "week4-facts": "물의 화학식을 알려줘.",
        "week4-study": "FFT를 간단히 설명해 줘.",
    }
    replies = {
        model: chat(model, prompt, options={"num_predict": 320})
        for model, prompt in prompts.items()
    }
    return {"definitions": definitions, "replies": replies, "models": api("/api/tags")["models"]}


def context():
    result = {}
    for model in ("week4-context-512", "week4-context-4096"):
        history = [{"role": "user", "content": "The project code is ORANGE-731. Reply only OK."}]
        measurements = []
        for turn in range(13):
            if turn:
                history.append({
                    "role": "user",
                    "content": (
                        f"Entry {turn}. The lab documented antenna calibration, sample rate, "
                        "signal strength, local storage, power supply, and a review plan. "
                        "It compared dry runs on Monday with longer recordings on Tuesday. "
                        "Confirm this entry by replying only OK."
                    ),
                })
            reply = chat(model, messages=history, options={"num_predict": 12})
            history.append({"role": "assistant", "content": reply["message"]["content"]})
            if turn in (0, 3, 6, 9, 10, 11, 12):
                question = {"role": "user", "content": "What is the project code from the start of this chat? Reply with only the code or UNKNOWN."}
                probe = chat(model, messages=[*history, question], options={"num_predict": 24})
                measurements.append({"messages_sent": len(history) + 1, "probe": probe})
        result[model] = measurements
    return result


def offline():
    return {
        "prompts": OFFLINE,
        "first_model": "qwen2.5:3b",
        "first": [chat("qwen2.5:3b", q, options={"temperature": 0, "num_predict": 90}) for q in OFFLINE],
        "switched_model": "llama3.2:3b",
        "switched": chat("llama3.2:3b", OFFLINE[0], options={"temperature": 0, "num_predict": 32}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    exercises = {
        "compare": compare, "resource": resource, "custom": custom,
        "context": context, "tuning": tuning, "offline": offline,
    }
    parser.add_argument("exercise", choices=exercises)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = exercises[args.exercise]()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
