import ollama
from openai import OpenAI

from config import LLM_BACKEND, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL


def build_prompt(query: str, context: str) -> str:
    return (
        "You are a helpful assistant that answers questions based only on "
        "the provided video transcript context.\n"
        "If the answer is not in the context, say \"I could not find this in the video.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


def query_ollama(prompt: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def query_openai(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def answer(query: str, context: str) -> str:
    prompt = build_prompt(query, context)
    if LLM_BACKEND == "ollama":
        return query_ollama(prompt)
    if LLM_BACKEND == "openai":
        return query_openai(prompt)
    raise ValueError(f"Unsupported LLM_BACKEND: '{LLM_BACKEND}'. Must be 'ollama' or 'openai'.")
