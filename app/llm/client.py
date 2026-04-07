from typing import List, Dict

import ollama

from app.utils.config import LLM_MODEL_NAME

def chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    num_predict: int = 512,
) -> str:
    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=messages,
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
    )
    return response["message"]["content"]


def generate_answer(prompt: str) -> str:
    return chat(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        num_predict=512,
    )
