import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "Explain RAG in simple terms"}
    ]
)

print(response["message"]["content"])