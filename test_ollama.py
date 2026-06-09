from ollama import chat

response = chat(
    model='llama3',
    messages=[
        {
            'role': 'user',
            'content': 'Explain machine learning in one sentence.'
        }
    ]
)

print(response['message']['content'])