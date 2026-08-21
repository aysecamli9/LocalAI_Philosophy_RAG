from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:53508/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="phi-4-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain in one short sentence what Socrates is doing in his defense."
        }
    ],
    temperature=0.1,
    max_tokens=100
)

print(response.choices[0].message.content)