from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

import os

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-pro",
    messages=[{"role": "user", "content": "Hello, confirm you are working."}],
    temperature=1,
    top_p=0.95,
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"thinking": False}}
)

print(completion.choices[0].message.content)
