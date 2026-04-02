import httpx
from openai import OpenAI

#$ curl -v https://api.chatanywhere.tech 2>&1 | grep -i 'CAfile' || true
#*   CAfile: /etc/ssl/certs/ca-bundle.crt
client = OpenAI(
    api_key="sk-w1jJibJZIbgJ0LQysYwxgxV5PMZk5YR8PdXKGVY8tgPrxdHr",
    base_url="https://api.chatanywhere.tech/v1",
    http_client=httpx.Client(verify=False),
)

resp = client.chat.completions.create(
    model="gpt-4o-mini-ca",
    messages=[{"role": "user", "content": "Say this is a test!"}],
    temperature=0.7,
)
print(resp)