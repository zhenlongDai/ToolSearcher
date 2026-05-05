
curl https://api.chatanywhere.tech/v1/chat/completions \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-w1jJibJZIbgJ0LQysYwxgxV5PMZk5YR8PdXKGVY8tgPrxdHr' \
-d '{
"model": "gpt-5-mini-ca",
"messages": [{"role": "user", "content": "Say this is a test!"}],
"temperature": 0.7
}'