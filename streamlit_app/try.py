import requests
GEMINI_API_KEY = "AIzaSyBhJAwUaR4yCBKQi8Px_nBjMZ0XQnzGcTo"
response = requests.post(
    "https://api.generativeai.google/v1beta2/models/text-bison-001:generate",
    headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
    json={"prompt": "Hello", "temperature": 0.7, "max_output_tokens": 50}
)
print(response.json())
