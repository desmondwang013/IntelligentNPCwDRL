"""Quick timing test"""
import ollama
import time

prompt = """You are an intent parser. Convert user instructions to JSON.
Output format: {"action": "navigate"|"wait"|"stop"|"unknown", "target": "..."}"""

print("Testing Qwen3 response time (no num_predict limit)...")
print()

start = time.time()
response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "go to the red box"},
    ],
    format="json",
    options={"temperature": 0.5}
)
elapsed = time.time() - start

content = response.message.content
print(f"Time: {elapsed:.1f}s")
print(f"Length: {len(content)} chars")
print(f"Content: {content[:500]}")  # First 500 chars
