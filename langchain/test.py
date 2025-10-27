import os

print(os.environ.get("ANTHROPIC_API_KEY"))

print(os.getenv("ANTHROPIC_API_KEY"))  # 应该能输出 key
print(os.getenv("ANTHROPIC_PROXY"))  # 应该能输出 key
