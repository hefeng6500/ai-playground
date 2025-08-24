import os

# 输出所有环境变量，帮助调试
for key, value in os.environ.items():
    print(f"{key}: {value}")
