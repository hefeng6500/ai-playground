#!/usr/bin/env python3
"""
示例：如何在代码中使用 install_packages_by_date 模块生成 requirements.txt
"""

from install_packages_by_date import generate_requirements_txt

# 示例1: 生成 requirements.txt
if __name__ == "__main__":
    # 日期
    date = "20250914"
    
    # 包列表
    packages = [
        "langchain_community",
        "unstructured",
        "charset-normalizer==3.4.3",  # 指定版本
        "markdown",
        "pi_heif",
        "unstructured_inference",
        "pdf2image",
        "unstructured_pytesseract",
        "python-docx",
        "langchain_huggingface",
        "sentence-transformers",
        "langchain_chroma",
        "dashscope",
        "jieba",
        "langchain_openai",
        "faiss-cpu",
        "ragas",
        "bitsandbytes",
        "rank_bm25",
        "pymysql",
        "sqlacodegen",
        "fastapi",
        "torch",
        "langchain_core",
        "langchain_openai",
        "langchain_chroma"
    ]
    
    # 生成 requirements.txt
    print("正在生成 requirements.txt...")
    success = generate_requirements_txt(
        date_str=date,
        packages=packages,
        output_file="requirements.txt",  # 输出文件名，默认为 requirements.txt
        verbose=True  # 显示详细信息
    )
    
    if success:
        print("\n✓ 成功生成 requirements.txt")
    else:
        print("\n✗ 生成失败")

