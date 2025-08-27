"""
第9节：文档处理与 RAG 基础
=============================================

学习目标：
- 理解 RAG（检索增强生成）的核心概念
- 掌握文档加载和预处理技术
- 学会文本分割策略和技巧
- 了解向量化和相似性检索基础

前置知识：
- 完成第1-8节基础内容

重点概念：
- RAG = 检索 + 生成，让AI有"外部记忆"
- 文档预处理是RAG系统的基础
- 好的分割策略影响检索质量
"""

import os
from typing import List, Dict, Any
import tempfile


def explain_rag_concept():
    """
    解释 RAG 的核心概念
    """
    print("\n" + "="*60)
    print("🧠 什么是 RAG（检索增强生成）？")
    print("="*60)
    
    print("""
📚 想象你在写论文：

传统方式（纯生成）：
👨‍🎓 学生：凭记忆写论文
🧠 大脑：只能用已有知识
📝 结果：可能信息过时、不准确

RAG 方式（检索+生成）：
👨‍🎓 学生：查阅资料后写论文  
📚 图书馆：查找相关资料
🧠 大脑：结合资料和知识
📝 结果：信息更新、更准确

RAG 的工作流程：
1. 📄 文档准备：将知识库文档向量化存储
2. 🔍 相关检索：根据用户问题检索相关文档
3. 🧠 增强生成：结合检索结果生成回答
4. 📤 返回答案：给出有根据的回答
    """)
    
    print("🎯 RAG 的优势：")
    advantages = [
        "📈 知识更新：随时添加新文档",
        "🎯 准确性高：基于实际文档回答",
        "🔍 可追溯：可以指出信息来源",
        "💾 成本低：不用重新训练模型",
        "🔒 私有数据：可以处理企业内部文档"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")


def document_loading_demo():
    """
    文档加载演示
    """
    print("\n" + "="*60)
    print("📄 文档加载与预处理")
    print("="*60)
    
    try:
        print("🎯 支持的文档类型")
        print("-" * 30)
        
        # 模拟不同类型的文档内容
        documents = {
            "纯文本": """
LangChain是一个用于开发由语言模型驱动的应用程序的框架。
它提供了以下核心功能：
1. 模型集成：支持多种LLM
2. 链式调用：组合不同组件
3. 内存管理：保持对话状态
4. 工具集成：连接外部API
            """.strip(),
            
            "Markdown": """
# LangChain 使用指南

## 安装
```bash
pip install langchain
```

## 快速开始
LangChain 的核心是**链式调用**：
- 提示模板 → 模型 → 输出解析器

## 特性
- ✅ 模块化设计
- ✅ 丰富的生态
            """.strip(),
            
            "JSON": {
                "title": "LangChain API文档",
                "version": "0.1.0",
                "endpoints": [
                    {"path": "/chat", "method": "POST", "description": "聊天接口"},
                    {"path": "/embed", "method": "POST", "description": "向量化接口"}
                ]
            }
        }
        
        print("📋 文档示例：")
        for doc_type, content in documents.items():
            print(f"\n📝 {doc_type} 文档：")
            if isinstance(content, str):
                print(content[:150] + "..." if len(content) > 150 else content)
            else:
                import json
                print(json.dumps(content, ensure_ascii=False, indent=2))
        
        # 文档预处理步骤
        print(f"\n🔧 文档预处理流程：")
        preprocessing_steps = [
            "📥 加载：读取各种格式文档",
            "🧹 清理：去除无关格式和噪声",
            "📏 标准化：统一文本格式",
            "🏷️  元数据：提取文档信息",
            "✂️  分割：切分成合适大小的块"
        ]
        
        for step in preprocessing_steps:
            print(f"   {step}")
        
        # 简单的文档处理示例
        def simple_document_processor(text: str, doc_type: str = "text"):
            """简单的文档处理器"""
            
            # 基础清理
            if doc_type == "markdown":
                # 移除markdown标记
                import re
                text = re.sub(r'#+ ', '', text)  # 移除标题标记
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 移除粗体
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # 移除代码块
            
            # 标准化换行和空格
            text = ' '.join(text.split())
            
            # 计算基础统计
            stats = {
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": text.count('\n') + 1
            }
            
            return {
                "processed_text": text,
                "statistics": stats,
                "doc_type": doc_type
            }
        
        print(f"\n🧪 测试文档处理器：")
        
        for doc_type, content in documents.items():
            if isinstance(content, str):
                result = simple_document_processor(content, doc_type.lower())
                stats = result["statistics"]
                print(f"\n📊 {doc_type} 处理结果：")
                print(f"   字符数：{stats['char_count']}")
                print(f"   词数：{stats['word_count']}")
                print(f"   处理后预览：{result['processed_text'][:100]}...")
        
    except Exception as e:
        print(f"❌ 文档加载演示失败：{e}")


def text_splitting_demo():
    """
    文本分割演示
    """
    print("\n" + "="*60)
    print("✂️  文本分割策略")
    print("="*60)
    
    print("""
🎯 为什么需要文本分割？

大文档的问题：
- 📏 超出模型上下文长度限制
- 🎯 检索精度下降（信息太杂）
- 💾 向量存储效率低
- 🔍 相关性判断困难

分割的目标：
- 📦 合适大小：不超过模型限制
- 🧩 语义完整：保持内容连贯性
- 🔗 重叠处理：避免信息断裂
- 📊 均匀分布：大小尽量一致
    """)
    
    try:
        # 测试文档
        sample_text = """
        人工智能的发展历程可以分为几个重要阶段。

        第一阶段：符号主义时期（1950-1980年代）
        这个时期的AI主要基于逻辑推理和符号操作。科学家们认为智能可以通过符号和规则来表示。
        代表性成果包括专家系统和知识图谱。然而，这种方法在处理不确定性和常识推理方面遇到了困难。

        第二阶段：连接主义兴起（1980-2000年代）
        神经网络的重新兴起标志着这个时期的开始。多层感知机、反向传播算法的发明使得神经网络能够学习复杂的模式。
        但由于计算能力和数据的限制，神经网络的发展一度陷入低谷。

        第三阶段：深度学习革命（2000年代至今）
        随着大数据、云计算和GPU计算能力的提升，深度学习迎来了爆发式增长。
        卷积神经网络在图像识别领域取得突破，循环神经网络在序列处理方面表现优异。
        Transformer架构的提出更是推动了自然语言处理的重大进展。

        第四阶段：大模型时代（2018年至今）
        BERT、GPT等大型预训练模型的出现，标志着AI进入了新的时代。
        这些模型通过在海量文本上预训练，获得了强大的语言理解和生成能力。
        ChatGPT的成功更是将AI应用推向了新的高度。

        未来展望
        AI的发展仍在加速，多模态AI、通用人工智能等概念正在逐步实现。
        同时，AI的安全性、可解释性和伦理问题也越来越受到关注。
        """
        
        print("🧪 测试不同的分割策略")
        print("-" * 30)
        
        # 策略1：按字符长度分割
        def split_by_chars(text: str, chunk_size: int = 200, overlap: int = 50):
            """按字符长度分割"""
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                
                # 尝试在句号处分割，保持语义完整
                if end < len(text) and '。' in chunk:
                    last_period = chunk.rfind('。')
                    if last_period > chunk_size // 2:  # 确保块不会太小
                        chunk = chunk[:last_period + 1]
                        end = start + last_period + 1
                
                chunks.append(chunk.strip())
                start = end - overlap  # 重叠
            
            return chunks
        
        # 策略2：按段落分割
        def split_by_paragraphs(text: str, max_chars: int = 300):
            """按段落分割，合并小段落"""
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_chunk) + len(para) <= max_chars:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # 策略3：按语义分割（简化版）
        def split_by_semantic(text: str, max_chars: int = 400):
            """按语义单元分割"""
            # 按标题和段落分割
            sections = []
            current_section = ""
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检测标题（包含"阶段"、"时期"等关键词）
                if any(keyword in line for keyword in ['阶段', '时期', '展望']):
                    if current_section:
                        sections.append(current_section.strip())
                    current_section = line + "\n"
                else:
                    current_section += line + "\n"
            
            if current_section:
                sections.append(current_section.strip())
            
            return sections
        
        # 测试三种策略
        strategies = {
            "字符分割": split_by_chars,
            "段落分割": split_by_paragraphs,
            "语义分割": split_by_semantic
        }
        
        for strategy_name, split_func in strategies.items():
            print(f"\n📊 {strategy_name}结果：")
            chunks = split_func(sample_text)
            
            print(f"   分块数量：{len(chunks)}")
            for i, chunk in enumerate(chunks, 1):
                char_count = len(chunk)
                preview = chunk.replace('\n', ' ')[:50] + "..."
                print(f"   块{i}: {char_count}字符 - {preview}")
            
            # 计算统计信息
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"   平均大小：{avg_size:.0f}字符")
            print(f"   大小范围：{min(chunk_sizes)}-{max(chunk_sizes)}字符")
        
        print(f"\n💡 分割策略选择建议：")
        suggestions = [
            "📝 技术文档：按段落或语义分割",
            "📰 新闻文章：按字符长度分割",
            "📚 学术论文：按章节和段落分割",
            "💬 对话记录：按对话轮次分割",
            "📋 列表数据：按条目分割"
        ]
        
        for suggestion in suggestions:
            print(f"   {suggestion}")
        
    except Exception as e:
        print(f"❌ 文本分割演示失败：{e}")


def vectorization_basics():
    """
    向量化基础概念
    """
    print("\n" + "="*60)
    print("🔢 向量化：文本变数字")
    print("="*60)
    
    print("""
🎯 什么是向量化？

文本 → 数字向量的转换过程：

例子：
"我喜欢苹果" → [0.2, -0.1, 0.8, 0.3, ...]
"苹果很甜" → [0.3, -0.2, 0.7, 0.4, ...]

相似的文本 → 相似的向量：
- 余弦相似度：计算向量夹角
- 欧几里得距离：计算向量距离
- 点积：计算向量相关性

🔍 检索原理：
1. 问题向量化：用户问题 → 向量
2. 计算相似度：与所有文档向量比较
3. 排序返回：最相似的文档排在前面
    """)
    
    # 简单的向量化演示（使用随机向量模拟）
    import random
    import math
    
    def simple_vectorize(text: str, dimension: int = 5) -> List[float]:
        """简单的文本向量化（仅演示用）"""
        # 基于文本内容生成确定性向量
        random.seed(hash(text) % 1000000)
        return [random.uniform(-1, 1) for _ in range(dimension)]
    
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    print("🧪 简单向量化演示：")
    print("-" * 30)
    
    # 测试文档
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的重要组成部分", 
        "深度学习使用神经网络进行训练",
        "今天天气很好，适合出门散步",
        "我喜欢吃水果，特别是苹果和香蕉"
    ]
    
    # 向量化所有文档
    doc_vectors = {}
    for i, doc in enumerate(documents):
        vec = simple_vectorize(doc)
        doc_vectors[f"文档{i+1}"] = {"text": doc, "vector": vec}
        print(f"📄 文档{i+1}: {doc}")
        print(f"   向量: [{', '.join(f'{x:.2f}' for x in vec)}]")
    
    # 测试查询
    print(f"\n🔍 相似性检索测试：")
    
    queries = [
        "什么是人工智能？",
        "神经网络如何工作？",
        "天气怎么样？"
    ]
    
    for query in queries:
        query_vector = simple_vectorize(query)
        print(f"\n❓ 查询: {query}")
        print(f"   查询向量: [{', '.join(f'{x:.2f}' for x in query_vector)}]")
        
        # 计算与所有文档的相似度
        similarities = []
        for doc_name, doc_info in doc_vectors.items():
            similarity = cosine_similarity(query_vector, doc_info["vector"])
            similarities.append((doc_name, similarity, doc_info["text"]))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("   📊 检索结果（按相似度排序）：")
        for rank, (doc_name, sim, text) in enumerate(similarities[:3], 1):
            print(f"      {rank}. {doc_name}: {sim:.3f} - {text}")
    
    print(f"\n✅ 向量化的关键要点：")
    key_points = [
        "🎯 语义理解：相似含义的文本向量接近",
        "📊 数值计算：可以进行数学运算",
        "🔍 快速检索：通过向量运算快速找到相关内容",
        "📈 可扩展：支持海量文档的高效检索"
    ]
    
    for point in key_points:
        print(f"   {point}")


def basic_rag_demo():
    """
    基础 RAG 系统演示
    """
    print("\n" + "="*60)
    print("🧠 基础 RAG 系统构建")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 DEEPSEEK_API_KEY")
            return
        
        print("🎯 构建简单的RAG系统")
        print("-" * 30)
        
        # 知识库文档
        knowledge_base = [
            {
                "id": "doc1",
                "title": "LangChain简介",
                "content": "LangChain是一个用于开发语言模型应用的框架，提供了模型集成、链式调用、内存管理等功能。"
            },
            {
                "id": "doc2", 
                "title": "LCEL语法",
                "content": "LCEL（LangChain Expression Language）使用管道操作符|连接不同组件，如prompt | model | parser。"
            },
            {
                "id": "doc3",
                "title": "提示模板",
                "content": "提示模板用于构建发送给语言模型的消息，支持变量插值和复杂的对话格式。"
            },
            {
                "id": "doc4",
                "title": "输出解析",
                "content": "输出解析器将模型的原始输出转换为结构化数据，如JSON、列表等格式。"
            }
        ]
        
        # 简单的检索函数
        def simple_retriever(query: str, top_k: int = 2):
            """简单的关键词检索器"""
            scores = []
            query_lower = query.lower()
            
            for doc in knowledge_base:
                score = 0
                content_lower = doc["content"].lower()
                title_lower = doc["title"].lower()
                
                # 简单的关键词匹配评分
                for word in query_lower.split():
                    if word in title_lower:
                        score += 2  # 标题匹配权重更高
                    if word in content_lower:
                        score += 1
                
                scores.append((score, doc))
            
            # 按分数排序，返回前k个
            scores.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scores[:top_k] if score > 0]
        
        # 创建RAG提示模板
        rag_prompt = ChatPromptTemplate.from_template("""
基于以下参考文档回答用户问题：

参考文档：
{context}

用户问题：{question}

请基于参考文档回答，如果文档中没有相关信息，请说明无法从给定文档中找到答案。
""")
        
        # 创建模型和解析器
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3
        )
        parser = StrOutputParser()
        
        # RAG链条
        def rag_chain(question: str):
            """完整的RAG处理流程"""
            
            # 1. 检索相关文档
            print(f"🔍 检索相关文档...")
            retrieved_docs = simple_retriever(question)
            
            print(f"   找到 {len(retrieved_docs)} 个相关文档：")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"   {i}. {doc['title']}")
            
            # 2. 构建上下文
            if retrieved_docs:
                context = "\n\n".join([
                    f"文档{i}: {doc['title']}\n{doc['content']}" 
                    for i, doc in enumerate(retrieved_docs, 1)
                ])
            else:
                context = "没有找到相关文档。"
            
            # 3. 生成回答
            print(f"🧠 生成回答...")
            chain = rag_prompt | model | parser
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response, retrieved_docs
        
        # 测试RAG系统
        print("🧪 测试RAG系统：")
        
        test_questions = [
            "什么是LangChain？",
            "如何使用LCEL语法？",
            "什么是区块链技术？",  # 知识库中没有的问题
            "提示模板有什么作用？"
        ]
        
        for question in test_questions:
            print(f"\n" + "="*50)
            print(f"❓ 问题: {question}")
            
            response, docs = rag_chain(question)
            
            print(f"🤖 回答: {response}")
            
            if docs:
                print(f"📚 参考文档: {[doc['title'] for doc in docs]}")
            else:
                print(f"📚 参考文档: 无相关文档")
        
        print(f"\n✅ RAG系统的核心组件：")
        components = [
            "📄 知识库：存储领域文档",
            "🔍 检索器：找到相关文档",
            "🧠 生成器：基于文档生成回答",
            "📊 评估器：评估回答质量"
        ]
        
        for component in components:
            print(f"   {component}")
        
    except Exception as e:
        print(f"❌ RAG系统演示失败：{e}")


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第9节总结 & 第10节预告")
    print("="*60)
    
    print("🎉 第9节你掌握了：")
    learned = [
        "✅ 理解RAG系统的核心概念",
        "✅ 掌握文档加载和预处理",
        "✅ 学会多种文本分割策略",
        "✅ 了解向量化和相似性检索",
        "✅ 构建基础RAG系统"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第10节预告：《向量存储与检索》")
    print("你将学到：")
    next_topics = [
        "🗄️  向量数据库介绍",
        "⚡ FAISS 向量存储",
        "🔍 高级检索策略",
        "📊 检索性能优化"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第9节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第9节")
    print("📄 文档处理与 RAG 基础")
    print("📚 前置：完成第1-8节")
    
    # 1. 解释RAG概念
    explain_rag_concept()
    
    # 2. 文档加载演示
    document_loading_demo()
    
    # 3. 文本分割演示
    text_splitting_demo()
    
    # 4. 向量化基础
    vectorization_basics()
    
    # 5. 基础RAG系统
    basic_rag_demo()
    
    # 6. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第9节完成！")
    print("📄 你已经掌握了RAG基础技术！")
    print("="*60)


if __name__ == "__main__":
    # 检查环境
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  请先配置 DEEPSEEK_API_KEY")
        import getpass
        temp_key = getpass.getpass("请输入 DeepSeek API Key: ")
        if temp_key:
            os.environ["DEEPSEEK_API_KEY"] = temp_key
    
    # 运行主程序
    main()
    
    print("\n🔗 本节参考资源：")
    print("   📖 RAG 论文和最佳实践")
    print("   💻 文档处理工具库")
    print("   🔍 向量检索算法原理")