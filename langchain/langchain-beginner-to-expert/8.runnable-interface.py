"""
第8节：Runnable 接口深入
=============================================

学习目标：
- 深入理解 Runnable 接口的设计理念
- 掌握自定义 Runnable 组件开发
- 学会构建复杂的处理链条
- 了解并行和分支处理技术

前置知识：
- 完成第1-7节基础内容

重点概念：
- Runnable 是 LangChain 的核心抽象
- 所有组件都实现了 Runnable 接口
- 支持组合、并行、分支等高级操作
"""

import os
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod


def explain_runnable_concept():
    """
    解释 Runnable 接口的核心概念
    """
    print("\n" + "="*60)
    print("🔧 Runnable 接口：LangChain 的核心")
    print("="*60)
    
    print("""
🧩 什么是 Runnable？

想象乐高积木：
- 每个积木都有标准的连接接口 🔌
- 不同形状的积木可以随意组合 🧱
- 组合后的积木仍然可以继续组合 🏗️

Runnable 就是 LangChain 的"连接接口"：
- 所有组件都实现 Runnable 接口
- 可以用 | 操作符连接
- 组合后的链条也是 Runnable

🎯 核心方法：
- invoke()：同步执行
- stream()：流式执行  
- batch()：批量执行
- ainvoke()：异步执行
- astream()：异步流式执行
- abatch()：异步批量执行
    """)
    
    print("📊 Runnable 生态系统：")
    components = [
        "🧠 LLM：语言模型",
        "📝 PromptTemplate：提示模板", 
        "🔧 OutputParser：输出解析器",
        "🔗 RunnableSequence：序列链",
        "🔀 RunnableParallel：并行链",
        "🎯 RunnableLambda：自定义函数"
    ]
    
    for component in components:
        print(f"   {component}")


def basic_runnable_demo():
    """
    基础 Runnable 操作演示
    """
    print("\n" + "="*60)
    print("🎯 基础 Runnable 操作")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 DEEPSEEK_API_KEY")
            return
        
        print("🔧 创建基础组件")
        print("-" * 30)
        
        # 创建组件
        prompt = ChatPromptTemplate.from_template("请简要回答：{question}")
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7
        )
        parser = StrOutputParser()
        
        # 每个组件都是 Runnable
        print(f"✅ Prompt 是 Runnable: {hasattr(prompt, 'invoke')}")
        print(f"✅ Model 是 Runnable: {hasattr(model, 'invoke')}")
        print(f"✅ Parser 是 Runnable: {hasattr(parser, 'invoke')}")
        
        print("\n🔗 组合成链条")
        print("-" * 30)
        
        # 组合成链条
        chain = prompt | model | parser
        print(f"✅ Chain 也是 Runnable: {hasattr(chain, 'invoke')}")
        
        print(f"🎯 测试不同的执行方法...")
        
        test_input = {"question": "什么是人工智能？"}
        
        # 1. invoke：同步执行
        print("\n1️⃣ invoke() - 同步执行：")
        result = chain.invoke(test_input)
        print(f"结果：{result[:100]}...")
        
        # 2. batch：批量执行
        print("\n2️⃣ batch() - 批量执行：")
        batch_inputs = [
            {"question": "什么是机器学习？"},
            {"question": "什么是深度学习？"},
            {"question": "什么是神经网络？"}
        ]
        
        batch_results = chain.batch(batch_inputs)
        for i, result in enumerate(batch_results, 1):
            print(f"   结果{i}：{result[:50]}...")
        
        # 3. stream：流式执行
        print("\n3️⃣ stream() - 流式执行：")
        print("流式输出：", end="", flush=True)
        for chunk in chain.stream({"question": "什么是区块链？"}):
            print(chunk, end="", flush=True)
        print("\n")
        
        print("✅ Runnable 接口的统一性让操作变得简单！")
        
    except Exception as e:
        print(f"❌ 基础 Runnable 演示失败：{e}")


def custom_runnable_demo():
    """
    自定义 Runnable 组件演示
    """
    print("\n" + "="*60)
    print("🛠️  自定义 Runnable 组件")
    print("="*60)
    
    try:
        from langchain_core.runnables import RunnableLambda
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json
        import re
        
        print("🎯 场景：文本分析处理链")
        print("-" * 30)
        
        # 自定义函数1：文本预处理
        def preprocess_text(input_data):
            """文本预处理"""
            text = input_data.get("text", "")
            
            # 清理文本
            cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 只保留中英文和数字
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # 标准化空格
            
            result = {
                "original_text": text,
                "cleaned_text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split())
            }
            
            print(f"🧹 预处理完成：{len(text)} → {len(cleaned)} 字符")
            return result
        
        # 自定义函数2：情感分析
        def analyze_sentiment(input_data):
            """简单的情感分析"""
            text = input_data.get("cleaned_text", "")
            
            # 简单的关键词情感分析
            positive_words = ["好", "棒", "优秀", "喜欢", "开心", "满意", "推荐"]
            negative_words = ["差", "坏", "糟糕", "讨厌", "愤怒", "失望", "不好"]
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                sentiment = "积极"
                score = min(0.8, 0.5 + pos_count * 0.1)
            elif neg_count > pos_count:
                sentiment = "消极" 
                score = max(0.2, 0.5 - neg_count * 0.1)
            else:
                sentiment = "中性"
                score = 0.5
            
            result = input_data.copy()
            result.update({
                "sentiment": sentiment,
                "sentiment_score": score,
                "positive_words": pos_count,
                "negative_words": neg_count
            })
            
            print(f"😊 情感分析：{sentiment} (得分: {score:.2f})")
            return result
        
        # 自定义函数3：生成报告
        def generate_report(input_data):
            """生成分析报告"""
            report = f"""
📊 文本分析报告
==================
📝 原文长度：{input_data['char_count']} 字符，{input_data['word_count']} 词
😊 情感倾向：{input_data['sentiment']} (置信度: {input_data['sentiment_score']:.2f})
📈 积极词汇：{input_data['positive_words']} 个
📉 消极词汇：{input_data['negative_words']} 个

💡 分析建议：
"""
            if input_data['sentiment'] == "积极":
                report += "文本整体情感积极，传达正面信息。"
            elif input_data['sentiment'] == "消极":
                report += "文本情感偏消极，可能需要关注相关问题。"
            else:
                report += "文本情感中性，客观描述为主。"
            
            result = input_data.copy()
            result["analysis_report"] = report
            
            print("📋 报告生成完成")
            return result
        
        # 将函数转为 Runnable
        preprocessor = RunnableLambda(preprocess_text)
        sentiment_analyzer = RunnableLambda(analyze_sentiment)
        report_generator = RunnableLambda(generate_report)
        
        # 组合成分析链
        analysis_chain = preprocessor | sentiment_analyzer | report_generator
        
        print("🧪 测试文本分析链...")
        
        test_texts = [
            "这个产品真的很棒！我非常喜欢，强烈推荐给大家。",
            "服务太差了，完全不满意，很失望。",
            "今天天气不错，适合出门散步。"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 测试文本 {i}：{text}")
            print("="*50)
            
            result = analysis_chain.invoke({"text": text})
            print(result["analysis_report"])
        
        print("\n✅ 自定义 Runnable 的优势：")
        advantages = [
            "🔧 高度定制：实现特定业务逻辑",
            "🔗 无缝集成：与其他组件完美配合",
            "🧪 易于测试：每个组件可单独测试",
            "📈 可扩展：容易添加新的处理步骤"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ 自定义 Runnable 演示失败：{e}")


def parallel_runnable_demo():
    """
    并行 Runnable 演示
    """
    print("\n" + "="*60)
    print("🔀 并行处理：RunnableParallel")
    print("="*60)
    
    try:
        from langchain_core.runnables import RunnableParallel, RunnableLambda
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import time
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 场景：多角度文本分析")
        print("-" * 30)
        
        # 创建模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.5
        )
        parser = StrOutputParser()
        
        # 创建不同的分析链
        summary_chain = (
            ChatPromptTemplate.from_template("请用一句话总结：{text}") |
            model | parser
        )
        
        keywords_chain = (
            ChatPromptTemplate.from_template("提取3个关键词：{text}") |
            model | parser
        )
        
        sentiment_chain = (
            ChatPromptTemplate.from_template("分析情感倾向：{text}") |
            model | parser
        )
        
        # 创建并行处理链
        parallel_chain = RunnableParallel({
            "summary": summary_chain,
            "keywords": keywords_chain, 
            "sentiment": sentiment_chain
        })
        
        print("🧪 测试并行处理...")
        
        test_text = """
        人工智能技术在近年来发展迅速，深度学习和神经网络的突破
        让AI在图像识别、自然语言处理等领域取得了显著进展。
        然而，AI的发展也带来了一些挑战，如数据隐私、算法偏见等问题
        需要我们认真对待和解决。
        """
        
        print(f"📝 分析文本：{test_text.strip()}")
        
        # 测试串行 vs 并行的性能差异
        print("\n⏱️  性能对比：")
        
        # 串行执行
        start_time = time.time()
        serial_results = {
            "summary": summary_chain.invoke({"text": test_text}),
            "keywords": keywords_chain.invoke({"text": test_text}),
            "sentiment": sentiment_chain.invoke({"text": test_text})
        }
        serial_time = time.time() - start_time
        
        print(f"🔄 串行执行时间：{serial_time:.2f} 秒")
        
        # 并行执行
        start_time = time.time()
        parallel_results = parallel_chain.invoke({"text": test_text})
        parallel_time = time.time() - start_time
        
        print(f"⚡ 并行执行时间：{parallel_time:.2f} 秒")
        print(f"🚀 性能提升：{((serial_time - parallel_time) / serial_time * 100):.1f}%")
        
        print("\n📊 并行分析结果：")
        print("="*50)
        
        for key, value in parallel_results.items():
            print(f"📋 {key.upper()}:")
            print(f"   {value}")
            print()
        
        print("✅ 并行处理的优势：")
        advantages = [
            "⚡ 更快速度：同时执行多个任务",
            "💾 资源利用：充分利用网络带宽",
            "🎯 独立性：每个分支独立处理",
            "🔧 易扩展：容易添加新的并行分支"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ 并行 Runnable 演示失败：{e}")


def conditional_runnable_demo():
    """
    条件分支 Runnable 演示
    """
    print("\n" + "="*60)
    print("🎯 条件分支处理")
    print("="*60)
    
    try:
        from langchain_core.runnables import RunnableLambda, RunnableBranch
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 场景：智能客服路由系统")
        print("-" * 30)
        
        # 创建模型和解析器
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3
        )
        parser = StrOutputParser()
        
        # 分类函数
        def classify_query(input_data):
            """分类用户查询"""
            query = input_data.get("query", "").lower()
            
            if any(word in query for word in ["价格", "费用", "多少钱", "收费"]):
                category = "pricing"
            elif any(word in query for word in ["技术", "如何", "怎么", "教程"]):
                category = "technical"
            elif any(word in query for word in ["投诉", "问题", "故障", "不满"]):
                category = "complaint"
            else:
                category = "general"
            
            result = input_data.copy()
            result["category"] = category
            print(f"🏷️  查询分类：{category}")
            return result
        
        # 不同类型的处理链
        pricing_chain = (
            ChatPromptTemplate.from_template(
                "作为销售代表，回答价格问题：{query}"
            ) | model | parser
        )
        
        technical_chain = (
            ChatPromptTemplate.from_template(
                "作为技术支持，提供技术帮助：{query}"
            ) | model | parser
        )
        
        complaint_chain = (
            ChatPromptTemplate.from_template(
                "作为客服主管，处理投诉问题：{query}"
            ) | model | parser
        )
        
        general_chain = (
            ChatPromptTemplate.from_template(
                "作为客服代表，回答一般问题：{query}"
            ) | model | parser
        )
        
        # 创建分类器
        classifier = RunnableLambda(classify_query)
        
        # 创建条件分支
        def route_query(input_data):
            """路由查询到对应的处理链"""
            category = input_data.get("category", "general")
            
            if category == "pricing":
                return pricing_chain.invoke(input_data)
            elif category == "technical":
                return technical_chain.invoke(input_data)
            elif category == "complaint":
                return complaint_chain.invoke(input_data)
            else:
                return general_chain.invoke(input_data)
        
        router = RunnableLambda(route_query)
        
        # 完整的客服系统
        customer_service_chain = classifier | router
        
        print("🧪 测试智能客服路由...")
        
        test_queries = [
            "你们的产品多少钱？有优惠吗？",
            "如何配置Python开发环境？",
            "我对你们的服务很不满意，要投诉！",
            "你们公司在哪里？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📞 客户查询 {i}：{query}")
            print("="*50)
            
            response = customer_service_chain.invoke({"query": query})
            print(f"🤖 客服回复：{response}")
        
        print("\n✅ 条件分支的应用场景：")
        use_cases = [
            "🎯 智能路由：根据内容分发到不同处理器",
            "🔧 错误处理：根据错误类型选择处理方式",
            "📊 数据处理：根据数据格式选择解析器",
            "🎮 游戏逻辑：根据玩家行为触发不同剧情"
        ]
        
        for use_case in use_cases:
            print(f"   {use_case}")
        
    except Exception as e:
        print(f"❌ 条件分支演示失败：{e}")


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第8节总结 & 第9节预告")
    print("="*60)
    
    print("🎉 第8节你掌握了：")
    learned = [
        "✅ 深入理解 Runnable 接口设计",
        "✅ 掌握自定义 Runnable 组件开发",
        "✅ 学会并行处理提升性能",
        "✅ 了解条件分支和智能路由"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第9节预告：《文档处理与 RAG 基础》")
    print("你将学到：")
    next_topics = [
        "📄 文档加载和预处理",
        "✂️  文本分割策略",
        "🔍 相似性检索基础",
        "🧠 RAG 系统构建入门"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第8节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第8节")
    print("🔧 Runnable 接口深入")
    print("📚 前置：完成第1-7节")
    
    # 1. 解释 Runnable 概念
    explain_runnable_concept()
    
    # 2. 基础 Runnable 操作
    basic_runnable_demo()
    
    # 3. 自定义 Runnable 组件
    custom_runnable_demo()
    
    # 4. 并行处理
    parallel_runnable_demo()
    
    # 5. 条件分支
    conditional_runnable_demo()
    
    # 6. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第8节完成！")
    print("🔧 你已经是 Runnable 接口专家了！")
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
    print("   📖 LangChain Runnable 官方文档")
    print("   💻 函数式编程最佳实践")
    print("   🔧 组件化架构设计模式")