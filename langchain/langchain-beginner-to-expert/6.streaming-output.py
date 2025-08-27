"""
第6节：流式输出与实时交互
=============================================

学习目标：
- 理解流式输出的原理和优势
- 掌握流式API的使用方法
- 学会构建实时交互应用
- 了解性能优化技巧

前置知识：
- 完成第1-5节基础内容

重点概念：
- 流式输出让用户立即看到响应
- 改善用户体验，减少等待时间
- 适合长文本生成和实时对话
"""

import os
import asyncio
import time
from typing import AsyncGenerator, Iterator


def explain_streaming():
    """
    解释流式输出的概念和优势
    """
    print("\n" + "="*60)
    print("⚡ 什么是流式输出？")
    print("="*60)
    
    print("""
🍔 想象在快餐店点餐：

传统方式（非流式）：
👤 顾客：我要一个汉堡套餐
🏪 店员：好的，请等待...
   ⏳ [制作汉堡...]
   ⏳ [准备薯条...]
   ⏳ [倒饮料...]
🏪 店员：您的套餐好了！
👤 顾客：等了10分钟才拿到...

流式方式：
👤 顾客：我要一个汉堡套餐
🏪 店员：好的！汉堡正在制作...
   ✅ [汉堡做好了] → 立即给顾客
   ✅ [薯条做好了] → 立即给顾客
   ✅ [饮料准备好] → 立即给顾客
👤 顾客：边等边吃，体验更好！

在 AI 中：
- 传统：等AI写完整篇文章才显示
- 流式：AI写一句就显示一句，立即可见
    """)
    
    print("🎯 流式输出的优势：")
    advantages = [
        "⚡ 响应更快：用户立即看到结果",
        "💫 体验更好：降低等待焦虑",
        "🔄 可中断：用户可以随时停止",
        "📱 移动友好：适合小屏幕逐步显示"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")


def basic_streaming_demo():
    """
    基础流式输出演示
    """
    print("\n" + "="*60)
    print("🌊 基础流式输出演示")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 DEEPSEEK_API_KEY")
            return
        
        print("🎯 对比：普通输出 vs 流式输出")
        print("-" * 40)
        
        # 创建组件
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7,
            streaming=True  # 启用流式输出
        )
        
        prompt = ChatPromptTemplate.from_template(
            "请写一篇200字的文章介绍：{topic}"
        )
        parser = StrOutputParser()
        
        # 创建链条
        chain = prompt | model | parser
        
        test_topic = "人工智能的未来发展"
        
        print(f"📝 测试主题：{test_topic}")
        
        # 1. 普通输出（一次性返回）
        print("\n🔴 普通输出模式：")
        print("等待中...", end="", flush=True)
        start_time = time.time()
        
        # 注意：这里仍然是一次性返回，但我们模拟等待
        normal_result = chain.invoke({"topic": test_topic})
        end_time = time.time()
        
        print(f"\r用时 {end_time - start_time:.1f} 秒")
        print(normal_result)
        
        # 2. 流式输出（逐步返回）
        print("\n🟢 流式输出模式：")
        print("实时显示：", flush=True)
        
        start_time = time.time()
        for chunk in chain.stream({"topic": test_topic}):
            print(chunk, end="", flush=True)
            time.sleep(0.05)  # 模拟处理时间
        
        end_time = time.time()
        print(f"\n\n✅ 流式输出完成，总用时 {end_time - start_time:.1f} 秒")
        
        print("\n📊 流式输出的特点：")
        features = [
            "🔄 逐步显示：内容一点点出现",
            "⚡ 更快感知：立即看到开始",
            "💭 思维过程：看到AI的'思考'过程",
            "🛑 可中断：随时可以停止生成"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
    except Exception as e:
        print(f"❌ 流式输出演示失败：{e}")


def async_streaming_demo():
    """
    异步流式输出演示
    """
    print("\n" + "="*60)
    print("🚀 异步流式输出")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 高性能异步处理")
        print("-" * 30)
        
        async def async_streaming_example():
            """异步流式输出示例"""
            
            # 创建异步模型
            model = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com",
                temperature=0.6,
                streaming=True
            )
            
            prompt = ChatPromptTemplate.from_template(
                "请用3句话总结：{topic}"
            )
            parser = StrOutputParser()
            
            chain = prompt | model | parser
            
            topics = [
                "机器学习的基本原理",
                "区块链技术的应用",
                "云计算的发展趋势"
            ]
            
            print("🔄 同时处理多个请求...")
            
            # 并发处理多个流式请求
            async def process_topic(topic: str, index: int):
                print(f"\n📝 任务{index}: {topic}")
                print("回答: ", end="", flush=True)
                
                async for chunk in chain.astream({"topic": topic}):
                    print(chunk, end="", flush=True)
                    await asyncio.sleep(0.03)  # 模拟处理时间
                
                print(f"\n✅ 任务{index}完成")
            
            # 并发执行所有任务
            tasks = [
                process_topic(topic, i+1) 
                for i, topic in enumerate(topics)
            ]
            
            await asyncio.gather(*tasks)
            
            print("\n🎉 所有异步任务完成！")
        
        # 运行异步示例
        print("开始异步流式处理...")
        asyncio.run(async_streaming_example())
        
        print("\n✅ 异步流式的优势：")
        advantages = [
            "🚀 并发处理：同时处理多个请求",
            "⚡ 高吞吐：充分利用网络和CPU",
            "📱 响应式：适合Web应用和聊天机器人",
            "🔧 可扩展：容易扩展到更多并发"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ 异步流式演示失败：{e}")


def streaming_chat_demo():
    """
    流式聊天演示
    """
    print("\n" + "="*60)
    print("💬 流式聊天体验")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import HumanMessage, AIMessage
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 模拟实时聊天应用")
        print("-" * 30)
        
        class StreamingChatBot:
            """流式聊天机器人"""
            
            def __init__(self):
                self.model = ChatOpenAI(
                    model="deepseek-chat",
                    openai_api_key=api_key,
                    openai_api_base="https://api.deepseek.com",
                    temperature=0.7,
                    streaming=True
                )
                
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个友善的AI助手，回答要简洁明了。"),
                    ("human", "{message}")
                ])
                
                self.parser = StrOutputParser()
                self.chain = self.prompt | self.model | self.parser
            
            def chat_stream(self, message: str):
                """流式聊天方法"""
                print(f"👤 用户: {message}")
                print("🤖 AI: ", end="", flush=True)
                
                response = ""
                for chunk in self.chain.stream({"message": message}):
                    print(chunk, end="", flush=True)
                    response += chunk
                    # 模拟打字机效果
                    time.sleep(0.02)
                
                print("\n")
                return response
            
            def demo_conversation(self):
                """演示对话"""
                conversations = [
                    "你好，你是谁？",
                    "什么是LangChain？",
                    "能给我推荐一本编程书吗？",
                    "谢谢你的建议！"
                ]
                
                print("🎭 开始模拟对话...")
                print("="*50)
                
                for msg in conversations:
                    self.chat_stream(msg)
                    print("-"*50)
                    time.sleep(1)  # 暂停一下模拟真实对话
        
        # 创建并运行聊天机器人
        chatbot = StreamingChatBot()
        chatbot.demo_conversation()
        
        print("\n💡 流式聊天的实现要点：")
        points = [
            "⚡ 实时响应：用户发送消息后立即开始回复",
            "📝 打字效果：模拟真人打字的感觉",
            "🛑 可中断：用户可以随时打断AI",
            "💾 状态管理：保持对话历史和上下文"
        ]
        
        for point in points:
            print(f"   {point}")
        
    except Exception as e:
        print(f"❌ 流式聊天演示失败：{e}")


def performance_optimization():
    """
    性能优化技巧
    """
    print("\n" + "="*60)
    print("⚡ 流式输出性能优化")
    print("="*60)
    
    optimization_tips = {
        "1. 网络优化": [
            "使用 HTTP/2 连接复用",
            "启用 gzip 压缩减少传输",
            "设置合理的超时时间",
            "实现断线重连机制"
        ],
        
        "2. 缓冲策略": [
            "设置适当的缓冲区大小",
            "批量处理小块数据",
            "避免频繁的UI更新",
            "使用队列管理数据流"
        ],
        
        "3. 用户体验": [
            "显示loading状态指示器",
            "提供停止/暂停功能",
            "处理网络异常情况",
            "保存部分结果防止丢失"
        ],
        
        "4. 资源管理": [
            "及时关闭流连接",
            "控制并发连接数量",
            "监控内存使用情况",
            "实现连接池管理"
        ]
    }
    
    for category, tips in optimization_tips.items():
        print(f"\n🎯 {category}")
        print("-" * 30)
        for tip in tips:
            print(f"   💡 {tip}")
    
    print("\n🧪 性能测试示例：")
    print("-" * 30)
    
    def measure_streaming_performance():
        """测量流式输出性能"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                return
            
            model = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com",
                streaming=True
            )
            
            prompt = ChatPromptTemplate.from_template("简单回答：{question}")
            chain = prompt | model
            
            # 测试指标
            first_chunk_time = None
            total_chunks = 0
            total_chars = 0
            start_time = time.time()
            
            print("📊 性能测试进行中...")
            
            for chunk in chain.stream({"question": "什么是人工智能？"}):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                
                total_chunks += 1
                if hasattr(chunk, 'content'):
                    total_chars += len(chunk.content)
            
            total_time = time.time() - start_time
            
            print(f"⏱️  首次响应时间: {first_chunk_time:.2f}秒")
            print(f"🔢 总块数: {total_chunks}")
            print(f"📝 总字符数: {total_chars}")
            print(f"⚡ 平均速度: {total_chars/total_time:.1f} 字符/秒")
            print(f"🕐 总耗时: {total_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ 性能测试失败：{e}")
    
    measure_streaming_performance()


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第6节总结 & 第7节预告")
    print("="*60)
    
    print("🎉 第6节你掌握了：")
    learned = [
        "✅ 理解流式输出的原理和优势",
        "✅ 掌握基础和异步流式API",
        "✅ 构建流式聊天应用",
        "✅ 学会性能优化技巧"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第7节预告：《内存与上下文管理》")
    print("你将学到：")
    next_topics = [
        "🧠 对话历史管理",
        "💾 不同类型的内存",
        "🔄 上下文窗口控制",
        "📚 长对话优化策略"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第6节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第6节")
    print("⚡ 流式输出与实时交互")
    print("📚 前置：完成第1-5节")
    
    # 1. 解释流式输出
    explain_streaming()
    
    # 2. 基础流式演示
    basic_streaming_demo()
    
    # 3. 异步流式演示
    async_streaming_demo()
    
    # 4. 流式聊天演示
    streaming_chat_demo()
    
    # 5. 性能优化
    performance_optimization()
    
    # 6. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第6节完成！")
    print("⚡ 你已经掌握了流式输出技术！")
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
    print("   📖 WebSocket 实时通信教程")
    print("   💻 异步编程最佳实践")
    print("   🎥 流式UI框架（如Streamlit）")