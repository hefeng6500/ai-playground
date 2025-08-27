"""
第7节：内存与上下文管理
=============================================

学习目标：
- 理解对话内存的重要性
- 掌握不同类型的内存机制
- 学会管理上下文窗口
- 了解长对话优化策略

前置知识：
- 完成第1-6节基础内容

重点概念：
- 内存让AI记住对话历史
- 不同场景需要不同的内存策略
- 上下文窗口有长度限制
"""

import os
from typing import List, Dict, Any


def explain_memory_importance():
    """
    解释内存管理的重要性
    """
    print("\n" + "="*60)
    print("🧠 为什么需要内存管理？")
    print("="*60)
    
    print("""
🤖 想象和一个失忆症患者对话：

没有内存的AI：
👤 用户: 我叫张三，今年25岁
🤖 AI: 你好！很高兴认识你
👤 用户: 我的年龄是多少？
🤖 AI: 抱歉，我不知道你的年龄 😵

有内存的AI：
👤 用户: 我叫张三，今年25岁  
🤖 AI: 你好张三！很高兴认识你
👤 用户: 我的年龄是多少？
🤖 AI: 根据你刚才说的，你今年25岁 😊

内存的作用：
✅ 记住用户信息
✅ 保持对话连贯性
✅ 理解上下文关系
✅ 提供个性化体验
    """)
    
    print("📚 内存类型概览：")
    memory_types = [
        "💬 对话缓冲内存：记住所有对话",
        "📊 对话摘要内存：压缩长对话",
        "🔢 对话窗口内存：只记住最近几轮",
        "🏷️  实体内存：记住重要实体信息",
        "🧠 知识图谱内存：结构化知识存储"
    ]
    
    for memory_type in memory_types:
        print(f"   {memory_type}")


def conversation_buffer_memory_demo():
    """
    对话缓冲内存演示
    """
    print("\n" + "="*60)
    print("💬 对话缓冲内存 (ConversationBufferMemory)")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.output_parsers import StrOutputParser
        from langchain.memory import ConversationBufferMemory
        from langchain_core.runnables import RunnablePassthrough
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 DEEPSEEK_API_KEY")
            return
        
        print("🎯 最简单的内存：记住所有对话")
        print("-" * 40)
        
        # 创建内存
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建带内存的提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友善的AI助手，能记住对话历史。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # 创建模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7
        )
        parser = StrOutputParser()
        
        def get_memory_variables():
            """获取内存变量"""
            return memory.load_memory_variables({})
        
        # 创建带内存的链条
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: get_memory_variables()["chat_history"]
            )
            | prompt
            | model
            | parser
        )
        
        # 模拟多轮对话
        conversations = [
            "你好，我叫李明，是一名程序员",
            "我最喜欢的编程语言是Python",
            "请问你还记得我的名字吗？",
            "我的职业是什么？",
            "我最喜欢的编程语言是什么？"
        ]
        
        print("🎭 开始多轮对话测试...")
        print("="*50)
        
        for i, user_input in enumerate(conversations, 1):
            print(f"回合 {i}:")
            print(f"👤 用户: {user_input}")
            
            # 调用链条
            response = chain.invoke({"input": user_input})
            print(f"🤖 AI: {response}")
            
            # 保存到内存
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)
            
            # 显示当前内存状态
            memory_vars = memory.load_memory_variables({})
            print(f"📚 内存中的对话数: {len(memory_vars['chat_history'])}")
            print("-"*50)
        
        print("\n📊 缓冲内存特点：")
        features = [
            "✅ 简单直接：直接存储所有对话",
            "✅ 完整保存：不丢失任何信息",
            "❌ 内存占用：对话越长占用越多",
            "❌ 长度限制：受模型上下文窗口限制"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
    except Exception as e:
        print(f"❌ 对话缓冲内存演示失败：{e}")


def conversation_summary_memory_demo():
    """
    对话摘要内存演示
    """
    print("\n" + "="*60)
    print("📊 对话摘要内存 (ConversationSummaryMemory)")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.memory import ConversationSummaryMemory
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 智能压缩：把长对话总结成摘要")
        print("-" * 40)
        
        # 创建模型（用于摘要）
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3  # 摘要用较低温度
        )
        
        # 创建摘要内存
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=False
        )
        
        print("🧪 测试摘要内存...")
        
        # 模拟一段较长的对话
        long_conversation = [
            ("Human", "你好，我叫王小明，是一名软件工程师"),
            ("AI", "你好王小明！很高兴认识你，听说你是软件工程师，很棒的职业！"),
            ("Human", "是的，我主要做Web开发，用的是React和Node.js"),
            ("AI", "React和Node.js是很流行的技术栈！你在这个领域工作多长时间了？"),
            ("Human", "大概3年了，目前在一家创业公司工作"),
            ("AI", "创业公司很有挑战性！你们公司主要做什么产品？"),
            ("Human", "我们做在线教育平台，帮助学生学习编程"),
            ("AI", "在线教育是个很有意义的领域！你们平台有多少用户了？"),
            ("Human", "目前有大约5万注册用户，还在快速增长中"),
            ("AI", "这个增长速度很不错！你在项目中主要负责哪些模块？")
        ]
        
        # 逐步添加对话到内存
        for speaker, message in long_conversation:
            if speaker == "Human":
                memory.chat_memory.add_user_message(message)
            else:
                memory.chat_memory.add_ai_message(message)
        
        print("💬 原始对话长度:", len(long_conversation), "轮")
        
        # 获取摘要
        summary = memory.load_memory_variables({})["chat_history"]
        print("\n📝 生成的摘要:")
        print(summary)
        
        print(f"\n📊 压缩效果:")
        original_length = sum(len(msg[1]) for msg in long_conversation)
        summary_length = len(summary)
        compression_ratio = (1 - summary_length / original_length) * 100
        
        print(f"   原始长度: {original_length} 字符")
        print(f"   摘要长度: {summary_length} 字符") 
        print(f"   压缩率: {compression_ratio:.1f}%")
        
        # 测试摘要内存的效果
        print("\n🧪 测试摘要记忆效果...")
        
        # 添加新的对话
        memory.chat_memory.add_user_message("请提醒我，我在哪家公司工作？")
        
        new_summary = memory.load_memory_variables({})["chat_history"]
        print("🔍 更新后的摘要:")
        print(new_summary)
        
        print("\n✅ 摘要内存优势：")
        advantages = [
            "💾 节省内存：大幅压缩对话内容",
            "🎯 保留关键信息：提取重要事实",
            "📈 支持长对话：突破上下文限制",
            "🧠 智能总结：AI自动提取要点"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ 对话摘要内存演示失败：{e}")


def conversation_window_memory_demo():
    """
    对话窗口内存演示
    """
    print("\n" + "="*60)
    print("🔢 对话窗口内存 (ConversationBufferWindowMemory)")
    print("="*60)
    
    try:
        from langchain.memory import ConversationBufferWindowMemory
        
        print("🎯 只记住最近几轮对话")
        print("-" * 40)
        
        # 创建窗口内存（只保留最近3轮对话）
        memory = ConversationBufferWindowMemory(
            k=3,  # 窗口大小
            memory_key="chat_history",
            return_messages=True
        )
        
        print(f"📊 窗口大小: {memory.k} 轮对话")
        
        # 模拟多轮对话
        conversations = [
            ("用户", "我叫张三"),
            ("AI", "你好张三！"),
            ("用户", "我今年30岁"),
            ("AI", "30岁正是好年华！"),
            ("用户", "我是医生"),
            ("AI", "医生是很崇高的职业！"),
            ("用户", "我住在北京"),
            ("AI", "北京是个很棒的城市！"),
            ("用户", "我有两个孩子"),
            ("AI", "有孩子真幸福！"),
        ]
        
        print("🧪 观察窗口内存的工作原理...")
        
        for i, (speaker, message) in enumerate(conversations):
            if speaker == "用户":
                memory.chat_memory.add_user_message(message)
            else:
                memory.chat_memory.add_ai_message(message)
            
            # 显示当前内存状态
            current_memory = memory.load_memory_variables({})["chat_history"]
            
            print(f"\n第{i+1}条消息后:")
            print(f"💬 添加: {speaker}: {message}")
            print(f"📚 内存中的对话数: {len(current_memory)}")
            
            if len(current_memory) > 0:
                print("🔍 当前内存内容:")
                for msg in current_memory[-2:]:  # 只显示最后2条
                    msg_type = "👤" if msg.type == "human" else "🤖"
                    print(f"   {msg_type} {msg.content}")
            
            if len(current_memory) >= memory.k * 2:  # k轮对话 = 2k条消息
                print("⚠️  已达到窗口上限，旧消息将被移除")
        
        print(f"\n📊 窗口内存特点：")
        features = [
            "⚡ 内存固定：占用内存恒定",
            "🎯 关注最近：保持对话即时性",
            "❌ 丢失历史：早期信息会丢失",
            "✅ 性能稳定：不受对话长度影响"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print(f"\n💡 适用场景：")
        use_cases = [
            "📱 客服聊天：只需记住当前问题",
            "🎮 游戏对话：关注当前任务",
            "📝 短文本生成：不需要长期记忆",
            "⚡ 高频交互：需要快速响应"
        ]
        
        for use_case in use_cases:
            print(f"   {use_case}")
        
    except Exception as e:
        print(f"❌ 对话窗口内存演示失败：{e}")


def entity_memory_demo():
    """
    实体内存演示
    """
    print("\n" + "="*60)
    print("🏷️  实体内存：记住重要信息")
    print("="*60)
    
    print("🎯 概念：智能提取和记住重要实体信息")
    print("-" * 40)
    
    # 模拟实体内存的工作原理
    class SimpleEntityMemory:
        """简化的实体内存实现"""
        
        def __init__(self):
            self.entities = {}
        
        def extract_entities(self, text: str) -> Dict[str, str]:
            """简单的实体提取（实际应用中会用NER模型）"""
            entities = {}
            
            # 简单的关键词匹配
            patterns = {
                "姓名": ["我叫", "我的名字是", "我是"],
                "年龄": ["我今年", "岁", "年龄"],
                "职业": ["我是", "我的工作是", "职业"],
                "地点": ["我住在", "我在", "城市"],
                "爱好": ["我喜欢", "我的爱好是", "喜好"]
            }
            
            for entity_type, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in text:
                        # 简单提取（实际会更复杂）
                        start = text.find(keyword)
                        if start != -1:
                            value = text[start:start+20].split("，")[0]
                            entities[entity_type] = value.replace(keyword, "").strip()
                            break
            
            return entities
        
        def update_entities(self, text: str):
            """更新实体信息"""
            new_entities = self.extract_entities(text)
            self.entities.update(new_entities)
        
        def get_entity_summary(self) -> str:
            """获取实体摘要"""
            if not self.entities:
                return "暂无已知信息"
            
            summary_parts = []
            for entity_type, value in self.entities.items():
                summary_parts.append(f"{entity_type}: {value}")
            
            return "已知信息 - " + ", ".join(summary_parts)
    
    print("🧪 测试实体内存...")
    
    entity_memory = SimpleEntityMemory()
    
    # 模拟对话
    user_inputs = [
        "你好，我叫李小红",
        "我今年28岁，是一名设计师",
        "我住在上海，喜欢画画",
        "对了，我还喜欢旅游",
        "请总结一下我的信息"
    ]
    
    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n回合 {i}:")
        print(f"👤 用户: {user_input}")
        
        # 更新实体信息
        entity_memory.update_entities(user_input)
        
        # 显示当前已知实体
        summary = entity_memory.get_entity_summary()
        print(f"🧠 实体记忆: {summary}")
        
        # 模拟AI回复
        if "总结" in user_input:
            print(f"🤖 AI: 根据我的记忆，{summary}")
        else:
            print(f"🤖 AI: 明白了！我已经记住这个信息。")
    
    print(f"\n✅ 实体内存优势：")
    advantages = [
        "🎯 重点突出：只记住重要信息",
        "📊 结构化：以实体-属性形式组织",
        "🔄 可更新：新信息覆盖旧信息",
        "💾 高效存储：占用内存很少"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")


def memory_strategy_comparison():
    """
    内存策略对比
    """
    print("\n" + "="*60)
    print("📊 内存策略对比分析")
    print("="*60)
    
    strategies = {
        "缓冲内存": {
            "优点": ["完整保存", "简单实现", "信息无损"],
            "缺点": ["内存增长", "长度限制", "处理缓慢"],
            "适用": ["短对话", "重要信息", "调试测试"],
            "性能": "★★☆☆☆"
        },
        
        "摘要内存": {
            "优点": ["智能压缩", "突破限制", "保留要点"],
            "缺点": ["信息丢失", "计算开销", "依赖LLM"],
            "适用": ["长对话", "知识问答", "客服系统"],
            "性能": "★★★☆☆"
        },
        
        "窗口内存": {
            "优点": ["内存固定", "性能稳定", "关注当前"],
            "缺点": ["丢失历史", "上下文断裂", "信息有限"],
            "适用": ["实时聊天", "游戏对话", "简单任务"],
            "性能": "★★★★☆"
        },
        
        "实体内存": {
            "优点": ["结构化", "高效存储", "重点突出"],
            "缺点": ["提取复杂", "信息有限", "依赖NER"],
            "适用": ["个人助手", "CRM系统", "用户画像"],
            "性能": "★★★★★"
        }
    }
    
    print("📋 详细对比表：")
    print("-" * 80)
    
    for strategy, details in strategies.items():
        print(f"\n🔍 {strategy}")
        print(f"   ✅ 优点: {', '.join(details['优点'])}")
        print(f"   ❌ 缺点: {', '.join(details['缺点'])}")
        print(f"   🎯 适用: {', '.join(details['适用'])}")
        print(f"   ⚡ 性能: {details['性能']}")
    
    print(f"\n💡 选择建议：")
    recommendations = [
        "🔰 新手项目：使用缓冲内存或窗口内存",
        "📈 生产环境：结合使用摘要+实体内存",
        "⚡ 高性能要求：优选窗口内存",
        "🧠 智能应用：推荐摘要+实体组合"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第7节总结 & 第8节预告")
    print("="*60)
    
    print("🎉 第7节你掌握了：")
    learned = [
        "✅ 理解内存管理的重要性",
        "✅ 掌握4种主要内存类型",
        "✅ 学会选择合适的内存策略",
        "✅ 了解不同场景的应用方案"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第8节预告：《Runnable 接口深入》")
    print("你将学到：")
    next_topics = [
        "🔧 Runnable 接口详解",
        "🎯 自定义 Runnable 组件",
        "🔗 复杂链条构建",
        "⚡ 并行和分支处理"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第7节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第7节")
    print("🧠 内存与上下文管理")
    print("📚 前置：完成第1-6节")
    
    # 1. 解释内存重要性
    explain_memory_importance()
    
    # 2. 对话缓冲内存
    conversation_buffer_memory_demo()
    
    # 3. 对话摘要内存
    conversation_summary_memory_demo()
    
    # 4. 对话窗口内存
    conversation_window_memory_demo()
    
    # 5. 实体内存
    entity_memory_demo()
    
    # 6. 内存策略对比
    memory_strategy_comparison()
    
    # 7. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第7节完成！")
    print("🧠 你已经掌握了内存管理技术！")
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
    print("   📖 LangChain Memory 官方文档")
    print("   💻 对话系统设计模式")
    print("   🧠 上下文管理最佳实践")