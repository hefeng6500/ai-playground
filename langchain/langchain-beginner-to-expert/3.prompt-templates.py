"""
第3节：提示模板进阶
=============================================

学习目标：
- 深入理解提示模板的设计原理
- 掌握系统消息和用户消息的使用
- 学会 Few-shot 学习模板
- 了解动态模板和条件逻辑

前置知识：
- 完成第1-2节基础内容

重点概念：
- 提示模板是 AI 的"指令手册"
- 好的模板 = 好的结果
"""

import os
from datetime import datetime


def explain_prompt_importance():
    """
    解释提示模板的重要性
    """
    print("\n" + "="*60)
    print("🎯 为什么提示模板如此重要？")
    print("="*60)
    
    print("""
🎭 想象 AI 是一个演员：

没有好剧本：演员不知道怎么演，表演很奇怪
有好剧本：演员知道角色、背景、情感，演出精彩

在 AI 中：
- 差的提示 = 差的回答
- 好的提示 = 好的回答
- 详细的提示 = 准确的回答
    """)


def system_vs_human_messages():
    """
    系统消息和用户消息的区别
    """
    print("\n" + "="*60)
    print("🎭 系统消息 vs 用户消息")
    print("="*60)
    
    print("""
系统消息 = 老板给员工的工作说明书
用户消息 = 客户的具体问题

系统消息定义：AI 的角色、行为规范、回答风格
用户消息包含：具体的问题和需求
    """)
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 DEEPSEEK_API_KEY")
            return
        
        # 创建模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7
        )
        parser = StrOutputParser()
        
        # 对比实验：有无系统消息
        template_without_system = ChatPromptTemplate.from_template("{question}")
        
        template_with_system = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的编程导师，特点：
1. 用简单语言解释复杂概念
2. 总是提供代码示例
3. 语气友善且专业"""),
            ("human", "{question}")
        ])
        
        chain_without = template_without_system | model | parser
        chain_with = template_with_system | model | parser
        
        test_question = "什么是递归？"
        
        print(f"测试问题：{test_question}")
        
        print("\n🔴 无系统消息：")
        answer1 = chain_without.invoke({"question": test_question})
        print(answer1[:150] + "...")
        
        print("\n🟢 有系统消息：")
        answer2 = chain_with.invoke({"question": test_question})
        print(answer2[:150] + "...")
        
        print("\n✅ 有系统消息的回答更专业、结构化！")
        
    except Exception as e:
        print(f"❌ 实验失败：{e}")


def few_shot_learning():
    """
    Few-shot 学习示例
    """
    print("\n" + "="*60)
    print("🎯 Few-shot 学习：教 AI 如何回答")
    print("="*60)
    
    print("""
🎓 Few-shot = 给几个标准答案作示例，让 AI 学习模式

比如教 AI 写评价：
示例1：输入"质量很好" → 输出"★★★★★ 产品质量优秀"
示例2：输入"价格贵了" → 输出"★★★☆☆ 性价比一般"
然后输入"发货很快" → AI 学会输出类似格式
    """)
    
    try:
        from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 API 密钥")
            return
        
        # 定义示例
        examples = [
            {
                "customer": "你们的产品质量怎么样？",
                "assistant": "感谢咨询！我们产品都经过严格质检，质量可靠。提供7天无理由退换服务。"
            },
            {
                "customer": "发货速度快吗？",
                "assistant": "我们承诺24小时内发货！与顺丰、京东合作，通常1-3个工作日收到。"
            }
        ]
        
        # 创建示例模板
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{customer}"),
            ("ai", "{assistant}")
        ])
        
        # 创建 few-shot 模板
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )
        
        # 完整模板
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业客服代表，回答要礼貌、准确、详细。"),
            few_shot_prompt,
            ("human", "{customer_question}")
        ])
        
        # 创建链条
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3
        )
        parser = StrOutputParser()
        chain = final_prompt | model | parser
        
        # 测试新问题
        test_question = "你们支持货到付款吗？"
        print(f"🤖 测试问题：{test_question}")
        
        answer = chain.invoke({"customer_question": test_question})
        print(f"🤖 AI回复：{answer}")
        
        print("\n✅ Few-shot 让 AI 学会了客服的回答风格！")
        
    except Exception as e:
        print(f"❌ Few-shot 示例失败：{e}")


def dynamic_templates():
    """
    动态模板示例
    """
    print("\n" + "="*60)
    print("🔀 动态模板：根据情况变化")
    print("="*60)
    
    print("""
🎪 动态模板 = 根据用户信息调整回答方式

例如：
- 初学者 → 用简单语言 + 多举例子
- 专家级 → 直接讲重点 + 技术细节
    """)
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        def create_user_prompt(user_level):
            """根据用户级别创建提示"""
            if user_level == "beginner":
                style = "用最简单的语言，多用比喻和例子"
            elif user_level == "intermediate":
                style = "用适中的技术语言，提供实践建议"
            else:  # expert
                style = "用专业术语，直接讲重点"
            
            return ChatPromptTemplate.from_messages([
                ("system", f"你是编程导师。回答风格：{style}"),
                ("human", "{question}")
            ])
        
        # 创建模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.6
        )
        parser = StrOutputParser()
        
        # 测试不同级别
        test_question = "什么是API？"
        levels = ["beginner", "expert"]
        
        for level in levels:
            prompt = create_user_prompt(level)
            chain = prompt | model | parser
            answer = chain.invoke({"question": test_question})
            
            print(f"\n👤 {level}级用户回答：")
            print(answer[:200] + "...")
        
        print("\n✅ 可以看到不同级别的回答风格完全不同！")
        
    except Exception as e:
        print(f"❌ 动态模板失败：{e}")


def template_best_practices():
    """
    模板设计最佳实践
    """
    print("\n" + "="*60)
    print("🏆 模板设计最佳实践")
    print("="*60)
    
    practices = {
        "1. 清晰角色定义": {
            "好": "你是有10年经验的Python工程师",
            "坏": "请回答编程问题"
        },
        "2. 具体任务描述": {
            "好": "写一个Python函数，输入两个数字，返回它们的和",
            "坏": "写个函数"
        },
        "3. 明确输出格式": {
            "好": "用JSON格式返回，包含name、age、city字段",
            "坏": "给我用户信息"
        },
        "4. 提供示例": {
            "好": "格式如：问题：xxx 答案：xxx 置信度：85%",
            "坏": "回答要有置信度"
        }
    }
    
    for practice, examples in practices.items():
        print(f"\n🎯 {practice}")
        print(f"   ✅ 好例子：{examples['好']}")
        print(f"   ❌ 坏例子：{examples['坏']}")


def common_template_patterns():
    """
    常用模板模式
    """
    print("\n" + "="*60)
    print("📚 常用模板模式")
    print("="*60)
    
    patterns = {
        "分析型": '''你是数据分析专家。按步骤分析：
1. 数据概览 2. 关键发现 3. 深入分析 4. 结论建议
分析内容：{content}''',
        
        "教学型": '''你是{subject}老师，给{level}学生授课。
要求：简单易懂、提供例子、鼓励思考
问题：{question}''',
        
        "问题解决型": '''你是问题解决专家。按STAR方法：
Situation（情况）Task（任务）Action（行动）Result（结果）
问题：{problem}'''
    }
    
    for name, template in patterns.items():
        print(f"\n📝 {name}模板：")
        print(template)


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第3节总结 & 第4节预告")
    print("="*60)
    
    print("🎉 第3节你掌握了：")
    learned = [
        "✅ 理解提示模板的重要性",
        "✅ 掌握系统消息和用户消息",
        "✅ 学会 Few-shot 学习技术",
        "✅ 能够创建动态模板",
        "✅ 了解最佳实践和常用模式"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第4节预告：《多模型接入与切换》")
    print("你将学到：")
    next_topics = [
        "🔌 接入不同 LLM 服务",
        "🔄 模型无缝切换",
        "💰 成本优化策略",
        "⚖️  模型选择指南"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第3节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第3节")
    print("📝 提示模板进阶")
    print("📚 前置：完成第1-2节")
    
    # 1. 解释重要性
    explain_prompt_importance()
    
    # 2. 系统消息 vs 用户消息
    system_vs_human_messages()
    
    # 3. Few-shot 学习
    few_shot_learning()
    
    # 4. 动态模板
    dynamic_templates()
    
    # 5. 最佳实践
    template_best_practices()
    
    # 6. 常用模式
    common_template_patterns()
    
    # 7. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第3节完成！")
    print("🚀 你已经是提示模板专家了！")
    print("="*60)


if __name__ == "__main__":
    # 检查环境
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  未检测到 DEEPSEEK_API_KEY")
        print("💡 请先完成第1节的环境配置")
        import getpass
        temp_key = getpass.getpass("请输入 DeepSeek API Key: ")
        if temp_key:
            os.environ["DEEPSEEK_API_KEY"] = temp_key
    
    # 运行主程序
    main()
    
    print("\n🔗 本节参考资源：")
    print("   📖 Prompt Engineering Guide：https://www.promptingguide.ai/")
    print("   🎥 提示工程视频教程")
    print("   💻 练习：构建不同领域的专业模板")