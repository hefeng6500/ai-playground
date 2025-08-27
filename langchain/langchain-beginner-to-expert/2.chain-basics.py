"""
第2节：理解链式调用（LCEL）
=============================================

学习目标：
- 理解什么是链式调用（Chain）
- 掌握 LCEL (LangChain Expression Language) 语法
- 学会使用管道操作符 | 连接不同组件
- 理解数据在链中的流转过程

前置知识：
- 完成第1节：LangChain 简介与环境搭建
- 熟悉 Python 基础语法

重点概念：
- 链式调用就像工厂流水线，每个环节处理一部分工作
- LCEL 是 LangChain 的"胶水"，用来粘合不同组件
- 管道操作符 | 让代码更简洁易读
"""

import os
from typing import Dict, Any


def explain_chain_concept():
    """
    用生活化的例子解释链式调用
    """
    print("\n" + "="*60)
    print("🔗 什么是链式调用？")
    print("="*60)
    
    print("""
🏭 想象一个汽车制造工厂：

传统方式（不用链）：
┌─────────────────────────────────────────────────────────┐
│ 你要亲自：                                              │
│ 1. 🔧 取零件                                           │
│ 2. 🔨 组装车身                                         │
│ 3. 🎨 喷漆                                             │
│ 4. 🔍 质检                                             │
│ 5. 📦 包装                                             │
│ 每一步都要手动操作，累死累活！                           │
└─────────────────────────────────────────────────────────┘

链式调用方式（LangChain）：
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 🔧 零件  │───▶│ 🔨 组装  │───▶│ 🎨 喷漆  │───▶│ 📦 包装  │
│   站    │    │   站    │    │   站    │    │   站    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
       ↓            ↓            ↓            ↓
    输入原料      组装车身      美化外观      最终产品

在 LangChain 中：
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 📝 输入  │───▶│ 🧠 AI   │───▶│ 🔧 处理  │───▶│ 📤 输出  │
│  文本   │    │  模型   │    │  结果   │    │  答案   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
    """)
    
    print("🎯 链式调用的核心优势：")
    advantages = [
        "🔄 可重用：每个环节都可以在其他地方使用",
        "🛠️  易维护：修改一个环节不影响其他部分",
        "📈 可扩展：随时可以添加新的处理环节",
        "🔍 易调试：可以单独测试每个环节"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")


def explain_lcel():
    """
    详细解释 LCEL 语法
    """
    print("\n" + "="*60)
    print("⚡ LCEL：LangChain 表达式语言")
    print("="*60)
    
    print("""
🤔 什么是 LCEL？

LCEL = LangChain Expression Language
就像是 LangChain 的"专用语法"，让你可以用很简洁的方式组合不同功能。

🔧 核心语法：管道操作符 |

就像 Linux 命令行一样：
cat file.txt | grep "error" | sort

在 LangChain 中：
prompt | model | output_parser

意思是：输入 → 提示模板 → AI模型 → 结果解析 → 输出
    """)
    
    print("📚 LCEL 的基本组件：")
    components = {
        "Prompt（提示模板）": "告诉 AI 要做什么，像给员工的工作指令",
        "Model（模型）": "AI 大脑，负责思考和生成回答",
        "OutputParser（输出解析器）": "把 AI 的回答整理成我们想要的格式",
        "Runnable（可运行对象）": "所有可以被串联的组件的基类"
    }
    
    for component, description in components.items():
        print(f"   🧩 {component}: {description}")


def basic_chain_example():
    """
    最基础的链式调用示例
    """
    print("\n" + "="*60)
    print("🚀 基础链式调用示例")
    print("="*60)
    
    try:
        # 导入必要的模块
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 检查 API 密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先设置 DEEPSEEK_API_KEY 环境变量")
            print("💡 回到第1节学习如何配置 API 密钥")
            return
        
        print("🔧 第1步：创建提示模板")
        print("-" * 30)
        
        # 创建提示模板 - 这是链的第一环
        prompt = ChatPromptTemplate.from_template(
            "你是一个友善的助手。请用简单易懂的语言回答用户的问题：{question}"
        )
        
        print("✅ 提示模板创建成功")
        print(f"   模板内容：{prompt.template}")
        
        print("\n🧠 第2步：创建 AI 模型")
        print("-" * 30)
        
        # 创建 AI 模型 - 这是链的第二环
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7,
            max_tokens=500
        )
        
        print("✅ AI 模型创建成功")
        print(f"   模型类型：{model.model_name}")
        
        print("\n🔧 第3步：创建输出解析器")
        print("-" * 30)
        
        # 创建输出解析器 - 这是链的第三环
        output_parser = StrOutputParser()
        
        print("✅ 输出解析器创建成功")
        print("   功能：将 AI 回答转为纯文本字符串")
        
        print("\n🔗 第4步：组装链条")
        print("-" * 30)
        print("魔法时刻到了！用 | 操作符连接所有组件：")
        
        # 这里就是 LCEL 的核心：用 | 连接组件
        chain = prompt | model | output_parser
        
        print("✅ 链条组装完成！")
        print("   链条结构：prompt | model | output_parser")
        
        print("\n🎯 第5步：测试链条")
        print("-" * 30)
        
        # 测试我们的链条
        test_question = "什么是机器学习？请用简单的话解释。"
        print(f"🤖 输入问题：{test_question}")
        
        # 调用链条 - 注意这里传入的是字典，键名要和模板中的变量名一致
        result = chain.invoke({"question": test_question})
        
        print("\n" + "="*50)
        print("🎉 链条执行结果：")
        print("="*50)
        print(result)
        
        # 解释整个流程
        print(f"\n📊 执行流程解析：")
        print("1. 📝 输入：{'question': '什么是机器学习？...'}")
        print("2. 🔧 提示模板处理：生成完整的对话消息")
        print("3. 🧠 AI 模型处理：理解问题并生成回答")
        print("4. 🔧 输出解析：提取纯文本内容")
        print("5. 📤 最终输出：干净的字符串结果")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败：{e}")
        print("💡 请安装必要的包：pip install langchain-openai langchain-core")
        return False
    except Exception as e:
        print(f"❌ 执行失败：{e}")
        return False


def advanced_chain_example():
    """
    进阶链式调用：带条件逻辑的链
    """
    print("\n" + "="*60)
    print("🚀 进阶示例：多步骤链条")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 API 密钥")
            return
        
        print("🎯 场景：智能翻译助手")
        print("功能：自动检测语言 → 翻译 → 优化表达")
        
        # 第一步：语言检测
        detect_prompt = ChatPromptTemplate.from_template(
            "请判断这段文字是什么语言，只回答语言名称（如：中文、英文、日文等）：{text}"
        )
        
        # 第二步：翻译
        translate_prompt = ChatPromptTemplate.from_template(
            """请将以下{source_lang}文本翻译成{target_lang}：
            
原文：{text}

要求：
1. 翻译要准确自然
2. 保持原文语气
3. 如果有专业术语要准确翻译"""
        )
        
        # 第三步：优化表达
        polish_prompt = ChatPromptTemplate.from_template(
            "请优化以下翻译，让它更自然流畅：{translation}"
        )
        
        # 创建模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3  # 翻译任务用较低的温度
        )
        
        # 创建解析器
        parser = StrOutputParser()
        
        # 自定义处理函数
        def prepare_translation_input(x):
            """
            准备翻译输入
            x 是前一步的输出结果
            """
            source_lang = x.strip()
            target_lang = "英文" if "中文" in source_lang else "中文"
            
            return {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": x  # 这里需要保存原始文本
            }
        
        print("\n🔧 构建复杂链条...")
        
        # 第一条链：检测语言
        detect_chain = detect_prompt | model | parser
        
        # 第二条链：翻译
        translate_chain = translate_prompt | model | parser
        
        # 第三条链：优化
        polish_chain = polish_prompt | model | parser
        
        print("✅ 子链条创建完成")
        
        # 测试简单的检测链
        test_text = "你好，世界！这是一个简单的测试。"
        print(f"\n🧪 测试文本：{test_text}")
        
        # 检测语言
        detected_lang = detect_chain.invoke({"text": test_text})
        print(f"🔍 检测到的语言：{detected_lang}")
        
        # 根据检测结果进行翻译
        target_lang = "英文" if "中文" in detected_lang else "中文"
        translation = translate_chain.invoke({
            "source_lang": detected_lang,
            "target_lang": target_lang,
            "text": test_text
        })
        print(f"🔄 翻译结果：{translation}")
        
        # 优化翻译
        polished = polish_chain.invoke({"translation": translation})
        print(f"✨ 优化后：{polished}")
        
        print("\n📈 链条执行流程：")
        print("1. 🔍 detect_chain: 检测输入文本的语言")
        print("2. 🔄 translate_chain: 根据检测结果进行翻译")
        print("3. ✨ polish_chain: 优化翻译结果")
        print("4. 📤 输出最终的高质量翻译")
        
    except Exception as e:
        print(f"❌ 进阶示例执行失败：{e}")


def chain_debugging():
    """
    链条调试技巧
    """
    print("\n" + "="*60)
    print("🔍 链条调试技巧")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 API 密钥")
            return
        
        print("🎯 调试技巧1：单独测试每个组件")
        print("-" * 40)
        
        # 创建组件
        prompt = ChatPromptTemplate.from_template("翻译这句话：{text}")
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.1
        )
        parser = StrOutputParser()
        
        # 测试输入
        test_input = {"text": "Hello World"}
        
        # 1. 单独测试提示模板
        print("📝 测试提示模板：")
        prompt_result = prompt.invoke(test_input)
        print(f"   输入：{test_input}")
        print(f"   输出：{prompt_result}")
        
        # 2. 测试模型
        print("\n🧠 测试模型：")
        model_result = model.invoke(prompt_result)
        print(f"   输入类型：{type(prompt_result)}")
        print(f"   输出类型：{type(model_result)}")
        print(f"   输出内容：{model_result.content[:100]}...")
        
        # 3. 测试解析器
        print("\n🔧 测试解析器：")
        parser_result = parser.invoke(model_result)
        print(f"   输入类型：{type(model_result)}")
        print(f"   输出类型：{type(parser_result)}")
        print(f"   最终结果：{parser_result}")
        
        print("\n🎯 调试技巧2：添加中间输出")
        print("-" * 40)
        
        def debug_function(x):
            """调试用的中间函数"""
            print(f"🔍 调试点 - 数据类型：{type(x)}")
            print(f"🔍 调试点 - 数据内容：{str(x)[:100]}...")
            return x
        
        # 带调试输出的链条
        debug_chain = (
            prompt | 
            RunnableLambda(debug_function) |  # 插入调试点
            model | 
            RunnableLambda(debug_function) |  # 再插入一个调试点
            parser
        )
        
        print("\n执行带调试的链条：")
        result = debug_chain.invoke(test_input)
        print(f"\n🎉 最终结果：{result}")
        
        print("\n🎯 调试技巧3：错误处理")
        print("-" * 40)
        
        def safe_invoke_chain(chain, input_data):
            """安全执行链条的函数"""
            try:
                result = chain.invoke(input_data)
                print(f"✅ 执行成功：{result}")
                return result
            except Exception as e:
                print(f"❌ 执行失败：{e}")
                print(f"🔍 输入数据：{input_data}")
                print(f"🔍 错误类型：{type(e).__name__}")
                return None
        
        # 测试错误处理
        safe_invoke_chain(debug_chain, test_input)
        
    except Exception as e:
        print(f"❌ 调试示例失败：{e}")


def performance_tips():
    """
    性能优化技巧
    """
    print("\n" + "="*60)
    print("⚡ 性能优化技巧")
    print("="*60)
    
    tips = {
        "1. 合理设置参数": [
            "temperature: 创造性任务用 0.7-0.9，精确任务用 0.1-0.3",
            "max_tokens: 根据需要设置，避免浪费",
            "timeout: 设置合理的超时时间"
        ],
        
        "2. 批量处理": [
            "使用 batch() 方法处理多个输入",
            "避免在循环中单独调用 invoke()",
            "考虑使用异步方法 ainvoke() 和 abatch()"
        ],
        
        "3. 缓存策略": [
            "对相同输入的结果进行缓存",
            "使用 LangChain 的内置缓存功能",
            "考虑外部缓存（Redis 等）"
        ],
        
        "4. 链条优化": [
            "避免不必要的中间步骤",
            "合并相似的处理逻辑",
            "使用流式输出提升用户体验"
        ]
    }
    
    for category, tip_list in tips.items():
        print(f"\n🎯 {category}")
        print("-" * 30)
        for tip in tip_list:
            print(f"   💡 {tip}")


def common_mistakes():
    """
    常见错误和解决方案
    """
    print("\n" + "="*60)
    print("⚠️  常见错误和解决方案")
    print("="*60)
    
    mistakes = {
        "❌ 错误1：变量名不匹配": {
            "问题": "提示模板中的变量名和输入字典的键名不一致",
            "示例": "模板用 {question}，输入用 {'query': '..'}",
            "解决": "确保变量名完全一致，区分大小写"
        },
        
        "❌ 错误2：类型不匹配": {
            "问题": "链条期望的输入类型和实际提供的不一致",
            "示例": "某个组件期望字典，但收到了字符串",
            "解决": "使用 RunnableLambda 进行类型转换"
        },
        
        "❌ 错误3：API 调用失败": {
            "问题": "网络问题、API 密钥错误、余额不足等",
            "示例": "连接超时、401 错误等",
            "解决": "添加重试机制、检查网络、验证密钥"
        },
        
        "❌ 错误4：链条太复杂": {
            "问题": "一条链包含太多步骤，难以调试",
            "示例": "prompt | model | parser | processor | validator | ...",
            "解决": "拆分成多个简单的子链，逐步组合"
        }
    }
    
    for error, details in mistakes.items():
        print(f"\n{error}")
        print("-" * 40)
        print(f"   🔍 问题：{details['问题']}")
        print(f"   📝 示例：{details['示例']}")
        print(f"   💡 解决：{details['解决']}")


def next_lesson_preview():
    """
    下一课预告
    """
    print("\n" + "="*60)
    print("🎓 第2节总结 & 第3节预告")
    print("="*60)
    
    print("🎉 第2节你学会了：")
    learned = [
        "✅ 理解链式调用的核心概念",
        "✅ 掌握 LCEL 管道语法 |",
        "✅ 创建基础的三段式链条",
        "✅ 学会链条调试技巧",
        "✅ 了解性能优化方法"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第3节预告：《提示模板进阶》")
    print("你将学到：")
    next_topics = [
        "🎨 更复杂的提示模板设计",
        "💬 系统消息 vs 用户消息",
        "🔀 条件模板和动态内容",
        "📋 Few-shot 示例模板",
        "🛠️  自定义模板变量处理"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")
    
    print("\n💪 课后练习：")
    exercises = [
        "修改基础链条，添加不同的提示模板",
        "尝试创建一个4步骤的复杂链条",
        "实验不同的 temperature 值对结果的影响",
        "练习链条调试，故意制造一些错误然后修复"
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"   {i}. {exercise}")


def main():
    """
    主函数：第2节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第2节")
    print("🔗 理解链式调用（LCEL）")
    print("📚 前置：完成第1节基础环境搭建")
    
    # 1. 核心概念解释
    explain_chain_concept()
    
    # 2. LCEL 语法详解
    explain_lcel()
    
    # 3. 基础链式调用示例
    basic_chain_example()
    
    # 4. 进阶多步骤链条
    advanced_chain_example()
    
    # 5. 调试技巧
    chain_debugging()
    
    # 6. 性能优化
    performance_tips()
    
    # 7. 常见错误
    common_mistakes()
    
    # 8. 下一课预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第2节完成！")
    print("🚀 你已经掌握了 LangChain 的核心：链式调用！")
    print("💪 继续前进，向模板专家进阶！")
    print("="*60)


# 实用代码片段
"""
🧰 常用代码模板

1. 基础三段式链条：
   chain = prompt | model | output_parser

2. 带调试的链条：
   chain = prompt | RunnableLambda(debug_func) | model | parser

3. 安全执行链条：
   try:
       result = chain.invoke(input_data)
   except Exception as e:
       print(f"Error: {e}")

4. 批量处理：
   results = chain.batch([input1, input2, input3])

5. 异步执行：
   result = await chain.ainvoke(input_data)
"""


if __name__ == "__main__":
    # 检查环境
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  未检测到 DEEPSEEK_API_KEY")
        print("💡 请先完成第1节的环境配置")
        print("🔗 或临时设置：")
        import getpass
        temp_key = getpass.getpass("请输入 DeepSeek API Key: ")
        if temp_key:
            os.environ["DEEPSEEK_API_KEY"] = temp_key
    
    # 运行主程序
    main()
    
    print("\n🔗 本节参考资源：")
    print("   📖 LCEL 官方文档：https://python.langchain.com/docs/expression_language/")
    print("   🎥 链式调用视频教程：搜索 'LangChain LCEL tutorial'")
    print("   💻 实践项目：尝试构建一个多语言翻译链")
    print("   🤝 社区讨论：LangChain Discord/GitHub Discussions")