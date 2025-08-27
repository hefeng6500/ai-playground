"""
第1节：LangChain 简介与环境搭建
=============================================

学习目标：
- 理解 LangChain 是什么，能解决什么问题
- 学会搭建 LangChain 开发环境
- 完成第一个 Hello World 程序
- 掌握基本的错误处理和调试方法

前置知识：
- Python 基础语法
- 对 AI 大语言模型有基本了解

适合人群：
- 零基础小白
- 想要快速上手 AI 应用开发的开发者
"""

import os
import getpass
from typing import Optional


def explain_langchain():
    """
    用通俗易懂的方式解释 LangChain 是什么
    """
    print("\n" + "="*60)
    print("🤖 什么是 LangChain？")
    print("="*60)
    
    explanation = """
    想象一下，你要做一道复杂的菜：
    
    1. 传统方式：你需要自己买菜、洗菜、切菜、炒菜、装盘...
       每一步都要亲自动手，很累很复杂。
    
    2. LangChain 方式：就像有了一个智能厨房助手，
       你只需要说"我想要宫保鸡丁"，它会：
       - 自动规划制作流程
       - 调用不同的"工具"（切菜机、炒锅等）
       - 把各个步骤串联起来
       - 最终给你一盘美味的宫保鸡丁
    
    LangChain 就是这样一个"AI 应用开发助手"，它帮你：
    ✅ 连接各种 AI 模型（GPT、Claude、国产大模型等）
    ✅ 处理复杂的对话流程
    ✅ 整合不同的数据源
    ✅ 构建智能应用，而不用从零开始写代码
    
    简单说：LangChain = 乐高积木 for AI 应用开发
    """
    
    print(explanation)
    print("\n🎯 核心优势：")
    advantages = [
        "🔗 链式调用：像搭积木一样组合 AI 功能",
        "🔌 模型无关：支持各种大语言模型",
        "📚 丰富生态：内置大量实用工具",
        "🚀 快速开发：几行代码实现复杂功能"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")


def setup_environment():
    """
    环境搭建指南
    """
    print("\n" + "="*60)
    print("⚙️  环境搭建步骤")
    print("="*60)
    
    print("\n📦 第一步：安装必要的包")
    print("-" * 30)
    
    packages = [
        "langchain-openai",      # OpenAI 兼容的模型接口
        "langchain-core",        # LangChain 核心功能
        "langchain-community",   # 社区扩展
        "python-dotenv",         # 环境变量管理
        "pydantic"               # 数据验证
    ]
    
    print("需要安装的包：")
    for i, package in enumerate(packages, 1):
        print(f"   {i}. {package}")
    
    print(f"\n💻 安装命令：")
    print("   pip install " + " ".join(packages))
    
    print("\n🔑 第二步：配置 API 密钥")
    print("-" * 30)
    print("""
    你需要申请以下任一服务的 API 密钥：
    
    1. DeepSeek（推荐，便宜好用）：
       - 官网：https://platform.deepseek.com/
       - 注册并获取 API Key
    
    2. 硅基流动（SiliconFlow）：
       - 官网：https://siliconflow.cn/
       - 提供多种开源模型
    
    3. OpenAI（功能最强，但较贵）：
       - 官网：https://platform.openai.com/
       - 需要国外信用卡
    """)


def setup_api_key() -> Optional[str]:
    """
    引导用户设置 API 密钥
    这里我们使用 DeepSeek 作为示例，因为它便宜且好用
    """
    print("\n🔐 API 密钥配置")
    print("-" * 30)
    
    # 首先检查环境变量中是否已经有密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if api_key:
        print("✅ 检测到已配置的 DEEPSEEK_API_KEY")
        return api_key
    
    print("❌ 未检测到 DEEPSEEK_API_KEY 环境变量")
    print("\n有两种配置方法：")
    print("1. 临时配置（仅本次运行有效）")
    print("2. 永久配置（推荐）")
    
    print("\n选择 1：临时配置")
    print("请输入你的 DeepSeek API Key:")
    temp_key = getpass.getpass("API Key (输入时不会显示): ")
    
    if temp_key:
        # 临时设置环境变量
        os.environ["DEEPSEEK_API_KEY"] = temp_key
        print("✅ 临时 API Key 配置成功！")
        
        print("\n💡 如何永久配置（下次就不用重复输入了）：")
        print("   1. 在项目根目录创建 .env 文件")
        print("   2. 在文件中添加：DEEPSEEK_API_KEY=你的密钥")
        print("   3. 使用 python-dotenv 加载环境变量")
        
        return temp_key
    
    return None


def first_hello_world():
    """
    第一个 LangChain 程序：Hello World
    """
    print("\n" + "="*60)
    print("🚀 第一个 LangChain 程序")
    print("="*60)
    
    try:
        # 导入必要的模块
        from langchain_openai import ChatOpenAI
        
        print("✅ 成功导入 LangChain 模块")
        
        # 检查 API 密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 API 密钥")
            return False
        
        # 创建 LLM 实例
        # 这里我们使用 DeepSeek 的模型，它兼容 OpenAI 的接口
        llm = ChatOpenAI(
            model="deepseek-chat",                    # 模型名称
            openai_api_key=api_key,                   # API 密钥
            openai_api_base="https://api.deepseek.com",  # API 基础 URL
            temperature=0.7,                           # 创造性参数（0-1，越高越有创意）
            max_tokens=1000                           # 最大输出长度
        )
        
        print("✅ 成功创建 LLM 实例")
        
        # 发送第一个请求
        print("\n🤖 发送第一个 AI 请求...")
        response = llm.invoke("你好！请简单介绍一下你自己，并解释什么是人工智能。")
        
        print("\n" + "="*60)
        print("🎉 AI 回复：")
        print("="*60)
        print(response.content)
        
        print("\n✅ 恭喜！你的第一个 LangChain 程序运行成功！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入模块失败：{e}")
        print("💡 解决方案：请安装必要的包")
        print("   pip install langchain-openai langchain-core")
        return False
        
    except Exception as e:
        print(f"❌ 程序执行失败：{e}")
        print("\n🔍 常见问题排查：")
        
        error_solutions = {
            "API key": [
                "检查 API 密钥是否正确",
                "确认 API 密钥有足够的余额",
                "检查网络连接是否正常"
            ],
            "network": [
                "检查网络连接",
                "确认能访问 api.deepseek.com",
                "如果在国内，可能需要配置代理"
            ],
            "timeout": [
                "网络超时，请稍后重试",
                "可以增加 timeout 参数",
                "检查网络稳定性"
            ]
        }
        
        for error_type, solutions in error_solutions.items():
            if error_type.lower() in str(e).lower():
                print(f"\n🛠️  针对 '{error_type}' 错误的解决方案：")
                for i, solution in enumerate(solutions, 1):
                    print(f"   {i}. {solution}")
                break
        
        return False


def understanding_concepts():
    """
    理解 LangChain 核心概念
    """
    print("\n" + "="*60)
    print("📚 核心概念理解")
    print("="*60)
    
    concepts = {
        "LLM (Large Language Model)": {
            "解释": "大语言模型，就是 AI 大脑，负责理解和生成文本",
            "比喻": "像一个博学的助手，可以回答各种问题",
            "例子": "ChatOpenAI、Claude、文心一言等"
        },
        
        "Chain (链)": {
            "解释": "把多个步骤串联起来，形成一个完整的处理流程",
            "比喻": "像工厂的流水线，每个环节处理一部分任务",
            "例子": "输入处理 -> AI 推理 -> 结果格式化"
        },
        
        "Prompt (提示)": {
            "解释": "给 AI 的指令，告诉它要做什么",
            "比喻": "像给助手的工作说明书",
            "例子": "'请总结这篇文章的要点'"
        },
        
        "Token": {
            "解释": "文本的最小单位，AI 按 token 计费",
            "比喻": "像出租车的里程数，用多少付多少",
            "例子": "一个汉字通常是 2-3 个 token"
        }
    }
    
    for concept, details in concepts.items():
        print(f"\n🔍 {concept}")
        print("-" * 40)
        print(f"   📖 解释：{details['解释']}")
        print(f"   🎭 比喻：{details['比喻']}")
        print(f"   💡 例子：{details['例子']}")


def advanced_example():
    """
    稍微进阶的例子：带参数的 AI 调用
    """
    print("\n" + "="*60)
    print("🚀 进阶示例：参数化 AI 调用")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 API 密钥")
            return
        
        # 创建 LLM 实例
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3,  # 降低创造性，让回答更稳定
            max_tokens=500
        )
        
        # 使用消息列表的方式，更专业的对话格式
        messages = [
            SystemMessage(content="你是一个友善的 Python 编程助手，专门帮助初学者学习编程。"),
            HumanMessage(content="请用简单的语言解释什么是变量，并给一个 Python 例子。")
        ]
        
        print("🤖 发送结构化消息...")
        response = llm.invoke(messages)
        
        print("\n" + "="*50)
        print("🎯 AI 专业回复：")
        print("="*50)
        print(response.content)
        
        # 显示一些有用的信息
        print(f"\n📊 本次调用信息：")
        print(f"   🔢 输入 token 数：约 {len(str(messages)) // 4}")
        print(f"   📝 输出字符数：{len(response.content)}")
        print(f"   ⚡ 模型类型：{response.response_metadata.get('model', '未知')}")
        
    except Exception as e:
        print(f"❌ 进阶示例执行失败：{e}")


def next_steps():
    """
    下一步学习指南
    """
    print("\n" + "="*60)
    print("🎓 下一步学习计划")
    print("="*60)
    
    print("恭喜完成第一节！现在你已经：")
    achievements = [
        "✅ 理解了 LangChain 是什么",
        "✅ 搭建了开发环境",
        "✅ 完成了第一个 AI 程序",
        "✅ 学会了基本的错误处理"
    ]
    
    for achievement in achievements:
        print(f"   {achievement}")
    
    print("\n📚 第二节预告：《理解链式调用（LCEL）》")
    print("你将学到：")
    next_topics = [
        "🔗 什么是链式调用",
        "⚡ LCEL 表达式语言",
        "🔀 管道操作符的使用",
        "📊 数据在链中的流转"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")
    
    print("\n💡 课后练习：")
    exercises = [
        "尝试修改 temperature 参数，观察 AI 回答的变化",
        "试试不同的问题，看看 AI 的回答质量",
        "阅读 DeepSeek API 文档，了解更多参数",
        "思考：在什么场景下会用到 AI 助手？"
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"   {i}. {exercise}")


def main():
    """
    主函数：按顺序执行所有学习内容
    """
    print("🎯 LangChain 入门到精通 - 第1节")
    print("🎓 LangChain 简介与环境搭建")
    print("👨‍🏫 适合：零基础小白")
    
    # 1. 解释 LangChain 是什么
    explain_langchain()
    
    # 2. 环境搭建指南
    setup_environment()
    
    # 3. API 密钥配置
    api_key = setup_api_key()
    
    if api_key:
        # 4. 第一个 Hello World 程序
        success = first_hello_world()
        
        if success:
            # 5. 核心概念理解
            understanding_concepts()
            
            # 6. 进阶示例
            advanced_example()
    
    # 7. 下一步学习指南
    next_steps()
    
    print("\n" + "="*60)
    print("🎉 第1节完成！")
    print("💪 继续加油，向 LangChain 专家进阶！")
    print("="*60)


# 特别说明区域
"""
📝 重要提醒：

1. API 密钥安全：
   - 不要把 API 密钥写在代码里
   - 使用环境变量或 .env 文件
   - 不要把密钥上传到 Git

2. 费用控制：
   - DeepSeek 很便宜，但也要注意用量
   - 可以在官网设置消费限额
   - 测试时使用较小的 max_tokens

3. 网络问题：
   - 确保能访问 api.deepseek.com
   - 如果网络不稳定，可以增加重试机制
   - 考虑使用代理（如果需要）

4. 错误处理：
   - 总是包装 try-except
   - 提供有用的错误信息
   - 引导用户解决问题

5. 学习建议：
   - 不要急于求成，基础很重要
   - 多动手实践，修改参数试试
   - 遇到问题多查文档
   - 加入社区交流学习
"""


if __name__ == "__main__":
    # 运行主程序
    main()
    
    print("\n🔗 相关资源：")
    print("   📖 LangChain 官方文档：https://python.langchain.com/")
    print("   🐍 DeepSeek API 文档：https://platform.deepseek.com/api-docs/")
    print("   💬 LangChain 中文社区：https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide")
    print("   🎥 视频教程：搜索 'LangChain 入门教程'")