"""
第4节：多模型接入与切换
=============================================

学习目标：
- 了解主流 LLM 服务商和模型特点
- 学会接入不同的模型服务
- 掌握模型切换和配置管理
- 理解模型选择策略和成本优化

前置知识：
- 完成第1-3节基础内容

重点概念：
- 不同模型有不同的特点和价格
- 统一接口让切换变得简单
- 根据任务选择合适的模型
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    provider: str
    api_base: str
    price_input: float  # 每1K tokens价格(元)
    price_output: float
    max_tokens: int
    strengths: List[str]


def explain_model_landscape():
    """
    介绍 LLM 模型格局
    """
    print("\n" + "="*60)
    print("🌍 LLM 模型生态全景")
    print("="*60)
    
    print("""
🏢 主要服务商分类：

国际巨头：
  🇺🇸 OpenAI：GPT-4, GPT-3.5（功能最强，价格较高）
  🇺🇸 Anthropic：Claude（安全性强，推理能力好）
  🇺🇸 Google：Gemini（多模态能力强）

国产力量：
  🇨🇳 DeepSeek：便宜实用，性价比之王
  🇨🇳 阿里：通义千问，中文理解好
  🇨🇳 百度：文心一言，国内生态完善
  🇨🇳 字节：豆包，短文本处理快

开源模型：
  🔓 Llama：Meta开源，可私有部署
  🔓 Mistral：欧洲开源，效率很高
  🔓 Qwen：阿里开源，中文友好
    """)
    
    # 模型配置对比表
    models = [
        ModelConfig("gpt-4", "OpenAI", "api.openai.com", 0.21, 0.42, 8192, 
                   ["推理能力强", "通用性好", "文档丰富"]),
        ModelConfig("deepseek-chat", "DeepSeek", "api.deepseek.com", 0.0014, 0.0028, 4096,
                   ["性价比高", "响应快", "中文友好"]),
        ModelConfig("claude-3", "Anthropic", "api.anthropic.com", 0.21, 0.42, 200000,
                   ["安全性强", "长文本", "推理准确"]),
        ModelConfig("qwen-max", "阿里云", "dashscope.aliyuncs.com", 0.12, 0.12, 8192,
                   ["中文优秀", "国内访问", "生态完善"])
    ]
    
    print("\n📊 模型对比表：")
    print("-" * 80)
    print(f"{'模型名称':<15} {'服务商':<10} {'输入价格':<10} {'输出价格':<10} {'主要优势'}")
    print("-" * 80)
    
    for model in models:
        strengths = ", ".join(model.strengths[:2])
        print(f"{model.name:<15} {model.provider:<10} {model.price_input:<10.4f} {model.price_output:<10.4f} {strengths}")


def setup_multi_models():
    """
    配置多个模型
    """
    print("\n" + "="*60)
    print("🔧 配置多个模型服务")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        print("🎯 步骤1：配置 DeepSeek 模型")
        print("-" * 30)
        
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            deepseek_model = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=deepseek_key,
                openai_api_base="https://api.deepseek.com",
                temperature=0.7,
                max_tokens=1000
            )
            print("✅ DeepSeek 配置成功")
        else:
            print("❌ 未找到 DEEPSEEK_API_KEY")
            deepseek_model = None
        
        print("\n🎯 步骤2：配置 SiliconFlow 模型")
        print("-" * 30)
        
        siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
        if siliconflow_key:
            siliconflow_model = ChatOpenAI(
                model="Qwen/Qwen2.5-7B-Instruct",
                openai_api_key=siliconflow_key,
                openai_api_base="https://api.siliconflow.cn/v1",
                temperature=0.7,
                max_tokens=1000
            )
            print("✅ SiliconFlow 配置成功")
        else:
            print("❌ 未找到 SILICONFLOW_API_KEY")
            siliconflow_model = None
        
        # 创建模型管理器
        print("\n🎯 步骤3：创建模型管理器")
        print("-" * 30)
        
        class ModelManager:
            """模型管理器"""
            
            def __init__(self):
                self.models = {}
                self.current_model = None
            
            def add_model(self, name: str, model, description: str = ""):
                """添加模型"""
                self.models[name] = {
                    "model": model,
                    "description": description
                }
                if self.current_model is None:
                    self.current_model = name
                print(f"✅ 添加模型：{name} - {description}")
            
            def switch_model(self, name: str):
                """切换模型"""
                if name in self.models:
                    self.current_model = name
                    print(f"🔄 切换到模型：{name}")
                    return True
                else:
                    print(f"❌ 模型不存在：{name}")
                    return False
            
            def get_current_model(self):
                """获取当前模型"""
                if self.current_model and self.current_model in self.models:
                    return self.models[self.current_model]["model"]
                return None
            
            def list_models(self):
                """列出所有模型"""
                print("\n📋 可用模型列表：")
                for name, info in self.models.items():
                    current = "👆 当前" if name == self.current_model else "  "
                    print(f"   {current} {name}: {info['description']}")
        
        # 初始化管理器
        manager = ModelManager()
        
        if deepseek_model:
            manager.add_model("deepseek", deepseek_model, "便宜实用，中文友好")
        
        if siliconflow_model:
            manager.add_model("qwen", siliconflow_model, "开源模型，功能全面")
        
        manager.list_models()
        
        return manager
        
    except ImportError as e:
        print(f"❌ 导入失败：{e}")
        return None


def model_comparison_test():
    """
    模型对比测试
    """
    print("\n" + "="*60)
    print("🔬 模型能力对比测试")
    print("="*60)
    
    try:
        manager = setup_multi_models()
        if not manager or not manager.models:
            print("❌ 没有可用的模型")
            return
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 测试题目
        test_cases = [
            {
                "name": "创意写作",
                "prompt": "写一个50字的科幻小故事",
                "评价标准": ["创意性", "语言流畅度", "故事完整性"]
            },
            {
                "name": "逻辑推理", 
                "prompt": "小明比小红高，小红比小李高，问谁最高？请说明推理过程。",
                "评价标准": ["逻辑正确性", "推理清晰度"]
            },
            {
                "name": "代码生成",
                "prompt": "写一个Python函数，计算斐波那契数列的第n项",
                "评价标准": ["代码正确性", "效率", "可读性"]
            }
        ]
        
        # 创建测试链
        template = ChatPromptTemplate.from_template("{prompt}")
        parser = StrOutputParser()
        
        print("🧪 开始对比测试...")
        
        for test_case in test_cases:
            print(f"\n📝 测试项目：{test_case['name']}")
            print(f"题目：{test_case['prompt']}")
            print("-" * 50)
            
            for model_name in manager.models.keys():
                manager.switch_model(model_name)
                model = manager.get_current_model()
                
                if model:
                    chain = template | model | parser
                    try:
                        result = chain.invoke({"prompt": test_case['prompt']})
                        print(f"\n🤖 {model_name} 回答：")
                        print(result[:200] + "..." if len(result) > 200 else result)
                    except Exception as e:
                        print(f"\n❌ {model_name} 调用失败：{e}")
            
            print("\n" + "="*50)
        
        print("\n💡 对比建议：")
        print("   📊 根据测试结果选择最适合你任务的模型")
        print("   💰 考虑成本因素，日常使用推荐便宜的模型")
        print("   🎯 重要任务使用效果最好的模型")
        
    except Exception as e:
        print(f"❌ 对比测试失败：{e}")


def smart_model_router():
    """
    智能模型路由
    """
    print("\n" + "="*60)
    print("🧠 智能模型路由：根据任务自动选择模型")
    print("="*60)
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        
        class SmartRouter:
            """智能模型路由器"""
            
            def __init__(self, manager):
                self.manager = manager
                self.routing_rules = {
                    "creative": "deepseek",  # 创意任务用 DeepSeek
                    "coding": "qwen",        # 编程任务用 Qwen
                    "analysis": "deepseek",  # 分析任务用 DeepSeek
                    "translation": "qwen",   # 翻译任务用 Qwen
                    "default": "deepseek"    # 默认使用 DeepSeek
                }
            
            def classify_task(self, prompt: str) -> str:
                """分类任务类型"""
                prompt_lower = prompt.lower()
                
                if any(word in prompt_lower for word in ["写作", "故事", "创意", "想象"]):
                    return "creative"
                elif any(word in prompt_lower for word in ["代码", "编程", "函数", "python", "code"]):
                    return "coding"
                elif any(word in prompt_lower for word in ["分析", "总结", "归纳", "解释"]):
                    return "analysis"
                elif any(word in prompt_lower for word in ["翻译", "translate", "英文", "中文"]):
                    return "translation"
                else:
                    return "default"
            
            def route_request(self, request):
                """路由请求到合适的模型"""
                prompt = request.get("prompt", "")
                task_type = self.classify_task(prompt)
                
                # 选择模型
                model_name = self.routing_rules.get(task_type, "default")
                available_models = list(self.manager.models.keys())
                
                if model_name not in available_models:
                    model_name = available_models[0] if available_models else None
                
                if model_name:
                    self.manager.switch_model(model_name)
                    print(f"🎯 任务类型：{task_type} → 选择模型：{model_name}")
                    return request
                else:
                    raise ValueError("没有可用的模型")
        
        # 创建路由器
        manager = setup_multi_models()
        if not manager:
            return
        
        router = SmartRouter(manager)
        
        # 创建智能链条
        template = ChatPromptTemplate.from_template("{prompt}")
        parser = StrOutputParser()
        
        def create_smart_chain():
            return (
                RunnableLambda(router.route_request) |
                template |
                RunnableLambda(lambda x: manager.get_current_model().invoke(x)) |
                parser
            )
        
        smart_chain = create_smart_chain()
        
        # 测试智能路由
        test_prompts = [
            "写一个关于机器人的小故事",
            "写一个计算阶乘的Python函数", 
            "分析一下人工智能的发展趋势",
            "将'Hello World'翻译成中文"
        ]
        
        print("🧪 测试智能路由...")
        
        for prompt in test_prompts:
            print(f"\n📝 输入：{prompt}")
            try:
                result = smart_chain.invoke({"prompt": prompt})
                print(f"🤖 输出：{result[:150]}...")
            except Exception as e:
                print(f"❌ 失败：{e}")
        
    except Exception as e:
        print(f"❌ 智能路由失败：{e}")


def cost_optimization():
    """
    成本优化策略
    """
    print("\n" + "="*60)
    print("💰 成本优化策略")
    print("="*60)
    
    strategies = {
        "1. 模型分层使用": [
            "日常任务：DeepSeek（便宜）",
            "重要任务：GPT-4（效果好）",
            "大批量：开源模型（免费）"
        ],
        
        "2. 提示优化": [
            "精简提示词，减少输入 tokens",
            "设置合理的 max_tokens 限制", 
            "使用缓存避免重复调用"
        ],
        
        "3. 批量处理": [
            "合并相似请求一起处理",
            "使用 batch 方法减少网络开销",
            "异步处理提高效率"
        ],
        
        "4. 智能降级": [
            "主模型失败时自动切换备用模型",
            "根据任务重要性选择模型级别",
            "高峰期使用便宜模型"
        ]
    }
    
    for strategy, tips in strategies.items():
        print(f"\n💡 {strategy}")
        print("-" * 30)
        for tip in tips:
            print(f"   • {tip}")
    
    print("\n📊 成本计算示例：")
    print("-" * 30)
    
    # 成本计算器
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int):
        costs = {
            "deepseek": {"input": 0.0014, "output": 0.0028},
            "gpt-4": {"input": 0.21, "output": 0.42},
            "qwen": {"input": 0.12, "output": 0.12}
        }
        
        if model_name in costs:
            input_cost = (input_tokens / 1000) * costs[model_name]["input"]
            output_cost = (output_tokens / 1000) * costs[model_name]["output"]
            return input_cost + output_cost
        return 0
    
    # 示例计算
    scenario = {
        "任务": "1000次客服对话",
        "平均输入": 100,
        "平均输出": 200
    }
    
    print(f"场景：{scenario['任务']}")
    print(f"每次对话：{scenario['平均输入']} tokens输入，{scenario['平均输出']} tokens输出")
    print()
    
    for model in ["deepseek", "gpt-4", "qwen"]:
        cost_per_call = calculate_cost(model, scenario['平均输入'], scenario['平均输出'])
        total_cost = cost_per_call * 1000
        print(f"{model:<10}: 单次 ¥{cost_per_call:.4f}, 总计 ¥{total_cost:.2f}")


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第4节总结 & 第5节预告")
    print("="*60)
    
    print("🎉 第4节你掌握了：")
    learned = [
        "✅ 了解主流 LLM 模型生态",
        "✅ 学会配置多个模型服务",
        "✅ 掌握模型切换和管理",
        "✅ 理解智能路由和成本优化"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第5节预告：《输出解析与格式化》")
    print("你将学到：")
    next_topics = [
        "🔧 结构化输出解析",
        "📋 JSON/XML 格式处理",
        "🎯 自定义解析器开发",
        "📊 数据验证和错误处理"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第4节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第4节")
    print("🔌 多模型接入与切换")
    print("📚 前置：完成第1-3节")
    
    # 1. 模型生态介绍
    explain_model_landscape()
    
    # 2. 配置多个模型
    setup_multi_models()
    
    # 3. 模型对比测试
    model_comparison_test()
    
    # 4. 智能模型路由
    smart_model_router()
    
    # 5. 成本优化
    cost_optimization()
    
    # 6. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第4节完成！")
    print("🚀 你已经是多模型管理专家了！")
    print("="*60)


if __name__ == "__main__":
    # 检查环境
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  建议配置 DEEPSEEK_API_KEY")
        print("💡 如需配置其他模型，请设置对应的 API 密钥")
    
    # 运行主程序
    main()
    
    print("\n🔗 本节参考资源：")
    print("   📖 各模型官方文档和价格页面")
    print("   💻 模型性能对比网站：https://chat.lmsys.org/")
    print("   🤝 社区模型评测和推荐")