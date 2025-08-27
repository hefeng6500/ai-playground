"""
第5节：输出解析与格式化
=============================================

学习目标：
- 理解输出解析器的作用和重要性
- 掌握各种内置解析器的使用
- 学会创建自定义解析器
- 了解数据验证和错误处理

前置知识：
- 完成第1-4节基础内容

重点概念：
- 输出解析器是 AI 输出的"翻译官"
- 结构化输出让程序更容易处理
- 数据验证确保输出质量
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


def explain_output_parsing():
    """
    解释输出解析的重要性
    """
    print("\n" + "="*60)
    print("🔧 为什么需要输出解析？")
    print("="*60)
    
    print("""
🎭 想象 AI 是一个外国朋友：

没有翻译官的对话：
👤 你：请给我一些电影推荐
🤖 AI：我推荐你看《阿甘正传》，这是一部很棒的电影，讲述了...还有《肖申克的救赎》，这部电影...
👤 你：😵 我只想要电影名字列表啊！

有翻译官的对话：
👤 你：请给我一些电影推荐，用JSON格式
🤖 AI：["阿甘正传", "肖申克的救赎", "泰坦尼克号"]
👤 你：😊 完美！程序可以直接处理

输出解析器就是这个"翻译官"：
✅ 把 AI 的自然语言输出转换为程序能处理的格式
✅ 验证数据格式是否正确
✅ 处理解析错误和异常情况
    """)
    
    print("📊 常见的输出格式：")
    formats = [
        "📝 纯文本：最简单，但难以处理",
        "📋 JSON：结构化，程序友好",
        "📊 CSV：表格数据，Excel 友好",
        "🏷️  XML：标记语言，配置文件",
        "🎯 自定义：特殊业务需求"
    ]
    
    for fmt in formats:
        print(f"   {fmt}")


def string_output_parser_demo():
    """
    字符串输出解析器演示
    """
    print("\n" + "="*60)
    print("📝 字符串输出解析器 (StrOutputParser)")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 DEEPSEEK_API_KEY")
            return
        
        print("🎯 最基础的解析器：提取纯文本内容")
        print("-" * 30)
        
        # 创建组件
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7
        )
        
        prompt = ChatPromptTemplate.from_template(
            "请用一句话总结：{topic}"
        )
        
        # 对比：有无解析器的区别
        print("🔬 对比实验：")
        
        # 不使用解析器
        chain_raw = prompt | model
        raw_result = chain_raw.invoke({"topic": "人工智能"})
        
        print(f"不用解析器的输出类型：{type(raw_result)}")
        print(f"内容：{raw_result.content}")
        print(f"元数据：{raw_result.response_metadata}")
        
        # 使用字符串解析器
        parser = StrOutputParser()
        chain_parsed = prompt | model | parser
        parsed_result = chain_parsed.invoke({"topic": "人工智能"})
        
        print(f"\n用解析器的输出类型：{type(parsed_result)}")
        print(f"内容：{parsed_result}")
        
        print("\n✅ 字符串解析器的作用：")
        print("   🔧 提取 AI 回复的纯文本内容")
        print("   🎯 去除元数据和格式信息")
        print("   📦 返回简单的字符串，便于后续处理")
        
    except Exception as e:
        print(f"❌ 字符串解析器演示失败：{e}")


def json_output_parser_demo():
    """
    JSON 输出解析器演示
    """
    print("\n" + "="*60)
    print("📋 JSON 输出解析器")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.pydantic_v1 import BaseModel, Field
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请先配置 API 密钥")
            return
        
        print("🎯 结构化数据输出：让 AI 返回 JSON 格式")
        print("-" * 40)
        
        # 定义数据模型
        class MovieRecommendation(BaseModel):
            """电影推荐数据模型"""
            title: str = Field(description="电影标题")
            genre: str = Field(description="电影类型")
            year: int = Field(description="上映年份")
            rating: float = Field(description="评分(1-10)")
            reason: str = Field(description="推荐理由")
        
        # 创建 JSON 解析器
        parser = JsonOutputParser(pydantic_object=MovieRecommendation)
        
        # 创建提示模板，包含格式指令
        prompt = ChatPromptTemplate.from_template(
            """请推荐一部{genre}类型的电影。

{format_instructions}

用户偏好：{preferences}"""
        )
        
        # 创建模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3  # JSON 输出用较低温度
        )
        
        # 创建链条
        chain = prompt | model | parser
        
        # 测试 JSON 输出
        print("🧪 测试 JSON 输出...")
        
        result = chain.invoke({
            "genre": "科幻",
            "preferences": "喜欢有深度的剧情",
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"✅ 解析成功！输出类型：{type(result)}")
        print("📋 JSON 数据：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 验证数据结构
        print("\n🔍 数据验证：")
        required_fields = ["title", "genre", "year", "rating", "reason"]
        for field in required_fields:
            if field in result:
                print(f"   ✅ {field}: {result[field]}")
            else:
                print(f"   ❌ 缺失字段: {field}")
        
        print("\n💡 JSON 解析器的优势：")
        advantages = [
            "🏗️  结构化：数据有明确的字段和类型",
            "🔍 可验证：可以检查必需字段是否存在",
            "🔧 易处理：程序可以直接使用字典访问",
            "📦 可扩展：容易添加新字段"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ JSON 解析器演示失败：{e}")


def list_output_parser_demo():
    """
    列表输出解析器演示
    """
    print("\n" + "="*60)
    print("📝 列表输出解析器")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import CommaSeparatedListOutputParser
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 从文本中提取列表数据")
        print("-" * 30)
        
        # 创建列表解析器
        parser = CommaSeparatedListOutputParser()
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(
            """请列出5个{category}。

{format_instructions}

要求：{requirements}"""
        )
        
        # 创建模型和链条
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.5
        )
        
        chain = prompt | model | parser
        
        # 测试不同类型的列表
        test_cases = [
            {
                "category": "编程语言",
                "requirements": "按流行度排序"
            },
            {
                "category": "中国城市",
                "requirements": "包含一线和新一线城市"
            },
            {
                "category": "学习方法",
                "requirements": "适合程序员的学习技巧"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 测试{i}：{test_case['category']}")
            
            result = chain.invoke({
                "category": test_case["category"],
                "requirements": test_case["requirements"],
                "format_instructions": parser.get_format_instructions()
            })
            
            print(f"输出类型：{type(result)}")
            print(f"列表内容：{result}")
            print(f"列表长度：{len(result)}")
            
            # 展示列表操作
            print("🔧 列表操作示例：")
            for idx, item in enumerate(result, 1):
                print(f"   {idx}. {item.strip()}")
        
        print("\n✅ 列表解析器的应用场景：")
        use_cases = [
            "📋 提取关键词列表",
            "🏷️  生成标签云",
            "📊 创建选项菜单",
            "🔍 分类结果展示"
        ]
        
        for use_case in use_cases:
            print(f"   {use_case}")
        
    except Exception as e:
        print(f"❌ 列表解析器演示失败：{e}")


def custom_output_parser_demo():
    """
    自定义输出解析器演示
    """
    print("\n" + "="*60)
    print("🛠️  自定义输出解析器")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import BaseOutputParser
        from langchain_core.exceptions import OutputParserException
        import re
        
        print("🎯 场景：解析代码评审结果")
        print("-" * 30)
        
        class CodeReviewParser(BaseOutputParser[Dict[str, Any]]):
            """自定义代码评审解析器"""
            
            def parse(self, text: str) -> Dict[str, Any]:
                """解析代码评审文本"""
                try:
                    # 使用正则表达式提取各部分内容
                    
                    # 提取评分
                    score_match = re.search(r'评分[：:]\s*(\d+(?:\.\d+)?)', text)
                    score = float(score_match.group(1)) if score_match else 0.0
                    
                    # 提取问题列表
                    problems_section = re.search(r'问题[：:](.+?)(?=建议|$)', text, re.DOTALL)
                    problems = []
                    if problems_section:
                        problem_lines = problems_section.group(1).strip().split('\n')
                        problems = [line.strip('- •').strip() for line in problem_lines if line.strip()]
                    
                    # 提取建议列表
                    suggestions_section = re.search(r'建议[：:](.+?)(?=总结|$)', text, re.DOTALL)
                    suggestions = []
                    if suggestions_section:
                        suggestion_lines = suggestions_section.group(1).strip().split('\n')
                        suggestions = [line.strip('- •').strip() for line in suggestion_lines if line.strip()]
                    
                    # 提取总结
                    summary_match = re.search(r'总结[：:](.+)', text, re.DOTALL)
                    summary = summary_match.group(1).strip() if summary_match else ""
                    
                    return {
                        "score": score,
                        "problems": problems[:3],  # 最多3个问题
                        "suggestions": suggestions[:3],  # 最多3个建议
                        "summary": summary,
                        "grade": self._get_grade(score)
                    }
                    
                except Exception as e:
                    raise OutputParserException(f"解析失败: {e}")
            
            def _get_grade(self, score: float) -> str:
                """根据评分获取等级"""
                if score >= 9.0:
                    return "优秀"
                elif score >= 7.0:
                    return "良好"
                elif score >= 5.0:
                    return "及格"
                else:
                    return "需改进"
            
            def get_format_instructions(self) -> str:
                """返回格式指令"""
                return """请按以下格式输出代码评审结果：

评分：[0-10分的数字]

问题：
- [问题1]
- [问题2]
- [问题3]

建议：
- [建议1]
- [建议2]
- [建议3]

总结：[整体评价]"""
        
        # 创建自定义解析器
        parser = CodeReviewParser()
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(
            """请对以下代码进行评审：

代码：
```python
{code}
```

{format_instructions}

评审要点：代码质量、可读性、性能、安全性"""
        )
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        # 创建模型和链条
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3
        )
        
        chain = prompt | model | parser
        
        # 测试代码
        test_code = '''
def calculate_factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
'''
        
        print("🧪 测试自定义解析器...")
        print(f"测试代码：{test_code}")
        
        result = chain.invoke({
            "code": test_code,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("\n📋 解析结果：")
        print(f"   📊 评分：{result['score']}/10")
        print(f"   🏆 等级：{result['grade']}")
        
        print("   ❌ 发现的问题：")
        for i, problem in enumerate(result['problems'], 1):
            print(f"      {i}. {problem}")
        
        print("   💡 改进建议：")
        for i, suggestion in enumerate(result['suggestions'], 1):
            print(f"      {i}. {suggestion}")
        
        print(f"   📝 总结：{result['summary']}")
        
        print("\n✅ 自定义解析器的优势：")
        advantages = [
            "🎯 针对性强：专门处理特定格式",
            "🔧 灵活性高：可以实现复杂的解析逻辑",
            "🛡️  错误处理：可以自定义异常处理",
            "📊 数据转换：可以进行复杂的数据转换"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ 自定义解析器演示失败：{e}")


def error_handling_demo():
    """
    错误处理演示
    """
    print("\n" + "="*60)
    print("🛡️  输出解析错误处理")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.output_parsers import OutputFixingParser
        from langchain_core.pydantic_v1 import BaseModel, Field
        
        print("🎯 处理 AI 输出格式不规范的问题")
        print("-" * 40)
        
        # 定义数据模型
        class ProductInfo(BaseModel):
            name: str = Field(description="产品名称")
            price: float = Field(description="价格")
            category: str = Field(description="分类")
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        # 基础解析器
        base_parser = JsonOutputParser(pydantic_object=ProductInfo)
        
        # 带错误修复的解析器
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3
        )
        
        fixing_parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=model
        )
        
        print("🧪 测试错误处理...")
        
        # 模拟不规范的 JSON 输出
        bad_json_examples = [
            '{"name": "iPhone 15", "price": "999", "category": "手机"}',  # price 是字符串
            '{"name": "MacBook", price: 1999, "category": "电脑"}',      # 缺少引号
            '{"name": "iPad", "price": 599.99, "category": "平板", "extra": "多余字段"}',  # 多余字段
        ]
        
        for i, bad_json in enumerate(bad_json_examples, 1):
            print(f"\n🔬 测试{i}：格式错误的 JSON")
            print(f"原始数据：{bad_json}")
            
            try:
                # 尝试基础解析器
                print("   🔧 基础解析器：", end="")
                result_basic = base_parser.parse(bad_json)
                print("✅ 成功")
                print(f"   结果：{result_basic}")
            except Exception as e:
                print(f"❌ 失败 - {e}")
                
                # 使用修复解析器
                print("   🛠️  修复解析器：", end="")
                try:
                    result_fixed = fixing_parser.parse(bad_json)
                    print("✅ 成功修复")
                    print(f"   结果：{result_fixed}")
                except Exception as e2:
                    print(f"❌ 修复失败 - {e2}")
        
        print("\n💡 错误处理策略：")
        strategies = [
            "🔄 自动重试：解析失败时重新生成",
            "🛠️  格式修复：使用 LLM 修复格式错误",
            "📋 默认值：提供合理的默认值",
            "🚨 优雅降级：返回部分解析结果"
        ]
        
        for strategy in strategies:
            print(f"   {strategy}")
        
    except Exception as e:
        print(f"❌ 错误处理演示失败：{e}")


def practical_example():
    """
    实用案例：智能简历解析器
    """
    print("\n" + "="*60)
    print("🎯 实战案例：智能简历解析器")
    print("="*60)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.pydantic_v1 import BaseModel, Field
        from typing import List
        
        # 定义简历数据模型
        class Experience(BaseModel):
            company: str = Field(description="公司名称")
            position: str = Field(description="职位")
            duration: str = Field(description="工作时长")
            
        class Resume(BaseModel):
            name: str = Field(description="姓名")
            email: str = Field(description="邮箱")
            phone: str = Field(description="电话")
            skills: List[str] = Field(description="技能列表")
            experience: List[Experience] = Field(description="工作经历")
            education: str = Field(description="教育背景")
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return
        
        print("🎯 从文本中提取结构化简历信息")
        print("-" * 40)
        
        # 创建解析器
        parser = JsonOutputParser(pydantic_object=Resume)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(
            """请从以下简历文本中提取结构化信息：

简历内容：
{resume_text}

{format_instructions}

要求：
1. 尽可能提取完整信息
2. 如果某些信息缺失，用"未提供"填充
3. 技能列表要详细，包含编程语言、框架等
4. 工作经历按时间顺序排列"""
        )
        
        # 创建模型和链条
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.1  # 信息提取用很低的温度
        )
        
        chain = prompt | model | parser
        
        # 测试简历文本
        resume_text = """
        张三
        软件工程师
        
        联系方式：
        邮箱：zhangsan@email.com
        电话：138-0000-0000
        
        技能：
        • Python、Java、JavaScript
        • Django、Flask、React
        • MySQL、Redis、Docker
        • Git、Linux、AWS
        
        工作经历：
        2021-2023 腾讯科技 高级软件工程师
        负责微信支付系统的后端开发，使用Python和Django
        
        2019-2021 百度 软件工程师
        参与搜索引擎优化，使用Java和Spring框架
        
        教育背景：
        2015-2019 清华大学 计算机科学与技术 本科
        """
        
        print("🧪 测试简历解析...")
        print("原始简历文本：")
        print(resume_text)
        
        result = chain.invoke({
            "resume_text": resume_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("\n📋 解析结果：")
        print("="*40)
        print(f"姓名：{result['name']}")
        print(f"邮箱：{result['email']}")
        print(f"电话：{result['phone']}")
        print(f"教育：{result['education']}")
        
        print("\n💼 工作经历：")
        for i, exp in enumerate(result['experience'], 1):
            print(f"  {i}. {exp['position']} @ {exp['company']}")
            print(f"     时间：{exp['duration']}")
        
        print(f"\n🛠️  技能列表：")
        for i, skill in enumerate(result['skills'], 1):
            print(f"  {i}. {skill}")
        
        print("\n✅ 应用价值：")
        values = [
            "🚀 自动化：HR 可以快速处理大量简历",
            "📊 标准化：统一的数据格式便于分析",
            "🔍 检索：可以按技能、经验快速筛选",
            "📈 匹配：与职位要求进行智能匹配"
        ]
        
        for value in values:
            print(f"   {value}")
        
    except Exception as e:
        print(f"❌ 简历解析案例失败：{e}")


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第5节总结 & 第6节预告")
    print("="*60)
    
    print("🎉 第5节你掌握了：")
    learned = [
        "✅ 理解输出解析器的作用和重要性",
        "✅ 掌握各种内置解析器的使用",
        "✅ 学会创建自定义解析器",
        "✅ 了解错误处理和数据验证",
        "✅ 完成实用的简历解析案例"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n📚 第6节预告：《流式输出与实时交互》")
    print("你将学到：")
    next_topics = [
        "⚡ 流式输出原理和实现",
        "🔄 实时数据处理",
        "💬 流式对话体验",
        "🎯 性能优化技巧"
    ]
    
    for topic in next_topics:
        print(f"   {topic}")


def main():
    """
    主函数：第5节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第5节")
    print("🔧 输出解析与格式化")
    print("📚 前置：完成第1-4节")
    
    # 1. 解释重要性
    explain_output_parsing()
    
    # 2. 字符串解析器
    string_output_parser_demo()
    
    # 3. JSON 解析器
    json_output_parser_demo()
    
    # 4. 列表解析器
    list_output_parser_demo()
    
    # 5. 自定义解析器
    custom_output_parser_demo()
    
    # 6. 错误处理
    error_handling_demo()
    
    # 7. 实战案例
    practical_example()
    
    # 8. 总结预告
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第5节完成！")
    print("🚀 你已经是输出解析专家了！")
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
    print("   📖 Pydantic 官方文档：https://pydantic-docs.helpmanual.io/")
    print("   💻 JSON Schema 规范")
    print("   🛠️  正则表达式教程")