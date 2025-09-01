"""
示例数据和环境配置脚本
基于 official-tutorials 的知识点
"""

import os
from typing import Dict, List

def setup_environment():
    """设置环境变量示例"""
    
    # API 密钥配置示例 (需要用户自行配置)
    env_examples = {
        "DEEPSEEK_API_KEY": "sk-your-deepseek-api-key",
        "SILICONFLOW_API_KEY": "sk-your-siliconflow-api-key", 
        "OPENAI_API_KEY": "sk-your-openai-api-key",
        "TAVILY_API_KEY": "tvly-your-tavily-api-key"
    }
    
    print("🔧 环境变量配置示例:")
    print("=" * 40)
    
    for key, example in env_examples.items():
        current_value = os.environ.get(key)
        if current_value:
            print(f"✅ {key}: 已配置")
        else:
            print(f"❌ {key}: 未配置")
            print(f"   示例: export {key}='{example}'")
    
    print("\n💡 配置方法:")
    print("1. 在终端中设置: export API_KEY='your-key'")
    print("2. 在 .env 文件中设置 (需要 python-dotenv)")
    print("3. 在系统环境变量中设置")


def create_sample_documents() -> List[Dict]:
    """创建示例文档数据 (参考 3.vector_stores.py)"""
    
    documents = [
        {
            "content": """
            OpenAI 是一家专注于人工智能研究的公司，由 Sam Altman 领导。
            公司开发了 GPT 系列模型，包括 ChatGPT 和 GPT-4。
            OpenAI 的使命是确保人工通用智能 (AGI) 造福全人类。
            公司总部位于美国旧金山，成立于 2015 年。
            """,
            "metadata": {
                "source": "openai_company_info",
                "category": "technology_company",
                "date": "2024-01-01"
            }
        },
        {
            "content": """
            DeepSeek 是一家中国的人工智能公司，专注于大语言模型研发。
            公司开发了 DeepSeek-Chat、DeepSeek-Coder 等模型。
            DeepSeek 的模型在代码生成和数学推理任务上表现优秀。
            公司致力于推动 AI 技术的发展和应用。
            """,
            "metadata": {
                "source": "deepseek_company_info", 
                "category": "technology_company",
                "date": "2024-01-02"
            }
        },
        {
            "content": """
            LangChain 是一个用于构建 LLM 应用的开源框架。
            它提供了提示模板、链式调用、向量存储、代理等核心组件。
            LangChain 支持多种模型提供商和向量数据库。
            开发者可以使用 LangChain 快速构建 RAG 系统和 AI 代理。
            """,
            "metadata": {
                "source": "langchain_framework_info",
                "category": "development_framework", 
                "date": "2024-01-03"
            }
        },
        {
            "content": """
            机器学习是人工智能的核心分支，通过算法从数据中学习模式。
            常见算法包括线性回归、逻辑回归、决策树、随机森林、SVM 等。
            深度学习作为机器学习的子集，使用神经网络处理复杂问题。
            机器学习广泛应用于图像识别、自然语言处理、推荐系统等领域。
            """,
            "metadata": {
                "source": "machine_learning_intro",
                "category": "technology_concept",
                "date": "2024-01-04"
            }
        }
    ]
    
    return documents


def create_test_queries() -> List[str]:
    """创建测试查询示例 (参考 4.similarity_search.py)"""
    
    queries = [
        "OpenAI 公司的基本信息",
        "中国的 AI 公司有哪些？",
        "LangChain 框架的主要功能", 
        "机器学习算法有哪些？",
        "如何构建 AI 代理应用？",
        "向量数据库的应用场景",
        "提示模板的使用方法",
        "RAG 系统的构建步骤"
    ]
    
    return queries


def create_extraction_examples() -> List[Dict]:
    """创建信息提取示例 (参考 7.extraction.py)"""
    
    examples = [
        {
            "text": """
            微软公司 (Microsoft Corporation) 是一家美国跨国科技公司，
            总部位于华盛顿州雷德蒙德。现任CEO是萨蒂亚·纳德拉 (Satya Nadella)。
            公司成立于1975年，主要业务包括软件开发、云计算服务和硬件制造。
            """,
            "expected_extraction": {
                "companies": ["Microsoft Corporation"],
                "people": ["Satya Nadella"],
                "locations": ["华盛顿州雷德蒙德"],
                "industries": ["科技", "软件", "云计算"]
            }
        },
        {
            "text": """
            谷歌 (Google) 是 Alphabet Inc. 的子公司，由拉里·佩奇和谢尔盖·布林创立。
            公司总部位于加利福尼亚州山景城，现任CEO是桑达尔·皮查伊。
            谷歌的主要业务包括搜索引擎、在线广告、云计算和移动操作系统。
            """,
            "expected_extraction": {
                "companies": ["Google", "Alphabet Inc."],
                "people": ["拉里·佩奇", "谢尔盖·布林", "桑达尔·皮查伊"],
                "locations": ["加利福尼亚州山景城"],
                "industries": ["搜索", "广告", "云计算", "移动"]
            }
        }
    ]
    
    return examples


if __name__ == "__main__":
    print("📊 Agent Demo 示例数据")
    print("=" * 30)
    
    # 环境配置检查
    setup_environment()
    
    # 示例数据统计
    documents = create_sample_documents()
    queries = create_test_queries() 
    examples = create_extraction_examples()
    
    print(f"\n📋 数据统计:")
    print(f"  示例文档: {len(documents)} 个")
    print(f"  测试查询: {len(queries)} 个") 
    print(f"  提取示例: {len(examples)} 个")
    
    print(f"\n🔍 示例查询:")
    for i, query in enumerate(queries[:3], 1):
        print(f"  {i}. {query}")
    
    print(f"\n✨ 提示: 运行 python main.py 开始演示")