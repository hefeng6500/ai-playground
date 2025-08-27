"""
第10节：向量存储与检索
=============================================

学习目标：
- 理解向量数据库的原理和优势
- 掌握 FAISS 向量存储的使用
- 学会高级检索策略和技巧
- 了解检索性能优化方法

前置知识：
- 完成第1-9节基础内容

重点概念：
- 向量数据库专为相似性检索优化
- FAISS 是高性能的向量检索库
- 不同检索策略适用不同场景
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
import tempfile
import json


def explain_vector_database():
    """
    解释向量数据库的概念
    """
    print("\n" + "="*60)
    print("🗄️  向量数据库：为AI检索而生")
    print("="*60)
    
    print("""
🏪 想象两种商店的区别：

传统数据库（关系型）：
🏬 超市：按类别整齐摆放
📋 查找：根据名称、价格等精确查找
🎯 优势：结构化查询，事务支持
❌ 劣势：不支持"相似性"查找

向量数据库：
🎨 艺术品市场：按风格、色彩相似度摆放
🔍 查找：找"类似莫奈风格"的画作
🎯 优势：语义相似性检索
❌ 劣势：不适合精确查询

🚀 向量数据库的核心优势：
- ⚡ 高速检索：专门为向量优化
- 📈 可扩展：支持百万级向量
- 🎯 精确度高：多种相似度算法
- 💾 内存优化：压缩存储技术
    """)
    
    print("📊 主流向量数据库：")
    databases = [
        "🔥 FAISS：Facebook开源，性能最强",
        "🌊 Weaviate：开源图向量数据库",
        "📌 Pinecone：云服务，易于使用",
        "🌟 Qdrant：Rust编写，高性能",
        "🔍 Milvus：分布式向量数据库"
    ]
    
    for db in databases:
        print(f"   {db}")


def faiss_basic_demo():
    """
    FAISS 基础使用演示
    """
    print("\n" + "="*60)
    print("⚡ FAISS 向量存储基础")
    print("="*60)
    
    try:
        # 检查是否有必要的包
        print("🔧 环境检查...")
        try:
            import faiss
            print("✅ FAISS 已安装")
        except ImportError:
            print("❌ FAISS 未安装，请执行：pip install faiss-cpu")
            print("💡 继续使用模拟演示...")
            faiss = None
        
        print("\n🎯 FAISS 核心概念")
        print("-" * 30)
        
        # 创建示例向量数据
        dimension = 128  # 向量维度
        n_vectors = 1000  # 向量数量
        
        print(f"📊 创建测试数据：{n_vectors}个{dimension}维向量")
        
        if faiss:
            # 真实的FAISS演示
            np.random.seed(42)
            vectors = np.random.random((n_vectors, dimension)).astype('float32')
            
            print("\n🏗️  构建FAISS索引...")
            
            # 1. 创建索引
            index = faiss.IndexFlatL2(dimension)  # L2距离索引
            print(f"   索引类型：{type(index).__name__}")
            print(f"   是否训练：{index.is_trained}")
            
            # 2. 添加向量
            index.add(vectors)
            print(f"   已添加向量数：{index.ntotal}")
            
            # 3. 搜索测试
            print("\n🔍 向量检索测试...")
            k = 5  # 返回前5个最相似的
            query_vector = np.random.random((1, dimension)).astype('float32')
            
            distances, indices = index.search(query_vector, k)
            
            print(f"   查询向量维度：{query_vector.shape}")
            print(f"   返回结果数：{len(indices[0])}")
            print(f"   相似向量索引：{indices[0]}")
            print(f"   对应距离：{distances[0]}")
            
            # 4. 索引统计
            print(f"\n📊 索引统计信息：")
            print(f"   总向量数：{index.ntotal}")
            print(f"   向量维度：{index.d}")
            print(f"   索引大小：约{index.ntotal * dimension * 4 / 1024 / 1024:.1f}MB")
            
        else:
            # 模拟演示
            print("🎭 模拟FAISS工作流程...")
            
            class MockFAISS:
                def __init__(self, dimension):
                    self.dimension = dimension
                    self.vectors = []
                    self.ntotal = 0
                
                def add(self, vectors):
                    self.vectors.extend(vectors.tolist())
                    self.ntotal = len(self.vectors)
                
                def search(self, query, k):
                    # 简单的模拟搜索
                    distances = np.random.random(k)
                    indices = np.random.randint(0, self.ntotal, k)
                    return distances.reshape(1, -1), indices.reshape(1, -1)
            
            # 模拟数据
            vectors = np.random.random((n_vectors, dimension)).astype('float32')
            
            # 模拟索引
            mock_index = MockFAISS(dimension)
            mock_index.add(vectors)
            
            # 模拟搜索
            query_vector = np.random.random((1, dimension)).astype('float32')
            distances, indices = mock_index.search(query_vector, 5)
            
            print(f"   模拟索引已创建：{n_vectors}个向量")
            print(f"   模拟搜索结果：{indices[0]}")
            print(f"   模拟距离：{distances[0]}")
        
        print("\n✅ FAISS 的主要特点：")
        features = [
            "⚡ 极速检索：毫秒级响应",
            "🎯 多种算法：L2、余弦、内积等",
            "📈 可扩展：支持GPU加速",
            "💾 内存友好：支持索引压缩"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
    except Exception as e:
        print(f"❌ FAISS演示失败：{e}")


def langchain_vectorstore_demo():
    """
    LangChain 向量存储演示
    """
    print("\n" + "="*60)
    print("🔗 LangChain 向量存储集成")
    print("="*60)
    
    try:
        print("🎯 使用 LangChain 的向量存储抽象")
        print("-" * 40)
        
        # 模拟文档数据
        documents = [
            {"content": "LangChain是一个强大的AI应用开发框架", "metadata": {"source": "doc1", "type": "intro"}},
            {"content": "LCEL提供了链式调用的简洁语法", "metadata": {"source": "doc2", "type": "syntax"}},
            {"content": "提示模板帮助构建高质量的AI交互", "metadata": {"source": "doc3", "type": "template"}},
            {"content": "向量存储是RAG系统的核心组件", "metadata": {"source": "doc4", "type": "storage"}},
            {"content": "FAISS提供了高性能的相似性检索", "metadata": {"source": "doc5", "type": "tech"}},
        ]
        
        print(f"📄 准备文档：{len(documents)}个")
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc['content'][:30]}...")
        
        # 模拟嵌入函数
        def mock_embed_documents(texts: List[str]) -> List[List[float]]:
            """模拟文档嵌入"""
            embeddings = []
            for text in texts:
                # 基于文本内容生成确定性向量
                np.random.seed(hash(text) % 1000000)
                embedding = np.random.random(384).tolist()  # 384维向量
                embeddings.append(embedding)
            return embeddings
        
        def mock_embed_query(text: str) -> List[float]:
            """模拟查询嵌入"""
            np.random.seed(hash(text) % 1000000)
            return np.random.random(384).tolist()
        
        # 模拟向量存储类
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.embeddings = []
                self.metadatas = []
            
            def add_documents(self, documents, embeddings=None):
                """添加文档"""
                if embeddings is None:
                    texts = [doc["content"] for doc in documents]
                    embeddings = mock_embed_documents(texts)
                
                self.documents.extend([doc["content"] for doc in documents])
                self.embeddings.extend(embeddings)
                self.metadatas.extend([doc["metadata"] for doc in documents])
                
                print(f"✅ 已添加 {len(documents)} 个文档到向量存储")
            
            def similarity_search(self, query: str, k: int = 3):
                """相似性搜索"""
                query_embedding = mock_embed_query(query)
                
                # 计算相似度（简化版）
                similarities = []
                for i, doc_embedding in enumerate(self.embeddings):
                    # 简单的点积相似度
                    similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    similarities.append((similarity, i))
                
                # 排序并返回top-k
                similarities.sort(reverse=True)
                
                results = []
                for similarity, idx in similarities[:k]:
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadatas[idx],
                        "similarity": similarity
                    })
                
                return results
            
            def similarity_search_with_score(self, query: str, k: int = 3):
                """带分数的相似性搜索"""
                return self.similarity_search(query, k)
        
        # 创建向量存储
        print("\n🏗️  创建向量存储...")
        vectorstore = MockVectorStore()
        
        # 添加文档
        vectorstore.add_documents(documents)
        
        # 测试检索
        print(f"\n🔍 测试相似性检索...")
        
        test_queries = [
            "什么是LangChain框架？",
            "如何使用向量检索？",
            "提示模板的作用",
            "FAISS性能如何？"
        ]
        
        for query in test_queries:
            print(f"\n❓ 查询：{query}")
            results = vectorstore.similarity_search(query, k=2)
            
            print("📋 检索结果：")
            for i, result in enumerate(results, 1):
                content = result["content"][:40] + "..."
                similarity = result["similarity"]
                source = result["metadata"]["source"]
                print(f"   {i}. [{source}] {content} (相似度: {similarity:.3f})")
        
        print(f"\n✅ LangChain 向量存储的优势：")
        advantages = [
            "🔧 统一接口：支持多种向量数据库",
            "📚 文档管理：自动处理文本和元数据",
            "🔍 检索方法：多种搜索策略",
            "🎯 过滤功能：基于元数据过滤"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
    except Exception as e:
        print(f"❌ LangChain向量存储演示失败：{e}")


def advanced_retrieval_strategies():
    """
    高级检索策略演示
    """
    print("\n" + "="*60)
    print("🎯 高级检索策略")
    print("="*60)
    
    print("💡 检索策略对比")
    print("-" * 30)
    
    strategies = {
        "基础相似性检索": {
            "原理": "直接计算查询与文档的向量相似度",
            "优点": "简单快速，易于实现",
            "缺点": "可能返回重复或相关性低的结果",
            "适用": "简单问答，文档相似度查找"
        },
        
        "MMR检索": {
            "原理": "在相似性和多样性之间平衡",
            "优点": "结果多样化，避免重复信息",
            "缺点": "计算复杂度稍高",
            "适用": "综合性问题，需要多角度信息"
        },
        
        "混合检索": {
            "原理": "结合关键词和向量检索",
            "优点": "精确匹配+语义理解",
            "缺点": "需要维护两套索引",
            "适用": "专业术语查找，精确+模糊匹配"
        },
        
        "重排序检索": {
            "原理": "先粗检索，再用模型精确排序",
            "优点": "检索质量最高",
            "缺点": "计算成本高",
            "适用": "高质量要求，批量处理"
        }
    }
    
    for strategy, details in strategies.items():
        print(f"\n🔍 {strategy}")
        print(f"   📖 原理：{details['原理']}")
        print(f"   ✅ 优点：{details['优点']}")
        print(f"   ❌ 缺点：{details['缺点']}")
        print(f"   🎯 适用：{details['适用']}")
    
    # MMR算法演示
    print(f"\n🧪 MMR (最大边际相关性) 算法演示")
    print("-" * 40)
    
    def mmr_search(query_embedding, doc_embeddings, documents, lambda_param=0.5, k=3):
        """MMR检索算法"""
        
        def cosine_similarity(a, b):
            """计算余弦相似度"""
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
        
        selected_docs = []
        remaining_docs = list(range(len(documents)))
        
        while len(selected_docs) < k and remaining_docs:
            mmr_scores = []
            
            for i in remaining_docs:
                # 与查询的相似度
                query_sim = cosine_similarity(query_embedding, doc_embeddings[i])
                
                # 与已选文档的最大相似度
                max_sim_selected = 0
                for selected_idx in selected_docs:
                    sim = cosine_similarity(doc_embeddings[i], doc_embeddings[selected_idx])
                    max_sim_selected = max(max_sim_selected, sim)
                
                # MMR分数：平衡相似性和多样性
                mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_sim_selected
                mmr_scores.append((mmr_score, i))
            
            # 选择MMR分数最高的文档
            mmr_scores.sort(reverse=True)
            best_idx = mmr_scores[0][1]
            selected_docs.append(best_idx)
            remaining_docs.remove(best_idx)
        
        return selected_docs
    
    # 模拟数据
    np.random.seed(42)
    query_embedding = np.random.random(10).tolist()
    doc_embeddings = [np.random.random(10).tolist() for _ in range(6)]
    documents = [
        "LangChain框架介绍",
        "LangChain基础教程", 
        "Python编程入门",
        "机器学习概述",
        "深度学习原理",
        "神经网络结构"
    ]
    
    print("📄 候选文档：")
    for i, doc in enumerate(documents):
        print(f"   {i}. {doc}")
    
    # 对比普通检索和MMR检索
    print(f"\n🔍 普通相似性检索 vs MMR检索：")
    
    # 普通检索：只按相似度排序
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
    
    similarities = [(cosine_similarity(query_embedding, emb), i) for i, emb in enumerate(doc_embeddings)]
    similarities.sort(reverse=True)
    normal_results = [i for _, i in similarities[:3]]
    
    # MMR检索
    mmr_results = mmr_search(query_embedding, doc_embeddings, documents, lambda_param=0.7, k=3)
    
    print(f"\n📊 检索结果对比：")
    print("普通检索结果：")
    for i, idx in enumerate(normal_results, 1):
        print(f"   {i}. {documents[idx]}")
    
    print("MMR检索结果：")
    for i, idx in enumerate(mmr_results, 1):
        print(f"   {i}. {documents[idx]}")
    
    print(f"\n💡 MMR的优势：通过多样性参数平衡相关性和多样性，避免返回过于相似的文档。")


def performance_optimization():
    """
    检索性能优化
    """
    print("\n" + "="*60)
    print("⚡ 检索性能优化技巧")
    print("="*60)
    
    optimization_tips = {
        "1. 索引优化": [
            "选择合适的索引类型（Flat, IVF, HNSW等）",
            "调整索引参数（nlist, M, efConstruction等）", 
            "使用GPU加速（如果可用）",
            "定期重建索引优化性能"
        ],
        
        "2. 向量优化": [
            "降维技术（PCA, t-SNE）减少计算",
            "向量量化压缩存储空间",
            "选择合适的向量维度",
            "批量处理向量操作"
        ],
        
        "3. 检索优化": [
            "缓存热门查询结果",
            "使用检索池限制搜索范围",
            "异步处理提高吞吐量",
            "分布式部署处理高并发"
        ],
        
        "4. 数据优化": [
            "文档去重减少冗余",
            "文档分块策略优化",
            "元数据索引加速过滤",
            "定期清理无效数据"
        ]
    }
    
    for category, tips in optimization_tips.items():
        print(f"\n🎯 {category}")
        print("-" * 30)
        for tip in tips:
            print(f"   💡 {tip}")
    
    # 性能测试示例
    print(f"\n🧪 性能测试模拟")
    print("-" * 30)
    
    import time
    
    def benchmark_search(n_docs, n_queries, vector_dim):
        """模拟检索性能测试"""
        print(f"📊 测试配置：")
        print(f"   文档数量：{n_docs:,}")
        print(f"   查询数量：{n_queries}")
        print(f"   向量维度：{vector_dim}")
        
        # 模拟数据准备时间
        prep_time = 0.1 * n_docs / 1000  # 假设每1000个文档需要0.1秒
        print(f"\n⏱️  数据准备时间：{prep_time:.2f}秒")
        
        # 模拟检索时间
        search_time_per_query = 0.01 + (n_docs / 100000) * 0.05  # 基础时间 + 规模影响
        total_search_time = search_time_per_query * n_queries
        
        print(f"🔍 平均检索时间：{search_time_per_query*1000:.1f}毫秒/查询")
        print(f"📈 总检索时间：{total_search_time:.2f}秒")
        
        # 计算吞吐量
        qps = n_queries / total_search_time if total_search_time > 0 else float('inf')
        print(f"⚡ 检索吞吐量：{qps:.1f} QPS (查询/秒)")
        
        # 内存估算
        memory_mb = (n_docs * vector_dim * 4) / (1024 * 1024)  # float32占4字节
        print(f"💾 内存占用：约{memory_mb:.1f}MB")
        
        return {
            "qps": qps,
            "avg_latency": search_time_per_query * 1000,
            "memory_mb": memory_mb
        }
    
    # 不同规模的性能测试
    test_configs = [
        (1000, 100, 384),      # 小规模
        (10000, 100, 384),     # 中规模  
        (100000, 100, 384),    # 大规模
        (1000000, 100, 384),   # 超大规模
    ]
    
    print("📊 不同规模性能对比：")
    print("-" * 60)
    print(f"{'规模':<10} {'QPS':<8} {'延迟(ms)':<10} {'内存(MB)':<10}")
    print("-" * 60)
    
    for n_docs, n_queries, vector_dim in test_configs:
        result = benchmark_search(n_docs, n_queries, vector_dim)
        scale = f"{n_docs//1000}K" if n_docs >= 1000 else str(n_docs)
        print(f"{scale:<10} {result['qps']:<8.1f} {result['avg_latency']:<10.1f} {result['memory_mb']:<10.1f}")
    
    print(f"\n💡 性能优化建议：")
    recommendations = [
        "📈 小规模(<10K)：使用简单的Flat索引",
        "🚀 中规模(10K-100K)：使用IVF索引",
        "⚡ 大规模(100K+)：使用HNSW索引",
        "🔧 超大规模(1M+)：考虑分布式方案"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")


def next_lesson_preview():
    """
    总结和预告
    """
    print("\n" + "="*60)
    print("🎓 第10节总结 & 后续学习建议")
    print("="*60)
    
    print("🎉 第10节你掌握了：")
    learned = [
        "✅ 理解向量数据库的原理和优势",
        "✅ 掌握FAISS向量存储的使用",
        "✅ 学会高级检索策略（MMR等）",
        "✅ 了解检索性能优化方法"
    ]
    
    for item in learned:
        print(f"   {item}")
    
    print("\n🎊 恭喜完成基础入门课程！")
    print("你已经从零基础成长为LangChain入门专家！")
    
    print("\n🚀 进阶学习建议：")
    next_steps = [
        "🤖 Agent开发：构建智能助手",
        "🔧 工具集成：连接外部API",
        "📊 生产部署：性能优化和监控", 
        "🏢 企业应用：实际项目开发",
        "🌟 开源贡献：参与社区建设"
    ]
    
    for step in next_steps:
        print(f"   {step}")
    
    print("\n📚 推荐资源：")
    resources = [
        "📖 LangChain官方文档深入阅读",
        "💻 GitHub开源项目实践",
        "🎥 技术会议和在线课程",
        "🤝 加入开发者社区讨论",
        "🏗️  构建个人项目作品集"
    ]
    
    for resource in resources:
        print(f"   {resource}")


def main():
    """
    主函数：第10节完整学习流程
    """
    print("🎯 LangChain 入门到精通 - 第10节")
    print("🗄️  向量存储与检索")
    print("📚 前置：完成第1-9节")
    
    # 1. 向量数据库概念
    explain_vector_database()
    
    # 2. FAISS基础演示
    faiss_basic_demo()
    
    # 3. LangChain向量存储
    langchain_vectorstore_demo()
    
    # 4. 高级检索策略
    advanced_retrieval_strategies()
    
    # 5. 性能优化
    performance_optimization()
    
    # 6. 总结和下一步
    next_lesson_preview()
    
    print("\n" + "="*60)
    print("🎉 第10节完成！")
    print("🎊 整个入门课程系列完成！")
    print("🚀 恭喜你成为LangChain专家！")
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
    print("   📖 FAISS官方文档和教程")
    print("   💻 向量数据库性能对比")
    print("   🏗️  大规模检索系统架构")
    print("\n🎓 课程完成！感谢你的学习！")