"""
RAG 更原生的实现方式
使用  sentence_transformers 和 sklearn.neighbors 实现向量检索
使用 openai 实现对话
"""

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

# ===== 1. 加载知识库文本 =====
docs = open("./data/LangChain.md", "r", encoding="utf8").read()
chunks = [c.strip() for c in docs.split("\n\n") if c.strip()]

# ===== 2. 嵌入模型 =====
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)  # all-MiniLM-L6-v2 模型：384 维向量，速度快，效果好
embeddings = model.encode(chunks)

# ===== 3. 建立向量检索器 =====
nn = NearestNeighbors(
    n_neighbors=3, metric="cosine"
)  # metric="cosine" 使用余弦相似度，cos = A · B / (|A|*|B|)
nn.fit(embeddings)  # 训练向量检索器


def retrieve(q):
    q_emb = model.encode([q])
    # return_distance=False 返回索引
    idx = nn.kneighbors(q_emb, return_distance=False)[0]
    return [chunks[i] for i in idx]


client = OpenAI()


def rag(query):
    ctx = "\n".join(retrieve(query))
    prompt = f"根据以下知识回答问题：\n{ctx}\n\n问题：{query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


print(rag("什么是 LangChain?"))
