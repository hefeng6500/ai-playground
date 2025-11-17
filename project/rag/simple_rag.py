from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# 1. 加载文档
loader = TextLoader("./data/LangChain.md")
docs = loader.load()

# print(docs)
# print(len(docs))
# print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content)

# 2.文档切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_text(docs[0].page_content)


# 3. 创建向量存储和检索器
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
database = FAISS.from_texts(texts, embeddings)
retriever = database.as_retriever()

# 4. 创建 prompt 模板
prompt = """
  你是一个 AI 助手，你需要根据提供的上下文回答问题。
  请勿使用任何外部信息。
  请使用中文回答。
  上下文: {context}
  问题: {question}
"""

# 5. 查询

query = "什么是 LangChain?"

temp_docs = retriever.invoke(query)

print(temp_docs)

context = "".join([doc.page_content for doc in temp_docs])

print(context)

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.3)

result = llm.invoke(prompt.format(context=context, question=query))

result.pretty_print()
