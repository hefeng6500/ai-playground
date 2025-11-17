"""
RAG (Retrieval-Augmented Generation) 应用
==========================================

这是一个完整的 RAG 应用，支持：
- 文档加载和预处理
- 向量存储和检索
- 基于检索的生成
- 交互式问答

使用方法：
1. 安装依赖：pip install -r requirements.txt
2. 设置环境变量（如 OPENAI_API_KEY 或其他模型 API key）
3. 运行：python main.py
"""

import os
import getpass
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)


class RAGApplication:
    """RAG 应用主类"""

    def __init__(
        self,
        embeddings_model=None,
        llm_model=None,
        vector_store_path: str = "./vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化 RAG 应用

        Args:
            embeddings_model: 嵌入模型实例
            llm_model: LLM 模型实例
            vector_store_path: 向量存储路径
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
        """
        self.embeddings = embeddings_model
        self.llm = llm_model
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # RAG 提示模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个有用的AI助手。请基于以下上下文信息回答用户的问题。
如果上下文中没有相关信息，请诚实地说你不知道，不要编造答案。

上下文信息：
{context}

请用中文回答。""",
                ),
                ("human", "{question}"),
            ]
        )

    def _setup_models(self):
        """设置模型（如果未提供）"""
        if self.embeddings is None:
            # 尝试使用 OpenAI
            try:
                from langchain_openai import OpenAIEmbeddings

                if not os.environ.get("OPENAI_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = getpass.getpass(
                        "Enter OpenAI API key: "
                    )
                self.embeddings = OpenAIEmbeddings()
                print("✅ 使用 OpenAI Embeddings")
            except Exception as e:
                # 尝试使用本地模型或其他服务
                try:
                    from langchain_siliconflow import SiliconFlowEmbeddings

                    if not os.environ.get("SILICONFLOW_API_KEY"):
                        os.environ["SILICONFLOW_API_KEY"] = getpass.getpass(
                            "Enter SiliconFlow API key: "
                        )
                    self.embeddings = SiliconFlowEmbeddings()
                    print("✅ 使用 SiliconFlow Embeddings")
                except Exception:
                    print(f"❌ 无法初始化嵌入模型: {e}")
                    print("请安装 langchain-openai 或 langchain-siliconflow")
                    raise

        if self.llm is None:
            # 尝试使用 OpenAI
            try:
                from langchain_openai import ChatOpenAI

                if not os.environ.get("OPENAI_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = getpass.getpass(
                        "Enter OpenAI API key: "
                    )
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                print("✅ 使用 OpenAI Chat Model")
            except Exception as e:
                # 尝试使用其他模型
                try:
                    from langchain.chat_models import init_chat_model

                    self.llm = init_chat_model("gpt-4o-mini")
                    print("✅ 使用 LangChain Chat Model")
                except Exception as ex:
                    print(f"❌ 无法初始化 LLM 模型: {e}")
                    print(f"详细错误: {ex}")
                    print("请安装 langchain-openai 或配置其他模型")
                    raise

    def load_documents(
        self,
        source: str,
        file_type: Optional[str] = None,
    ) -> List[Document]:
        """
        加载文档

        Args:
            source: 文档路径（文件或目录）
            file_type: 文件类型（可选，自动检测）

        Returns:
            文档列表
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"路径不存在: {source}")

        documents = []

        # 如果是目录，加载所有支持的文档
        if source_path.is_dir():
            # 支持的文档类型
            loaders = {
                "*.txt": TextLoader,
                "*.pdf": PyPDFLoader,
                "*.md": UnstructuredMarkdownLoader,
            }

            for pattern, loader_class in loaders.items():
                try:
                    loader = DirectoryLoader(
                        str(source_path),
                        glob=pattern,
                        loader_cls=loader_class,
                        show_progress=True,
                    )
                    docs = loader.load()
                    documents.extend(docs)
                    if docs:
                        print(f"✅ 加载了 {len(docs)} 个 {pattern} 文件")
                except Exception as e:
                    print(f"⚠️ 加载 {pattern} 文件时出错: {e}")

        # 如果是单个文件
        elif source_path.is_file():
            file_ext = source_path.suffix.lower()

            loader_map = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader,
                ".md": UnstructuredMarkdownLoader,
            }

            loader_class = loader_map.get(file_ext)
            if loader_class is None:
                raise ValueError(f"不支持的文件类型: {file_ext}")

            try:
                loader = loader_class(str(source_path))
                documents = loader.load()
                print(f"✅ 加载了文档: {source_path.name}")
            except Exception as e:
                raise Exception(f"加载文档失败: {e}")

        return documents

    def build_vector_store(
        self,
        documents: List[Document],
        force_rebuild: bool = False,
    ):
        """
        构建向量存储

        Args:
            documents: 文档列表
            force_rebuild: 是否强制重建
        """
        if not documents:
            raise ValueError("文档列表为空")

        self._setup_models()

        # 检查是否已存在向量存储
        vector_store_file = Path(self.vector_store_path) / "index.faiss"
        if vector_store_file.exists() and not force_rebuild:
            print(f"📂 加载现有向量存储: {self.vector_store_path}")
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("✅ 向量存储加载成功")
            except Exception as e:
                print(f"⚠️ 加载失败，将重新构建: {e}")
                force_rebuild = True

        if force_rebuild or self.vector_store is None:
            print("🔄 开始构建向量存储...")

            # 分割文档
            print(f"📄 分割 {len(documents)} 个文档...")
            splits = self.text_splitter.split_documents(documents)
            print(f"✅ 分割为 {len(splits)} 个文档块")

            # 创建向量存储
            print("🔢 生成向量嵌入...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)

            # 保存向量存储
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            print(f"✅ 向量存储已保存到: {self.vector_store_path}")

        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},  # 检索前 4 个最相关的文档
        )

        # 构建 RAG 链
        self._build_rag_chain()

    def _build_rag_chain(self):
        """构建 RAG 链"""

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG 链构建完成")

    def query(self, question: str, verbose: bool = True) -> str:
        """
        查询问题

        Args:
            question: 用户问题
            verbose: 是否显示详细信息

        Returns:
            回答
        """
        if self.rag_chain is None:
            raise ValueError("请先构建向量存储（调用 build_vector_store）")

        if verbose:
            print(f"\n🔍 检索相关文档...")
            # 先检索相关文档
            relevant_docs = self.retriever.get_relevant_documents(question)
            print(f"📚 找到 {len(relevant_docs)} 个相关文档片段")
            for i, doc in enumerate(relevant_docs[:3], 1):  # 只显示前3个
                preview = doc.page_content[:100].replace("\n", " ")
                print(f"   {i}. {preview}...")

        if verbose:
            print(f"🧠 生成回答...")

        # 生成回答
        answer = self.rag_chain.invoke(question)

        return answer

    def add_documents(self, documents: List[Document]):
        """
        添加新文档到向量存储

        Args:
            documents: 新文档列表
        """
        if self.vector_store is None:
            raise ValueError("请先构建向量存储")

        # 分割新文档
        splits = self.text_splitter.split_documents(documents)

        # 添加到向量存储
        self.vector_store.add_documents(splits)

        # 保存更新后的向量存储
        self.vector_store.save_local(self.vector_store_path)

        # 重新创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # 重新构建 RAG 链
        self._build_rag_chain()

        print(f"✅ 已添加 {len(splits)} 个文档块到向量存储")


def main():
    """主函数 - 交互式 RAG 应用"""
    print("=" * 60)
    print("🚀 RAG 应用启动")
    print("=" * 60)

    # 创建 RAG 应用实例
    rag = RAGApplication(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # 检查是否已有向量存储
    vector_store_file = Path(rag.vector_store_path) / "index.faiss"

    if not vector_store_file.exists():
        print("\n📚 首次使用，需要加载文档构建知识库")
        print("请输入文档路径（文件或目录）：")
        doc_path = input("> ").strip()

        if not doc_path:
            # 使用示例文档
            print("⚠️ 未提供路径，创建示例文档...")
            example_doc = Document(
                page_content="""
                LangChain 是一个用于构建 LLM 应用的框架。
                
                主要功能包括：
                1. 文档加载和处理
                2. 向量存储和检索
                3. 链式调用和组合
                4. Agent 和工具集成
                
                RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术。
                它通过检索相关文档来增强 LLM 的生成能力，使回答更加准确和可靠。
                """,
                metadata={"source": "example.txt"},
            )
            documents = [example_doc]
        else:
            try:
                documents = rag.load_documents(doc_path)
            except Exception as e:
                print(f"❌ 加载文档失败: {e}")
                return

        # 构建向量存储
        try:
            rag.build_vector_store(documents, force_rebuild=True)
        except Exception as e:
            print(f"❌ 构建向量存储失败: {e}")
            return
    else:
        print(f"\n📂 检测到现有向量存储: {rag.vector_store_path}")
        print("加载向量存储...")
        try:
            rag._setup_models()
            rag.vector_store = FAISS.load_local(
                rag.vector_store_path,
                rag.embeddings,
                allow_dangerous_deserialization=True,
            )
            rag.retriever = rag.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )
            rag._build_rag_chain()
            print("✅ 向量存储加载成功")
        except Exception as e:
            print(f"❌ 加载向量存储失败: {e}")
            return

    # 交互式问答
    print("\n" + "=" * 60)
    print("💬 开始问答（输入 'quit' 或 'exit' 退出）")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "退出", "q"]:
                print("\n👋 再见！")
                break

            # 查询并显示回答
            answer = rag.query(question, verbose=True)
            print(f"\n🤖 回答: {answer}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
