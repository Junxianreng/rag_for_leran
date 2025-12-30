import os
import warnings
import logging
import uuid
from typing import Dict, List, Generator, Optional
from pathlib import Path

# 屏蔽警告
try:
    from langchain._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    warnings.filterwarnings("ignore", message=".*GuardrailsOutputParser.*deprecated.*")

# 基础配置
CONFIG = {
    "zhipu_api_key": "your_keys",  # 替换为你的key
    "embedding_model_path": r"D:\python_work\master\1\backend\models\bge-large-zh-v1.5",# orther_embedding_model
    "persist_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage"),
    "device": "cpu",  # 有GPU改cuda
    # 支持切换的模型配置（含温度参数）
    "supported_models": {
        "glm-4.5-air": {"model_name": "glm-4.5-air", "temperature": 0.3},
        "glm-3-turbo": {"model_name": "glm-3-turbo", "temperature": 0.5},
    },
    "document_chunk": {"chunk_size": 500, "chunk_overlap": 50},
}

# 基础日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 导入核心依赖
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI # 其他模型需换结构
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 关闭Chroma遥测
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "True"


class RAGService:
    """增强版RAG服务：支持模型切换、流式输出、基础文档管理"""

    def __init__(self):
        self._init_embeddings()
        self._init_vector_db()
        self._init_llm_config()
        logger.info("RAG服务初始化完成")

    def _init_embeddings(self):
        """初始化Embedding模型"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model_path"],
            model_kwargs={"device": CONFIG["device"]}
        )

    def _init_vector_db(self):
        """初始化向量库"""
        self.vectordb = Chroma(
            persist_directory=CONFIG["persist_path"],
            embedding_function=self.embeddings,
            collection_metadata={"description": "RAG文档库"}
        )

    def _init_llm_config(self):
        """初始化LLM配置"""
        self.default_model = "glm-4.5-air"
        self.llm_cache = {}  # 缓存不同模型的LLM实例

    def _get_llm(self, model_name: str = None) -> ChatZhipuAI:
        """获取指定模型的LLM实例（缓存复用）"""
        model_name = model_name or self.default_model
        if model_name not in CONFIG["supported_models"]:
            raise ValueError(f"不支持的模型：{model_name}，可选：{list(CONFIG['supported_models'].keys())}")

        if model_name not in self.llm_cache:
            model_config = CONFIG["supported_models"][model_name]
            self.llm_cache[model_name] = ChatZhipuAI(
                api_key=CONFIG["zhipu_api_key"],
                model=model_config["model_name"],
                temperature=model_config["temperature"],
                streaming=True,  # 开启流式输出
                callbacks=[StreamingStdOutCallbackHandler()]  # 流式回调
            )
        return self.llm_cache[model_name]

    # ---------------------- 文档管理功能 ----------------------
    def upload_document(self, file_path: str) -> Optional[int]:
        """上传并入库文档（带元数据标记，支持PDF/TXT/DOCX）"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在：{file_path}")
            return None

        # 1. 选择加载器
        suffix = Path(file_path).suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif suffix == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif suffix == ".docx":
                loader = Docx2txtLoader(file_path)
            else:
                logger.error(f"不支持的格式：{suffix}")
                return None

            # 2. 加载+分割
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CONFIG["document_chunk"]["chunk_size"],
                chunk_overlap=CONFIG["document_chunk"]["chunk_overlap"]
            )
            splits = splitter.split_documents(docs)

            # 3. 添加文档元数据（用于管理）
            doc_id = str(uuid.uuid4())  # 唯一文档ID
            file_name = Path(file_path).name
            for split in splits:
                split.metadata.update({
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "upload_time": os.path.getctime(file_path)
                })

            # 4. 入库
            self.vectordb.add_documents(splits)
            logger.info(f"文档[{file_name}]上传成功，分割为{len(splits)}段，doc_id={doc_id}")
            return len(splits)
        except Exception as e:
            logger.error(f"上传文档失败：{e}")
            return None

    def list_documents(self) -> List[Dict]:
        """列出已上传的文档（去重）"""
        try:
            # 获取所有文档元数据
            all_metadata = self.vectordb.get()["metadatas"]
            # 按doc_id去重，提取关键信息
            doc_map = {}
            for meta in all_metadata:
                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "file_name": meta.get("file_name", "未知文件"),
                        "upload_time": meta.get("upload_time", 0)
                    }
            return list(doc_map.values())
        except Exception as e:
            logger.error(f"查询文档列表失败：{e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """根据doc_id删除文档（删除所有关联片段）"""
        try:
            # 筛选该文档的所有片段ID
            all_ids = self.vectordb.get()["ids"]
            all_metas = self.vectordb.get()["metadatas"]
            delete_ids = [
                idx for idx, meta in zip(all_ids, all_metas)
                if meta.get("doc_id") == doc_id
            ]
            if not delete_ids:
                logger.warning(f"未找到doc_id={doc_id}的文档片段")
                return False

            # 删除片段
            self.vectordb.delete(ids=delete_ids)
            logger.info(f"删除doc_id={doc_id}的文档，共删除{len(delete_ids)}段")
            return True
        except Exception as e:
            logger.error(f"删除文档失败：{e}")
            return False

    # ---------------------- 流式问答功能 ----------------------
    def chat_stream(self, query: str, model_name: str = None) -> Generator[str, None, None]:
        """真流式问答：LLM实时生成、逐token返回（能感觉到打字效果）"""
        if not query.strip():
            yield "请输入有效的问题！"
            return

        try:
            # 1. 先检索上下文（和原来一样，获取文档内容）
            retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # 2. 构建Prompt
            prompt_template = """使用以下上下文来回答问题。不知道就说不知道。
    上下文: {context}
    问题: {question}
    回答:"""
            final_prompt = prompt_template.format(context=context, question=query)

            # 3. 调用LLM原生流式接口（关键：不用QA链，直接stream）
            llm = self._get_llm(model_name)
            # 实时获取LLM生成的每个token（字符/词）
            for token in llm.stream(final_prompt):
                # 逐token返回，粒度最小，流式效果最明显
                yield token.content

            # 4. 补充参考来源（单独返回，不影响主回答的流式效果）
            sources = [f"来源{idx + 1}：{doc.page_content[:100]}..." for idx, doc in enumerate(relevant_docs)]
            yield "\n\n参考来源：\n" + "\n".join(sources)

        except Exception as e:
            logger.error(f"流式问答失败：{e}")
            yield f"问答出错：{str(e)}"

    # 兼容非流式调用
    def chat(self, query: str, model_name: str = None) -> Dict:
        """非流式问答（一次性返回）"""
        full_answer = ""
        for chunk in self.chat_stream(query, model_name):
            full_answer += chunk
        return {"answer": full_answer}


# 单例实例
rag_service = RAGService()
