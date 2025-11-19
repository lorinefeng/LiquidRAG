# RAG系统配置文件
import os

class RAGConfig:
    """RAG系统配置类"""
    
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 目录配置
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
    DOCS_SOURCE_DIR = os.path.join(PROJECT_ROOT, "learn-nlp-with-transformers-main", 
                                   "learn-nlp-with-transformers-main", "docs")
    
    # 嵌入模型配置
    EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "Qwen3-Embedding-0.6B")
    EMBEDDING_DIMENSION = 1024
    
    # 文本分割配置
    CHUNK_SIZE = 800  # 中文适配，参考文档建议500-800字符
    CHUNK_OVERLAP = 100  # 重叠部分，参考文档建议50-100字符
    
    # ChromaDB配置
    CHROMA_DB_PATH = os.path.join(INDEX_DIR, "chroma_db")
    COLLECTION_NAME = "nlp_transformers_docs"
    
    # 检索配置
    TOP_K = 5  # 检索返回的文档数量
    SIMILARITY_THRESHOLD = 0.3  # 相似度阈值，降低以获得更多结果
    
    # LFM2模型配置
    LLM_MODEL_NAME = "LiquidAI/LFM2-1.2B"
    LLM_MODEL_PATH = os.path.join(MODELS_DIR, "LiquidAI_LFM2-1.2B")  # 本地LFM2模型路径
    
    # LFM2模型加载配置（与evaluate_lfm2.py保持一致）
    LFM2_DTYPE = "fp16"  # 数据类型：bf16/fp16/fp32
    LFM2_MAX_MEMORY_GB = 7.0  # 最大显存使用量
    LFM2_USE_LOCAL = True  # 优先使用本地模型
    LFM2_FORCE_DOWNLOAD = False  # 是否强制重新下载
    LLM_MAX_NEW_TOKENS = 1024  # 服务生成默认长度（影响GPU负载）
    
    # 性能配置
    MAX_RESPONSE_TIME = 2.0  # 最大响应时间（秒）
    BATCH_SIZE = 32  # 批处理大小
    
    # 支持的文件格式
    SUPPORTED_EXTENSIONS = ['.md', '.pdf', '.docx', '.txt']
    
    # 缓存配置
    ENABLE_CACHE = True
    CACHE_SIZE = 1000
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必需的目录存在"""
        directories = [
            cls.MODELS_DIR,
            cls.DATA_DIR, 
            cls.INDEX_DIR,
            cls.CHROMA_DB_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    @classmethod
    def get_config_summary(cls):
        """获取配置摘要"""
        return {
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "top_k": cls.TOP_K,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "supported_formats": cls.SUPPORTED_EXTENSIONS
        }