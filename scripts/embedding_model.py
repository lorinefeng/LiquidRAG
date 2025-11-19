#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型模块
使用Qwen3-Embedding-0.6B进行文本向量化
"""

import os
import sys
import logging
import torch
from typing import List, Union, Optional, Dict, Any

# 设置环境变量解决兼容性问题
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.rag_config import RAGConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EmbeddingModel:
    """文本嵌入模型类"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化嵌入模型
        
        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()
        self.model = None
        self.device = self._get_device()
        
        logging.info(f"使用设备: {self.device}")
        
        # 延迟加载模型
        try:
            self._load_model()
            
            # 测试模型
            if self.test_model():
                logging.info("✅ 嵌入模型初始化成功")
            else:
                logging.warning("⚠️ 模型测试失败，但模型已加载")
                
        except Exception as e:
            logging.error(f"加载嵌入模型失败: {e}")
            raise
    
    def _get_device(self) -> str:
        """
        获取计算设备
        
        Returns:
            设备名称
        """
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            # 延迟导入以避免NumPy问题
            from sentence_transformers import SentenceTransformer
            import time
            
            model_path = self.config.EMBEDDING_MODEL_PATH
            
            if not os.path.exists(model_path):
                logging.error(f"模型路径不存在: {model_path}")
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            logging.info(f"正在加载嵌入模型: {model_path}")
            start_time = time.time()
            
            # 加载模型
            self.model = SentenceTransformer(model_path, device=self.device)
            
            # 设置模型为评估模式
            self.model.eval()
            
            load_time = time.time() - start_time
            logging.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            logging.error(f"加载嵌入模型失败: {e}")
            raise
    
    def test_model(self) -> bool:
        """
        测试模型功能
        
        Returns:
            测试是否成功
        """
        try:
            test_texts = [
                "这是一个测试文本",
                "机器学习是人工智能的重要分支"
            ]
            
            logging.info("开始模型功能测试...")
            
            # 测试编码
            embeddings = self.encode_texts(test_texts)
            
            if embeddings is not None and embeddings.shape[0] == len(test_texts):
                logging.info(f"✅ 编码测试成功，向量形状: {embeddings.shape}")
                
                # 测试相似度计算
                similarity = self.compute_similarity(test_texts[0], test_texts[1])
                logging.info(f"✅ 相似度计算成功: {similarity:.4f}")
                
                return True
            else:
                logging.error("❌ 编码测试失败")
                return False
                
        except Exception as e:
            logging.error(f"❌ 模型测试失败: {e}")
            return False
    
    def encode_texts(self, texts: Union[str, List[str]], 
                    batch_size: int = None,
                    show_progress: bool = False) -> torch.Tensor:
        """
        编码文本为向量
        
        Args:
            texts: 文本或文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度
            
        Returns:
            文本向量数组
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            # 确保输入是列表
            if isinstance(texts, str):
                texts = [texts]
            
            # 设置批处理大小
            if batch_size is None:
                batch_size = min(32, len(texts))
            
            # 编码文本
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            return embeddings
            
        except Exception as e:
            logging.error(f"文本编码失败: {e}")
            # 返回零向量作为fallback
            if isinstance(texts, str):
                texts = [texts]
            return torch.zeros((len(texts), self.config.EMBEDDING_DIMENSION), dtype=torch.float32, device=self.device)
    
    def encode_single(self, text: str) -> torch.Tensor:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本嵌入向量
        """
        embeddings = self.encode_texts([text], show_progress=False)
        return embeddings[0] if len(embeddings) > 0 else torch.tensor([])
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            余弦相似度 (0-1之间)
        """
        embeddings = self.encode_texts([text1, text2], show_progress=False)
        
        if len(embeddings) != 2:
            return 0.0
        
        # 计算余弦相似度
        similarity = torch.dot(embeddings[0], embeddings[1])
        return float(similarity.item())
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {}
        
        return {
            'model_name': self.config.EMBEDDING_MODEL_NAME,
            'model_path': self.config.EMBEDDING_MODEL_PATH,
            'embedding_dim': self.config.EMBEDDING_DIMENSION,
            'device': str(self.device),
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown')
        }

def main():
    """测试嵌入模型"""
    try:
        # 初始化模型
        embedding_model = EmbeddingModel()
        
        # 测试文本
        test_texts = [
            "自然语言处理是人工智能的重要分支",
            "Transformer模型在NLP任务中表现出色",
            "BERT是基于Transformer的预训练模型",
            "今天天气很好，适合出门散步"
        ]
        
        # 编码测试
        logging.info("开始编码测试...")
        embeddings = embedding_model.encode_texts(test_texts)
        
        logging.info(f"编码结果形状: {embeddings.shape}")
        
        # 相似度测试
        logging.info("\n相似度测试:")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                similarity = embedding_model.compute_similarity(test_texts[i], test_texts[j])
                logging.info(f"文本{i+1} vs 文本{j+1}: {similarity:.4f}")
        
        # 模型信息
        model_info = embedding_model.get_model_info()
        logging.info(f"\n模型信息: {model_info}")
        
        logging.info("嵌入模型测试完成！")
        
    except Exception as e:
        logging.error(f"测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()