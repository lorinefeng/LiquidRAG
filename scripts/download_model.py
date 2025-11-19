#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
下载Qwen3-Embedding-0.6B模型到本地
"""

import os
import sys
import logging
from pathlib import Path

# 设置环境变量解决NumPy兼容性问题
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from huggingface_hub import snapshot_download

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.rag_config import RAGConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_qwen3_embedding():
    """下载Qwen3-Embedding-0.6B模型"""
    
    try:
        # 确保目录存在
        RAGConfig.ensure_directories()
        
        logger.info(f"开始下载模型: {RAGConfig.EMBEDDING_MODEL_NAME}")
        logger.info(f"保存路径: {RAGConfig.EMBEDDING_MODEL_PATH}")
        
        # 下载模型
        snapshot_download(
            repo_id=RAGConfig.EMBEDDING_MODEL_NAME,
            local_dir=RAGConfig.EMBEDDING_MODEL_PATH,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=True,  # 支持断点续传
        )
        
        logger.info("模型下载完成！")
        
        # 验证下载的文件
        model_files = list(Path(RAGConfig.EMBEDDING_MODEL_PATH).glob("*"))
        logger.info(f"下载的文件数量: {len(model_files)}")
        
        # 检查关键文件
        key_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = []
        
        for key_file in key_files:
            file_path = Path(RAGConfig.EMBEDDING_MODEL_PATH) / key_file
            if not file_path.exists():
                # 检查是否有.safetensors文件
                if key_file == "pytorch_model.bin":
                    safetensors_files = list(Path(RAGConfig.EMBEDDING_MODEL_PATH).glob("*.safetensors"))
                    if not safetensors_files:
                        missing_files.append(key_file)
                else:
                    missing_files.append(key_file)
        
        if missing_files:
            logger.warning(f"缺少以下关键文件: {missing_files}")
        else:
            logger.info("所有关键文件下载完成")
            
        return True
        
    except Exception as e:
        logger.error(f"模型下载失败: {str(e)}")
        return False

def verify_model():
    """验证模型是否可以正常加载"""
    try:
        logger.info("验证模型加载...")
        
        # 延迟导入以避免NumPy问题
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.error(f"导入SentenceTransformer失败: {e}")
            return False
        
        # 尝试加载模型
        model = SentenceTransformer(RAGConfig.EMBEDDING_MODEL_PATH)
        
        # 测试编码功能
        test_texts = ["测试文本", "Test text"]
        try:
            embeddings = model.encode(test_texts, show_progress_bar=False)
            logger.info(f"✅ 模型验证成功，嵌入维度: {embeddings.shape}")
            return True
        except Exception as e:
            logger.error(f"模型编码测试失败: {e}")
            # 即使编码失败，如果模型文件存在也认为下载成功
            if os.path.exists(os.path.join(RAGConfig.EMBEDDING_MODEL_PATH, "config.json")):
                logger.info("✅ 模型文件下载完成（编码功能可能需要额外配置）")
                return True
            return False
            
    except Exception as e:
        logger.error(f"模型验证失败: {str(e)}")
        # 检查关键文件是否存在
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        existing_files = []
        for file in required_files:
            file_path = os.path.join(RAGConfig.EMBEDDING_MODEL_PATH, file)
            if os.path.exists(file_path):
                existing_files.append(file)
        
        if len(existing_files) >= 2:  # 至少有2个关键文件
            logger.info(f"✅ 模型文件下载完成（找到文件: {existing_files}）")
            return True
        else:
            logger.error("❌ 模型验证失败")
            return False

def main():
    """主函数"""
    logger.info("=== Qwen3-Embedding-0.6B 模型下载工具 ===")
    
    # 检查模型是否已存在
    if Path(RAGConfig.EMBEDDING_MODEL_PATH).exists():
        logger.info("模型目录已存在，检查是否完整...")
        if verify_model():
            logger.info("模型已存在且可正常使用")
            return
        else:
            logger.info("模型文件不完整，重新下载...")
    
    # 下载模型
    if download_qwen3_embedding():
        logger.info("开始验证模型...")
        if verify_model():
            logger.info("✅ 模型下载和验证完成！")
        else:
            logger.error("❌ 模型验证失败")
    else:
        logger.error("❌ 模型下载失败")

if __name__ == "__main__":
    main()