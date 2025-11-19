#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库构建脚本
处理learn-nlp-with-transformers文档，构建向量索引
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.rag_config import RAGConfig
from scripts.document_loader import DocumentLoader, DocumentChunk
from scripts.vector_store import VectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, config: RAGConfig = None):
        """
        初始化知识库构建器
        
        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()
        self.document_loader = DocumentLoader(self.config)
        self.vector_store = VectorStore(self.config)
    
    def build_knowledge_base(self, source_dir: str, 
                           reset_existing: bool = False) -> bool:
        """
        构建知识库
        
        Args:
            source_dir: 源文档目录
            reset_existing: 是否重置现有数据
            
        Returns:
            是否成功
        """
        try:
            logging.info("=" * 60)
            logging.info("开始构建知识库")
            logging.info("=" * 60)
            
            start_time = time.time()
            
            # 检查源目录
            if not os.path.exists(source_dir):
                logging.error(f"源目录不存在: {source_dir}")
                return False
            
            # 重置现有数据（如果需要）
            if reset_existing:
                logging.info("重置现有向量数据库...")
                self.vector_store.reset_collection()
            
            # 1. 加载文档
            logging.info(f"从目录加载文档: {source_dir}")
            documents = self.document_loader.load_documents(source_dir)
            
            if not documents:
                logging.warning("没有找到任何文档")
                return False
            
            # 显示文档统计
            self._show_document_stats(documents)
            
            # 添加到向量数据库
            logging.info("添加文档到向量数据库...")
            success = self.vector_store.add_documents(documents)
            
            if not success:
                logging.error("添加文档到向量数据库失败")
                return False
            
            # 显示最终统计
            self._show_final_stats()
            
            total_time = time.time() - start_time
            logging.info("=" * 60)
            logging.info(f"知识库构建完成！总耗时: {total_time:.2f}秒")
            logging.info("=" * 60)
            
            return True
            
        except Exception as e:
            logging.error(f"构建知识库失败: {e}")
            return False
    
    def _show_document_stats(self, documents: List[DocumentChunk]):
        """显示文档统计信息"""
        stats = self.document_loader.get_document_stats(documents)
        
        logging.info("\n文档加载统计:")
        logging.info(f"  总文档块数: {stats['total_chunks']}")
        logging.info(f"  总字符数: {stats['total_characters']}")
        logging.info(f"  唯一文件数: {stats['unique_files']}")
        logging.info(f"  平均块大小: {stats['avg_chunk_size']:.1f} 字符")
        
        logging.info("  文件类型分布:")
        for file_type, count in stats['file_types'].items():
            logging.info(f"    {file_type}: {count} 个块")
        
        # 显示文件列表
        sources = set(doc.source_file for doc in documents)
        logging.info(f"\n处理的文件列表 ({len(sources)} 个文件):")
        for i, source in enumerate(sorted(sources), 1):
            logging.info(f"  {i:2d}. {source}")
    
    def _show_final_stats(self):
        """显示最终统计信息"""
        stats = self.vector_store.get_collection_stats()
        
        logging.info("\n向量数据库统计:")
        logging.info(f"  总文档数: {stats.get('total_documents', 0)}")
        logging.info(f"  唯一源文件: {stats.get('unique_sources', 0)}")
        logging.info(f"  集合名称: {stats.get('collection_name', 'N/A')}")
        logging.info(f"  索引目录: {stats.get('index_directory', 'N/A')}")
        
        if stats.get('file_types'):
            logging.info("  文件类型分布:")
            for file_type, count in stats['file_types'].items():
                logging.info(f"    {file_type}: {count}")
    
    def test_retrieval(self, test_queries: List[str] = None) -> bool:
        """
        测试检索功能
        
        Args:
            test_queries: 测试查询列表
            
        Returns:
            是否成功
        """
        try:
            # 默认测试查询
            if test_queries is None:
                test_queries = [
                    "transformer模型的注意力机制",
                    "BERT预训练模型",
                    "自然语言处理任务",
                    "文本分类方法",
                    "序列标注技术",
                    "问答系统实现",
                    "机器翻译模型",
                    "摘要生成算法"
                ]
            
            logging.info("\n" + "=" * 60)
            logging.info("开始检索测试")
            logging.info("=" * 60)
            
            total_time = 0
            
            for i, query in enumerate(test_queries, 1):
                logging.info(f"\n测试查询 {i}: '{query}'")
                
                start_time = time.time()
                results = self.vector_store.search(query, top_k=3)
                search_time = time.time() - start_time
                total_time += search_time
                
                logging.info(f"检索耗时: {search_time:.3f}秒")
                logging.info(f"找到 {len(results)} 个相关结果")
                
                # 显示前3个结果
                for j, result in enumerate(results[:3], 1):
                    logging.info(f"  结果 {j}:")
                    logging.info(f"    相似度: {result['similarity']:.4f}")
                    logging.info(f"    来源: {result['metadata']['source']}")
                    logging.info(f"    内容: {result['content'][:150]}...")
                
                # 检查性能要求
                if search_time > self.config.MAX_RESPONSE_TIME:
                    logging.warning(f"检索时间超过阈值 ({self.config.MAX_RESPONSE_TIME}秒)")
            
            avg_time = total_time / len(test_queries)
            logging.info(f"\n检索性能统计:")
            logging.info(f"  总查询数: {len(test_queries)}")
            logging.info(f"  总耗时: {total_time:.3f}秒")
            logging.info(f"  平均耗时: {avg_time:.3f}秒")
            logging.info(f"  性能要求: < {self.config.MAX_RESPONSE_TIME}秒")
            
            if avg_time <= self.config.MAX_RESPONSE_TIME:
                logging.info("✅ 性能测试通过")
            else:
                logging.warning("⚠️ 性能测试未通过")
            
            return True
            
        except Exception as e:
            logging.error(f"检索测试失败: {e}")
            return False

def main():
    """主函数"""
    try:
        # 初始化配置
        config = RAGConfig()
        
        # 检查源文档目录
        source_dir = config.DOCS_SOURCE_DIR
        if not os.path.exists(source_dir):
            logging.error(f"源文档目录不存在: {source_dir}")
            logging.info("请确保已正确设置 DOCS_SOURCE_DIR 路径")
            return False
        
        # 创建知识库构建器
        builder = KnowledgeBaseBuilder(config)
        
        # 询问是否重置现有数据
        stats = builder.vector_store.get_collection_stats()
        existing_docs = stats.get('total_documents', 0)
        
        if existing_docs > 0:
            logging.info(f"检测到现有数据: {existing_docs} 个文档")
            reset = input("是否重置现有数据？(y/N): ").lower().strip() == 'y'
        else:
            reset = False
        
        # 构建知识库
        success = builder.build_knowledge_base(source_dir, reset_existing=reset)
        
        if not success:
            logging.error("知识库构建失败")
            return False
        
        # 测试检索功能
        test_retrieval = input("是否进行检索测试？(Y/n): ").lower().strip() != 'n'
        
        if test_retrieval:
            builder.test_retrieval()
        
        logging.info("知识库构建和测试完成！")
        return True
        
    except KeyboardInterrupt:
        logging.info("用户中断操作")
        return False
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        return False

if __name__ == "__main__":
    main()