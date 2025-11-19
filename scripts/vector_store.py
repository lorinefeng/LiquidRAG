#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储模块
基于ChromaDB构建向量数据库，支持文档存储、检索和管理
"""

import os
import sys
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import chromadb
from chromadb.config import Settings
import time
import hashlib  # 引入hashlib用于内容哈希，目的：计算文本内容的稳定指纹以进行去重

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.rag_config import RAGConfig
from scripts.document_loader import DocumentChunk
from scripts.document_loader import DocumentLoader
from scripts.embedding_model import EmbeddingModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VectorStore:
    """向量存储类"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化向量存储
        
        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        self._init_chromadb()
        self._init_embedding_model()
    
    def _init_chromadb(self):
        """初始化ChromaDB"""
        try:
            # 确保索引目录存在
            os.makedirs(self.config.INDEX_DIR, exist_ok=True)
            
            # 配置ChromaDB
            settings = Settings(
                persist_directory=self.config.INDEX_DIR,
                anonymized_telemetry=False
            )
            
            # 创建客户端
            self.client = chromadb.PersistentClient(
                path=self.config.INDEX_DIR,
                settings=settings
            )
            
            # 获取或创建集合
            try:
                self.collection = self.client.get_collection(
                    name=self.config.COLLECTION_NAME
                )
                logging.info(f"加载现有集合: {self.config.COLLECTION_NAME}")
            except:
                self.collection = self.client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    metadata={"description": "RAG system document collection"}
                )
                logging.info(f"创建新集合: {self.config.COLLECTION_NAME}")
            
            logging.info("ChromaDB初始化成功")
            
        except Exception as e:
            logging.error(f"ChromaDB初始化失败: {e}")
            raise
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            self.embedding_model = EmbeddingModel(self.config)
            logging.info("嵌入模型初始化成功")
        except Exception as e:
            logging.error(f"嵌入模型初始化失败: {e}")
            raise
    
    def add_document_file(self, file_path: str) -> int:
        """
        解析单文件并写入索引，仅支持 .md/.txt/.pdf/.docx
        Returns: 写入的块数量
        """
        loader = DocumentLoader(self.config)
        try:
            from pathlib import Path
            chunks = loader._load_single_document(Path(file_path))
            ok = self.add_documents(chunks)
            return len(chunks) if ok else 0
        except Exception as e:
            logging.error(f"添加文件失败 {file_path}: {e}")
            return 0
    
    def add_documents(self, documents: List[DocumentChunk], 
                     batch_size: int = 100) -> bool:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档块列表
            batch_size: 批处理大小
            
        Returns:
            是否成功
        """
        if not documents:
            logging.warning("没有文档需要添加")
            return True
        
        try:
            logging.info(f"开始添加 {len(documents)} 个文档块到向量数据库")
            start_time = time.time()
            
            # 读取现有集合中的内容哈希集合，用于去重
            # 说明：通过遍历集合中已存在的文档文本，计算其SHA256哈希，构建集合，避免重复添加完全相同的内容。
            existing_hashes = self._get_existing_content_hashes()
            logging.info(f"现有内容指纹数量: {len(existing_hashes)}")
            
            # 先对输入文档进行过滤，只保留全新内容
            filtered_docs, skipped = self._filter_new_documents(documents, existing_hashes)
            if skipped > 0:
                logging.info(f"检测到 {skipped} 个重复文档块，已跳过，仅添加 {len(filtered_docs)} 个新文档块")
            else:
                logging.info("未发现重复文档块")
            
            if not filtered_docs:
                logging.info("没有需要添加的新文档块，结束操作")
                return True
            
            # 分批处理
            for i in range(0, len(filtered_docs), batch_size):
                batch_docs = filtered_docs[i:i + batch_size]
                self._add_batch(batch_docs)
                logging.info(f"已处理 {min(i + batch_size, len(filtered_docs))}/{len(filtered_docs)} 个文档块")
            
            # ChromaDB PersistentClient 会自动持久化数据
            
            total_time = time.time() - start_time
            logging.info(f"文档添加完成，总耗时: {total_time:.2f}秒")
            
            return True
            
        except Exception as e:
            logging.error(f"添加文档失败: {e}")
            return False

    def _add_batch(self, documents: List[DocumentChunk]):
        """添加一批文档"""
        # 提取文本内容
        texts = [doc.content for doc in documents]
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode_texts(texts, show_progress=False)
        
        # 准备数据
        ids = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # 计算内容哈希（稳定ID）
            # 说明：使用SHA256对文本内容进行哈希，作为稳定的文档块ID，保证同样内容不会重复写入。
            content_hash = self._hash_content(doc.content)
            doc_id = content_hash
            ids.append(doc_id)
            
            # 准备元数据（添加内容哈希与相对路径等信息）
            metadata = {
                'source': doc.source_file,
                'relative_path': doc.metadata.get('relative_path', ''),
                'chunk_index': doc.metadata.get('chunk_index', i),
                'total_chunks': doc.metadata.get('total_chunks', len(documents)),
                'char_count': len(doc.content),
                'file_type': os.path.splitext(doc.source_file)[1].lower(),
                'content_hash': content_hash
            }
            metadatas.append(metadata)
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def _hash_content(self, text: str) -> str:
        """
        计算文本内容的SHA256哈希
        目的：为文档块生成稳定的唯一指纹，用于去重
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _get_existing_content_hashes(self, page_size: int = 1000) -> Set[str]:
        """
        获取集合中已存在的内容哈希集合（通过遍历读取已存文档并计算内容哈希）
        说明：ChromaDB返回原始文档内容，我们对其进行SHA256计算，构建集合用于快速去重。
        Args:
            page_size: 每次读取的条数（分页）
        Returns:
            set[str]: 已存在内容的哈希集合
        """
        hashes: Set[str] = set()
        try:
            total = self.collection.count()
            if total == 0:
                return hashes
            
            # 分页读取所有已有文档
            offset = 0
            while offset < total:
                batch = self.collection.get(limit=min(page_size, total - offset), offset=offset, include=["documents"])
                docs = batch.get('documents') or []
                # 计算并加入哈希集合
                for d in docs:
                    if isinstance(d, str) and d.strip():
                        hashes.add(self._hash_content(d))
                offset += len(docs)
            return hashes
        except Exception as e:
            logging.warning(f"读取现有内容哈希集合失败，将不进行去重: {e}")
            return hashes

    def _filter_new_documents(self, documents: List[DocumentChunk], existing_hashes: Set[str]) -> Tuple[List[DocumentChunk], int]:
        """
        根据已有内容哈希集合过滤重复文档，只保留全新内容
        Args:
            documents: 待添加的文档块列表
            existing_hashes: 已存在内容哈希集合
        Returns:
            (filtered_docs, skipped_count): 过滤后的文档块列表与跳过数量
        """
        if not existing_hashes:
            # 若无已存在哈希，则全部保留
            return documents, 0
        
        filtered = []
        skipped = 0
        for doc in documents:
            h = self._hash_content(doc.content)
            if h in existing_hashes:
                skipped += 1
            else:
                filtered.append(doc)
        return filtered, skipped
    
    def search(self, query: str, 
               top_k: Optional[int] = None,
               similarity_threshold: Optional[float] = None,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            filter_metadata: 元数据过滤条件
            
        Returns:
            搜索结果列表
        """
        if not query.strip():
            return []
        
        try:
            # 使用配置的默认值
            top_k = top_k or self.config.TOP_K
            similarity_threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
            
            logging.info(f"搜索查询: '{query}', top_k: {top_k}, 阈值: {similarity_threshold}")
            start_time = time.time()
            
            # 生成查询向量
            query_embedding = self.embedding_model.encode_single(query)
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata
            )
            
            search_time = time.time() - start_time
            
            # 处理结果
            processed_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    # 计算相似度分数 (ChromaDB返回的是距离，需要转换)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # 转换为相似度
                    
                    # 应用相似度阈值
                    if similarity >= similarity_threshold:
                        result = {
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity,
                            'id': results['ids'][0][i]
                        }
                        processed_results.append(result)
            
            logging.info(f"搜索完成，找到 {len(processed_results)} 个相关结果，耗时: {search_time:.3f}秒")
            
            return processed_results
            
        except Exception as e:
            logging.error(f"搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            count = self.collection.count()
            
            # 获取样本数据来分析
            sample_results = self.collection.peek(limit=min(100, count))
            
            file_types = {}
            sources = set()
            
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    # 统计文件类型
                    file_type = metadata.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    
                    # 统计源文件
                    source = metadata.get('source', 'unknown')
                    sources.add(source)
            
            stats = {
                'total_documents': count,
                'unique_sources': len(sources),
                'file_types': file_types,
                'collection_name': self.config.COLLECTION_NAME,
                'index_directory': self.config.INDEX_DIR
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"获取统计信息失败: {e}")
            return {}
    
    def list_documents(self, page: int = 1, page_size: int = 20, query: Optional[str] = None) -> Dict[str, Any]:
        """
        列出已索引文档（按源文件聚合），支持分页与简单搜索
        返回：{ items: [{source, file_type, chunks, preview}], page, page_size, total }
        """
        try:
            total = self.collection.count()
            # 全量扫描聚合到源文件维度
            offset = 0
            agg: Dict[str, Dict[str, Any]] = {}
            while offset < total:
                batch = self.collection.get(limit=min(1000, total - offset), offset=offset, include=["documents", "metadatas"])
                docs = batch.get('documents') or []
                metas = batch.get('metadatas') or []
                ids = batch.get('ids') or []
                for i in range(len(docs)):
                    m = metas[i] if i < len(metas) else {}
                    src = m.get('source', 'unknown')
                    if query and query.strip():
                        if query.lower() not in str(src).lower():
                            continue
                    item = agg.get(src)
                    if not item:
                        item = {
                            'source': src,
                            'file_type': m.get('file_type', 'unknown'),
                            'chunks': 0,
                            'ids': [],
                            'preview': docs[i][:200] if isinstance(docs[i], str) else ''
                        }
                        agg[src] = item
                    item['chunks'] += 1
                    if i < len(ids):
                        item['ids'].append(ids[i])
                offset += len(docs)
            items = list(agg.values())
            total_sources = len(items)
            start = max(0, (page - 1) * page_size)
            end = start + page_size
            return {
                'items': items[start:end],
                'page': page,
                'page_size': page_size,
                'total': total_sources
            }
        except Exception as e:
            logging.error(f"列出文档失败: {e}")
            return {'items': [], 'page': page, 'page_size': page_size, 'total': 0}
    
    def delete_by_source(self, source: str) -> int:
        """根据源文件路径删除其所有块"""
        try:
            total = self.collection.count()
            offset = 0
            to_delete: List[str] = []
            import os
            source_norm = os.path.normpath(source)
            source_name = os.path.basename(source_norm)
            while offset < total:
                batch = self.collection.get(limit=min(1000, total - offset), offset=offset, include=["metadatas"]) 
                metas = batch.get('metadatas') or []
                ids = batch.get('ids') or []
                for i, m in enumerate(metas):
                    src_meta = os.path.normpath(str(m.get('source', '')))
                    rel_meta = os.path.normpath(str(m.get('relative_path', '')))
                    file_meta = os.path.normpath(str(m.get('file_name', '')))
                    if (src_meta == source_norm) or (rel_meta == source_norm) or (file_meta == source_name):
                        if i < len(ids):
                            to_delete.append(ids[i])
                offset += len(ids)
            if to_delete:
                self.collection.delete(ids=to_delete)
            return len(to_delete)
        except Exception as e:
            logging.error(f"按来源删除失败 {source}: {e}")
            return 0
    
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否成功
        """
        try:
            self.client.delete_collection(name=self.config.COLLECTION_NAME)
            logging.info(f"集合 {self.config.COLLECTION_NAME} 已删除")
            return True
        except Exception as e:
            logging.error(f"删除集合失败: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        重置集合（删除后重新创建）
        
        Returns:
            是否成功
        """
        try:
            # 删除现有集合
            try:
                self.client.delete_collection(name=self.config.COLLECTION_NAME)
            except:
                pass  # 集合可能不存在
            
            # 重新创建集合
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"description": "RAG system document collection"}
            )
            
            logging.info(f"集合 {self.config.COLLECTION_NAME} 已重置")
            return True
            
        except Exception as e:
            logging.error(f"重置集合失败: {e}")
            return False

def main():
    """测试向量存储功能"""
    try:
        # 初始化向量存储
        vector_store = VectorStore()
        
        # 获取统计信息
        stats = vector_store.get_collection_stats()
        logging.info(f"集合统计信息: {stats}")
        
        # 测试搜索（如果有数据）
        if stats.get('total_documents', 0) > 0:
            test_queries = [
                "transformer模型",
                "自然语言处理",
                "BERT模型",
                "注意力机制"
            ]
            
            for query in test_queries:
                logging.info(f"\n搜索测试: '{query}'")
                results = vector_store.search(query, top_k=3)
                
                for i, result in enumerate(results):
                    logging.info(f"结果 {i+1}:")
                    logging.info(f"  相似度: {result['similarity']:.4f}")
                    logging.info(f"  来源: {result['metadata']['source']}")
                    logging.info(f"  内容预览: {result['content'][:100]}...")
        else:
            logging.info("集合中没有数据，跳过搜索测试")
        
        logging.info("向量存储测试完成！")
        
    except Exception as e:
        logging.error(f"测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()