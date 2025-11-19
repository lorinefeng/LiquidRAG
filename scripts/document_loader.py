#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档加载与预处理模块
支持PDF、Word、Markdown、TXT文件格式
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# 文档处理库
import PyPDF2
import markdown
from docx import Document

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.rag_config import RAGConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """文档块数据结构"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str

class DocumentLoader:
    """文档加载器类"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.supported_extensions = self.config.SUPPORTED_EXTENSIONS
        
    def load_documents(self, source_dir: str) -> List[DocumentChunk]:
        """
        从指定目录加载所有支持的文档
        
        Args:
            source_dir: 源文档目录路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        documents = []
        source_path = Path(source_dir)
        
        if not source_path.exists():
            logger.error(f"源目录不存在: {source_dir}")
            return documents
            
        logger.info(f"开始扫描目录: {source_dir}")
        
        # 递归扫描所有支持的文件
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    logger.info(f"处理文件: {file_path}")
                    doc_chunks = self._load_single_document(file_path)
                    documents.extend(doc_chunks)
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {str(e)}")
                    
        logger.info(f"总共加载了 {len(documents)} 个文档块")
        return documents
    
    def _load_single_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        extension = file_path.suffix.lower()
        
        # 根据文件扩展名选择处理方法
        if extension == '.md':
            return self._load_markdown(file_path)
        elif extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension == '.docx':
            return self._load_docx(file_path)
        elif extension == '.txt':
            return self._load_txt(file_path)
        else:
            logger.warning(f"不支持的文件格式: {extension}")
            return []
    
    def _load_markdown(self, file_path: Path) -> List[DocumentChunk]:
        """加载Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 转换Markdown为纯文本
            md = markdown.Markdown()
            html = md.convert(content)
            # 简单的HTML标签清理
            import re
            text = re.sub(r'<[^>]+>', '', html)
            
            return self._split_text_into_chunks(text, file_path)
            
        except Exception as e:
            logger.error(f"加载Markdown文件失败 {file_path}: {str(e)}")
            return []
    
    def _load_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """加载PDF文件"""
        try:
            text_content = ""
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"\n\n=== 第{page_num + 1}页 ===\n\n"
                            text_content += page_text
                    except Exception as e:
                        logger.warning(f"提取PDF第{page_num + 1}页失败: {str(e)}")
            
            return self._split_text_into_chunks(text_content, file_path)
            
        except Exception as e:
            logger.error(f"加载PDF文件失败 {file_path}: {str(e)}")
            return []
    
    def _load_docx(self, file_path: Path) -> List[DocumentChunk]:
        """加载Word文档"""
        try:
            doc = Document(file_path)
            text_content = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n"
            
            return self._split_text_into_chunks(text_content, file_path)
            
        except Exception as e:
            logger.error(f"加载Word文档失败 {file_path}: {str(e)}")
            return []
    
    def _load_txt(self, file_path: Path) -> List[DocumentChunk]:
        """加载文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._split_text_into_chunks(content, file_path)
            
        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {str(e)}")
            return []
    
    def _split_text_into_chunks(self, text: str, file_path: Path) -> List[DocumentChunk]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            file_path: 源文件路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        if not text.strip():
            return []
        
        chunks = []
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP
        
        # 简单的文本分割策略
        text = text.strip()
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在句号、换行符等位置分割
            if end < len(text):
                # 寻找合适的分割点
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '。！？\n\r':
                        end = i + 1
                        break
            
            # 提取当前块
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # 创建文档块
                try:
                    rel = str(file_path.relative_to(Path(self.config.DOCS_SOURCE_DIR)))
                except Exception:
                    rel = file_path.name
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        'source_file': str(file_path),
                        'file_name': file_path.name,
                        'file_extension': file_path.suffix,
                        'chunk_index': chunk_index,
                        'chunk_size': len(chunk_text),
                        'relative_path': rel
                    },
                    chunk_id=f"{file_path.stem}_{chunk_index}",
                    source_file=str(file_path)
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # 移动到下一个块的起始位置（考虑重叠）
            start = max(start + chunk_size - overlap, end)
        
        logger.info(f"文件 {file_path.name} 分割为 {len(chunks)} 个块")
        return chunks
    
    def get_document_stats(self, documents: List[DocumentChunk]) -> Dict[str, Any]:
        """
        获取文档统计信息
        
        Args:
            documents: 文档块列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not documents:
            return {}
        
        # 按文件类型统计
        file_types = {}
        total_chars = 0
        source_files = set()
        
        for doc in documents:
            ext = doc.metadata.get('file_extension', 'unknown')
            file_types[ext] = file_types.get(ext, 0) + 1
            total_chars += len(doc.content)
            source_files.add(doc.metadata.get('file_name', 'unknown'))
        
        return {
            'total_chunks': len(documents),
            'total_characters': total_chars,
            'unique_files': len(source_files),
            'file_types': file_types,
            'avg_chunk_size': total_chars / len(documents) if documents else 0
        }

def main():
    """测试文档加载功能"""
    logger.info("=== 文档加载器测试 ===")
    
    loader = DocumentLoader()
    
    # 加载learn-nlp-with-transformers文档
    docs_dir = RAGConfig.DOCS_SOURCE_DIR
    logger.info(f"加载文档目录: {docs_dir}")
    
    documents = loader.load_documents(docs_dir)
    
    # 显示统计信息
    stats = loader.get_document_stats(documents)
    logger.info("文档统计信息:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # 显示前几个文档块的示例
    logger.info("\n前3个文档块示例:")
    for i, doc in enumerate(documents[:3]):
        logger.info(f"\n--- 块 {i+1} ---")
        logger.info(f"来源: {doc.metadata['file_name']}")
        logger.info(f"内容长度: {len(doc.content)}")
        logger.info(f"内容预览: {doc.content[:200]}...")

if __name__ == "__main__":
    main()