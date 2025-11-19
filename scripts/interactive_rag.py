#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式RAG问答系统
提供命令行界面供用户实时查询
"""

import sys
import os
import time
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.rag_pipeline import RAGPipeline
from configs.rag_config import RAGConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/interactive_rag.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InteractiveRAG:
    """交互式RAG问答系统"""
    
    def __init__(self):
        """初始化RAG系统"""
        self.config = RAGConfig()
        self.rag_pipeline = None
        self.is_initialized = False
        
    def initialize(self):
        """初始化RAG流程"""
        try:
            print("🚀 正在初始化RAG系统...")
            print("=" * 60)
            
            # 初始化RAG流程
            self.rag_pipeline = RAGPipeline(self.config)
            
            print("✅ RAG系统初始化完成！")
            print("=" * 60)
            self.is_initialized = True
            
        except Exception as e:
            print(f"❌ RAG系统初始化失败: {str(e)}")
            logger.error(f"RAG系统初始化失败: {str(e)}")
            return False
            
        return True
    
    def display_help(self):
        """显示帮助信息"""
        help_text = """
🤖 RAG问答系统 - 帮助信息
=" * 60
可用命令:
  help     - 显示此帮助信息
  status   - 显示系统状态
  clear    - 清屏
  quit     - 退出系统
  exit     - 退出系统

使用方法:
  直接输入您的问题，系统会自动检索相关文档并生成答案。

示例问题:
  - 什么是Transformer？
  - 如何使用BERT进行文本分类？
  - 注意力机制的原理是什么？
  - 如何进行模型微调？

=" * 60
        """
        print(help_text)
    
    def display_status(self):
        """显示系统状态"""
        print("\n📊 系统状态")
        print("=" * 40)
        print(f"初始化状态: {'✅ 已初始化' if self.is_initialized else '❌ 未初始化'}")
        
        if self.is_initialized and self.rag_pipeline:
            # 获取向量数据库状态
            try:
                collection_count = self.rag_pipeline.vector_store.get_collection_count()
                print(f"向量数据库: ✅ 已连接 ({collection_count} 个文档)")
            except:
                print("向量数据库: ❌ 连接失败")
            
            # 获取嵌入模型状态
            try:
                model_info = self.rag_pipeline.embedding_model.get_model_info()
                print(f"嵌入模型: ✅ {model_info['model_name']}")
                print(f"设备: {model_info['device']}")
            except:
                print("嵌入模型: ❌ 加载失败")
        
        print("=" * 40)
    
    def process_query(self, query: str):
        """处理用户查询"""
        if not self.is_initialized:
            print("❌ 系统未初始化，请重启程序")
            return
        
        print(f"\n🔍 正在处理查询: {query}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 执行RAG查询
            result = self.rag_pipeline.ask(query)
            # 使用返回的 sources 与 num_sources 字段确保文档数量显示准确
            docs = result.get('sources', [])
            num_docs = result.get('num_sources', len(docs))

            processing_time = time.time() - start_time

            # 显示结果
            print(f"⏱️  处理时间: {processing_time:.2f}秒")
            print(f"📚 检索到 {num_docs} 个相关文档")
            
            if docs:
                print("\n📖 相关文档:")
                for i, doc in enumerate(docs[:3], 1):
                    source = doc.get('metadata', {}).get('source', 'Unknown')
                    similarity = doc.get('similarity', 0)
                    print(f"  {i}. {os.path.basename(source)}")
                    print(f"     相似度: {similarity:.3f}")
            
            print(f"\n🤖 回答:")
            print("-" * 30)
            print(result.get('answer', '抱歉，无法生成答案'))
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 查询处理失败: {str(e)}")
            logger.error(f"查询处理失败: {str(e)}")
    
    def run(self):
        """运行交互式系统"""
        # 显示欢迎信息
        print("\n" + "=" * 60)
        print("🤖 欢迎使用RAG智能问答系统！")
        print("=" * 60)
        print("输入 'help' 查看帮助信息")
        print("输入 'quit' 或 'exit' 退出系统")
        print("=" * 60)
        
        # 初始化系统
        if not self.initialize():
            print("系统初始化失败，程序退出")
            return
        
        # 主循环
        while True:
            try:
                # 获取用户输入
                user_input = input("\n💬 请输入您的问题: ").strip()
                
                # 处理空输入
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 感谢使用RAG问答系统，再见！")
                    break
                elif user_input.lower() == 'help':
                    self.display_help()
                elif user_input.lower() == 'status':
                    self.display_status()
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                else:
                    # 处理查询
                    self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断信号，程序退出")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                logger.error(f"主循环错误: {str(e)}")

def main():
    """主函数"""
    try:
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        # 启动交互式系统
        interactive_rag = InteractiveRAG()
        interactive_rag.run()
        
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")
        logger.error(f"程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()