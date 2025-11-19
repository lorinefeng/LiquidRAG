#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG流程管道
集成文档检索和LFM2-1.2B问答功能的完整RAG系统
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.cuda.amp import autocast

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.rag_config import RAGConfig
from scripts.vector_store import VectorStore
from scripts.embedding_model import EmbeddingModel
from scripts.gpu_manager import GPUManager
from scripts.model_cache import ModelCacheManager
from scripts.error_handler import ErrorHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RAGPipeline:
    """RAG流程管道类"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化RAG流程
        
        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()
        self.vector_store = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查硬件配置
        self._check_hardware()
        
        # 初始化组件
        self._init_vector_store()
        self._init_llm()
    
    def _check_hardware(self):
        """检查硬件配置"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"检测到GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
            
            # 针对RTX4060 8GB显存的优化建议
            if gpu_memory < 10:
                logging.warning("检测到显存较小，将启用内存优化模式")
                self.config.USE_MEMORY_OPTIMIZATION = True
        else:
            logging.info("未检测到GPU，使用CPU进行推理")
    
    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            self.vector_store = VectorStore(self.config)
            
            # 检查是否有数据
            stats = self.vector_store.get_collection_stats()
            total_docs = stats.get('total_documents', 0)
            
            if total_docs == 0:
                logging.warning("向量数据库中没有文档，请先构建知识库")
            else:
                logging.info(f"向量数据库已加载，包含 {total_docs} 个文档")
                
        except Exception as e:
            logging.error(f"初始化向量存储失败: {e}")
            raise
    
    def _init_llm(self):
        """初始化大语言模型 - 使用与evaluate_lfm2.py相同的加载方法"""
        try:
            model_path = self.config.LLM_MODEL_PATH
            
            if not model_path or not Path(model_path).exists():
                logging.warning(f"LFM2模型路径未配置或不存在: {model_path}")
                logging.info("将使用模拟回答模式")
                return
            
            logging.info(f"正在加载LFM2模型: {model_path}")
            start_time = time.time()
            
            # 初始化管理器（与evaluate_lfm2.py保持一致）
            gpu_manager = GPUManager()
            cache_manager = ModelCacheManager(cache_dir=self.config.MODELS_DIR)
            error_handler = ErrorHandler("RAG_LFM2_Loader")
            
            # 优化GPU设置
            gpu_manager.optimize_for_inference()

            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                    torch.backends.cuda.sdp_kernel.enable_flash(False)
                    torch.backends.cuda.sdp_kernel.enable_mem_efficient(True)
                    torch.backends.cuda.sdp_kernel.enable_math(True)
            except Exception:
                pass
            
            # 数据类型映射（与evaluate_lfm2.py保持一致）
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32
            }
            torch_dtype = dtype_map.get(self.config.LFM2_DTYPE, torch.bfloat16)
            
            # 加载配置
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            
            # 获取最优设备映射
            device_map = self._get_optimal_device_map(config, self.config.LFM2_MAX_MEMORY_GB, gpu_manager)
            logging.info(f"使用设备映射: {device_map}")
            
            # 加载分词器
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 加载模型（避免对accelerate的依赖）
            target_device = None
            if isinstance(device_map, dict) and len(device_map) > 1:
                best_device = gpu_manager.get_best_device(min_memory_mb=int(self.config.LFM2_MAX_MEMORY_GB * 1024))
                target_device = f"cuda:{best_device}" if best_device is not None else "cpu"
            elif isinstance(device_map, str):
                target_device = device_map
            else:
                target_device = "cpu"

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if target_device:
                try:
                    self.llm_model.to(target_device)
                except Exception:
                    self.llm_model.to("cpu")
            # 推理模式
            try:
                self.llm_model.eval()
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            
            load_time = time.time() - start_time
            logging.info(f"LFM2模型加载完成，耗时: {load_time:.2f}秒")
            
            # 记录模型信息
            if gpu_manager.is_cuda_available():
                memory_info = gpu_manager.monitor_memory_usage(0)
                logging.info(f"模型加载后GPU内存使用: {memory_info}")
            
        except Exception as e:
            logging.error(f"加载LFM2模型失败: {e}")
            logging.info("将使用模拟回答模式")
            self.llm_model = None
            self.llm_tokenizer = None
    
    def _get_optimal_device_map(self, model_config, available_memory_gb: float, gpu_manager: GPUManager):
        """获取最优设备映射（与evaluate_lfm2.py保持一致）"""
        if not gpu_manager.is_cuda_available():
            return "cpu"
        
        try:
            # 估算模型大小
            if hasattr(model_config, 'num_parameters'):
                num_params = model_config.num_parameters
            elif hasattr(model_config, 'n_params'):
                num_params = model_config.n_params
            else:
                # 粗略估算：基于hidden_size和层数
                hidden_size = getattr(model_config, 'hidden_size', 2048)
                num_layers = getattr(model_config, 'num_hidden_layers', 24)
                vocab_size = getattr(model_config, 'vocab_size', 32000)
                
                # 简化估算：embedding + transformer layers + head
                num_params = vocab_size * hidden_size + num_layers * (hidden_size ** 2 * 4) + hidden_size * vocab_size
            
            # 转换为MB (假设bf16，每个参数2字节)
            model_size_mb = (num_params * 2) / (1024**2)
            
        except Exception as e:
            logging.warning(f"无法估算模型大小: {e}")
            model_size_mb = 2400  # LFM2-1.2B的大概大小
        
        # 使用GPU管理器获取设备映射
        device_map = gpu_manager.get_memory_efficient_device_map(
            model_size_mb=model_size_mb,
            max_memory_per_gpu_mb=int(available_memory_gb * 1024)
        )
        
        # 如果返回的是内存限制字典，转换为最佳单设备
        if isinstance(device_map, dict) and all(isinstance(v, str) and v.endswith('MB') for v in device_map.values()):
            best_device = gpu_manager.get_best_device(min_memory_mb=int(available_memory_gb * 1024))
            return f"cuda:{best_device}" if best_device is not None else "cpu"
        
        return device_map
    
    def retrieve_documents(self, query: str, 
                          top_k: Optional[int] = None,
                          similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            检索结果列表
        """
        if not self.vector_store:
            logging.error("向量存储未初始化")
            return []
        
        try:
            start_time = time.time()
            
            results = self.vector_store.search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            retrieve_time = time.time() - start_time
            logging.info(f"文档检索完成，找到 {len(results)} 个相关文档，耗时: {retrieve_time:.3f}秒")
            
            return results
            
        except Exception as e:
            logging.error(f"文档检索失败: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        基于检索到的文档生成回答
        
        Args:
            query: 用户查询
            context_docs: 检索到的相关文档
            
        Returns:
            生成的回答
        """
        try:
            # 构建上下文
            context = self._build_context(context_docs)
            
            # 构建提示词
            prompt = self._build_prompt(query, context)
            
            # 生成回答
            if self.llm_model and self.llm_tokenizer:
                answer = self._generate_with_llm(prompt)
            else:
                # 模拟回答模式
                answer = self._generate_mock_answer(query, context_docs)
            
            return answer
            
        except Exception as e:
            logging.error(f"生成回答失败: {e}")
            return "抱歉，生成回答时出现错误。"
    
    def _build_context(self, docs: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        构建上下文文本
        
        Args:
            docs: 文档列表
            max_length: 最大长度
            
        Returns:
            上下文文本
        """
        if not docs:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs):
            content = doc['content']
            source = doc['metadata']['source']
            
            # 添加来源信息
            doc_text = f"[文档{i+1}] 来源: {source}\n{content}\n"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建提示词
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            完整提示词
        """
        prompt_template = """你是一个专业的自然语言处理助手，请基于以下提供的文档内容回答用户的问题。

相关文档内容：
{context}

用户问题：{query}

请基于上述文档内容，用中文详细回答用户的问题。如果文档中没有相关信息，请说明无法从提供的文档中找到答案。

回答："""
        
        return prompt_template.format(context=context, query=query)
    
    def _generate_with_llm(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        使用LLM生成回答 - 与evaluate_lfm2.py保持一致的生成方法
        
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            
        Returns:
            生成的回答
        """
        try:
            start_time = time.time()
            
            # 编码输入（与evaluate_lfm2.py保持一致）
            inputs = self.llm_tokenizer(
                prompt, 
                return_tensors='pt',
                truncation=True,
                max_length=2048,  # 限制输入长度
                padding=True
            )
            
            # 移动到正确设备
            device = next(self.llm_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if torch.cuda.is_available() and str(device) == 'cpu':
                try:
                    self.llm_model.to('cuda')
                    device = next(self.llm_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
            
            # 生成参数（与evaluate_lfm2.py保持一致）
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': False,
                'repetition_penalty': 1.1,
                'pad_token_id': self.llm_tokenizer.eos_token_id,
                'eos_token_id': self.llm_tokenizer.eos_token_id,
                'use_cache': True
            }
            
            # 生成回答（混合精度/AMP）
            with torch.no_grad():
                # 默认长度来自配置
                generation_kwargs['max_new_tokens'] = max_new_tokens or getattr(self.config, 'LLM_MAX_NEW_TOKENS', 512)
                try:
                    logging.info(f"生成配置: {generation_kwargs}, 输入长度: {inputs['input_ids'].shape[1]}, 模型设备: {device}")
                except Exception:
                    pass
                try:
                    test_ids = torch.full((1, 4), self.llm_tokenizer.eos_token_id, dtype=torch.long, device=device)
                    t0 = time.time()
                    _ = self.llm_model(input_ids=test_ids)
                    logging.info(f"小测试推理时间: {time.time()-t0:.3f}秒")
                except Exception:
                    pass
                if torch.cuda.is_available():
                    from torch.cuda.amp import autocast as cuda_autocast
                    with cuda_autocast(dtype=torch.bfloat16):
                        outputs = self.llm_model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask'),
                            **generation_kwargs
                        )
                else:
                    outputs = self.llm_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        **generation_kwargs
                    )
            
            # 解码输出（只取新生成的部分）
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            generated_text = self.llm_tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            generation_time = time.time() - start_time
            try:
                gen_tokens = int(generation_kwargs.get('max_new_tokens') or 0)
                if 'input_ids' in inputs:
                    gen_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                tps = (gen_tokens / generation_time) if generation_time > 0 else 0
                logging.info(f"生成速度: {tps:.2f} tokens/s")
            except Exception:
                pass
            # GPU监控
            try:
                gpu_manager = GPUManager()
                if gpu_manager.is_cuda_available():
                    mem = gpu_manager.monitor_memory_usage(0)
                    logging.info(f"GPU监控: 利用率 {mem.get('utilization_percent',0)}% | 内存 {mem.get('used_mb',0)}/{mem.get('total_mb',0)}MB")
            except Exception:
                pass
            logging.info(f"LFM2生成完成，耗时: {generation_time:.2f}秒，生成长度: {len(generated_tokens)}tokens")
            
            return generated_text.strip()
            
        except Exception as e:
            logging.error(f"LFM2生成失败: {e}")
            return "抱歉，生成回答时出现错误。"
    
    def _generate_mock_answer(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        生成模拟回答（当LLM不可用时）
        
        Args:
            query: 用户查询
            docs: 相关文档
            
        Returns:
            模拟回答
        """
        if not docs:
            return "抱歉，没有找到与您的问题相关的文档内容。"
        
        # 简单的基于关键词匹配的回答生成
        answer_parts = [
            f"基于检索到的 {len(docs)} 个相关文档，我为您整理了以下信息：\n"
        ]
        
        for i, doc in enumerate(docs[:3], 1):  # 只使用前3个最相关的文档
            source = doc['metadata']['source']
            content = doc['content'][:300]  # 截取前300字符
            similarity = doc['similarity']
            
            answer_parts.append(
                f"{i}. 来源：{source} (相似度: {similarity:.3f})\n"
                f"   内容摘要：{content}...\n"
            )
        
        answer_parts.append(
            "\n注意：以上回答基于文档检索结果生成。如需更详细的解答，"
            "建议查看完整的源文档内容。"
        )
        
        return "\n".join(answer_parts)
    
    def ask(self, query: str, 
            top_k: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            return_sources: bool = True) -> Dict[str, Any]:
        """
        完整的RAG问答流程
        
        Args:
            query: 用户查询
            top_k: 检索文档数量
            similarity_threshold: 相似度阈值
            return_sources: 是否返回源文档
            
        Returns:
            包含回答和相关信息的字典
        """
        if not query.strip():
            return {
                'answer': "请输入有效的问题。",
                'sources': [],
                'retrieval_time': 0,
                'generation_time': 0,
                'total_time': 0
            }
        
        start_time = time.time()
        
        try:
            logging.info(f"处理查询: '{query}'")
            
            # 1. 检索相关文档
            retrieve_start = time.time()
            docs = self.retrieve_documents(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            retrieve_time = time.time() - retrieve_start
            
            # 2. 生成回答
            generate_start = time.time()
            answer = self.generate_answer(query, docs)
            generate_time = time.time() - generate_start
            
            total_time = time.time() - start_time
            
            # 3. 构建结果
            result = {
                'answer': answer,
                'retrieval_time': retrieve_time,
                'generation_time': generate_time,
                'total_time': total_time,
                'num_sources': len(docs)
            }
            
            if return_sources:
                result['sources'] = docs
            
            logging.info(f"查询处理完成，总耗时: {total_time:.3f}秒")
            
            return result
            
        except Exception as e:
            logging.error(f"RAG查询失败: {e}")
            return {
                'answer': "抱歉，处理您的问题时出现错误。",
                'sources': [],
                'retrieval_time': 0,
                'generation_time': 0,
                'total_time': time.time() - start_time,
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        vector_stats = self.vector_store.get_collection_stats() if self.vector_store else {}
        
        return {
            'vector_store': {
                'total_documents': vector_stats.get('total_documents', 0),
                'unique_sources': vector_stats.get('unique_sources', 0),
                'collection_name': vector_stats.get('collection_name', 'N/A')
            },
            'llm': {
                'model_available': self.llm_model is not None,
                'model_path': self.config.LLM_MODEL_PATH,
                'device': str(self.device)
            },
            'config': {
                'top_k': self.config.TOP_K,
                'similarity_threshold': self.config.SIMILARITY_THRESHOLD,
                'max_response_time': self.config.MAX_RESPONSE_TIME
            }
        }

def main():
    """测试RAG流程"""
    try:
        # 初始化RAG流程
        rag = RAGPipeline()
        
        # 显示系统信息
        system_info = rag.get_system_info()
        logging.info(f"系统信息: {system_info}")
        
        # 测试查询
        test_queries = [
            "什么是Transformer模型？",
            "BERT模型的主要特点是什么？",
            "如何进行文本分类？",
            "注意力机制是如何工作的？"
        ]
        
        logging.info("\n" + "=" * 60)
        logging.info("开始RAG测试")
        logging.info("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            logging.info(f"\n测试查询 {i}: {query}")
            
            result = rag.ask(query, return_sources=False)
            
            logging.info(f"回答: {result['answer'][:200]}...")
            logging.info(f"检索时间: {result['retrieval_time']:.3f}秒")
            logging.info(f"生成时间: {result['generation_time']:.3f}秒")
            logging.info(f"总时间: {result['total_time']:.3f}秒")
            logging.info(f"相关文档数: {result['num_sources']}")
        
        logging.info("\nRAG流程测试完成！")
        
    except Exception as e:
        logging.error(f"测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()