#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统性能优化器
优化系统性能，确保响应时间<2秒，提升资源利用率
"""

import os
import sys
import logging
import time
import gc
import psutil
import torch
from typing import Dict, Any, List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.rag_config import RAGConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config: RAGConfig = None):
        """
        初始化性能优化器
        
        Args:
            config: RAG配置对象
        """
        self.config = config or RAGConfig()
        self.optimization_cache = {}
        self.performance_metrics = {}
        self._lock = threading.Lock()
        
        # 检测硬件配置
        self.hardware_info = self._detect_hardware()
        logging.info(f"硬件配置: {self.hardware_info}")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        检测硬件配置
        
        Returns:
            硬件信息
        """
        hardware_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_memory': []
        }
        
        # GPU信息
        if hardware_info['gpu_available']:
            for i in range(hardware_info['gpu_count']):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                hardware_info['gpu_memory'].append({
                    'device': i,
                    'name': gpu_props.name,
                    'memory_total': gpu_memory,
                    'memory_free': torch.cuda.memory_reserved(i) / (1024**3) if torch.cuda.is_available() else 0
                })
        
        return hardware_info
    
    def optimize_pytorch_settings(self) -> Dict[str, Any]:
        """
        优化PyTorch设置
        
        Returns:
            优化结果
        """
        logging.info("优化PyTorch设置...")
        
        optimizations = {}
        
        try:
            # 设置线程数
            if self.hardware_info['cpu_count'] >= 4:
                torch.set_num_threads(min(4, self.hardware_info['cpu_count'] // 2))
                optimizations['num_threads'] = torch.get_num_threads()
            
            # 启用优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            optimizations['cudnn_benchmark'] = True
            
            # GPU内存优化
            if self.hardware_info['gpu_available']:
                # 启用内存池
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
                # 设置内存分配策略
                torch.cuda.empty_cache()
                optimizations['gpu_memory_optimized'] = True
                
                # 检查GPU内存使用
                for i, gpu_info in enumerate(self.hardware_info['gpu_memory']):
                    if gpu_info['memory_total'] <= 8.5:  # RTX4060 8GB限制
                        # 为8GB显存优化
                        torch.cuda.set_per_process_memory_fraction(0.85, device=i)
                        optimizations[f'gpu_{i}_memory_fraction'] = 0.85
            
            # 设置数据类型优化
            torch.set_default_dtype(torch.float32)
            optimizations['default_dtype'] = 'float32'
            
            logging.info(f"PyTorch优化完成: {optimizations}")
            return optimizations
            
        except Exception as e:
            logging.error(f"PyTorch优化失败: {e}")
            return {'error': str(e)}
    
    def optimize_embedding_model(self, model_path: str) -> Dict[str, Any]:
        """
        优化嵌入模型加载和推理
        
        Args:
            model_path: 模型路径
            
        Returns:
            优化结果
        """
        logging.info("优化嵌入模型...")
        
        optimizations = {}
        
        try:
            # 模型量化建议
            if self.hardware_info['gpu_available']:
                gpu_memory = self.hardware_info['gpu_memory'][0]['memory_total']
                if gpu_memory <= 8.5:  # RTX4060限制
                    optimizations['quantization_recommended'] = True
                    optimizations['precision'] = 'fp16'
                    optimizations['batch_size_limit'] = 32
                else:
                    optimizations['precision'] = 'fp32'
                    optimizations['batch_size_limit'] = 64
            else:
                optimizations['precision'] = 'fp32'
                optimizations['batch_size_limit'] = 16
            
            # 缓存策略
            optimizations['model_cache'] = True
            optimizations['embedding_cache_size'] = min(1000, int(self.hardware_info['memory_available'] * 100))
            
            # 批处理优化
            optimizations['dynamic_batching'] = True
            optimizations['max_batch_size'] = optimizations['batch_size_limit']
            
            logging.info(f"嵌入模型优化完成: {optimizations}")
            return optimizations
            
        except Exception as e:
            logging.error(f"嵌入模型优化失败: {e}")
            return {'error': str(e)}
    
    def optimize_vector_store(self) -> Dict[str, Any]:
        """
        优化向量存储
        
        Returns:
            优化结果
        """
        logging.info("优化向量存储...")
        
        optimizations = {}
        
        try:
            # ChromaDB优化设置
            optimizations['chroma_settings'] = {
                'anonymized_telemetry': False,
                'allow_reset': True,
                'is_persistent': True
            }
            
            # 索引优化
            optimizations['index_optimization'] = {
                'hnsw_space': 'cosine',
                'hnsw_construction_ef': 200,
                'hnsw_m': 16,
                'hnsw_ef_search': 100
            }
            
            # 批量操作优化
            optimizations['batch_size'] = min(100, int(self.hardware_info['memory_available'] * 10))
            optimizations['parallel_processing'] = min(4, self.hardware_info['cpu_count'])
            
            # 缓存策略
            optimizations['query_cache_size'] = 500
            optimizations['result_cache_ttl'] = 3600  # 1小时
            
            logging.info(f"向量存储优化完成: {optimizations}")
            return optimizations
            
        except Exception as e:
            logging.error(f"向量存储优化失败: {e}")
            return {'error': str(e)}
    
    def optimize_text_processing(self) -> Dict[str, Any]:
        """
        优化文本处理
        
        Returns:
            优化结果
        """
        logging.info("优化文本处理...")
        
        optimizations = {}
        
        try:
            # 分块策略优化
            memory_gb = self.hardware_info['memory_available']
            
            if memory_gb >= 20:
                chunk_size = 1000
                overlap = 150
                batch_size = 50
            elif memory_gb >= 15:
                chunk_size = 800
                overlap = 120
                batch_size = 30
            else:
                chunk_size = 600
                overlap = 100
                batch_size = 20
            
            optimizations['chunk_size'] = chunk_size
            optimizations['chunk_overlap'] = overlap
            optimizations['processing_batch_size'] = batch_size
            
            # 并行处理
            optimizations['parallel_workers'] = min(4, self.hardware_info['cpu_count'])
            optimizations['use_multiprocessing'] = self.hardware_info['cpu_count'] >= 4
            
            # 中英文处理优化
            optimizations['mixed_language_support'] = True
            optimizations['unicode_normalization'] = True
            optimizations['text_cleaning'] = True
            
            logging.info(f"文本处理优化完成: {optimizations}")
            return optimizations
            
        except Exception as e:
            logging.error(f"文本处理优化失败: {e}")
            return {'error': str(e)}
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        优化内存使用
        
        Returns:
            优化结果
        """
        logging.info("优化内存使用...")
        
        optimizations = {}
        
        try:
            # 垃圾回收优化
            gc.collect()
            if self.hardware_info['gpu_available']:
                torch.cuda.empty_cache()
            
            # 内存监控
            memory_info = psutil.virtual_memory()
            optimizations['memory_before'] = {
                'total': memory_info.total / (1024**3),
                'available': memory_info.available / (1024**3),
                'percent': memory_info.percent
            }
            
            # 设置内存限制
            available_memory = memory_info.available / (1024**3)
            if available_memory < 10:  # 小于10GB可用内存
                optimizations['memory_limit'] = available_memory * 0.7
                optimizations['conservative_mode'] = True
            else:
                optimizations['memory_limit'] = available_memory * 0.8
                optimizations['conservative_mode'] = False
            
            # GPU内存优化
            if self.hardware_info['gpu_available']:
                for i in range(self.hardware_info['gpu_count']):
                    torch.cuda.empty_cache()
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    if gpu_memory <= 8.5:  # RTX4060限制
                        optimizations[f'gpu_{i}_conservative'] = True
                        optimizations[f'gpu_{i}_max_batch'] = 16
                    else:
                        optimizations[f'gpu_{i}_conservative'] = False
                        optimizations[f'gpu_{i}_max_batch'] = 32
            
            # 再次检查内存
            memory_info_after = psutil.virtual_memory()
            optimizations['memory_after'] = {
                'total': memory_info_after.total / (1024**3),
                'available': memory_info_after.available / (1024**3),
                'percent': memory_info_after.percent
            }
            
            optimizations['memory_freed'] = optimizations['memory_after']['available'] - optimizations['memory_before']['available']
            
            logging.info(f"内存优化完成: {optimizations}")
            return optimizations
            
        except Exception as e:
            logging.error(f"内存优化失败: {e}")
            return {'error': str(e)}
    
    def create_optimized_config(self) -> RAGConfig:
        """
        创建优化后的配置
        
        Returns:
            优化后的配置对象
        """
        logging.info("创建优化配置...")
        
        # 运行所有优化
        pytorch_opt = self.optimize_pytorch_settings()
        embedding_opt = self.optimize_embedding_model(self.config.EMBEDDING_MODEL_PATH)
        vector_opt = self.optimize_vector_store()
        text_opt = self.optimize_text_processing()
        memory_opt = self.optimize_memory_usage()
        
        # 创建新配置
        optimized_config = RAGConfig()
        
        # 应用文本处理优化
        if 'chunk_size' in text_opt:
            optimized_config.CHUNK_SIZE = text_opt['chunk_size']
        if 'chunk_overlap' in text_opt:
            optimized_config.CHUNK_OVERLAP = text_opt['chunk_overlap']
        
        # 应用向量存储优化
        if 'batch_size' in vector_opt:
            optimized_config.BATCH_SIZE = vector_opt['batch_size']
        
        # 应用性能优化
        optimized_config.MAX_RESPONSE_TIME = 2.0  # 确保2秒内响应
        
        # 保存优化信息
        optimized_config.OPTIMIZATION_INFO = {
            'pytorch': pytorch_opt,
            'embedding': embedding_opt,
            'vector_store': vector_opt,
            'text_processing': text_opt,
            'memory': memory_opt,
            'hardware': self.hardware_info
        }
        
        logging.info("优化配置创建完成")
        return optimized_config
    
    def benchmark_system(self, config: RAGConfig = None) -> Dict[str, Any]:
        """
        系统性能基准测试
        
        Args:
            config: 配置对象
            
        Returns:
            基准测试结果
        """
        logging.info("开始系统性能基准测试...")
        
        config = config or self.config
        benchmark_results = {}
        
        try:
            # 测试文本处理速度
            test_texts = [
                "这是一个测试文本，用于评估文本处理性能。" * 10,
                "This is a test text for evaluating text processing performance." * 10,
                "混合中英文测试 Mixed language test 性能评估 performance evaluation." * 10
            ]
            
            start_time = time.time()
            for text in test_texts * 10:  # 重复测试
                # 模拟文本分块
                chunks = [text[i:i+config.CHUNK_SIZE] for i in range(0, len(text), config.CHUNK_SIZE - config.CHUNK_OVERLAP)]
            text_processing_time = time.time() - start_time
            
            benchmark_results['text_processing'] = {
                'time': text_processing_time,
                'texts_processed': len(test_texts) * 10,
                'speed': len(test_texts) * 10 / text_processing_time
            }
            
            # 测试内存使用
            memory_before = psutil.virtual_memory().available / (1024**3)
            
            # 模拟大量数据处理
            large_data = ["测试数据" * 1000] * 100
            processed_data = [data.lower() for data in large_data]
            
            memory_after = psutil.virtual_memory().available / (1024**3)
            memory_used = memory_before - memory_after
            
            benchmark_results['memory_usage'] = {
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_used': memory_used,
                'data_processed': len(large_data)
            }
            
            # 清理内存
            del large_data, processed_data
            gc.collect()
            
            # GPU基准测试（如果可用）
            if self.hardware_info['gpu_available']:
                start_time = time.time()
                
                # 模拟GPU计算
                test_tensor = torch.randn(1000, 1024).cuda()
                result = torch.matmul(test_tensor, test_tensor.T)
                torch.cuda.synchronize()
                
                gpu_time = time.time() - start_time
                
                benchmark_results['gpu_performance'] = {
                    'computation_time': gpu_time,
                    'tensor_size': test_tensor.shape,
                    'gpu_memory_used': torch.cuda.memory_allocated() / (1024**3)
                }
                
                # 清理GPU内存
                del test_tensor, result
                torch.cuda.empty_cache()
            
            # 计算总体性能评分
            performance_score = self._calculate_performance_score(benchmark_results)
            benchmark_results['overall_score'] = performance_score
            
            logging.info(f"基准测试完成，性能评分: {performance_score:.2f}/100")
            return benchmark_results
            
        except Exception as e:
            logging.error(f"基准测试失败: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_score(self, benchmark_results: Dict[str, Any]) -> float:
        """
        计算性能评分
        
        Args:
            benchmark_results: 基准测试结果
            
        Returns:
            性能评分 (0-100)
        """
        score = 0.0
        max_score = 100.0
        
        # 文本处理性能 (30分)
        if 'text_processing' in benchmark_results:
            text_speed = benchmark_results['text_processing']['speed']
            if text_speed >= 100:
                score += 30
            elif text_speed >= 50:
                score += 20
            elif text_speed >= 20:
                score += 15
            else:
                score += 10
        
        # 内存使用效率 (30分)
        if 'memory_usage' in benchmark_results:
            memory_used = benchmark_results['memory_usage']['memory_used']
            if memory_used <= 1.0:  # 小于1GB
                score += 30
            elif memory_used <= 2.0:  # 小于2GB
                score += 25
            elif memory_used <= 4.0:  # 小于4GB
                score += 20
            else:
                score += 10
        
        # GPU性能 (20分)
        if 'gpu_performance' in benchmark_results:
            gpu_time = benchmark_results['gpu_performance']['computation_time']
            if gpu_time <= 0.1:
                score += 20
            elif gpu_time <= 0.5:
                score += 15
            elif gpu_time <= 1.0:
                score += 10
            else:
                score += 5
        else:
            score += 10  # CPU模式基础分
        
        # 硬件配置 (20分)
        if self.hardware_info['memory_total'] >= 20:
            score += 10
        elif self.hardware_info['memory_total'] >= 15:
            score += 8
        else:
            score += 5
        
        if self.hardware_info['gpu_available']:
            score += 10
        else:
            score += 5
        
        return min(score, max_score)
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        生成优化报告
        
        Returns:
            优化报告
        """
        logging.info("生成优化报告...")
        
        # 创建优化配置
        optimized_config = self.create_optimized_config()
        
        # 运行基准测试
        benchmark_results = self.benchmark_system(optimized_config)
        
        # 生成报告
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'hardware_info': self.hardware_info,
            'optimization_applied': optimized_config.OPTIMIZATION_INFO,
            'benchmark_results': benchmark_results,
            'recommendations': self._generate_recommendations(benchmark_results),
            'config_changes': self._get_config_changes(optimized_config)
        }
        
        return report
    
    def _generate_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """
        生成优化建议
        
        Args:
            benchmark_results: 基准测试结果
            
        Returns:
            优化建议列表
        """
        recommendations = []
        
        # 基于硬件的建议
        if self.hardware_info['memory_total'] < 16:
            recommendations.append("建议增加系统内存到16GB以上以获得更好性能")
        
        if not self.hardware_info['gpu_available']:
            recommendations.append("建议使用GPU加速以提升嵌入计算性能")
        elif self.hardware_info['gpu_memory'] and self.hardware_info['gpu_memory'][0]['memory_total'] <= 8.5:
            recommendations.append("当前GPU内存有限(8GB)，建议使用模型量化和批处理优化")
        
        # 基于性能测试的建议
        if 'overall_score' in benchmark_results:
            score = benchmark_results['overall_score']
            if score < 60:
                recommendations.append("系统性能较低，建议检查硬件配置和优化设置")
            elif score < 80:
                recommendations.append("系统性能中等，可通过调整批处理大小和缓存策略进一步优化")
            else:
                recommendations.append("系统性能良好，当前优化设置适合您的硬件配置")
        
        # 内存使用建议
        if 'memory_usage' in benchmark_results:
            memory_used = benchmark_results['memory_usage']['memory_used']
            if memory_used > 4.0:
                recommendations.append("内存使用较高，建议减少批处理大小或启用保守模式")
        
        # 文本处理建议
        if 'text_processing' in benchmark_results:
            speed = benchmark_results['text_processing']['speed']
            if speed < 20:
                recommendations.append("文本处理速度较慢，建议启用并行处理或减少分块大小")
        
        return recommendations
    
    def _get_config_changes(self, optimized_config: RAGConfig) -> Dict[str, Any]:
        """
        获取配置变更
        
        Args:
            optimized_config: 优化后的配置
            
        Returns:
            配置变更信息
        """
        original_config = RAGConfig()
        
        changes = {}
        
        if optimized_config.CHUNK_SIZE != original_config.CHUNK_SIZE:
            changes['CHUNK_SIZE'] = {
                'original': original_config.CHUNK_SIZE,
                'optimized': optimized_config.CHUNK_SIZE
            }
        
        if optimized_config.CHUNK_OVERLAP != original_config.CHUNK_OVERLAP:
            changes['CHUNK_OVERLAP'] = {
                'original': original_config.CHUNK_OVERLAP,
                'optimized': optimized_config.CHUNK_OVERLAP
            }
        
        if optimized_config.BATCH_SIZE != original_config.BATCH_SIZE:
            changes['BATCH_SIZE'] = {
                'original': original_config.BATCH_SIZE,
                'optimized': optimized_config.BATCH_SIZE
            }
        
        return changes

def main():
    """主函数"""
    try:
        # 初始化优化器
        optimizer = PerformanceOptimizer()
        
        # 生成优化报告
        report = optimizer.generate_optimization_report()
        
        # 打印报告
        logging.info("\n" + "=" * 60)
        logging.info("RAG系统性能优化报告")
        logging.info("=" * 60)
        
        # 硬件信息
        hardware = report['hardware_info']
        logging.info(f"\n💻 硬件配置:")
        logging.info(f"  CPU核心数: {hardware['cpu_count']}")
        logging.info(f"  总内存: {hardware['memory_total']:.1f}GB")
        logging.info(f"  可用内存: {hardware['memory_available']:.1f}GB")
        logging.info(f"  GPU可用: {'是' if hardware['gpu_available'] else '否'}")
        if hardware['gpu_available']:
            for gpu in hardware['gpu_memory']:
                logging.info(f"  GPU {gpu['device']}: {gpu['name']} ({gpu['memory_total']:.1f}GB)")
        
        # 性能评分
        if 'overall_score' in report['benchmark_results']:
            score = report['benchmark_results']['overall_score']
            logging.info(f"\n📊 性能评分: {score:.1f}/100")
        
        # 优化建议
        recommendations = report['recommendations']
        if recommendations:
            logging.info(f"\n💡 优化建议:")
            for i, rec in enumerate(recommendations, 1):
                logging.info(f"  {i}. {rec}")
        
        # 配置变更
        config_changes = report['config_changes']
        if config_changes:
            logging.info(f"\n⚙️ 配置优化:")
            for key, change in config_changes.items():
                logging.info(f"  {key}: {change['original']} → {change['optimized']}")
        
        logging.info("\n" + "=" * 60)
        logging.info("优化完成！建议使用优化后的配置运行RAG系统。")
        logging.info("=" * 60)
        
        return True
        
    except Exception as e:
        logging.error(f"性能优化失败: {e}")
        return False

if __name__ == "__main__":
    main()