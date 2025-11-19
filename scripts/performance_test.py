#!/usr/bin/env python3
"""
性能测试和验证脚本
测试模型缓存机制、错误处理和性能优化的效果
"""

import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import psutil
import gc

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.model_cache import ModelCacheManager
from scripts.error_handler import ErrorHandler, ErrorType
from scripts.gpu_manager import GPUManager
from evaluate_lfm2 import load_model_and_tokenizer, generate, apply_chat_template

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        
        # CPU内存
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024**3  # GB
        
        # GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        self.logger.info(f"开始监控 - CPU内存: {self.start_memory:.2f}GB, GPU内存: {self.start_gpu_memory:.2f}GB")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """停止监控并返回统计信息"""
        end_time = time.time()
        
        # CPU内存
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024**3  # GB
        
        # GPU内存
        end_gpu_memory = 0
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        stats = {
            "duration": end_time - self.start_time,
            "cpu_memory_start": self.start_memory,
            "cpu_memory_end": end_memory,
            "cpu_memory_delta": end_memory - self.start_memory,
            "gpu_memory_start": self.start_gpu_memory,
            "gpu_memory_end": end_gpu_memory,
            "gpu_memory_delta": end_gpu_memory - self.start_gpu_memory
        }
        
        self.logger.info(f"监控结束 - 耗时: {stats['duration']:.2f}s, "
                        f"CPU内存变化: {stats['cpu_memory_delta']:.2f}GB, "
                        f"GPU内存变化: {stats['gpu_memory_delta']:.2f}GB")
        
        return stats

class ModelLoadingTest:
    """模型加载测试"""
    
    def __init__(self, cache_dir: str = "models", log_level: str = "INFO"):
        self.cache_dir = cache_dir
        self.logger = self._setup_logging(log_level)
        self.cache_manager = ModelCacheManager(cache_dir)
        self.error_handler = ErrorHandler("PerformanceTest")
        self.gpu_manager = GPUManager()
        self.results = []
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def test_cache_performance(self, model_id: str = "LiquidAI/LFM2-1.2B") -> Dict[str, any]:
        """测试缓存性能"""
        self.logger.info(f"=== 测试缓存性能: {model_id} ===")
        
        # 清理缓存以确保测试准确性
        if self.cache_manager.is_model_cached(model_id):
            self.logger.info("清理现有缓存")
            cache_info = self.cache_manager.get_cache_info(model_id)
            if cache_info:
                cache_path = Path(cache_info.cache_path)
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
        
        results = {}
        
        # 第一次加载（无缓存）
        self.logger.info("第一次加载（无缓存）")
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            model, tokenizer, load_info = load_model_and_tokenizer(
                model_id=model_id,
                dtype="bf16",
                local_models_dir=self.cache_dir,
                use_local=True,
                force_download=False,
                max_memory_gb=7.0,
                gpu_manager=self.gpu_manager,
                cache_manager=self.cache_manager,
                error_handler=self.error_handler
            )
            
            first_load_stats = monitor.stop_monitoring()
            results["first_load"] = {
                "success": True,
                "stats": first_load_stats,
                "load_info": load_info
            }
            
            # 清理模型以释放内存
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            first_load_stats = monitor.stop_monitoring()
            results["first_load"] = {
                "success": False,
                "error": str(e),
                "stats": first_load_stats
            }
            self.logger.error(f"第一次加载失败: {e}")
        
        # 等待一段时间
        time.sleep(2)
        
        # 第二次加载（有缓存）
        self.logger.info("第二次加载（有缓存）")
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            model, tokenizer, load_info = load_model_and_tokenizer(
                model_id=model_id,
                dtype="bf16",
                local_models_dir=self.cache_dir,
                use_local=True,
                force_download=False,
                max_memory_gb=7.0,
                gpu_manager=self.gpu_manager,
                cache_manager=self.cache_manager,
                error_handler=self.error_handler
            )
            
            second_load_stats = monitor.stop_monitoring()
            results["second_load"] = {
                "success": True,
                "stats": second_load_stats,
                "load_info": load_info
            }
            
            # 清理模型
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            second_load_stats = monitor.stop_monitoring()
            results["second_load"] = {
                "success": False,
                "error": str(e),
                "stats": second_load_stats
            }
            self.logger.error(f"第二次加载失败: {e}")
        
        # 计算性能提升
        if results["first_load"]["success"] and results["second_load"]["success"]:
            first_time = results["first_load"]["stats"]["duration"]
            second_time = results["second_load"]["stats"]["duration"]
            speedup = first_time / second_time if second_time > 0 else float('inf')
            results["performance_improvement"] = {
                "speedup": speedup,
                "time_saved": first_time - second_time,
                "percentage_improvement": ((first_time - second_time) / first_time) * 100
            }
            
            self.logger.info(f"性能提升: {speedup:.2f}x, 节省时间: {first_time - second_time:.2f}s, "
                           f"提升百分比: {results['performance_improvement']['percentage_improvement']:.1f}%")
        
        return results
    
    def test_error_handling(self) -> Dict[str, any]:
        """测试错误处理"""
        self.logger.info("=== 测试错误处理 ===")
        
        results = {}
        
        # 测试无效模型ID
        self.logger.info("测试无效模型ID")
        try:
            model, tokenizer, load_info = load_model_and_tokenizer(
                model_id="invalid/model-id-12345",
                dtype="bf16",
                local_models_dir=self.cache_dir,
                use_local=False,  # 强制在线加载
                force_download=False,
                max_memory_gb=7.0,
                gpu_manager=self.gpu_manager,
                cache_manager=self.cache_manager,
                error_handler=self.error_handler
            )
            results["invalid_model"] = {"success": True, "unexpected": True}
        except Exception as e:
            results["invalid_model"] = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.logger.info(f"正确捕获无效模型错误: {e}")
        
        # 测试内存不足情况（模拟）
        self.logger.info("测试内存限制")
        try:
            model, tokenizer, load_info = load_model_and_tokenizer(
                model_id="LiquidAI/LFM2-1.2B",
                dtype="bf16",
                local_models_dir=self.cache_dir,
                use_local=True,
                force_download=False,
                max_memory_gb=0.1,  # 极小的内存限制
                gpu_manager=self.gpu_manager,
                cache_manager=self.cache_manager,
                error_handler=self.error_handler
            )
            results["memory_limit"] = {"success": True}
            # 清理
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            results["memory_limit"] = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.logger.info(f"内存限制测试结果: {e}")
        
        # 获取错误统计
        error_stats = self.error_handler.get_error_statistics()
        results["error_stats"] = error_stats
        
        return results
    
    def test_generation_performance(self, model_id: str = "LiquidAI/LFM2-1.2B") -> Dict[str, any]:
        """测试生成性能"""
        self.logger.info(f"=== 测试生成性能: {model_id} ===")
        
        try:
            # 加载模型
            model, tokenizer, load_info = load_model_and_tokenizer(
                model_id=model_id,
                dtype="bf16",
                local_models_dir=self.cache_dir,
                use_local=True,
                force_download=False,
                max_memory_gb=7.0,
                gpu_manager=self.gpu_manager,
                cache_manager=self.cache_manager,
                error_handler=self.error_handler
            )
            
            # 准备测试提示
            test_prompts = [
                "What is artificial intelligence?",
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint."
            ]
            
            results = {
                "model_info": load_info,
                "generation_tests": []
            }
            
            for i, prompt in enumerate(test_prompts):
                self.logger.info(f"测试提示 {i+1}: {prompt[:50]}...")
                
                # 准备输入
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                input_ids = apply_chat_template(tokenizer, messages)
                
                # 性能监控
                monitor = PerformanceMonitor()
                monitor.start_monitoring()
                
                try:
                    text, full_text, gen_time, new_tokens, tps = generate(
                        model, tokenizer, input_ids,
                        max_new_tokens=128,
                        temperature=0.3,
                        repetition_penalty=1.05,
                        stream=False
                    )
                    
                    gen_stats = monitor.stop_monitoring()
                    
                    test_result = {
                        "prompt": prompt,
                        "success": True,
                        "generated_text": text[:200] + "..." if len(text) > 200 else text,
                        "generation_time": gen_time,
                        "new_tokens": new_tokens,
                        "tokens_per_second": tps,
                        "performance_stats": gen_stats
                    }
                    
                    self.logger.info(f"生成成功 - 耗时: {gen_time:.2f}s, "
                                   f"新token数: {new_tokens}, TPS: {tps:.2f}")
                    
                except Exception as e:
                    gen_stats = monitor.stop_monitoring()
                    test_result = {
                        "prompt": prompt,
                        "success": False,
                        "error": str(e),
                        "performance_stats": gen_stats
                    }
                    self.logger.error(f"生成失败: {e}")
                
                results["generation_tests"].append(test_result)
            
            # 清理模型
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
            return results
            
        except Exception as e:
            self.logger.error(f"生成性能测试失败: {e}")
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_test(self, model_id: str = "LiquidAI/LFM2-1.2B") -> Dict[str, any]:
        """运行综合测试"""
        self.logger.info("=== 开始综合性能测试 ===")
        
        comprehensive_results = {
            "model_id": model_id,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # 缓存性能测试
        try:
            cache_results = self.test_cache_performance(model_id)
            comprehensive_results["cache_performance"] = cache_results
        except Exception as e:
            self.logger.error(f"缓存性能测试失败: {e}")
            comprehensive_results["cache_performance"] = {"error": str(e)}
        
        # 错误处理测试
        try:
            error_results = self.test_error_handling()
            comprehensive_results["error_handling"] = error_results
        except Exception as e:
            self.logger.error(f"错误处理测试失败: {e}")
            comprehensive_results["error_handling"] = {"error": str(e)}
        
        # 生成性能测试
        try:
            generation_results = self.test_generation_performance(model_id)
            comprehensive_results["generation_performance"] = generation_results
        except Exception as e:
            self.logger.error(f"生成性能测试失败: {e}")
            comprehensive_results["generation_performance"] = {"error": str(e)}
        
        return comprehensive_results
    
    def save_results(self, results: Dict[str, any], output_file: str = None):
        """保存测试结果"""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_test_results_{timestamp}.json"
        
        output_path = Path("outputs") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"测试结果已保存到: {output_path}")
        return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LFM2模型性能测试")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-1.2B",
                       help="要测试的模型ID")
    parser.add_argument("--cache_dir", type=str, default="models",
                       help="缓存目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出文件名")
    parser.add_argument("--test_type", type=str, default="comprehensive",
                       choices=["cache", "error", "generation", "comprehensive"],
                       help="测试类型")
    
    args = parser.parse_args()
    
    # 创建测试实例
    tester = ModelLoadingTest(args.cache_dir, args.log_level)
    
    try:
        if args.test_type == "cache":
            results = tester.test_cache_performance(args.model_id)
        elif args.test_type == "error":
            results = tester.test_error_handling()
        elif args.test_type == "generation":
            results = tester.test_generation_performance(args.model_id)
        else:  # comprehensive
            results = tester.run_comprehensive_test(args.model_id)
        
        # 保存结果
        output_path = tester.save_results(results, args.output)
        
        # 打印摘要
        print("\n=== 测试完成 ===")
        print(f"结果已保存到: {output_path}")
        
        if "cache_performance" in results:
            cache_perf = results["cache_performance"]
            if "performance_improvement" in cache_perf:
                perf = cache_perf["performance_improvement"]
                print(f"缓存性能提升: {perf['speedup']:.2f}x ({perf['percentage_improvement']:.1f}%)")
        
        if "error_handling" in results:
            error_stats = results["error_handling"].get("error_stats", {})
            if isinstance(error_stats, dict) and "error_counts" in error_stats:
                total_errors = sum(error_stats["error_counts"].values()) if error_stats["error_counts"] else 0
            else:
                total_errors = error_stats.get("total_errors", 0) if isinstance(error_stats, dict) else 0
            print(f"错误处理测试: 总计 {total_errors} 个错误被正确处理")
        
        if "generation_performance" in results:
            gen_perf = results["generation_performance"]
            if "generation_tests" in gen_perf:
                successful_tests = sum(1 for test in gen_perf["generation_tests"] if test["success"])
                total_tests = len(gen_perf["generation_tests"])
                print(f"生成测试: {successful_tests}/{total_tests} 成功")
        
    except Exception as e:
        tester.logger.error(f"测试过程中发生错误: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())