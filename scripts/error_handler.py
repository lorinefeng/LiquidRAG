"""
错误处理和回退机制模块
提供统一的错误处理、重试逻辑和回退策略
"""

import logging
import time
import traceback
from typing import Optional, Callable, Any, Dict, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import torch


class ErrorType(Enum):
    """错误类型枚举"""
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    MODEL_LOADING_ERROR = "model_loading_error"
    CUDA_ERROR = "cuda_error"
    FILE_ERROR = "file_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True


@dataclass
class ErrorInfo:
    """错误信息"""
    error_type: ErrorType
    message: str
    exception: Optional[Exception] = None
    timestamp: float = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, logger_name: str = "ErrorHandler"):
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[ErrorInfo] = []
        self.retry_configs: Dict[ErrorType, RetryConfig] = {
            ErrorType.NETWORK_ERROR: RetryConfig(max_retries=5, base_delay=2.0),
            ErrorType.MEMORY_ERROR: RetryConfig(max_retries=2, base_delay=5.0),
            ErrorType.MODEL_LOADING_ERROR: RetryConfig(max_retries=3, base_delay=3.0),
            ErrorType.CUDA_ERROR: RetryConfig(max_retries=2, base_delay=1.0),
            ErrorType.FILE_ERROR: RetryConfig(max_retries=3, base_delay=1.0),
            ErrorType.TIMEOUT_ERROR: RetryConfig(max_retries=2, base_delay=5.0),
            ErrorType.UNKNOWN_ERROR: RetryConfig(max_retries=1, base_delay=1.0),
        }
    
    def classify_error(self, exception: Exception) -> ErrorType:
        """分类错误类型"""
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # 网络相关错误
        if any(keyword in error_msg for keyword in [
            'connection', 'network', 'timeout', 'http', 'ssl', 'certificate',
            'resolve', 'unreachable', 'refused'
        ]):
            return ErrorType.NETWORK_ERROR
        
        # 内存相关错误
        if any(keyword in error_msg for keyword in [
            'memory', 'oom', 'out of memory', 'cuda out of memory',
            'allocation', 'insufficient'
        ]) or 'outofmemoryerror' in exception_type:
            return ErrorType.MEMORY_ERROR
        
        # CUDA相关错误
        if any(keyword in error_msg for keyword in [
            'cuda', 'gpu', 'device', 'cudnn', 'cublas', 'curand'
        ]) or 'cuda' in exception_type:
            return ErrorType.CUDA_ERROR
        
        # 文件相关错误
        if any(keyword in error_msg for keyword in [
            'file', 'directory', 'path', 'permission', 'not found',
            'exists', 'access'
        ]) or exception_type in ['filenotfounderror', 'permissionerror', 'ioerror']:
            return ErrorType.FILE_ERROR
        
        # 模型加载相关错误
        if any(keyword in error_msg for keyword in [
            'model', 'tokenizer', 'config', 'checkpoint', 'safetensors',
            'transformers', 'huggingface'
        ]):
            return ErrorType.MODEL_LOADING_ERROR
        
        # 超时错误
        if 'timeout' in error_msg or 'timeouterror' in exception_type:
            return ErrorType.TIMEOUT_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def log_error(self, error_info: ErrorInfo):
        """记录错误信息"""
        self.error_history.append(error_info)
        
        # 保持错误历史记录在合理范围内
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]
        
        self.logger.error(f"[{error_info.error_type.value}] {error_info.message}")
        if error_info.exception:
            self.logger.error(f"异常详情: {error_info.exception}")
            self.logger.debug(f"堆栈跟踪: {traceback.format_exc()}")
        
        if error_info.context:
            self.logger.debug(f"错误上下文: {error_info.context}")
    
    def get_retry_delay(self, error_type: ErrorType, attempt: int) -> float:
        """计算重试延迟时间"""
        config = self.retry_configs.get(error_type, RetryConfig())
        
        # 指数退避
        delay = min(
            config.base_delay * (config.backoff_factor ** attempt),
            config.max_delay
        )
        
        # 添加抖动
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """判断是否应该重试"""
        config = self.retry_configs.get(error_type, RetryConfig())
        return attempt < config.max_retries
    
    def handle_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """处理错误并返回错误信息"""
        error_type = self.classify_error(exception)
        error_info = ErrorInfo(
            error_type=error_type,
            message=str(exception),
            exception=exception,
            context=context
        )
        
        self.log_error(error_info)
        return error_info
    
    def retry_with_backoff(self, 
                          func: Callable,
                          *args,
                          error_context: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Any:
        """带退避的重试执行"""
        last_error = None
        
        for attempt in range(10):  # 最大尝试次数
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = self.handle_error(e, error_context)
                last_error = error_info
                
                if not self.should_retry(error_info.error_type, attempt):
                    self.logger.error(f"达到最大重试次数，放弃执行")
                    break
                
                delay = self.get_retry_delay(error_info.error_type, attempt)
                self.logger.info(f"第{attempt + 1}次重试失败，{delay:.1f}秒后重试...")
                time.sleep(delay)
        
        # 所有重试都失败了
        if last_error:
            raise last_error.exception
        else:
            raise RuntimeError("未知错误导致重试失败")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        recent_errors = []
        current_time = time.time()
        
        for error_info in self.error_history:
            error_type = error_info.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # 最近1小时的错误
            if current_time - error_info.timestamp < 3600:
                recent_errors.append(error_info)
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "recent_errors_count": len(recent_errors),
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }


class ModelLoadingFallback:
    """模型加载回退策略"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger("ModelLoadingFallback")
    
    def try_reduce_memory_usage(self):
        """尝试减少内存使用"""
        self.logger.info("尝试减少内存使用...")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("已清理GPU缓存")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        self.logger.info("已执行垃圾回收")
    
    def try_fallback_dtype(self, current_dtype: str) -> Optional[str]:
        """尝试回退到更低精度的数据类型"""
        fallback_map = {
            "bf16": "fp16",
            "fp16": "fp32",
            "fp32": None  # 无法进一步回退
        }
        
        fallback_dtype = fallback_map.get(current_dtype)
        if fallback_dtype:
            self.logger.info(f"回退数据类型: {current_dtype} -> {fallback_dtype}")
        else:
            self.logger.warning(f"无法从{current_dtype}进一步回退数据类型")
        
        return fallback_dtype
    
    def try_fallback_device(self, current_device_map) -> Optional[str]:
        """尝试回退到更简单的设备配置"""
        if isinstance(current_device_map, dict) and len(current_device_map) > 1:
            # 从多GPU回退到单GPU
            self.logger.info("从多GPU回退到单GPU")
            return "auto"
        elif current_device_map == "auto" and torch.cuda.is_available():
            # 从GPU回退到CPU
            self.logger.info("从GPU回退到CPU")
            return "cpu"
        else:
            self.logger.warning("无法进一步回退设备配置")
            return None
    
    def suggest_memory_reduction(self, available_memory_gb: float) -> Dict[str, Any]:
        """建议内存减少策略"""
        suggestions = {
            "reduce_max_memory": max(available_memory_gb * 0.8, 1.0),
            "use_cpu_offload": available_memory_gb < 4.0,
            "use_8bit": available_memory_gb < 6.0,
            "use_4bit": available_memory_gb < 3.0
        }
        
        self.logger.info(f"内存优化建议: {suggestions}")
        return suggestions


def create_error_handler(logger_name: str = "LFM2_ErrorHandler") -> ErrorHandler:
    """创建错误处理器的便捷函数"""
    return ErrorHandler(logger_name)


def handle_model_loading_error(error_handler: ErrorHandler, 
                              exception: Exception,
                              context: Dict[str, Any]) -> Dict[str, Any]:
    """处理模型加载错误的便捷函数"""
    error_info = error_handler.handle_error(exception, context)
    fallback = ModelLoadingFallback(error_handler)
    
    suggestions = {}
    
    if error_info.error_type == ErrorType.MEMORY_ERROR:
        fallback.try_reduce_memory_usage()
        suggestions.update(fallback.suggest_memory_reduction(
            context.get("available_memory_gb", 8.0)
        ))
        
        # 建议数据类型回退
        current_dtype = context.get("dtype", "bf16")
        fallback_dtype = fallback.try_fallback_dtype(current_dtype)
        if fallback_dtype:
            suggestions["fallback_dtype"] = fallback_dtype
    
    elif error_info.error_type == ErrorType.CUDA_ERROR:
        fallback.try_reduce_memory_usage()
        
        # 建议设备回退
        current_device_map = context.get("device_map", "auto")
        fallback_device = fallback.try_fallback_device(current_device_map)
        if fallback_device:
            suggestions["fallback_device_map"] = fallback_device
    
    return {
        "error_info": error_info,
        "suggestions": suggestions
    }


if __name__ == "__main__":
    # 测试错误处理器
    error_handler = create_error_handler()
    
    # 模拟各种错误
    test_errors = [
        ConnectionError("网络连接失败"),
        RuntimeError("CUDA out of memory"),
        FileNotFoundError("模型文件未找到"),
        ValueError("无效的模型配置")
    ]
    
    for error in test_errors:
        error_info = error_handler.handle_error(error, {"test": True})
        print(f"错误类型: {error_info.error_type.value}")
    
    # 显示统计信息
    stats = error_handler.get_error_statistics()
    print(f"错误统计: {stats}")