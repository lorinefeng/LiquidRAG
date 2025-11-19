"""
GPU管理模块 - 提供多GPU支持和设备管理功能
"""

import os
import logging
import torch
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class GPUInfo:
    """GPU信息数据类"""
    id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # %
    temperature: Optional[float] = None  # °C
    power_usage: Optional[float] = None  # W


class GPUManager:
    """GPU管理器 - 处理多GPU环境下的设备分配和监控"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.available_devices = list(range(self.device_count))
        self._gpu_info_cache = {}
        self._cache_timeout = 5.0  # 缓存5秒
        self._last_update = 0
        
    def is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        return torch.cuda.is_available()
    
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        return self.device_count
    
    def get_gpu_info(self, device_id: int) -> GPUInfo:
        """
        获取指定GPU的详细信息
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            GPUInfo对象
        """
        if not self.is_cuda_available() or device_id >= self.device_count:
            raise ValueError(f"无效的GPU设备ID: {device_id}")
        
        # 检查缓存
        current_time = time.time()
        cache_key = f"gpu_{device_id}"
        
        if (cache_key in self._gpu_info_cache and 
            current_time - self._last_update < self._cache_timeout):
            return self._gpu_info_cache[cache_key]
        
        # 获取GPU信息
        props = torch.cuda.get_device_properties(device_id)
        
        # 切换到指定设备获取内存信息
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        
        try:
            memory_total = props.total_memory // (1024**2)  # MB
            memory_allocated = torch.cuda.memory_allocated(device_id) // (1024**2)
            memory_reserved = torch.cuda.memory_reserved(device_id) // (1024**2)
            memory_free = memory_total - memory_reserved
            
            # 尝试获取GPU利用率（需要nvidia-ml-py）
            utilization = 0.0
            temperature = None
            power_usage = None
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # GPU利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
                
                # 温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass
                
                # 功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    pass
                    
            except ImportError:
                self.logger.debug("pynvml未安装，无法获取详细GPU信息")
            except Exception as e:
                self.logger.debug(f"获取GPU详细信息失败: {e}")
            
            gpu_info = GPUInfo(
                id=device_id,
                name=props.name,
                memory_total=memory_total,
                memory_free=memory_free,
                memory_used=memory_allocated,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage
            )
            
            # 更新缓存
            self._gpu_info_cache[cache_key] = gpu_info
            self._last_update = current_time
            
            return gpu_info
            
        finally:
            # 恢复原设备
            torch.cuda.set_device(current_device)
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有GPU信息"""
        if not self.is_cuda_available():
            return []
        
        return [self.get_gpu_info(i) for i in range(self.device_count)]
    
    def get_best_device(self, min_memory_mb: int = 1000) -> Optional[int]:
        """
        选择最佳GPU设备
        
        Args:
            min_memory_mb: 最小内存要求(MB)
            
        Returns:
            最佳设备ID，如果没有合适设备返回None
        """
        if not self.is_cuda_available():
            return None
        
        best_device = None
        max_free_memory = 0
        
        for i in range(self.device_count):
            try:
                gpu_info = self.get_gpu_info(i)
                
                # 检查内存要求
                if gpu_info.memory_free < min_memory_mb:
                    continue
                
                # 选择空闲内存最多的设备
                if gpu_info.memory_free > max_free_memory:
                    max_free_memory = gpu_info.memory_free
                    best_device = i
                    
            except Exception as e:
                self.logger.warning(f"检查GPU {i} 信息失败: {e}")
                continue
        
        return best_device
    
    def get_memory_efficient_device_map(
        self, 
        model_size_mb: float,
        max_memory_per_gpu_mb: Optional[int] = None,
        exclude_devices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        根据模型大小和GPU内存情况生成设备映射
        
        Args:
            model_size_mb: 模型大小(MB)
            max_memory_per_gpu_mb: 每个GPU最大使用内存(MB)
            exclude_devices: 排除的设备列表
            
        Returns:
            设备映射字典
        """
        if not self.is_cuda_available():
            return {"": "cpu"}
        
        exclude_devices = exclude_devices or []
        available_gpus = [i for i in range(self.device_count) if i not in exclude_devices]
        
        if not available_gpus:
            return {"": "cpu"}
        
        # 获取GPU信息
        gpu_infos = []
        for gpu_id in available_gpus:
            try:
                info = self.get_gpu_info(gpu_id)
                gpu_infos.append(info)
            except Exception as e:
                self.logger.warning(f"获取GPU {gpu_id} 信息失败: {e}")
                continue
        
        if not gpu_infos:
            return {"": "cpu"}
        
        # 单GPU情况
        if len(gpu_infos) == 1:
            gpu = gpu_infos[0]
            if max_memory_per_gpu_mb:
                available_memory = min(gpu.memory_free, max_memory_per_gpu_mb)
            else:
                available_memory = gpu.memory_free * 0.9  # 保留10%缓冲
            
            if model_size_mb <= available_memory:
                return {"": f"cuda:{gpu.id}"}
            else:
                self.logger.warning(f"模型大小({model_size_mb}MB)超过GPU可用内存({available_memory}MB)")
                return {"": "cpu"}
        
        # 多GPU情况 - 使用accelerate的自动分配
        max_memory = {}
        for gpu in gpu_infos:
            if max_memory_per_gpu_mb:
                memory_limit = min(gpu.memory_free, max_memory_per_gpu_mb)
            else:
                memory_limit = gpu.memory_free * 0.9
            
            max_memory[gpu.id] = f"{int(memory_limit)}MB"
        
        self.logger.info(f"多GPU内存分配: {max_memory}")
        return max_memory
    
    def optimize_for_inference(self, device_ids: Optional[List[int]] = None):
        """
        为推理优化GPU设置
        
        Args:
            device_ids: 要优化的设备ID列表，None表示所有设备
        """
        if not self.is_cuda_available():
            return
        
        if device_ids is None:
            device_ids = list(range(self.device_count))
        
        for device_id in device_ids:
            if device_id >= self.device_count:
                continue
                
            try:
                torch.cuda.set_device(device_id)
                
                # 清理缓存
                torch.cuda.empty_cache()
                
                # 设置内存分配策略
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.95, device_id)
                
                # 启用TF32与cudnn benchmark以提升吞吐
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
                
                self.logger.info(f"GPU {device_id} 推理优化完成")
                
            except Exception as e:
                self.logger.warning(f"优化GPU {device_id} 失败: {e}")
    
    def monitor_memory_usage(self, device_id: int) -> Dict[str, float]:
        """
        监控指定GPU的内存使用情况
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            内存使用信息字典
        """
        if not self.is_cuda_available() or device_id >= self.device_count:
            return {}
        
        try:
            gpu_info = self.get_gpu_info(device_id)
            
            return {
                "total_mb": gpu_info.memory_total,
                "used_mb": gpu_info.memory_used,
                "free_mb": gpu_info.memory_free,
                "utilization_percent": gpu_info.utilization,
                "usage_percent": (gpu_info.memory_used / gpu_info.memory_total) * 100
            }
            
        except Exception as e:
            self.logger.error(f"监控GPU {device_id} 内存失败: {e}")
            return {}
    
    def clear_cache(self, device_ids: Optional[List[int]] = None):
        """
        清理GPU缓存
        
        Args:
            device_ids: 要清理的设备ID列表，None表示所有设备
        """
        if not self.is_cuda_available():
            return
        
        if device_ids is None:
            device_ids = list(range(self.device_count))
        
        for device_id in device_ids:
            if device_id >= self.device_count:
                continue
                
            try:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                self.logger.debug(f"GPU {device_id} 缓存已清理")
                
            except Exception as e:
                self.logger.warning(f"清理GPU {device_id} 缓存失败: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "cuda_available": self.is_cuda_available(),
            "gpu_count": self.device_count,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if self.is_cuda_available():
            info["gpus"] = []
            for i in range(self.device_count):
                try:
                    gpu_info = self.get_gpu_info(i)
                    info["gpus"].append({
                        "id": gpu_info.id,
                        "name": gpu_info.name,
                        "memory_total_mb": gpu_info.memory_total,
                        "memory_free_mb": gpu_info.memory_free,
                        "utilization_percent": gpu_info.utilization
                    })
                except Exception as e:
                    self.logger.warning(f"获取GPU {i} 信息失败: {e}")
        
        return info
    
    def save_system_info(self, filepath: str):
        """保存系统信息到文件"""
        info = self.get_system_info()
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"系统信息已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存系统信息失败: {e}")


def get_optimal_batch_size(
    model_size_mb: float,
    sequence_length: int,
    available_memory_mb: float,
    dtype_size: int = 2  # bf16/fp16 = 2 bytes, fp32 = 4 bytes
) -> int:
    """
    根据模型大小和可用内存估算最优批处理大小
    
    Args:
        model_size_mb: 模型大小(MB)
        sequence_length: 序列长度
        available_memory_mb: 可用内存(MB)
        dtype_size: 数据类型大小(字节)
        
    Returns:
        建议的批处理大小
    """
    # 保留30%内存用于其他操作
    usable_memory_mb = available_memory_mb * 0.7
    
    # 减去模型占用的内存
    remaining_memory_mb = usable_memory_mb - model_size_mb
    
    if remaining_memory_mb <= 0:
        return 1
    
    # 估算每个样本的内存占用（输入+输出+梯度等）
    # 这是一个粗略估算，实际情况可能有所不同
    memory_per_sample_mb = (sequence_length * dtype_size * 4) / (1024**2)  # 4倍用于前向+反向
    
    batch_size = max(1, int(remaining_memory_mb / memory_per_sample_mb))
    
    # 限制最大批处理大小
    return min(batch_size, 32)


if __name__ == "__main__":
    # 测试GPU管理器
    logging.basicConfig(level=logging.INFO)
    
    gpu_manager = GPUManager()
    
    print("=== GPU管理器测试 ===")
    print(f"CUDA可用: {gpu_manager.is_cuda_available()}")
    print(f"GPU数量: {gpu_manager.get_gpu_count()}")
    
    if gpu_manager.is_cuda_available():
        print("\n=== GPU信息 ===")
        for gpu_info in gpu_manager.get_all_gpu_info():
            print(f"GPU {gpu_info.id}: {gpu_info.name}")
            print(f"  内存: {gpu_info.memory_used}MB / {gpu_info.memory_total}MB "
                  f"(空闲: {gpu_info.memory_free}MB)")
            print(f"  利用率: {gpu_info.utilization}%")
            if gpu_info.temperature:
                print(f"  温度: {gpu_info.temperature}°C")
            if gpu_info.power_usage:
                print(f"  功耗: {gpu_info.power_usage}W")
        
        print(f"\n最佳设备: GPU {gpu_manager.get_best_device()}")
        
        # 测试设备映射
        device_map = gpu_manager.get_memory_efficient_device_map(1000)  # 1GB模型
        print(f"设备映射: {device_map}")
    
    # 保存系统信息
    gpu_manager.save_system_info("system_info.json")