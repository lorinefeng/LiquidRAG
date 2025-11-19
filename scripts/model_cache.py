"""
模型缓存管理器
提供模型缓存、状态检查和元数据管理功能
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class ModelCacheInfo:
    """模型缓存信息"""
    model_id: str
    cache_path: str
    download_time: str
    file_size: int
    file_count: int
    last_accessed: str
    access_count: int
    checksum: Optional[str] = None
    config_hash: Optional[str] = None
    is_valid: bool = True
    load_time: Optional[float] = None
    device_map: Optional[str] = None


class ModelCacheManager:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir: str = "models", metadata_file: str = "cache_metadata.json"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            metadata_file: 元数据文件名
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / metadata_file
        self.logger = logging.getLogger("ModelCache")
        
        # 加载现有元数据
        self.metadata = self._load_metadata()
        
        # 清理过期缓存
        self._cleanup_expired_cache()
    
    def _load_metadata(self) -> Dict[str, ModelCacheInfo]:
        """加载缓存元数据"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = {}
            for model_id, info_dict in data.items():
                try:
                    metadata[model_id] = ModelCacheInfo(**info_dict)
                except Exception as e:
                    self.logger.warning(f"加载模型 {model_id} 元数据失败: {e}")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"加载元数据文件失败: {e}")
            return {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            data = {model_id: asdict(info) for model_id, info in self.metadata.items()}
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
    
    def _calculate_directory_hash(self, directory: Path) -> str:
        """计算目录内容的哈希值"""
        hash_md5 = hashlib.md5()
        
        try:
            for file_path in sorted(directory.rglob("*")):
                if file_path.is_file():
                    # 添加文件路径和大小到哈希
                    relative_path = file_path.relative_to(directory)
                    hash_md5.update(str(relative_path).encode())
                    hash_md5.update(str(file_path.stat().st_size).encode())
                    
                    # 对小文件添加内容哈希
                    if file_path.stat().st_size < 1024 * 1024:  # 1MB以下
                        with open(file_path, 'rb') as f:
                            hash_md5.update(f.read())
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"计算目录哈希失败: {e}")
            return ""
    
    def _get_directory_info(self, directory: Path) -> tuple:
        """获取目录信息"""
        total_size = 0
        file_count = 0
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return total_size, file_count
            
        except Exception as e:
            self.logger.warning(f"获取目录信息失败: {e}")
            return 0, 0
    
    def is_model_cached(self, model_id: str) -> bool:
        """检查模型是否已缓存"""
        if model_id not in self.metadata:
            # 尝试检测现有的模型目录
            model_dir = self.cache_dir / model_id.replace("/", "_")
            if model_dir.exists():
                self.logger.info(f"发现未记录的模型目录: {model_dir}")
                # 验证模型完整性
                if self._verify_model_files(model_dir):
                    # 添加到缓存元数据
                    self.add_cache_info(model_id, str(model_dir))
                    return True
            return False
        
        cache_info = self.metadata[model_id]
        cache_path = Path(cache_info.cache_path)
        
        # 检查路径是否存在
        if not cache_path.exists():
            self.logger.warning(f"缓存路径不存在: {cache_path}")
            self._remove_cache_info(model_id)
            return False
        
        # 验证模型文件
        if not self._verify_model_files(cache_path):
            self.logger.warning(f"模型文件验证失败: {cache_path}")
            cache_info.is_valid = False
            self._save_metadata()
            return False
        
        return cache_info.is_valid
    
    def _verify_model_files(self, model_path: Path) -> bool:
        """验证模型文件完整性"""
        # 检查Hugging Face缓存结构
        hf_cache_dir = model_path / ".cache" / "huggingface" / "download"
        if hf_cache_dir.exists():
            self.logger.info(f"检测到Hugging Face缓存结构: {hf_cache_dir}")
            # 检查缓存目录中的文件
            cache_files = list(hf_cache_dir.rglob("*"))
            if cache_files:
                # 检查是否有.incomplete文件（表示下载未完成）
                incomplete_files = [f for f in cache_files if f.name.endswith('.incomplete')]
                if incomplete_files:
                    self.logger.warning(f"发现未完成的下载文件: {len(incomplete_files)} 个")
                    return False
                
                # 检查是否有实际的模型文件
                model_files = [f for f in cache_files if f.is_file() and f.stat().st_size > 0]
                if len(model_files) >= 3:  # 至少应该有config、tokenizer和model文件
                    self.logger.info(f"Hugging Face缓存验证通过，找到 {len(model_files)} 个文件")
                    return True
        
        # 检查标准模型目录结构
        required_files = ["config.json"]
        for file_name in required_files:
            if not (model_path / file_name).exists():
                self.logger.warning(f"缺少必要文件: {file_name}")
                return False
        
        return True
    
    def get_cache_info(self, model_id: str) -> Optional[ModelCacheInfo]:
        """获取模型缓存信息"""
        return self.metadata.get(model_id)
    
    def add_cache_info(self, model_id: str, cache_path: str, **kwargs) -> ModelCacheInfo:
        """添加模型缓存信息"""
        cache_path_obj = Path(cache_path)
        
        # 获取目录信息
        file_size, file_count = self._get_directory_info(cache_path_obj)
        checksum = self._calculate_directory_hash(cache_path_obj)
        
        # 创建缓存信息
        cache_info = ModelCacheInfo(
            model_id=model_id,
            cache_path=str(cache_path_obj.absolute()),
            download_time=datetime.now().isoformat(),
            file_size=file_size,
            file_count=file_count,
            last_accessed=datetime.now().isoformat(),
            access_count=1,
            checksum=checksum,
            **kwargs
        )
        
        self.metadata[model_id] = cache_info
        self._save_metadata()
        
        self.logger.info(f"添加缓存信息: {model_id}")
        return cache_info
    
    def update_access_info(self, model_id: str, load_time: Optional[float] = None, 
                          device_map: Optional[str] = None):
        """更新访问信息"""
        if model_id in self.metadata:
            cache_info = self.metadata[model_id]
            cache_info.last_accessed = datetime.now().isoformat()
            cache_info.access_count += 1
            
            if load_time is not None:
                cache_info.load_time = load_time
            
            if device_map is not None:
                cache_info.device_map = str(device_map)
            
            self._save_metadata()
            self.logger.debug(f"更新访问信息: {model_id}")
    
    def verify_cache_integrity(self, model_id: str) -> bool:
        """验证缓存完整性"""
        if model_id not in self.metadata:
            return False
        
        cache_info = self.metadata[model_id]
        cache_path = Path(cache_info.cache_path)
        
        if not cache_path.exists():
            return False
        
        # 重新计算哈希
        current_checksum = self._calculate_directory_hash(cache_path)
        
        if cache_info.checksum and current_checksum != cache_info.checksum:
            self.logger.warning(f"缓存完整性验证失败: {model_id}")
            cache_info.is_valid = False
            self._save_metadata()
            return False
        
        # 检查文件数量
        current_size, current_count = self._get_directory_info(cache_path)
        if current_count != cache_info.file_count:
            self.logger.warning(f"文件数量不匹配: {model_id}")
            cache_info.is_valid = False
            self._save_metadata()
            return False
        
        return True
    
    def _remove_cache_info(self, model_id: str):
        """移除缓存信息"""
        if model_id in self.metadata:
            del self.metadata[model_id]
            self._save_metadata()
            self.logger.info(f"移除缓存信息: {model_id}")
    
    def remove_cache(self, model_id: str, remove_files: bool = False) -> bool:
        """移除模型缓存"""
        if model_id not in self.metadata:
            return False
        
        cache_info = self.metadata[model_id]
        
        if remove_files:
            cache_path = Path(cache_info.cache_path)
            if cache_path.exists():
                try:
                    import shutil
                    shutil.rmtree(cache_path)
                    self.logger.info(f"删除缓存文件: {cache_path}")
                except Exception as e:
                    self.logger.error(f"删除缓存文件失败: {e}")
                    return False
        
        self._remove_cache_info(model_id)
        return True
    
    def _cleanup_expired_cache(self, max_age_days: int = 30):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_models = []
        
        for model_id, cache_info in self.metadata.items():
            try:
                last_accessed = datetime.fromisoformat(cache_info.last_accessed)
                if current_time - last_accessed > timedelta(days=max_age_days):
                    expired_models.append(model_id)
            except Exception as e:
                self.logger.warning(f"解析访问时间失败: {model_id}, {e}")
        
        for model_id in expired_models:
            self.logger.info(f"清理过期缓存: {model_id}")
            self.remove_cache(model_id, remove_files=False)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(info.file_size for info in self.metadata.values())
        total_models = len(self.metadata)
        valid_models = sum(1 for info in self.metadata.values() if info.is_valid)
        
        # 最近访问的模型
        recent_models = sorted(
            self.metadata.items(),
            key=lambda x: x[1].last_accessed,
            reverse=True
        )[:5]
        
        return {
            "total_models": total_models,
            "valid_models": valid_models,
            "total_size_mb": total_size / (1024**2),
            "cache_directory": str(self.cache_dir),
            "recent_models": [
                {
                    "model_id": model_id,
                    "last_accessed": info.last_accessed,
                    "access_count": info.access_count,
                    "size_mb": info.file_size / (1024**2)
                }
                for model_id, info in recent_models
            ]
        }
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """列出所有缓存的模型"""
        models = []
        
        for model_id, cache_info in self.metadata.items():
            models.append({
                "model_id": model_id,
                "cache_path": cache_info.cache_path,
                "size_mb": cache_info.file_size / (1024**2),
                "file_count": cache_info.file_count,
                "download_time": cache_info.download_time,
                "last_accessed": cache_info.last_accessed,
                "access_count": cache_info.access_count,
                "is_valid": cache_info.is_valid,
                "load_time": cache_info.load_time
            })
        
        return sorted(models, key=lambda x: x["last_accessed"], reverse=True)


def main():
    """测试缓存管理器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型缓存管理器")
    parser.add_argument("--cache-dir", default="models", help="缓存目录")
    parser.add_argument("--list", action="store_true", help="列出缓存的模型")
    parser.add_argument("--stats", action="store_true", help="显示缓存统计")
    parser.add_argument("--verify", type=str, help="验证指定模型的缓存完整性")
    parser.add_argument("--remove", type=str, help="移除指定模型的缓存")
    parser.add_argument("--cleanup", action="store_true", help="清理过期缓存")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建缓存管理器
    cache_manager = ModelCacheManager(cache_dir=args.cache_dir)
    
    if args.list:
        models = cache_manager.list_cached_models()
        print(f"\n缓存的模型 ({len(models)} 个):")
        print("-" * 80)
        for model in models:
            print(f"模型ID: {model['model_id']}")
            print(f"  路径: {model['cache_path']}")
            print(f"  大小: {model['size_mb']:.1f} MB")
            print(f"  文件数: {model['file_count']}")
            print(f"  最后访问: {model['last_accessed']}")
            print(f"  访问次数: {model['access_count']}")
            print(f"  状态: {'有效' if model['is_valid'] else '无效'}")
            print()
    
    if args.stats:
        stats = cache_manager.get_cache_statistics()
        print(f"\n缓存统计:")
        print(f"  总模型数: {stats['total_models']}")
        print(f"  有效模型数: {stats['valid_models']}")
        print(f"  总大小: {stats['total_size_mb']:.1f} MB")
        print(f"  缓存目录: {stats['cache_directory']}")
        
        if stats['recent_models']:
            print(f"\n最近访问的模型:")
            for model in stats['recent_models']:
                print(f"  {model['model_id']} - {model['size_mb']:.1f}MB - 访问{model['access_count']}次")
    
    if args.verify:
        is_valid = cache_manager.verify_cache_integrity(args.verify)
        print(f"模型 {args.verify} 缓存完整性: {'有效' if is_valid else '无效'}")
    
    if args.remove:
        success = cache_manager.remove_cache(args.remove, remove_files=True)
        print(f"移除模型 {args.remove}: {'成功' if success else '失败'}")
    
    if args.cleanup:
        print("清理过期缓存...")
        cache_manager._cleanup_expired_cache()
        print("清理完成")


if __name__ == "__main__":
    main()