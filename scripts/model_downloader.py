"""
模型下载器模块
支持断点续传、完整性验证和多线程下载
"""

import os
import hashlib
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import time
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError


class ModelDownloader:
    """模型下载器类，支持断点续传和完整性验证"""
    
    def __init__(self, cache_dir: str = None, max_retries: int = 3):
        """
        初始化模型下载器
        
        Args:
            cache_dir: 模型缓存目录
            max_retries: 最大重试次数
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models")
        self.max_retries = max_retries
        self.api = HfApi()
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ModelDownloader")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 创建文件处理器
            log_file = os.path.join(self.cache_dir, "download.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 设置格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型信息字典
        """
        try:
            model_info = self.api.model_info(model_id)
            return {
                "model_id": model_id,
                "sha": model_info.sha,
                "files": [f.rfilename for f in model_info.siblings],
                "size": sum(f.size for f in model_info.siblings if f.size),
                "last_modified": model_info.lastModified
            }
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """
        计算文件哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法
            
        Returns:
            文件哈希值
        """
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _verify_file_integrity(self, file_path: str, expected_hash: str = None) -> bool:
        """
        验证文件完整性
        
        Args:
            file_path: 文件路径
            expected_hash: 期望的哈希值
            
        Returns:
            验证结果
        """
        if not os.path.exists(file_path):
            return False
            
        if expected_hash:
            actual_hash = self._calculate_file_hash(file_path)
            return actual_hash == expected_hash
            
        # 如果没有提供哈希值，只检查文件是否存在且大小大于0
        return os.path.getsize(file_path) > 0
    
    def _download_file_with_resume(self, url: str, local_path: str, 
                                 progress_callback: Callable = None) -> bool:
        """
        支持断点续传的文件下载
        
        Args:
            url: 下载URL
            local_path: 本地保存路径
            progress_callback: 进度回调函数
            
        Returns:
            下载是否成功
        """
        headers = {}
        initial_pos = 0
        
        # 检查是否存在部分下载的文件
        if os.path.exists(local_path):
            initial_pos = os.path.getsize(local_path)
            headers['Range'] = f'bytes={initial_pos}-'
        
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # 检查是否支持断点续传
            if initial_pos > 0 and response.status_code not in [206, 200]:
                self.logger.warning("服务器不支持断点续传，重新下载")
                initial_pos = 0
                response = requests.get(url, stream=True, timeout=30)
            
            total_size = int(response.headers.get('content-length', 0)) + initial_pos
            
            # 创建目录
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 写入文件
            mode = 'ab' if initial_pos > 0 else 'wb'
            with open(local_path, mode) as f:
                downloaded = initial_pos
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded, total_size)
            
            return True
            
        except Exception as e:
            self.logger.error(f"下载文件失败: {e}")
            return False
    
    def download_model(self, model_id: str, local_dir: str = None, 
                      force_download: bool = False) -> str:
        """
        下载模型到本地
        
        Args:
            model_id: 模型ID
            local_dir: 本地目录
            force_download: 是否强制重新下载
            
        Returns:
            本地模型路径
        """
        if local_dir is None:
            local_dir = os.path.join(self.cache_dir, model_id.replace("/", "_"))
        
        self.logger.info(f"开始下载模型: {model_id}")
        self.logger.info(f"目标目录: {local_dir}")
        
        # 检查本地是否已存在
        if not force_download and self.is_model_cached(local_dir):
            self.logger.info("模型已存在于本地，跳过下载")
            return local_dir
        
        # 创建本地目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 使用huggingface_hub进行下载，支持断点续传
        try:
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"下载尝试 {attempt + 1}/{self.max_retries}")
                    
                    # 使用snapshot_download下载整个模型
                    downloaded_path = snapshot_download(
                        repo_id=model_id,
                        cache_dir=self.cache_dir,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,  # 不使用符号链接
                        resume_download=True,  # 支持断点续传
                        max_workers=4  # 多线程下载
                    )
                    
                    self.logger.info(f"模型下载完成: {downloaded_path}")
                    
                    # 验证下载完整性
                    if self._verify_model_integrity(local_dir):
                        self.logger.info("模型完整性验证通过")
                        return local_dir
                    else:
                        self.logger.warning("模型完整性验证失败，重试下载")
                        continue
                        
                except HfHubHTTPError as e:
                    self.logger.error(f"下载失败 (尝试 {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # 指数退避
                    
        except Exception as e:
            self.logger.error(f"模型下载失败: {e}")
            raise
    
    def _verify_model_integrity(self, model_dir: str) -> bool:
        """
        验证模型完整性
        
        Args:
            model_dir: 模型目录
            
        Returns:
            验证结果
        """
        model_path = Path(model_dir)
        
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
        
        # 检查必需文件
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                self.logger.error(f"缺少必需文件: {file_name}")
                return False
        
        # 检查模型权重文件
        weight_files = [
            "pytorch_model.bin", "model.safetensors", 
            "pytorch_model-00001-of-*.bin"
        ]
        
        has_weights = False
        for pattern in weight_files:
            if "*" in pattern:
                # 处理分片模型
                import glob
                matches = glob.glob(str(model_path / pattern))
                if matches:
                    has_weights = True
                    break
            else:
                if (model_path / pattern).exists():
                    has_weights = True
                    break
        
        if not has_weights:
            self.logger.error("未找到模型权重文件")
            return False
        
        self.logger.info("模型完整性验证通过")
        return True
    
    def is_model_cached(self, model_path: str) -> bool:
        """
        检查模型是否已缓存
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否已缓存
        """
        if not os.path.exists(model_path):
            return False
            
        return self._verify_model_integrity(model_path)
    
    def get_model_size(self, model_dir: str) -> int:
        """
        获取模型大小
        
        Args:
            model_dir: 模型目录
            
        Returns:
            模型大小（字节）
        """
        total_size = 0
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    
    def cleanup_cache(self, keep_latest: int = 1):
        """
        清理缓存，保留最新的几个版本
        
        Args:
            keep_latest: 保留的最新版本数
        """
        self.logger.info(f"开始清理缓存，保留最新 {keep_latest} 个版本")
        
        # 获取所有模型目录
        model_dirs = []
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if os.path.isdir(item_path):
                model_dirs.append((item_path, os.path.getmtime(item_path)))
        
        # 按修改时间排序
        model_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # 删除旧版本
        for i, (model_dir, _) in enumerate(model_dirs):
            if i >= keep_latest:
                self.logger.info(f"删除旧版本: {model_dir}")
                import shutil
                shutil.rmtree(model_dir)


def download_lfm2_model(cache_dir: str = None, force_download: bool = False) -> str:
    """
    便捷函数：下载LFM2模型
    
    Args:
        cache_dir: 缓存目录
        force_download: 是否强制重新下载
        
    Returns:
        本地模型路径
    """
    downloader = ModelDownloader(cache_dir)
    return downloader.download_model("LiquidAI/LFM2-1.2B", force_download=force_download)


if __name__ == "__main__":
    # 测试下载功能
    import argparse
    
    parser = argparse.ArgumentParser(description="模型下载器测试")
    parser.add_argument("--model_id", default="LiquidAI/LFM2-1.2B", help="模型ID")
    parser.add_argument("--cache_dir", default="./models", help="缓存目录")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.cache_dir)
    model_path = downloader.download_model(args.model_id, force_download=args.force)
    print(f"模型下载完成: {model_path}")