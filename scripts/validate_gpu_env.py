#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU环境验证与利用率压力测试脚本（Windows 11）
目的：
- 验证 torch/cuda/pynvml 状态
- 进行一次生成压力测试，采样GPU利用率，期望峰值>=80%
"""

import time, logging, sys
import torch
from pathlib import Path

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gpu_manager import GPUManager
from configs.rag_config import RAGConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.cuda.amp import autocast

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

def check_env():
    gm = GPUManager()
    info = gm.get_system_info()
    logging.info(f"环境信息: {info}")
    if not info.get('cuda_available'):
        logging.warning("CUDA不可用，无法进行GPU压力测试")
    return gm

def run_stress(gm: GPUManager, max_new_tokens: int = 768):
    cfg = RAGConfig()
    model_path = cfg.LLM_MODEL_PATH
    if not model_path or not Path(model_path).exists():
        logging.warning(f"模型路径不存在: {model_path}，使用占位小模型进行测试")
        model_path = 'gpt2'
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    prompt = "你是一个专业助手，请详细介绍Transformer的工作原理，并给出示例。" * 8
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=2048, padding=True).to(device)

    util_samples = []
    start = time.time()
    with torch.no_grad():
        if device == 'cuda':
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
        else:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=True)

        # 采样期间每0.5s读取一次利用率
        for _ in range(10):
            mem = gm.monitor_memory_usage(0)
            util_samples.append(mem.get('utilization_percent', 0.0))
            time.sleep(0.5)

    dur = time.time()-start
    peak = max(util_samples) if util_samples else 0
    avg = sum(util_samples)/len(util_samples) if util_samples else 0
    logging.info(f"生成耗时: {dur:.2f}s | GPU峰值利用率: {peak:.1f}% | 平均: {avg:.1f}%")
    return {'duration': dur, 'peak_util': peak, 'avg_util': avg}

def main():
    gm = check_env()
    res = run_stress(gm)
    if res['peak_util'] < 80:
        logging.warning("GPU峰值利用率未达到80%，建议增大 max_new_tokens 或并发负载进行压力测试")
    else:
        logging.info("GPU峰值利用率达到目标")

if __name__ == '__main__':
    main()
