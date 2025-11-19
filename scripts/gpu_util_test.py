#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, logging, sys, torch
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gpu_manager import GPUManager
from configs.rag_config import RAGConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

def main():
    gm = GPUManager()
    info = gm.get_system_info()
    logging.info(f"环境信息: {info}")
    if not torch.cuda.is_available():
        logging.warning("CUDA不可用")
        return
    cfg = RAGConfig()
    path = cfg.LLM_MODEL_PATH if cfg.LLM_MODEL_PATH and Path(cfg.LLM_MODEL_PATH).exists() else 'gpt2'
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dtype = torch.bfloat16
    conf = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, config=conf, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    model.to('cuda')
    model.eval()
    prompt = "请详细介绍Transformer原理及典型应用。" * 10
    base = tok(prompt, return_tensors='pt', truncation=True, max_length=2048, padding=True).to('cuda')
    batch = 12
    inputs = {
        'input_ids': base['input_ids'].repeat(batch, 1),
        'attention_mask': base.get('attention_mask').repeat(batch, 1)
    }
    start = time.time()
    util_samples = []
    import threading
    stop_flag = False
    def sampler():
        while not stop_flag:
            m = gm.monitor_memory_usage(0)
            util = m.get('utilization_percent', 0.0) or m.get('usage_percent', 0.0)
            util_samples.append(util)
            time.sleep(0.5)
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
    stop_flag = True
    t.join(timeout=1)
    dur = time.time() - start
    peak = max(util_samples) if util_samples else 0
    avg = sum(util_samples)/len(util_samples) if util_samples else 0
    logging.info(f"生成耗时: {dur:.2f}s | 峰值利用率: {peak:.1f}% | 平均: {avg:.1f}%")

if __name__ == '__main__':
    main()