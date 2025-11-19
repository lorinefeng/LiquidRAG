#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests, json, time

def main():
    url = 'http://127.0.0.1:8000/api/v1/ask'
    payload = {"query":"请简述Transformer的核心机制与注意力的作用。","return_sources": True}
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=180)
    t1 = time.time()
    print('status:', r.status_code)
    data = r.json()
    print('total_time:', data.get('total_time'))
    print('answer_len:', len(data.get('answer','')))
    metrics = data.get('gpu_metrics', [])
    print('gpu_metrics_samples:', len(metrics))
    if metrics:
        print('first:', metrics[0])
        print('last:', metrics[-1])
    print('elapsed:', t1-t0)

if __name__ == '__main__':
    main()