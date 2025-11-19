---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
pipeline_tag: text-generation
tags:
- liquid
- lfm2
- edge
---

<center>
<div style="text-align: center;">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/7_6D7rWrLxp2hb6OHSV1p.png" 
    alt="Liquid AI"
    style="width: 100%; max-width: 66%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
</div>
<div style="display: flex; justify-content: center;">
<a href="https://playground.liquid.ai/chat">
<svg width="114.8" height="20" viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Playground" style="margin-bottom: 1em;">
  <title>Playground</title>
  <g>
    <rect fill="#fff" width="200" height="200"></rect>
    <rect fill="url(#x)" x="200" width="800" height="200"></rect>
  </g>
  <g transform="translate(35, 30) scale(0.45, 0.45)">
    <path d="M172.314 129.313L172.219 129.367L206.125 188.18C210.671 195.154 213.324 203.457 213.324 212.382C213.324 220.834 210.956 228.739 206.839 235.479L275.924 213.178L167.853 33.6L141.827 76.9614L172.314 129.313Z" fill="black"/>
    <path d="M114.217 302.4L168.492 257.003C168.447 257.003 168.397 257.003 168.352 257.003C143.515 257.003 123.385 237.027 123.385 212.387C123.385 203.487 126.023 195.204 130.55 188.24L162.621 132.503L135.966 86.7327L60.0762 213.183L114.127 302.4H114.217Z" fill="black"/>
    <path d="M191.435 250.681C191.435 250.681 191.43 250.681 191.425 250.686L129.71 302.4H221.294L267.71 226.593L191.435 250.686V250.681Z" fill="black"/>
  </g>
  <g transform="translate(50, 0)" aria-hidden="true" fill="#fff" text-anchor="start" font-family="Verdana,DejaVu Sans,sans-serif" font-size="110">
    <text x="255" y="148" textLength="619" fill="#000" opacity="0.1">Playground</text>
    <text x="245" y="138" textLength="619">Playground</text>
  </g>
  <linearGradient id="x" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" style="stop-color:#000000"></stop>
    <stop offset="100%" style="stop-color:#000000"></stop>
  </linearGradient>
</svg>
</a>
<a href="https://leap.liquid.ai/?utm_source=huggingface&utm_medium=modelcards">
<svg width="114.8" height="20" viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Leap" style="margin-bottom: 1em;">
  <title>Leap</title>
  <g>
    <rect fill="#000" width="500" height="200"></rect>
  </g>
  <g transform="translate(100, 45) scale(3.5, 3.5)" fill="#fff">
    <path d="M13.8512 28.0769C12.5435 28.0769 11.4025 27.8205 10.4281 27.3077C9.45375 26.7692 8.68452 26.0128 8.12042 25.0385C7.58196 24.0641 7.31273 22.9359 7.31273 21.6538V3.76923H0.389648V0H11.4666V21.6538C11.4666 22.4744 11.6973 23.1282 12.1589 23.6154C12.6204 24.0769 13.2486 24.3077 14.0435 24.3077H20.582V28.0769H13.8512Z"/>
    <path d="M29.6439 28.4615C27.9259 28.4615 26.4131 28.1282 25.1054 27.4615C23.8233 26.7692 22.8362 25.8077 22.1439 24.5769C21.4516 23.3462 21.1054 21.9103 21.1054 20.2692V14.7308C21.1054 13.0641 21.4516 11.6282 22.1439 10.4231C22.8362 9.19231 23.8233 8.24359 25.1054 7.57692C26.4131 6.88462 27.9259 6.53846 29.6439 6.53846C31.3875 6.53846 32.9003 6.88462 34.1823 7.57692C35.4644 8.24359 36.4516 9.19231 37.1439 10.4231C37.8362 11.6282 38.1823 13.0641 38.1823 14.7308V18.5H25.1054V20.2692C25.1054 21.8333 25.49 23.0256 26.2592 23.8462C27.0541 24.6667 28.1951 25.0769 29.6823 25.0769C30.8875 25.0769 31.8618 24.8718 32.6054 24.4615C33.349 24.0256 33.8105 23.3974 33.99 22.5769H38.1054C37.7977 24.3718 36.8746 25.8077 35.3362 26.8846C33.7977 27.9359 31.9003 28.4615 29.6439 28.4615ZM34.1823 16V14.6923C34.1823 13.1538 33.7977 11.9615 33.0285 11.1154C32.2592 10.2692 31.131 9.84615 29.6439 9.84615C28.1823 9.84615 27.0541 10.2692 26.2592 11.1154C25.49 11.9615 25.1054 13.1667 25.1054 14.7308V15.6923L34.49 15.6538L34.1823 16Z"/>
    <path d="M46.3596 28.4615C44.1545 28.4615 42.4109 27.8974 41.1288 26.7692C39.8724 25.6154 39.2442 24.0513 39.2442 22.0769C39.2442 20.0769 39.9109 18.5128 41.2442 17.3846C42.6032 16.2308 44.4622 15.6538 46.8211 15.6538H52.7058V13.6923C52.7058 12.5385 52.3468 11.641 51.6288 11C50.9109 10.359 49.8981 10.0385 48.5904 10.0385C47.4365 10.0385 46.475 10.2949 45.7058 10.8077C44.9365 11.2949 44.4878 11.9487 44.3596 12.7692H40.2827C40.5135 10.8718 41.3852 9.35897 42.8981 8.23077C44.4365 7.10256 46.3724 6.53846 48.7058 6.53846C51.2186 6.53846 53.2058 7.17949 54.6673 8.46154C56.1288 9.71795 56.8596 11.4359 56.8596 13.6154V28.0769H52.8211V24.1923H52.1288L52.8211 23.4231C52.8211 24.9615 52.2314 26.1923 51.0519 27.1154C49.8724 28.0128 48.3083 28.4615 46.3596 28.4615ZM47.5904 25.2692C49.0776 25.2692 50.2955 24.8974 51.2442 24.1538C52.2186 23.3846 52.7058 22.4103 52.7058 21.2308V18.4615H46.8981C45.8211 18.4615 44.9622 18.7564 44.3211 19.3462C43.7058 19.9359 43.3981 20.7436 43.3981 21.7692C43.3981 22.8462 43.7699 23.7051 44.5135 24.3462C45.257 24.9615 46.2827 25.2692 47.5904 25.2692Z"/>
    <path d="M58.9984 35V6.92308H63.1138V10.9615H63.9984L63.1138 11.9231C63.1138 10.2564 63.6266 8.94872 64.6523 8C65.7036 7.02564 67.101 6.53846 68.8446 6.53846C70.9728 6.53846 72.6651 7.25641 73.9215 8.69231C75.2036 10.1026 75.8446 12.0385 75.8446 14.5V20.4615C75.8446 22.1026 75.5497 23.5256 74.96 24.7308C74.3959 25.9103 73.5882 26.8333 72.5369 27.5C71.5113 28.141 70.2805 28.4615 68.8446 28.4615C67.1266 28.4615 65.742 27.9872 64.6907 27.0385C63.6395 26.0641 63.1138 24.7436 63.1138 23.0769L63.9984 24.0385H63.0369L63.1523 28.9615V35H58.9984ZM67.4215 24.8462C68.7805 24.8462 69.8318 24.4615 70.5754 23.6923C71.3446 22.8974 71.7292 21.7564 71.7292 20.2692V14.7308C71.7292 13.2436 71.3446 12.1154 70.5754 11.3462C69.8318 10.5513 68.7805 10.1538 67.4215 10.1538C66.1138 10.1538 65.0754 10.5641 64.3061 11.3846C63.5369 12.1795 63.1523 13.2949 63.1523 14.7308V20.2692C63.1523 21.7051 63.5369 22.8333 64.3061 23.6538C65.0754 24.4487 66.1138 24.8462 67.4215 24.8462Z"/>
  </g>
  <linearGradient id="y" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" style="stop-color:#000000"></stop>
  </linearGradient>
</svg>
</a>
</div>
</center>

# LFM2-1.2B

LFM2 is a new generation of hybrid models developed by [Liquid AI](https://www.liquid.ai/), specifically designed for edge AI and on-device deployment. It sets a new standard in terms of quality, speed, and memory efficiency. 

We're releasing the weights of four post-trained checkpoints with 350M, 700M, 1.2B, and 2.6 parameters. They provide the following key features to create AI-powered edge applications:

* **Fast training & inference** – LFM2 achieves 3x faster training compared to its previous generation. It also benefits from 2x faster decode and prefill speed on CPU compared to Qwen3.
* **Best performance** – LFM2 outperforms similarly-sized models across multiple benchmark categories, including knowledge, mathematics, instruction following, and multilingual capabilities.
* **New architecture** – LFM2 is a new hybrid Liquid model with multiplicative gates and short convolutions.
* **Flexible deployment** – LFM2 runs efficiently on CPU, GPU, and NPU hardware for flexible deployment on smartphones, laptops, or vehicles.

Find more information about LFM2 in our [blog post](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models).

## 📄 Model details

Due to their small size, **we recommend fine-tuning LFM2 models on narrow use cases** to maximize performance. 
They are particularly suited for agentic tasks, data extraction, RAG, creative writing, and multi-turn conversations. 
However, we do not recommend using them for tasks that are knowledge-intensive or require programming skills.

| Property            | [**LFM2-350M**](https://huggingface.co/LiquidAI/LFM2-350M) | [**LFM2-700M**](https://huggingface.co/LiquidAI/LFM2-700M) | [**LFM2-1.2B**](https://huggingface.co/LiquidAI/LFM2-1.2B) | [**LFM2-2.6B**](https://huggingface.co/LiquidAI/LFM2-2.6B) |
| ------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| **Parameters**      | 354,483,968                   | 742,489,344                   | 1,170,340,608                 | 2,569,272,320                 |
| **Layers**          | 16 (10 conv + 6 attn)         | 16 (10 conv + 6 attn)         | 16 (10 conv + 6 attn)         | 30 (22 conv + 8 attn)         |
| **Context length**  | 32,768 tokens                 | 32,768 tokens                 | 32,768 tokens                 | 32,768 tokens                 |
| **Vocabulary size** | 65,536                        | 65,536                        | 65,536                        | 65,536                        |
| **Precision**       | bfloat16                      | bfloat16                      | bfloat16                      | bfloat16                      |
| **Training budget** | 10 trillion tokens            | 10 trillion tokens            | 10 trillion tokens            | 10 trillion tokens            |
| **License**         | LFM Open License v1.0         | LFM Open License v1.0         | LFM Open License v1.0         | LFM Open License v1.0         |

**Supported languages**: English, Arabic, Chinese, French, German, Japanese, Korean, and Spanish.

**Generation parameters**: We recommend the following parameters:
* `temperature=0.3`
* `min_p=0.15`
* `repetition_penalty=1.05`

**Chat template**: LFM2 uses a ChatML-like chat template as follows:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
It's a tiny nematode that lives in temperate soil environments.<|im_end|>
```

You can automatically apply it using the dedicated [`.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_templating#applychattemplate) function from Hugging Face transformers.

**Tool use**: It consists of four main steps:
1. **Function definition**: LFM2 takes JSON function definitions as input (JSON objects between `<|tool_list_start|>` and `<|tool_list_end|>` special tokens), usually in the system prompt
2. **Function call**: LFM2 writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer.
3. **Function execution**: The function call is executed and the result is returned (string between `<|tool_response_start|>` and `<|tool_response_end|>` special tokens), as a "tool" role.
4. **Final answer**: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.

Here is a simple example of a conversation using tool use:

```
<|startoftext|><|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
<|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

**Architecture**: Hybrid model with multiplicative gates and short convolutions: 10 double-gated short-range LIV convolution blocks and 6 grouped query attention (GQA) blocks.

**Pre-training mixture**: Approximately 75% English, 20% multilingual, and 5% code data sourced from the web and licensed materials.

**Training approach**:
* Knowledge distillation using [LFM1-7B](https://www.liquid.ai/blog/introducing-lfm-7b-setting-new-standards-for-efficient-language-models) as teacher model
* Very large-scale SFT on 50% downstream tasks, 50% general domains
* Custom DPO with length normalization and semi-online datasets
* Iterative model merging

## 🏃 How to run LFM2

### 1. Transformers

To run LFM2, you need to install Hugging Face [`transformers`](https://github.com/huggingface/transformers) v4.55 or a more recent version as follows:

```bash
pip install -U transformers
```

Here is an example of how to generate an answer with transformers in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM2-1.2B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

# <|startoftext|><|im_start|>user
# What is C. elegans?<|im_end|>
# <|im_start|>assistant
# C. elegans, also known as Caenorhabditis elegans, is a small, free-living
# nematode worm (roundworm) that belongs to the phylum Nematoda.
```

You can directly run and test the model with this [Colab notebook](https://colab.research.google.com/drive/1_q3jQ6LtyiuPzFZv7Vw8xSfPU5FwkKZY?usp=sharing).

### 2. vLLM

You need to install [`vLLM`](https://github.com/vllm-project/vllm) v0.10.2 or a more recent version as follows:

```bash
uv pip install vllm==0.10.2 --extra-index-url https://wheels.vllm.ai/0.10.2/ --torch-backend=auto
```

Here is an example of how to use it for inference:

```python
from vllm import LLM, SamplingParams

prompts = [
    "What is C. elegans?",
    "Say hi in JSON format",
    "Define AI in Spanish"
]
sampling_params = SamplingParams(
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05
)

llm = LLM(model="LiquidAI/LFM2-1.2B")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 3. llama.cpp

You can run LFM2 with llama.cpp using its [GGUF checkpoint](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF). Find more information in the model card.

## 🔧 How to fine-tune LFM2

We recommend fine-tuning LFM2 models on your use cases to maximize performance.

| Notebook | Description | Link |
|-------|------|------|
| SFT (Unsloth) | Supervised Fine-Tuning (SFT) notebook with a LoRA adapter using Unsloth. | <a href="https://colab.research.google.com/drive/1HROdGaPFt1tATniBcos11-doVaH7kOI3?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT (Axolotl) | Supervised Fine-Tuning (SFT) notebook with a LoRA adapter using Axolotl. | <a href="https://colab.research.google.com/drive/155lr5-uYsOJmZfO6_QZPjbs8hA_v8S7t?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT (TRL) | Supervised Fine-Tuning (SFT) notebook with a LoRA adapter using TRL. | <a href="https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| DPO (TRL) | Preference alignment with Direct Preference Optimization (DPO) using TRL. | <a href="https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |

## 📈 Performance

LFM2 outperforms similar-sized models across different evaluation categories.

### 1. Automated benchmarks

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/3cB7VqMnrG9I8EqrL7k-q.png)

| Model | MMLU | GPQA | IFEval | IFBench | GSM8K | MGSM | MMMLU |
|-------|------|------|--------|---------|-------|------|-------|
| LFM2-350M | 43.43 | 27.46 | 65.12 | 16.41 | 30.1 | 29.52 | 37.99 |
| LFM2-700M | 49.9 | 28.48 | 72.23 | 20.56 | 46.4 | 45.36 | 43.28 |
| LFM2-1.2B | *55.23* | **31.47** | **74.89** | *20.7* | *58.3* | *55.04* | **46.73** |
| Qwen3-0.6B | 44.93 | 22.14 | 64.24 | 19.75 | 36.47 | 41.28 | 30.84 |
| Qwen3-1.7B | **59.11** | 27.72 | *73.98* | **21.27** | 51.4 | **66.56** | *46.51* |
| Llama-3.2-1B-Instruct | 46.6 | *28.84* | 52.39 | 16.86 | 35.71 | 29.12 | 38.15 |
| gemma-3-1b-it | 40.08 | 21.07 | 62.9 | 17.72 | **59.59** | 43.6 | 34.43 |

### 2. LLM-as-a-Judge

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/4Yxx0l9aQ6ATrps5GWHzv.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/lzpZOGwH-8bTlOWd3tv6M.png)

### 3. Inference

#### Throughput comparison on CPU in ExecuTorch

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/KoKcsXUOnkvz2dwZ99k08.png)

#### Throughput comparison on CPU in Llama.cpp

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/c7UYZ5nh6qJMB4rd6WKde.png)

## 📬 Contact

If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).