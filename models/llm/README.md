# LLM Models

Place your GGUF model files here.

## Recommended Model

**Qwen2.5-3B-Instruct** (quantized)

Download from: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

Recommended files (pick one):
- `qwen2.5-3b-instruct-q5_k_m.gguf` (~2.5GB) - Best quality/size balance
- `qwen2.5-3b-instruct-q4_k_m.gguf` (~2.0GB) - Faster, slightly lower quality

## Alternative Models

If you want to try different sizes:
- **Qwen2.5-1.5B-Instruct-GGUF** - Smaller, faster, less capable
- **Qwen2.5-7B-Instruct-GGUF** - Larger, slower, more capable

## Usage

After placing the model file here, test with:

```bash
python test_llm.py
```
