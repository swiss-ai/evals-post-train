# Model Registry

Reference of models for evaluation with `launch_evaluations.sh`.

```bash
# Usage
bash scripts/lm_eval_harness/launch_evaluations.sh <mode> --model <path> [options]
```

## Apertus

> Requires: `--tokenizer alehc/swissai-tokenizer --bos`
> Aligned variants also need: `--chat-template`

| Name | Path | Type |
|------|------|------|
| Apertus8B-tokens15T-it2627139 | `/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens15T-it2627139` | base |
| Apertus8B-tokens15T-longcontext64k | `/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens15T-longcontext64k` | base |
| Apertus-8B-sft-mixture-8e-aligned | `/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus-8B-sft-mixture-8e-aligned` | aligned |
| Apertus-70B-sft-mixture-8e-aligned | `/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus-70B-sft-mixture-8e-aligned` | aligned |

## Meta Llama

| Name | Path | Type |
|------|------|------|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | base |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | instruct |
| Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` | instruct |

## OLMo (AllenAI)

| Name | Path | Type |
|------|------|------|
| OLMo-2-1124-7B | `allenai/OLMo-2-1124-7B` | base |
| OLMo-2-1124-7B-Instruct | `allenai/OLMo-2-1124-7B-Instruct` | instruct |
| OLMo-2-1124-13B | `allenai/OLMo-2-1124-13B` | base |
| OLMo-2-1124-13B-Instruct | `allenai/OLMo-2-1124-13B-Instruct` | instruct |
| OLMo-2-0325-32B | `allenai/OLMo-2-0325-32B` | base |
| OLMo-2-0325-32B-Instruct | `allenai/OLMo-2-0325-32B-Instruct` | instruct |
| OLMo-3-7B-Instruct | `allenai/Olmo-3-7B-Instruct` | instruct |

## Qwen

| Name | Path | Type |
|------|------|------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | base |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | instruct |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | instruct |
| Qwen2.5-32B-Instruct | `Qwen/Qwen2.5-32B-Instruct` | instruct |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` | instruct |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | base |
| Qwen3-4B | `Qwen/Qwen3-4B` | base |
| Qwen3-8B | `Qwen/Qwen3-8B` | base |
| Qwen3-14B | `Qwen/Qwen3-14B` | base |
| Qwen3-32B | `Qwen/Qwen3-32B` | base |

## Google Gemma

| Name | Path | Type |
|------|------|------|
| gemma-3-4b-it | `google/gemma-3-4b-it` | instruct |
| gemma-3-12b-it | `google/gemma-3-12b-it` | instruct |
| gemma-3-27b-it | `google/gemma-3-27b-it` | instruct |

## EuroLLM

| Name | Path | Type |
|------|------|------|
| EuroLLM-1.7B | `utter-project/EuroLLM-1.7B` | base |
| EuroLLM-1.7B-Instruct | `utter-project/EuroLLM-1.7B-Instruct` | instruct |
| EuroLLM-9B | `utter-project/EuroLLM-9B` | base |
| EuroLLM-9B-Instruct | `utter-project/EuroLLM-9B-Instruct` | instruct |
| EuroLLM-22B-Preview | `utter-project/EuroLLM-22B-Preview` | base |
| EuroLLM-22B-Instruct-Preview | `utter-project/EuroLLM-22B-Instruct-Preview` | instruct |

## Mistral

| Name | Path | Type |
|------|------|------|
| Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` | instruct |

## SmolLM

| Name | Path | Type |
|------|------|------|
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B` | base |

## Marin

| Name | Path | Type |
|------|------|------|
| marin-8b-base | `marin-community/marin-8b-base` | base |
| marin-8b-instruct | `marin-community/marin-8b-instruct` | instruct |

## Other

| Name | Path | Type |
|------|------|------|
| salamandra-7b-instruct | `BSC-LT/salamandra-7b-instruct` | instruct |
| jais-13b-chat | `inceptionai/jais-13b-chat` | chat |
| jais-adapted-7b-chat | `inceptionai/jais-adapted-7b-chat` | chat |
| jais-adapted-13b-chat | `inceptionai/jais-adapted-13b-chat` | chat |
| Poro-34B-chat | `LumiOpen/Poro-34B-chat` | chat |
| Viking-7B | `LumiOpen/Viking-7B` | base |
| Viking-13B | `LumiOpen/Viking-13B` | base |
| Viking-33B | `LumiOpen/Viking-33B` | base |
