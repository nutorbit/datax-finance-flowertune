# FlowerTune LLM on Finance Dataset

This directory conducts federated instruction tuning with a pretrained [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) model on a [Finance dataset](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

and run the code using:

```bash
flwr run .
```

## Methodology

Our approach, we use [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B) as a pretrained LLM and fine-tune the model with 4bit quantization and [DoRA adapter](https://arxiv.org/abs/2402.09353) as Parameter-Efficient Fine-Tuning (PEFT) technique. Aggregation is performed using the **FedAvg** strategy.

Here is the checkpoint of the model: [link](https://drive.google.com/drive/folders/1ArOfC82H0E5GAqM4WxAAKlM7HcamHS61?usp=sharing), tested using `peft_10`.

Configs:
* Quantization: 4 bits
* LoRA-R: 8
* LoRA-Alpha: 32
* Training Batch Size Per Device: 8
* Number of Server Rounds: 10

## Accuracy

* FIQA: 60.85%
* FPB: 68.23%
* TFNS: 67.50%
* Average: 65.52%

You can check `/flowertune-eval-finance/benchmarks` folder to get raw results.