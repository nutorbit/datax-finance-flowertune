import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from unsloth import FastLanguageModel

from benchmarks import infer_fiqa, infer_fpb, infer_tfns

# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-model-name-path", type=str, default="mistralai/Mistral-7B-v0.3"
)
parser.add_argument("--run-name", type=str, default="fl")
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument("--datasets", type=str, default="fpb")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--quantization", type=int, default=4)
args = parser.parse_args()


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.peft_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


# Evaluate
model = model.eval()
with torch.no_grad():
    for dataset in args.datasets.split(","):
        if dataset == "fpb":
            infer_fpb(model, tokenizer, args.batch_size, args.run_name)
        elif dataset == "fiqa":
            infer_fiqa(model, tokenizer, args.batch_size, args.run_name)
        elif dataset == "tfns":
            infer_tfns(model, tokenizer, args.batch_size, args.run_name)
        else:
            raise ValueError("Undefined Dataset.")
