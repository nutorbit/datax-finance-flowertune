"""datax-flower-competition: A Flower / FlowerTune app."""

import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from unsloth import FastLanguageModel

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.
    """
    
    model, _ = FastLanguageModel.from_pretrained(
        model_name = model_cfg.name,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_cfg.lora.peft_lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = model_cfg.lora.peft_lora_alpha,
        lora_dropout = 0.075,
        use_gradient_checkpointing = model_cfg.gradient_checkpointing,
        random_state = 3407,
    )

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    return model


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

