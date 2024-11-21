from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, OPTForCausalLM
import logging
import torch


def load_model(model_name, device, checkpoint=143000):
    """Get the HuggingFace model for the current model

    Args:
        model_name (str): The model+scheme used to determine the tokenizer and model
        checkpoint (str): The checkpoint to load
        device (torch.device (or) str): Pytorch device to load the model into
    """

    if model_name.startswith("EleutherAI"):
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=f"step{checkpoint}", torch_dtype=torch.float16)
    else:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    return model


def load_tokenizer(model_name):
    """Get the HuggingFace tokenizer for the current model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def entity_to_prompt(name):
    """Get the name of the current entity and return the prompt for the model"""
    return f"Question: Tell me a bio of {name}."


def setup_logger():
    logging.basicConfig(filename='generation_log.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        filemode='a'
    )