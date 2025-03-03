# +
import os
import torch
import argparse

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    Trainer,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

os.environ['HF_HOME'] = '../../gemma/'
# -

parser = argparse.ArgumentParser()
parser.add_argument('--num_slices', 
                    type=int,
                    default=5,)
parser.add_argument('--num_loras', 
                    type=int,
                    default=4,)
parser.add_argument('--shard_id', 
                    type=int,
                    default=0,)
parser.add_argument('--rank', 
                    type=int,
                    default=32,)
parser.add_argument('--model', 
                    type=str,
                    default="llama-13b",)
parser.add_argument('--lr', 
                    type=float,
                    default=2e-5,)
args = parser.parse_args()

if args.model == 'llama-7b':
    base_model_name = "meta-llama/Llama-2-7b-hf"
    num_layers=32
elif args.model == 'llama-13b':
    base_model_name = "meta-llama/Llama-2-13b-hf"
    num_layers=40
elif args.model == 'llama-3-8b':
    num_layers=32
    base_model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

start = 0
end = int((args.shard_id + 1) * 100/args.num_slices)
train_dataset = load_dataset("tatsu-lab/alpaca", 
                             split=f"train[{start}%:{end}%]")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {data_point["instruction"]}

        ### Input:
        {data_point["input"]}

        ### Response:
        {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {data_point["instruction"]}

        ### Response:
        {data_point["output"]}"""



train_dataset = train_dataset.map(lambda data_point: {"prompt": (generate_prompt(data_point))})



lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=2*args.rank,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        # "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)

def get_subdirectory_name(dir_path):
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith("checkpoint")]
    return subdirectories[0]

def get_layers(idx, num_loras):
    layers = []
    for i in range(num_loras):
        layer_id = str(num_layers-1 - (idx*num_loras + i))
        layers.append(f"layers.{layer_id}")
    return layers

def check_if(name, layer_ids):
    for idx in layer_ids:
        if idx in name:
            return True
    return False

def model_init(idx, model_dir, num_loras=2):
  layer_ids = get_layers(idx, num_loras)
  print(layer_ids)

  if idx > 0:
    subpath = get_subdirectory_name(model_dir)
    print(f"Loading model from {model_dir}/{subpath}")
    model = AutoModelForCausalLM.from_pretrained(
        f'{model_dir}/{subpath}', 
        load_in_8bit=True,
        device_map="auto",
    )
  else:
    print("Initializing model!")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
    )

  model.config.use_cache = False
  model.config.pretraining_tp = 1
  model = prepare_model_for_int8_training(model)

  peft_model = get_peft_model(model, lora_config)
  for name, param in peft_model.named_parameters():
      if param.requires_grad:
          param.requires_grad = check_if(name, layer_ids)
  return peft_model

suff = args.model.split("-")[-1]
output_dir = f"./alpaca_lora_models_{suff}/models_loras_{args.num_loras}_rank_{args.rank}/"
model_dir = os.path.join(output_dir,
                         f'model_shard_{str(args.shard_id-1)}')
model = model_init(idx=args.shard_id, 
                   model_dir=model_dir, 
                   num_loras=args.num_loras)

# Training Params
train_params = TrainingArguments(
    output_dir=os.path.join(output_dir, f"model_shard_{args.shard_id}"),
    num_train_epochs=3,
    auto_find_batch_size=True,
#     per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,    
    save_total_limit=3,
    logging_steps=25,
    learning_rate=args.lr,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=train_params,
    dataset_text_field="prompt",
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

