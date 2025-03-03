# +
import os
import json
import argparse
import evaluate
import pandas as pd
import numpy as np

os.environ['HF_HOME']="../.cache/"


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)


from datasets import load_dataset
from datasets import concatenate_datasets

from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model


# -

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', 
                    type=str,
                    default='glue',)
parser.add_argument('--dataset', 
                    type=str,
                    default='wic',)
parser.add_argument('--model',
                    type=str,
                   default='roberta-large')
parser.add_argument('--lr',
                    type=float,
                   default=1e-4)
parser.add_argument('--run_id',
                    type=int,
                   default=5)
parser.add_argument('--num_slices',
                    type=int,
                   default=7)
parser.add_argument('--rank',
                    type=int,
                   default=16)
parser.add_argument('--epochs',
                    type=int,
                   default=30)
parser.add_argument('--save_strategy', 
                    type=str,
                    default='steps',)
parser.add_argument('--eval_strategy', 
                    type=str,
                    default='steps',)
parser.add_argument('--save_steps', 
                    type=int,
                    default=1000,)
parser.add_argument('--total_loras', 
                    type=int,
                    default=15,)
parser.add_argument('--stop_epochs', 
                    type=int,
                    default=30,)
parser.add_argument('--shard_id', 
                    type=int,
                    default=0,)
args = parser.parse_args()


def dump_json(content, filename):
    with open(filename, "w") as f:
        json.dump(content, f)


checkpoint = args.model
dataset_name = args.dataset
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, cache_dir='/playpen-ssd/gemma/')

config = LoraConfig(
    r=args.rank,
    lora_alpha=2*args.rank,
    target_modules=["query", "value"],
    lora_dropout=0.01,
    bias="none",
    task_type="SEQ_CLS",
    use_rslora=True,
)


# +
def boolq_tokenize_function(example):
    return tokenizer(example["passage"], example["question"],
                     padding='max_length',
                     truncation=True)

def rte_tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], 
                     truncation=True)

def generic_tokenize_function(example):
    return tokenizer(example["input"], truncation=True)

def sst2_tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

def qqp_tokenize_function(example):
    return tokenizer(example["question1"], example["question2"], 
                     truncation=True)

def stsb_tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], 
                     truncation=True)

def qnli_tokenize_function(example):
    return tokenizer(example["question"], example["sentence"], 
                     truncation=True)

def multirc_processor(example):
    ex = {}
    ex['input'] = example['paragraph'] + f" {tokenizer.sep_token} Question: "\
    + example['question']  + f" {tokenizer.sep_token} Answer: " + example['answer']
    ex['label'] = int(example['label'])
    return ex

def wic_processor(example):
    ex = {}
    ex['input'] = example["word"] + f": {tokenizer.sep_token} " + example["sentence1"]\
    + f" {tokenizer.sep_token} " + example["sentence2"]
    ex['label'] = int(example['label'])
    return ex

def wsc_processor(example):
    ex = {}
    pronoun, entity = example["span2_text"], example["span1_text"]
    qs = f"Question: In the passage above, does the pronoun '{pronoun}' refer to {entity}?"
    ex['input'] = example['text'] + f" {tokenizer.sep_token} " + qs
    ex['label'] = int(example['label'])
    return ex


def copa_processor(example):
    ex1, ex2 = {}, {}
    ex1['input'] = example['premise'] + f" {tokenizer.sep_token} " + example['choice1']
    ex1['label'] = int(example['label'] == 0)
    yield DatasetDict(ex1)
    
    ex2['input'] = example['premise'] + f"{tokenizer.sep_token} " + example['choice2']
    ex2['label'] = int(example['label'] == 1)
    yield DatasetDict(ex2)

    

def get_copa(data):
    dataset = []
    for x in data:
        dataset.extend(copa_processor(x))
    return dataset

def get_generic_dataset(data, processor):
    dataset = []
    for x in data:
        dataset.append(processor(x))
    return dataset

# +
tokenizer_func = {
      'boolq': boolq_tokenize_function,
      'copa': generic_tokenize_function,
      'rte': rte_tokenize_function,
      'wic': generic_tokenize_function,
      'cb': rte_tokenize_function,
      'multirc': generic_tokenize_function,
      'wsc': generic_tokenize_function,
      'sst2': sst2_tokenize_function,
      'cola': sst2_tokenize_function,
      'qqp': qqp_tokenize_function,
      'stsb': stsb_tokenize_function,
      'qnli': qnli_tokenize_function,
      'mrpc': stsb_tokenize_function,
      'mnli': rte_tokenize_function,
}

processor_func = {
      'copa': copa_processor,
      'wic': wic_processor,
      'multirc': multirc_processor,
      'wsc': wsc_processor,
}

metric_dict = {
    'cola': 'eval_matthews_correlation',
    'stsb': 'eval_pearson',
}

label_count = {
      'boolq': 2,
      'copa': 2,
      'rte': 2,
      'wic': 2,
      'cb': 3,
      'multirc': 2,
      'wsc': 2,
      'sst2': 2,
      'cola': 2,
      'qqp': 2,
      'qnli': 2,
      'mrpc': 2,
      'mnli': 3,
      'stsb': 1,
}




label_pad_token_id = -100
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# +
metric = evaluate.load(args.benchmark,
                       'cb' if dataset_name == 'multirc' else dataset_name)

def compute_metrics(eval_preds, task_name=dataset_name):
    logits, labels = eval_preds
    if args.dataset == 'stsb':
        predictions = np.array([l[0] for l in logits])
    else:
        predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


# +


metric_name = 'eval_accuracy' 
if dataset_name in ['cola', 'stsb']:
    metric_name = metric_dict[dataset_name]




# -

def get_subdirectory_name(dir_path):
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith("checkpoint")]
    return subdirectories[0]


def get_layers(idx, num_loras):
    layers = ['classifier']
    for i in range(num_loras):
        layer_id = str(23 - (idx*num_loras + i))
        layers.append(f"layer.{layer_id}")
    return layers


def check_if(name, layer_ids):
    for idx in layer_ids:
        if idx in name:
            return True
    return False


def model_init(idx, model_dir, num_loras=4):
    layer_ids = get_layers(idx, num_loras)
    print(layer_ids)
    num_labels = label_count[args.dataset]


    if idx > 0:
        subpath = get_subdirectory_name(model_dir)
        print(subpath)
        model = AutoModelForSequenceClassification.from_pretrained(
            f'{model_dir}/{subpath}', num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, 
            num_labels=num_labels,
            cache_dir='/playpen-ssd/gemma/',
        )
    peft_model = get_peft_model(model, config)
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.requires_grad = check_if(name, layer_ids)
    return peft_model



val_set_name = "validation_matched" if args.dataset == 'mnli' else "validation"
train_set, valid_set = load_dataset(
    args.benchmark, dataset_name, 
    split=[f"train", val_set_name])


save_set = train_set.shuffle().select(list(range(100)))
concatenate_datasets([valid_set, save_set])


num_loras = args.total_loras // args.num_slices

print(f"Number of shards: {args.num_slices}, Number of LoRAs: {num_loras}")
filename = f"results_{dataset_name}_{str(args.shard_id)}/results_nshards_{args.num_slices}_nloras_{num_loras}.json"
os.makedirs(f"results_{dataset_name}_{str(args.shard_id)}", exist_ok = True) 

exemplars = None
results = []

start = 0
end = int((args.shard_id + 1) * 100/ args.num_slices)

val_set_name = "validation_matched" if args.dataset == 'mnli' else "validation"
train_set, valid_set = load_dataset(
    args.benchmark, dataset_name, 
    split=[f"train[{start}%:{end}%]", val_set_name])


if dataset_name in ['copa']:
    train_set = Dataset.from_pandas(pd.DataFrame(data=get_copa(train_set)))
    valid_set = Dataset.from_pandas(pd.DataFrame(data=get_copa(valid_set)))
elif dataset_name in ['multirc', 'wsc', 'wic']:
    processor_f = processor_func[dataset_name]
    train_set = Dataset.from_pandas(pd.DataFrame(
        data=get_generic_dataset(train_set, processor_f)))
    valid_set = Dataset.from_pandas(pd.DataFrame(
        data=get_generic_dataset(valid_set, processor_f)))

tokenized_train = train_set.map(tokenizer_func[dataset_name], batched=True)
tokenized_valid = valid_set.map(tokenizer_func[dataset_name], batched=True)

output_dir = f'outputs_ilora_{args.dataset}_{str(args.shard_id)}'
model_dir = f'outputs_ilora_{args.dataset}_{str(args.shard_id-1)}'

model = model_init(args.shard_id, model_dir, num_loras)


training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate= args.lr, # higher learning rate
    num_train_epochs=args.epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy=args.save_strategy,
    evaluation_strategy=args.eval_strategy,
    save_steps=args.save_steps,
    eval_steps=args.save_steps,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(args.stop_epochs, 0.0)] if args.shard_id > 0 else [],
)

logs = (trainer.train(resume_from_checkpoint=False))

eval_acc = max([x[metric_name] for x in trainer.state.log_history if metric_name in x.keys()])
print(f"Shard ID: {args.shard_id}, {metric_name}: {eval_acc}")
results.append((args.shard_id, eval_acc))

# Dump the results
dump_json(results, filename)
