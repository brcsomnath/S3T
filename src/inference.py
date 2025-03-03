# +
import os
import json
import torch
import argparse
import random
import itertools
import numpy as np

from collections import defaultdict

import math
import networkx as nx

from copy import deepcopy

from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from transformers import (
    AutoModelForImageClassification, 
    TrainingArguments, 
    AutoImageProcessor,
    Trainer
)


os.environ['HF_HOME'] = '../../gemma/'
# -
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', 
                    type=int,
                    default=0,)
parser.add_argument('--device_id', 
                    type=int,
                    default=0,)
parser.add_argument('--method', 
                    type=str,
                    default="s3t",)
parser.add_argument('--data_key', 
                    type=str,
                    default="cifar10",)
parser.add_argument('--budget', 
                    type=int,
                    default=4,)
args = parser.parse_args()


def bms(num_slices=4):
    choices = list(range(num_slices))
    perms = [[x] for x in choices]
        
    for _ in range(num_slices):
        G = nx.Graph()
        edges = []
        
        
        stale_perms = deepcopy(perms)
        
        for p in stale_perms:
            for j in range(num_slices):
                if j not in p:
                    edges.append((p[-1], j+num_slices, 1))
        G.add_weighted_edges_from(edges)
        assignments = nx.max_weight_matching(G)
        
        
        
        for a in assignments:
            if a[0] >= num_slices:
                idx = [i for i, p in enumerate(stale_perms) if p[-1] == a[1]][0]
                perms[idx].append(a[0] - num_slices)
            else:
                idx = [i for i, p in enumerate(stale_perms) if p[-1] == a[0]][0]
                perms[idx].append(a[1] - num_slices)
    return perms


def get_random_perms(num_slices, budget):
    perms = []
    cache = []
    count = 0
    while count < min(budget, math.factorial(num_slices)):
        p = []
        nodes = list(range(num_slices))
        for i in range(num_slices):
            p.extend(random.sample(nodes, 1))
            nodes.remove(p[-1])
            
        p_str = "_".join([str(_) for _ in p])
        if p_str not in cache:
            perms.append(p)
            cache.append(p_str)
            count += 1
    return perms


img_key = "img"
label_key = "label"
model_ckpt = "google/vit-base-patch16-224-in21k"


def dump_results(content):
    path = f"perm_results/{args.data_key}-{args.method}-budget-{args.budget}/"
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, f"{args.run_id}.json", "w")) as f:
        json.dump(content, f)


device = torch.device(f"cuda:{args.device_id}")

train_ds = load_dataset(args.data_key, split="train")
eval_ds = load_dataset(args.data_key, split="test")

labels = train_ds.features[label_key].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


# +
def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch[img_key]]
    return example_batch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# +
def get_subdirectory_name(dir_path):
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith("checkpoint")]
    return subdirectories[0]
# -

def load_model(path):    
    model = AutoModelForImageClassification.from_pretrained(
        path, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model


num_shards, num_slices = 5, 4

model = load_model(f"cifar10-vit-allperms/shard_0/perm_0/slice_3/checkpoint-790/")
eval_ds.set_transform(preprocess_val)

# +
metric_name = "accuracy"

training_args = TrainingArguments(
    output_dir=os.path.join("output/"),
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=image_processor,
    data_collator=collate_fn,
)
# -

eval_dl = trainer.get_eval_dataloader()


def get_acc(models):
    device_models = [model.to(device) for model in models]
    
    correct = 0
    count = 0
    for batch in tqdm(eval_dl):
        count += 1
        img = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        logits = torch.stack([model(img)['logits'] for model in device_models])
        combined_logits = torch.mean(logits, dim=0)
        preds = torch.argmax(combined_logits, dim=-1)
        correct += torch.sum(preds==labels)
    return correct / len(eval_ds)



def get_perm_dict():
    order = list(range(num_slices))
    all_perms = list(itertools.permutations(order))

    perm_dict = {}

    for j, perm in enumerate(all_perms):
        perm_s = [str(p) for p in perm]
        full_perm = "".join(perm_s)
        for i in range(len(perm)):
            prefix = "".join(perm_s[:i+1])
            slice_id = f"slice_{i}"
            perm_id = f"perm_{j}"
            perm_dict[f"{full_perm}_{prefix}"] = f"{perm_id}/{slice_id}"
    return perm_dict


perm_path_dict = get_perm_dict()


def get_model_inventory(budget=4):
    order = list(range(num_slices))
    all_perms = list(itertools.permutations(order))
    
    
    if args.method == 'sisa':
        all_perms = [all_perms[0]]
    
    if args.method == 's3t':
        if budget <= num_slices:
            all_perms = bms(num_slices)[:budget]
        
        

    model_inventory = defaultdict(list)
    for _ in range(num_shards):
        for perm in all_perms:
            perm_s = [str(p) for p in perm]
            prefix = "".join(perm_s)
            model_inventory[_].append(f"{prefix}_{prefix}")
    return model_inventory

model_inventory = get_model_inventory(args.budget)


def delete(shard_id, slice_id, model_inventory):
    modified_models = []
    for p in model_inventory[shard_id]:
        prefix, suffix = p.split("_")
        try:
            idx = suffix.index(str(slice_id))
            new_suffix = suffix[:idx]
            if len(new_suffix) > 0:
                modified_models.append(f"{prefix}_{new_suffix}")
        except:
            modified_models.append(f"{prefix}_{suffix}")
    model_inventory[shard_id] = modified_models


def check_empty(model_inventory):
    for _ in range(5):
        if len(model_inventory[_]) > 0:
            return False
    return True


def random_deletion():
    scores = []
    count = 0
    
    while not check_empty(model_inventory):
        delete_id = random.randint(0, num_shards*num_slices-1)
        shard_id = delete_id // num_slices
        slice_id = delete_id % num_slices
        delete(shard_id, slice_id, model_inventory)
        count += 1
        
        # get models
        curr_models = []
        for _ in range(num_shards):
            if len(model_inventory[_]) > 0:
                idx = np.argmax([len(p) for p in model_inventory[_]])
                path = os.path.join(
                    f"cifar10-vit-allperms/shard_{_}",
                    f"{perm_path_dict[model_inventory[_][idx]]}"
                )
                subpath = get_subdirectory_name(path)
                model_path = os.path.join(path, subpath)
                model = load_model(model_path)
                curr_models.append(model)
        
        # evaluate
        if len(curr_models) == 0:
            break
        score = get_acc(curr_models).detach().cpu().numpy()
        scores.append(float(score))
        print(scores)
    return scores


scores = random_deletion()

dump_results(scores)


