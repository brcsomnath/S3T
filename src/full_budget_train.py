# +
import os
import json
import torch
import argparse
import evaluate
import itertools
import numpy as np

from datasets import load_dataset
from datasets import concatenate_datasets
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import (
    AutoModelForImageClassification, 
    TrainingArguments, 
    AutoImageProcessor,
    Trainer
)
from peft import get_peft_model, LoraConfig


# -

parser = argparse.ArgumentParser()
parser.add_argument('--num_shards', 
                    type=int,
                    default=5,)
parser.add_argument('--num_slices', 
                    type=int,
                    default=4,)
parser.add_argument('--shard_id', 
                    type=int,
                    default=0,)
parser.add_argument('--rank', 
                    type=int,
                    default=16,)
parser.add_argument('--model_checkpoint', 
                    type=str,
                    default="vit-base",)
parser.add_argument('--lr', 
                    type=float,
                    default=2e-3,)
parser.add_argument('--dataset', 
                    type=str,
                    default="cifar100",)
parser.add_argument('--epochs', 
                    type=int,
                    default=15,)
parser.add_argument('--batch_size', 
                    type=int,
                    default=128,)
args = parser.parse_args()

# +
MODEL_KEYS = {
    'vit-base': "google/vit-base-patch16-224-in21k",
    'vit-large': "google/vit-large-patch16-224-in21k",
}
DATASET_KEYS = {
    "cifar10": "cifar10", 
    "cifar100": "cifar100", 
    "imagenet-1k": "imagenet-1k", 
    "tinyimagenet": "zh-plus/tiny-imagenet"
}
IMG_KEYS = {
    "cifar10": "img", 
    "cifar100": "img", 
    "imagenet-1k": "image", 
    "tinyimagenet": "image"
}
LABEL_KEYS = {
    "cifar10": "label",
    "cifar100": "fine_label",
    "imagenet-1k": "label", 
    "tinyimagenet": "label"
}
NUM_LAYERS = {
    'vit-base': 12,
    'vit-large': 24,
}

img_key = IMG_KEYS[args.dataset]
label_key = LABEL_KEYS[args.dataset]
data_key = DATASET_KEYS[args.dataset]
model_ckpt = MODEL_KEYS[args.model_checkpoint]
# -

image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

# +
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


# +
def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch[img_key]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch[img_key]]
    return example_batch


# -

def dump_json(content, filename):
    with open(filename, "w") as f:
        json.dump(content, f)


def get_subdirectory_name(dir_path):
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith("checkpoint")]
    return subdirectories[0]


def get_layers(idx, num_loras):
    layers = ["classifier"]
    for i in range(num_loras):
        layer_id = str(NUM_LAYERS[args.model_checkpoint]-1 - (idx*num_loras + i))
        layers.append(f"layer.{layer_id}")
    return layers


def check_if(name, layer_ids):
    for idx in layer_ids:
        if idx in name:
            return True
    return False


def model_init(idx, model_dir, label2id, id2label, lora_config, num_loras=3):
    layer_ids = get_layers(idx, num_loras)
    print(layer_ids)

    if idx > 0:
        subpath = get_subdirectory_name(model_dir)
        print(f"Loading model from {model_dir}/{subpath}")
        model = AutoModelForImageClassification.from_pretrained(
            f'{model_dir}/{subpath}', 
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        print("Initializing model!")
        model = AutoModelForImageClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )


    peft_model = get_peft_model(model, lora_config)
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.requires_grad = check_if(name, layer_ids)
    return peft_model

metric = evaluate.load("accuracy")


# +
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# -

def train_model(perm_id, slice_id, train_ds, eval_ds, label2id, id2label):
    trainable_modules = ["query", "value"]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=trainable_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
        use_rslora=True,
    )
    
    num_loras = NUM_LAYERS[args.model_checkpoint] // args.num_slices
    output_dir = f"./{args.dataset}-vit-all_perms/shard_{args.shard_id}/perm_{perm_id}/"
    model_dir = os.path.join(output_dir, f'slice_{str(slice_id-1)}')


    model = model_init(
        idx=slice_id, 
        model_dir=model_dir, 
        label2id=label2id,
        id2label=id2label,
        lora_config=lora_config,
        num_loras=num_loras,
    )
    
    model_name = model_ckpt.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"slice_{slice_id}"),
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=25,
        save_total_limit=2,
        label_names=["labels"],
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    logs = (trainer.train(resume_from_checkpoint=False))
    logs = trainer.evaluate()
    
    metric_name = "eval_accuracy"
    eval_acc = max([x[metric_name] for x in trainer.state.log_history if metric_name in x.keys()])
    print(f"Slice ID: {args.shard_id}, {metric_name}: {eval_acc}")


def main():
    eval_ds = load_dataset(data_key, split="test")
    labels = eval_ds.features[label_key].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    start = int(args.shard_id * 100/args.num_shards)
    end = int((args.shard_id + 1) * 100/args.num_shards)
    shard_train_ds = load_dataset(data_key, split=f"train[{start}%:{end}%]")
    
    slice_size = len(shard_train_ds)//args.num_slices
    
    slice_datasets = []
    for i in range(args.num_slices):
        indices = list(range(i*slice_size, (i+1)*slice_size))
        slice_ds = shard_train_ds.select(indices)
        slice_datasets.append(slice_ds)
    
    order = list(range(args.num_slices))
    all_perms = list(itertools.permutations(order))
    
    eval_ds.set_transform(preprocess_val)
    for perm_id, perm in enumerate(all_perms):
        print(perm)
        ordered_ds = []
        for p in perm:
            ordered_ds.append(slice_datasets[p])

        for slice_id in range(args.num_slices):
            print(f"perm_id: {perm_id} slice_id: {slice_id}")
            curr_train_ds = concatenate_datasets(ordered_ds[:slice_id+1])
            curr_train_ds.set_transform(preprocess_train)
            train_model(perm_id, slice_id, curr_train_ds, eval_ds, label2id, id2label)


if __name__ == '__main__':
    main()


