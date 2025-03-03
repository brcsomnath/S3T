# +
import os
import json
import torch
import argparse
import evaluate
import numpy as np

from datasets import load_dataset
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
parser.add_argument('--num_slices', 
                    type=int,
                    default=6,)
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
                    default=2e-4,)
parser.add_argument('--dataset', 
                    type=str,
                    default="cifar10",)
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

# +
start = 0
end = int((args.shard_id + 1) * 100/args.num_slices)

if args.dataset == 'tinyimagenet':
    train_ds = load_dataset(data_key, split=f"train")
    train_ds = train_ds.shuffle(seed=42)
    N = len(train_ds)
    train_ds = train_ds.select(range(int(N * (args.shard_id + 1)/args.num_slices)))
else:
    train_ds = load_dataset(data_key, split=f"train[{start}%:{end}%]")
eval_ds = load_dataset(data_key, split="valid" if args.dataset == 'tinyimagenet' else "test")
# -

labels = train_ds.features[label_key].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# ### Image Processing

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

train_ds.set_transform(preprocess_train)
eval_ds.set_transform(preprocess_val)


# ### Model

def dump_json(content, filename):
    with open(filename, "w") as f:
        json.dump(content, f)


def get_subdirectory_name(dir_path):
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(
        os.path.join(dir_path, d)) and d.startswith("checkpoint")]
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


def model_init(idx, model_dir, num_loras=2):
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

# +
num_loras = NUM_LAYERS[args.model_checkpoint] // args.num_slices
output_dir = f"./vit-iloras-{args.dataset}/models_loras_{num_loras}_rank_{args.rank}/"
model_dir = os.path.join(output_dir, f'model_shard_{str(args.shard_id-1)}')


model = model_init(
    idx=args.shard_id, 
    model_dir=model_dir, 
    num_loras=num_loras
)

# +
model_name = model_ckpt.split("/")[-1]
metric_name = "accuracy"

training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, f"model_shard_{args.shard_id}"),
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
)
# -

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# +

logs = (trainer.train(resume_from_checkpoint=False))

# +
metric_name = "eval_accuracy"
eval_acc = max([x[metric_name] for x in trainer.state.log_history if metric_name in x.keys()])
print(f"Shard ID: {args.shard_id}, {metric_name}: {eval_acc}")

path = f"vit_results/{args.dataset}-{args.model_checkpoint}/"
os.makedirs(path, exist_ok=True)
dump_json(eval_acc, os.path.join(path, f"{args.shard_id}.txt"))



