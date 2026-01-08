import os

for v in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(v, None)

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from tirex_tracker import tracking
import torch

def train(modelname: str, bf16: bool=False, batchsize: int=256, grad_accumulation=1):
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("./datasets/countercausal-news-corpus", "causality detection").map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    print("Example entry of the dataset:")
    print(dataset["train"][0])

    labels = dataset["train"].features["label"].names
    id2label={i: l for i, l in enumerate(labels)}
    label2id={l: i for i, l in enumerate(labels)}

    def create_metrics(labels: list):
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        metrics = evaluate.combine([precision, recall, f1])

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metrics.compute(predictions=predictions, references=labels)

        return compute_metrics

    model_task1 = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=len(labels), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16 if bf16 else "auto")

    args = TrainingArguments(
        output_dir=f"output/causality_task1_{modelname.replace('/', '__')}",
        learning_rate=2e-5,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        gradient_accumulation_steps=grad_accumulation,
        num_train_epochs=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, # Save only the best model
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",
        metric_for_best_model="f1",
        logging_steps=50,
        bf16=bf16,
    )
    trainer = Trainer(
        model=model_task1,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=create_metrics(labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    with tracking(export_file_path=f"output/task1-{modelname.replace('/', '__')}.irmetadata.yaml") as result:
        trainer.train()

train("distilbert/distilbert-base-uncased")
train("roberta-base")
train("mistralai/Mistral-7B-Instruct-v0.3", bf16=True, batchsize=16, grad_accumulation=16)