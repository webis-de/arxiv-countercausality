import os

for v in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(v, None)

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from tirex_tracker import tracking
import torch

def train(modelname: str, bf16: bool=False, batchsize: int=96):
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    labels = ["B-Entity", "I-Entity", "O"]
    id2label={i: l for i, l in enumerate(labels)}
    label2id={l: i for i, l in enumerate(labels)}

    def process(x):
        toks = tokenizer(x["text"], truncation=True, return_offsets_mapping=True)
        labels = [label2id["O"]]*len(toks["offset_mapping"])
        for i, (tok_start, tok_end) in enumerate(toks["offset_mapping"]):
            if tok_start == tok_end:
                labels[i] = -100 # Don't classify special tokens
                continue
            for ent_start, ent_end in x["entity"]:
                if tok_start >= ent_start and tok_end <= ent_end:
                    if tok_start == ent_start:
                        labels[i] = label2id["B-Entity"]
                    else:
                        labels[i] = label2id["I-Entity"]

        return {"labels": labels, "input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

    dataset = load_dataset("./datasets/countercausal-news-corpus", "causal candidate extraction").map(process, batched=False, remove_columns=["entity", "index"]) #.shuffle()
    print("Example entry of the dataset:")
    print(dataset["train"][0])

    model_task2 = AutoModelForTokenClassification.from_pretrained(modelname, num_labels=len(labels), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16 if bf16 else "auto")
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, lbls = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, lbls)
        ]
        true_labels = [
            [labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, lbls)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    args = TrainingArguments(
        output_dir=f"output/causality_task2_{modelname.replace('/', '__')}",
        learning_rate=2e-5,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
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
        model=model_task2,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    with tracking(export_file_path=f"output/task2-{modelname.replace('/', '__')}.irmetadata.yaml") as result:
        trainer.train()

train("distilbert/distilbert-base-uncased")
train("roberta-base")