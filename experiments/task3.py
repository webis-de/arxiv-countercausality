import os

for v in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(v, None)

import re

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from tirex_tracker import tracking
import torch

def train(modelname: str, bf16: bool=False, batchsize: int=256, grad_accumulation=1, dsetname: str="countercausal-news-corpus"):
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenizer.add_tokens(['<e0>','</e0>','<e1>','</e1>'], special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    labels = ["no-rel", "causal", "countercausal"]
    id2label={i: l for i, l in enumerate(labels)}
    label2id={l: i for i, l in enumerate(labels)}

    def process(row):
        texts: list[str] = []
        labels: list[int] = []
        for i in range(len(row["index"])):
            entities = re.findall(r"<(e\d+)>", row["text"][i])
            for e1 in entities:
                for e2 in entities:
                    if e1 == e2:
                        continue
                    label = label2id["no-rel"]
                    for r in row["relations"][i]:
                        if r['first'] == e1 and r['second'] == e2:
                            label = r['relationship']
                            break
                    # Remove all entities except for e1 and e2
                    regex = "|".join(f"</?{e}>" for e in entities if e not in (e1, e2))
                    text = re.sub(regex, "", row["text"][i])
                    text = re.sub(f"<(/?){e1}>", r"<\1e0>", text)
                    text = re.sub(f"<(/?){e2}>", r"<\1e1>", text)
                    texts.append(text)
                    labels.append(label)
        return {"text": texts, "label": labels}

    dataset = load_dataset(f"./datasets/{dsetname}", "causality identification").map(process, batched=True, remove_columns=["index", "text", "relations"]).map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    print("Example entries of the dataset:")
    print(dataset["train"][0])
    print(dataset["train"][1])
    print(dataset["train"][2])
    print(dataset["train"][3])
    print(dataset["train"][4])
    print(dataset["train"][5])

    model_task3 = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=len(labels), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16 if bf16 else "auto")
    model_task3.resize_token_embeddings(len(tokenizer))

    def create_metrics(labels: list):
        precision = evaluate.load("precision")#, average="macro", labels=labels)
        recall = evaluate.load("recall")#, average="macro", labels=labels)
        f1 = evaluate.load("f1")#, average="macro", labels=labels)
        metrics = evaluate.combine([precision, recall, f1])

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            print(predictions)
            predictions = np.argmax(predictions, axis=-1)
            print(predictions)
            print(labels)
            return metrics.compute(predictions=predictions, references=labels, average="macro")

        return compute_metrics

    args = TrainingArguments(
        output_dir=f"output/causality_task3_{modelname.replace('/', '__')}_{dsetname.replace('/', '__')}",
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
        model=model_task3,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=create_metrics(labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    with tracking(export_file_path=f"output/task3-{modelname.replace('/', '__')}_{dsetname}.irmetadata.yaml") as result:
        trainer.train()


train("distilbert/distilbert-base-uncased")
train("roberta-base")
train("mistralai/Mistral-7B-Instruct-v0.3", bf16=True, batchsize=16, grad_accumulation=16)

train("distilbert/distilbert-base-uncased", dsetname="causal-news-corpus-v2")
train("roberta-base", dsetname="causal-news-corpus-v2")
train("mistralai/Mistral-7B-Instruct-v0.3", bf16=True, batchsize=16, grad_accumulation=16, dsetname="causal-news-corpus-v2")
