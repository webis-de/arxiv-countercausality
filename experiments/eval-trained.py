import os
import re

from datasets import load_dataset, Dataset
import evaluate
from evaluate import evaluator
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch

def load_task1_dataset() -> tuple[list[str], Dataset]:
    dataset = load_dataset("./datasets/countercausal-news-corpus", "causality detection")
    labels = dataset["train"].features["label"].names
    return labels, dataset

def load_task2_dataset(tokenizer) -> tuple[list[str], Dataset]:
    ####################
    # Same as in task2.py
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
    ####################

    dataset = load_dataset("./datasets/countercausal-news-corpus", "causal candidate extraction").map(process, batched=False, remove_columns=["entity", "index"])
    return labels, dataset

def load_task3_dataset() -> tuple[list[str], Dataset]:
    ####################
    # Same as in task3.py
    labels = ["no-rel", "causal", "countercausal"]
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
    ####################
    dataset = load_dataset("./datasets/countercausal-news-corpus", "causality identification").map(process, batched=True, remove_columns=["index", "text", "relations"])
    return labels, dataset

def get_classification_eval(path: str, data: tuple[list[str], Dataset]):
    tokenizer = AutoTokenizer.from_pretrained(path+"/"+os.listdir(path)[0])
    model = AutoModelForSequenceClassification.from_pretrained(path+"/"+os.listdir(path)[0])
    pipe = pipeline(model=model, tokenizer=tokenizer, task="text-classification")

    labels, dataset = data
    label2id={l: i for i, l in enumerate(labels)}

    task_evaluator = evaluator("text-classification")
    task_evaluator.METRIC_KWARGS = {"average": "macro"} # Workaround needed because of https://github.com/huggingface/evaluate/issues/423
    return task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset["dev"],
        label_mapping=label2id,
        metric=evaluate.combine([evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1")])
    )

def get_confusion_matrix(runname: str):
    path = f"./output/causality_task3_{runname}"
    assert len(os.listdir(path)) == 1

    tokenizer = AutoTokenizer.from_pretrained(path+"/"+os.listdir(path)[0])
    model = AutoModelForSequenceClassification.from_pretrained(path+"/"+os.listdir(path)[0])
    pipe = pipeline(model=model, tokenizer=tokenizer, task="text-classification", padding=True, truncation=True)

    labels, dataset = load_task3_dataset()
    label2id={l: i for i, l in enumerate(labels)}

    task_evaluator = evaluator("text-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset["dev"],
        label_mapping=label2id,
        metric=evaluate.load("confusion_matrix")
    )
    return pd.DataFrame(eval_results["confusion_matrix"], labels, labels)

print("DistilBERT trained on Countercausal News Corpus")
print(get_confusion_matrix("distilbert__distilbert-base-uncased_countercausal-news-corpus"))
print(get_classification_eval("./output/causality_task1_distilbert__distilbert-base-uncased", load_task1_dataset()))
print(get_classification_eval("./output/causality_task3_distilbert__distilbert-base-uncased_countercausal-news-corpus", load_task3_dataset()))
print("DistilBERT trained on Causal News Corpus v2")
print(get_confusion_matrix("distilbert__distilbert-base-uncased_causal-news-corpus-v2"))

print("RoBERTa trained on Countercausal News Corpus v2")
print(get_confusion_matrix("roberta-base_countercausal-news-corpus"))
print(get_classification_eval("./output/causality_task1_roberta-base", load_task1_dataset()))
print(get_classification_eval("./output/causality_task3_roberta-base_countercausal-news-corpus", load_task3_dataset()))
print("RoBERTa trained on Causal News Corpus v2")
print(get_confusion_matrix("roberta-base_causal-news-corpus-v2"))

print("Mistral Instruct trained on Countercausal News Corpus v2")
print(get_confusion_matrix("mistralai__Mistral-7B-Instruct-v0.3_countercausal-news-corpus"))
print(get_classification_eval("./output/causality_task1_mistralai__Mistral-7B-Instruct-v0.3", load_task1_dataset()))
print(get_classification_eval("./output/causality_task3_mistralai__Mistral-7B-Instruct-v0.3_countercausal-news-corpus", load_task3_dataset()))
print("Mistral Instruct trained on Causal News Corpus v2")
print(get_confusion_matrix("mistralai__Mistral-7B-Instruct-v0.3_causal-news-corpus-v2"))