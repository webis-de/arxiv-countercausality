# Investigating Counterclaims in Causality Extraction from Text

<p align="center" id="abstract">
	<b>Abstract</b>
</p>
<center>
	<p align="justify" style="max-width: 20cm;">
        Many causal claims, such as "sugar causes hyperactivity," are disputed or outdated. Yet research on causality extraction from text has almost entirely neglected counterclaims of causation. To close this gap, we conduct a thorough literature review of causality extraction, compile an extensive inventory of linguistic realizations of countercausal claims, and develop rigorous annotation guidelines that explicitly incorporate countercausal language. We also highlight how counterclaims of causation are an integral part of causal reasoning. Based on our guidelines, we construct a new dataset comprising 1028 causal claims, 952 counterclaims, and 1435 uncausal statements, achieving substantial inter-annotator agreement (Cohen's κ=0.74). In our experiments, state-of-the-art models trained solely on causal claims misclassify counterclaims more than 10 times as often as models trained on our dataset.
	</p>
</center>
<br/>

> [!TIP]
> This repository has a [Dev Container](https://containers.dev/overview) configuration.

# The Dataset
You can finde the dataset in `dataset`. Each subfolder corresponds to one of the three tasks: causality detection, causal candidate extraction, and causality identification. Each task contains a `.jsonl`-file 

## Format
### Causality detection
Each line of the file contains a JSON object which has the following fields:
- `label`: An integer that is `0` if the text is uncausal and `1` otherwise.
- `text`: A string that should be classified.

**Example:**
```json
{"label":0,"text":"The union also holds the Shahjahanpur toll plaza ."}
```

### Causal candidate extraction
Each line of the file contains a JSON object which has the following fields:
- `text`: A string containing the text in which entity spans should be marked.
- `entity`: A list of pairs of integer: the first integer marks the index in `text` at which the entity span starts and the second integer marks where it ends.

**Example:**
```json
{"text":"`` There were demonstrations outside , but the meeting of the PEC continued , '' he said .","entity":[[14,36],[43,75]]}
```

### Causality identification
Each line of the file contains a JSON object which has the following fields:
- `text`: A string with marked entity spans (`<e1>...</e1>`, `<e2>...</e2>`, ...).
- `relations`: A list of relationships, where each entry has the following fields:
  - `first`: A string representing the first entity as labeled in the `text` that parttakes in the relationship (`e1`, `e2`, ...).
  - `second`: A string representing the second entity as labeled in the `text` that parttakes in the relationship (`e1`, `e2`, ...).
  - `relationship`: An integer indicating the type of the relationship: `1` for causal and `2` for countercausal.

**Example:**
```json
{"text":"The company said <e1>negotiations were continuing between management and NUM officials<\/e1> with in a bid <e2>to bring an end to the strike<\/e2> .","relations":[{"first":"e2","relationship":1,"second":"e1"}]}
```

## Loading the Dataset
### Using HF Datasets
The Countercausal News Corpus is stored to be Hugging Face Datasets compatible. This means that you can simply load it using

```py
from datasets import load_dataset

dset = load_dataset("./dataset", "<task>")
```

from inside the repo root, where `"<task>"` is one of the supported tasks: `"causality detection"`, `"causal candidate extraction"`, and `"causality identification"`.

### Using Pandas
Altneratively you can load the jsonl files directly through
```py
import pandas as pd

df = pd.read_json("./dataset/<task>/<split>.jsonl", lines=True)
```
from inside the repo root, where `<task>` is one of the supported tasks: `"causality detection"`, `"causal candidate extraction"`, and `"causality identification"`, and `<split>` is one of `train` or `dev`.
