import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv("annotation_round1.csv", index_col="index")
print(cohen_kappa_score(list(df["annot1"]), list(df["annot2"]), labels=["Uncausal", "Causal", "Countercausal"]))

df = pd.read_csv("annotation_round2.csv", index_col="index")
print(cohen_kappa_score(list(df["annot1"]), list(df["annot2"]), labels=["Uncausal", "Causal", "Countercausal"]))