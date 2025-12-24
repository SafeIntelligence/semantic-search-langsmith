from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

# Download the dataset (only downloads if not already present)
dataset = "fiqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "./beir_data")

# Load the dataset
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Number of documents in corpus: {len(corpus)}")
print(f"Number of queries: {len(queries)}")
print(f"Number of qrels: {len(qrels)}")

print("\nSample document:")
print(next(iter(corpus.values())))
print("\nSample query:")
print(next(iter(queries.values())))
print("\nSample qrel:")
print(next(iter(qrels.items())))